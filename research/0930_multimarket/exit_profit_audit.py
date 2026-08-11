from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest_engine import (
    EntrySignal,
    PreparedMarket,
    ReferenceEvent,
    StrategyConfig,
    compute_stop,
    find_entry,
    simulate_period,
)
from data_pipeline import load_all_markets

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "exit_audit_work"
OUT = ROOT / "exit_audit_output"
OUT.mkdir(parents=True, exist_ok=True)
PERIODS = ["development", "validation", "holdout"]


@dataclass(frozen=True)
class ExitPolicy:
    policy_id: str
    description: str
    scale_kind: str = "none"
    scale_value: float = 0.0
    exit_kind: str = "target_r"
    exit_value: float = 3.0
    ema_span: int = 80
    activation_r: float = 0.0
    break_even_r: float = math.nan
    max_hold_days: int = 60


@dataclass
class PolicyFeatures:
    ema: Dict[int, np.ndarray]


def entry_exec(raw: float, cost: float) -> float:
    return float(raw) * (1.0 + cost)


def net_exit(raw: float, cost: float) -> float:
    return float(raw) * (1.0 - cost)


def build_features(mkt: PreparedMarket) -> PolicyFeatures:
    close = mkt.df["close"]
    return PolicyFeatures(
        ema={
            span: close.ewm(span=span, adjust=False, min_periods=span).mean().to_numpy(dtype=float)
            for span in [20, 50, 80, 160, 320]
        }
    )


def policy_catalog() -> List[ExitPolicy]:
    p: List[ExitPolicy] = [
        ExitPolicy(
            "CORRECTED_BASELINE",
            "Half after the first close below EMA20; remainder exits at 3R or a lagged EMA80 touch",
            scale_kind="half_below_ema20",
            exit_kind="target_or_ma_touch_lagged",
            exit_value=3.0,
            ema_span=80,
        ),
        ExitPolicy(
            "HALF_EMA20_REST_EMA80_CLOSE",
            "Half after close below EMA20; remainder exits next open after close below EMA80",
            scale_kind="half_below_ema20",
            exit_kind="ma_close",
            ema_span=80,
        ),
    ]
    for target in [2, 3, 5, 10, 20]:
        p.append(ExitPolicy(f"FULL_TARGET_{target}R", f"No partial; original stop or full exit at +{target}R", exit_kind="target_r", exit_value=float(target)))
    for pct in [0.03, 0.05, 0.10, 0.15, 0.20, 0.30]:
        p.append(ExitPolicy(f"FULL_TARGET_{pct*100:g}PCT", f"No partial; original stop or full exit at +{pct*100:g}%", exit_kind="target_pct", exit_value=pct, max_hold_days=120))
    for span in [20, 50, 80, 160, 320]:
        p.append(ExitPolicy(f"FULL_CLOSE_BELOW_EMA{span}", f"No partial; full next-open exit after close below EMA{span}", exit_kind="ma_close", ema_span=span, max_hold_days=120))
    for span in [80, 160, 320]:
        p.append(ExitPolicy(f"HALF_EMA20_REST_EMA{span}", f"Half below EMA20; remainder next-open exit below EMA{span}", scale_kind="half_below_ema20", exit_kind="ma_close", ema_span=span, max_hold_days=120))
    for target in [5, 10, 20]:
        p.append(ExitPolicy(f"HALF_EMA20_REST_{target}R", f"Half below EMA20; remainder held for +{target}R", scale_kind="half_below_ema20", exit_kind="target_r", exit_value=float(target), max_hold_days=120))
    for pct in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
        p.append(ExitPolicy(f"TRAIL_{pct*100:g}PCT_AFTER_2R", f"No partial; {pct*100:g}% trail activated after +2R", exit_kind="pct_trail", exit_value=pct, activation_r=2.0, max_hold_days=120))
    for multiple in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]:
        p.append(ExitPolicy(f"TRAIL_{multiple:g}ATR_AFTER_2R", f"No partial; {multiple:g} ATR trail activated after +2R", exit_kind="atr_trail", exit_value=multiple, activation_r=2.0, max_hold_days=120))
    for lookback in [1, 3, 5, 10]:
        p.append(ExitPolicy(f"TRAIL_LOW_{lookback}B_AFTER_2R", f"No partial; trail under lowest low of {lookback} completed bars after +2R", exit_kind="low_trail", exit_value=float(lookback), activation_r=2.0, max_hold_days=120))
    for trigger in [0.5, 1.0, 1.5, 2.0]:
        for target in [5.0, 10.0]:
            p.append(ExitPolicy(f"BE_{trigger:g}R_TARGET_{target:g}R", f"Break-even after +{trigger:g}R; full target +{target:g}R", exit_kind="target_r", exit_value=target, break_even_r=trigger, max_hold_days=120))
    for days in [1, 3, 5, 10, 20, 60, 120]:
        p.append(ExitPolicy(f"STOP_OR_HOLD_{days}D", f"Original stop or {days}-day time exit, no profit target", exit_kind="time_only", max_hold_days=days))
    for trigger in [2.0, 3.0, 5.0]:
        p.append(ExitPolicy(f"HALF_{trigger:g}R_REST_3ATR", f"Sell half at +{trigger:g}R; trail remainder by 3 ATR", scale_kind="half_at_r", scale_value=trigger, exit_kind="atr_trail", exit_value=3.0, activation_r=trigger, max_hold_days=120))
    for trigger_pct in [0.10, 0.20, 0.30]:
        p.append(ExitPolicy(f"HALF_{trigger_pct*100:g}PCT_REST_5PCT_TRAIL", f"Sell half at +{trigger_pct*100:g}%; trail remainder by 5%", scale_kind="half_at_pct", scale_value=trigger_pct, exit_kind="pct_trail", exit_value=0.05, max_hold_days=180))
    unique: Dict[str, ExitPolicy] = {}
    for item in p:
        unique[item.policy_id] = item
    return list(unique.values())


def target_raw(signal: EntrySignal, stop_raw: float, policy: ExitPolicy) -> Optional[float]:
    if policy.exit_kind in {"target_r", "target_or_ma_touch_lagged"}:
        return signal.raw_entry + policy.exit_value * (signal.raw_entry - stop_raw)
    if policy.exit_kind == "target_pct":
        return signal.raw_entry * (1.0 + policy.exit_value)
    return None


def simulate_policy_trade(
    mkt: PreparedMarket,
    features: PolicyFeatures,
    event: ReferenceEvent,
    signal: EntrySignal,
    cfg: StrategyConfig,
    policy: ExitPolicy,
    equity: float,
    period_end: int,
) -> Optional[dict]:
    entry_i = signal.entry_idx
    if entry_i > period_end:
        return None
    stop_raw = compute_stop(mkt, signal, cfg)
    if not (np.isfinite(stop_raw) and stop_raw > 0 and stop_raw < signal.raw_entry):
        return None
    cost = mkt.spec.base_cost_rate
    entry_price = entry_exec(signal.raw_entry, cost)
    risk_per_unit = entry_price - net_exit(stop_raw, cost)
    if not (risk_per_unit > 0 and np.isfinite(risk_per_unit)):
        return None
    risk_budget = equity * 0.01
    qty = risk_budget / risk_per_unit
    if qty <= 0:
        return None
    planned_risk = qty * risk_per_unit
    risk_raw = signal.raw_entry - stop_raw

    remaining = qty
    realized = 0.0
    fills: List[Tuple[int, float, float, str]] = []
    scheduled_amount = 0.0
    scheduled_reason: Optional[str] = None
    scaled = False
    highest = signal.raw_entry
    active_stop = stop_raw
    entry_ts = mkt.index[entry_i]
    last_i = entry_i
    exit_reason = "period_end"
    mfe_raw = signal.raw_entry
    mae_raw = signal.raw_entry
    fixed_target = target_raw(signal, stop_raw, policy)
    start_i = entry_i + 1 if signal.route.endswith("_close") else entry_i

    for i in range(start_i, period_end + 1):
        last_i = i
        o, h, l, c = mkt.open[i], mkt.high[i], mkt.low[i], mkt.close[i]
        mfe_raw = max(mfe_raw, h)
        mae_raw = min(mae_raw, l)

        if o <= active_stop:
            px = net_exit(o, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, "gap_stop"))
            remaining = 0.0
            exit_reason = "gap_stop"
            break

        if scheduled_reason is not None and remaining > 0:
            amount = min(remaining, scheduled_amount if scheduled_amount > 0 else remaining)
            px = net_exit(o, cost)
            realized += amount * (px - entry_price)
            fills.append((i, amount, px, scheduled_reason))
            remaining -= amount
            if amount < qty - 1e-12:
                scaled = True
            reason = scheduled_reason
            scheduled_reason = None
            scheduled_amount = 0.0
            if remaining <= 1e-12:
                remaining = 0.0
                exit_reason = reason
                break

        if l <= active_stop:
            px = net_exit(active_stop, cost)
            realized += remaining * (px - entry_price)
            managed = active_stop > stop_raw + 1e-12
            fills.append((i, remaining, px, "managed_stop" if managed else "stop"))
            remaining = 0.0
            exit_reason = "managed_stop" if managed else "stop"
            break

        if not scaled and policy.scale_kind == "half_at_r":
            level = signal.raw_entry + policy.scale_value * risk_raw
            if h >= level:
                amount = remaining * 0.5
                px = net_exit(level, cost)
                realized += amount * (px - entry_price)
                fills.append((i, amount, px, f"half_{policy.scale_value:g}R"))
                remaining -= amount
                scaled = True
        elif not scaled and policy.scale_kind == "half_at_pct":
            level = signal.raw_entry * (1.0 + policy.scale_value)
            if h >= level:
                amount = remaining * 0.5
                px = net_exit(level, cost)
                realized += amount * (px - entry_price)
                fills.append((i, amount, px, f"half_{policy.scale_value*100:g}pct"))
                remaining -= amount
                scaled = True

        if fixed_target is not None and remaining > 0 and h >= fixed_target:
            px = net_exit(fixed_target, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, "target"))
            remaining = 0.0
            exit_reason = "target"
            break

        if policy.exit_kind in {"ma_touch_lagged", "target_or_ma_touch_lagged"} and remaining > 0:
            prev_ma = features.ema[policy.ema_span][i - 1] if i > 0 else np.nan
            can_exit = policy.exit_kind == "ma_touch_lagged" or scaled
            if can_exit and np.isfinite(prev_ma) and l <= prev_ma and prev_ma > active_stop:
                px = net_exit(prev_ma, cost)
                realized += remaining * (px - entry_price)
                fills.append((i, remaining, px, f"lagged_ema{policy.ema_span}_touch"))
                remaining = 0.0
                exit_reason = f"lagged_ema{policy.ema_span}_touch"
                break

        ema20 = features.ema[20][i]
        if not scaled and policy.scale_kind in {"half_below_ema20", "full_below_ema20"} and np.isfinite(ema20) and c < ema20 and i < period_end:
            scheduled_reason = policy.scale_kind
            scheduled_amount = remaining if policy.scale_kind == "full_below_ema20" else remaining * 0.5

        if policy.exit_kind == "ma_close" and remaining > 0 and i < period_end:
            ema = features.ema[policy.ema_span][i]
            can_exit = policy.scale_kind == "none" or scaled
            if can_exit and np.isfinite(ema) and c < ema:
                scheduled_reason = f"close_below_ema{policy.ema_span}"
                scheduled_amount = remaining

        highest = max(highest, h)
        if np.isfinite(policy.break_even_r) and highest >= signal.raw_entry + policy.break_even_r * risk_raw:
            active_stop = max(active_stop, signal.raw_entry)
        activated = highest >= signal.raw_entry + policy.activation_r * risk_raw
        if policy.scale_kind in {"half_at_r", "half_at_pct"} and not scaled:
            activated = False
        if activated and policy.exit_kind == "pct_trail":
            active_stop = max(active_stop, highest * (1.0 - policy.exit_value))
        elif activated and policy.exit_kind == "atr_trail":
            atr = mkt.atr14[i]
            if np.isfinite(atr):
                active_stop = max(active_stop, highest - policy.exit_value * atr)
        elif activated and policy.exit_kind == "low_trail":
            lookback = int(policy.exit_value)
            left = max(entry_i, i - lookback + 1)
            active_stop = max(active_stop, float(np.nanmin(mkt.low[left : i + 1])))

        if mkt.index[i] >= entry_ts + pd.Timedelta(days=policy.max_hold_days):
            if i < period_end:
                scheduled_reason = "time_exit"
                scheduled_amount = remaining
            else:
                px = net_exit(c, cost)
                realized += remaining * (px - entry_price)
                fills.append((i, remaining, px, "time_exit_close"))
                remaining = 0.0
                exit_reason = "time_exit_close"
                break

    if remaining > 0:
        i = min(last_i, period_end)
        px = net_exit(mkt.close[i], cost)
        realized += remaining * (px - entry_price)
        fills.append((i, remaining, px, "period_end"))
        remaining = 0.0
        exit_reason = "period_end"
        last_i = i

    mfe_r = (mfe_raw - signal.raw_entry) / risk_raw
    return {
        "market": mkt.name,
        "policy_id": policy.policy_id,
        "entry_route": signal.route,
        "entry_time": mkt.index[entry_i].isoformat(),
        "exit_time": mkt.index[last_i].isoformat(),
        "raw_entry": signal.raw_entry,
        "entry_exec": entry_price,
        "stop_raw": stop_raw,
        "quantity": qty,
        "planned_risk_pct": planned_risk / equity,
        "net_pnl": realized,
        "r_multiple": realized / planned_risk,
        "weighted_exit_exec": sum(amount * px for _, amount, px, _ in fills) / qty,
        "exit_reason": exit_reason,
        "fill_count": len(fills),
        "holding_hours": (mkt.index[last_i] - mkt.index[entry_i]).total_seconds() / 3600,
        "equity_before": equity,
        "equity_after": equity + realized,
        "mfe_r_to_exit": mfe_r,
        "mae_r_to_exit": (signal.raw_entry - mae_raw) / risk_raw,
        "capture_ratio": (realized / planned_risk) / mfe_r if mfe_r > 0 and realized > 0 else np.nan,
        "fills": json.dumps([{"time": mkt.index[idx].isoformat(), "amount": amount, "price": px, "reason": reason} for idx, amount, px, reason in fills]),
    }


def simulate_policy_period(mkt: PreparedMarket, features: PolicyFeatures, entry_cfg: StrategyConfig, policy: ExitPolicy, split: Dict[str, int], period: str) -> Tuple[dict, List[dict]]:
    start, end = mkt.period_bounds(split, period)
    if start < 0:
        return {}, []
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    trades: List[dict] = []
    last_exit = start - 1
    for event in mkt.generate_events(entry_cfg):
        if event.ref_idx < start or event.ref_idx > end or event.ref_idx <= last_exit:
            continue
        signal = find_entry(mkt, event, entry_cfg, end)
        if signal is None or signal.entry_idx < start or signal.entry_idx <= last_exit:
            continue
        trade = simulate_policy_trade(mkt, features, event, signal, entry_cfg, policy, equity, end)
        if trade is None:
            continue
        trade["period"] = period
        trades.append(trade)
        equity = float(trade["equity_after"])
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        last_exit = int(mkt.index.searchsorted(pd.Timestamp(trade["exit_time"]), side="left"))
    if not trades:
        return {"market": mkt.name, "policy_id": policy.policy_id, "period": period, "trades": 0, "total_return": 0.0, "win_rate": np.nan, "profit_factor": np.nan, "mean_r": np.nan, "median_r": np.nan, "max_drawdown": 0.0, "avg_holding_hours": np.nan, "ending_equity": 10000.0}, []
    pnl = np.asarray([t["net_pnl"] for t in trades], dtype=float)
    rs = np.asarray([t["r_multiple"] for t in trades], dtype=float)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return {"market": mkt.name, "policy_id": policy.policy_id, "period": period, "trades": len(trades), "total_return": equity / 10000.0 - 1.0, "win_rate": float((pnl > 0).mean()), "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else np.nan), "mean_r": float(rs.mean()), "median_r": float(np.median(rs)), "max_drawdown": float(max_dd), "avg_holding_hours": float(np.mean([t["holding_hours"] for t in trades])), "ending_equity": equity}, trades


def aggregate_policy_metrics(market_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy_id, period), group in market_metrics.groupby(["policy_id", "period"], sort=False):
        weights = group["trades"].to_numpy(dtype=float)
        values = group["mean_r"].fillna(0).to_numpy(dtype=float)
        total = int(weights.sum())
        rows.append({"policy_id": policy_id, "period": period, "trades": total, "equal_weight_return": float(group["total_return"].mean()), "positive_markets": int((group["total_return"] > 0).sum()), "trade_weighted_mean_r": float(np.average(values, weights=weights)) if total else np.nan, "worst_market_return": float(group["total_return"].min()), "best_market_return": float(group["total_return"].max()), "max_market_drawdown": float(group["max_drawdown"].max()), "avg_holding_hours": float(np.average(group["avg_holding_hours"].fillna(0), weights=weights)) if total else np.nan})
    return pd.DataFrame(rows)


def current_trade_diagnostics(mkt: PreparedMarket, trade: dict, period_end: int) -> dict:
    entry_ts = pd.Timestamp(trade["entry_time"])
    exit_ts = pd.Timestamp(trade["exit_time"])
    entry_i = int(mkt.index.searchsorted(entry_ts, side="left"))
    exit_i = int(mkt.index.searchsorted(exit_ts, side="left"))
    entry = float(trade["raw_entry"])
    stop = float(trade["stop_raw"])
    risk = entry - stop
    high_to_exit = float(np.nanmax(mkt.high[entry_i : exit_i + 1]))
    low_to_exit = float(np.nanmin(mkt.low[entry_i : exit_i + 1]))
    result = dict(trade)
    mfe_r = (high_to_exit - entry) / risk
    result.update({"mfe_r_to_exit": mfe_r, "mae_r_to_exit": (entry - low_to_exit) / risk, "capture_ratio": float(trade["r_multiple"]) / mfe_r if mfe_r > 0 and float(trade["r_multiple"]) > 0 else np.nan})
    for days in [1, 3, 5, 10, 20, 60]:
        end_i = min(int(mkt.index.searchsorted(exit_ts + pd.Timedelta(days=days), side="right") - 1), period_end)
        if end_i <= exit_i:
            result[f"post_exit_max_r_{days}d"] = np.nan
            result[f"post_exit_max_pct_{days}d"] = np.nan
        else:
            max_high = float(np.nanmax(mkt.high[exit_i + 1 : end_i + 1]))
            result[f"post_exit_max_r_{days}d"] = (max_high - entry) / risk
            result[f"post_exit_max_pct_{days}d"] = max_high / entry - 1.0
    targets = {"3r": entry + 3 * risk, "5r": entry + 5 * risk, "10r": entry + 10 * risk, "20r": entry + 20 * risk}
    reached = {key: False for key in targets}
    stop_seen = False
    end_i = min(int(mkt.index.searchsorted(exit_ts + pd.Timedelta(days=60), side="right") - 1), period_end)
    for i in range(exit_i + 1, end_i + 1):
        o, h, l = mkt.open[i], mkt.high[i], mkt.low[i]
        if o <= stop or l <= stop:
            stop_seen = True
            break
        for key, level in targets.items():
            if not reached[key] and (o >= level or h >= level):
                reached[key] = True
    result["original_stop_hit_within_60d_after_exit"] = stop_seen
    for key, value in reached.items():
        result[f"reached_{key}_before_stop_after_exit"] = value
    return result


def code_audit_notes() -> List[dict]:
    return [
        {"severity": "material", "issue": "Current MA80 exit calculates MA80 with the current bar close, then assumes an intrabar fill at that MA80.", "effect": "The level was not known when the alleged touch occurred. The corrected baseline uses the previous completed bar's MA80."},
        {"severity": "material for ATR variants", "issue": "Current ATR trail uses the current bar high to raise the trail and the current bar low to trigger that newly raised trail.", "effect": "The high may have occurred after the low. Corrected trails take effect on the next bar."},
        {"severity": "moderate", "issue": "The bearish-cross variant exits at the same close that confirms the cross.", "effect": "That assumes ideal market-on-close execution; the audit uses next-open execution for close signals."},
        {"severity": "design", "issue": "The exact strategy sells half after the first 15-minute close below EMA20.", "effect": "This fast intraday management can cut a weekly-scale trend very early."},
    ]


def main() -> None:
    started = time.time()
    frames, quality, splits = load_all_markets(WORK)
    prepared = {name: PreparedMarket(name, frame) for name, frame in frames.items()}
    features = {name: build_features(mkt) for name, mkt in prepared.items()}
    entry_cfg = StrategyConfig(config_id="EXACT_BASELINE", family="exact_baseline")
    policies = policy_catalog()
    pd.DataFrame([asdict(p) for p in policies]).to_csv(OUT / "exit_policy_catalog.csv", index=False)

    market_rows: List[dict] = []
    policy_trades: List[dict] = []
    for number, policy in enumerate(policies, start=1):
        for name, mkt in prepared.items():
            for period in PERIODS:
                metrics, trades = simulate_policy_period(mkt, features[name], entry_cfg, policy, splits[name], period)
                market_rows.append(metrics)
                policy_trades.extend(trades)
        if number == 1 or number % 10 == 0 or number == len(policies):
            print(f"[{number}/{len(policies)}] {policy.policy_id}", flush=True)

    policy_market = pd.DataFrame(market_rows)
    policy_market.to_csv(OUT / "exit_policy_market_metrics.csv", index=False)
    pd.DataFrame(policy_trades).to_csv(OUT / "exit_policy_trade_log.csv", index=False)
    policy_aggregate = aggregate_policy_metrics(policy_market)
    policy_aggregate.to_csv(OUT / "exit_policy_aggregate_metrics.csv", index=False)

    descriptions = {p.policy_id: p.description for p in policies}
    ranking_rows = []
    for policy_id in policy_aggregate["policy_id"].unique():
        row = {"policy_id": policy_id, "description": descriptions[policy_id]}
        for period in PERIODS:
            sub = policy_aggregate[(policy_aggregate.policy_id == policy_id) & (policy_aggregate.period == period)]
            if sub.empty:
                continue
            s = sub.iloc[0]
            for col in ["trades", "equal_weight_return", "positive_markets", "trade_weighted_mean_r", "worst_market_return", "best_market_return", "max_market_drawdown", "avg_holding_hours"]:
                row[f"{period}_{col}"] = s[col]
        ranking_rows.append(row)
    ranking = pd.DataFrame(ranking_rows)
    ranking["development_score"] = ranking["development_equal_weight_return"].fillna(-99) + 0.01 * ranking["development_trade_weighted_mean_r"].fillna(-99) - 0.25 * ranking["development_max_market_drawdown"].fillna(0)
    ranking = ranking.sort_values(["development_score", "development_trades"], ascending=[False, False])
    ranking["development_rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(OUT / "exit_policy_rankings.csv", index=False)

    current_market_rows, current_trades, current_diagnostics = [], [], []
    for name, mkt in prepared.items():
        for period in PERIODS:
            metrics, trades = simulate_period(mkt, entry_cfg, splits[name], period)
            metrics["policy_id"] = "CURRENT_CODE_BASELINE"
            current_market_rows.append(metrics)
            _, period_end = mkt.period_bounds(splits[name], period)
            for trade in trades:
                current_trades.append(trade)
                current_diagnostics.append(current_trade_diagnostics(mkt, trade, period_end))
    current_market_df = pd.DataFrame(current_market_rows)
    current_trade_df = pd.DataFrame(current_trades)
    diagnostics_df = pd.DataFrame(current_diagnostics)
    current_market_df.to_csv(OUT / "current_code_market_metrics.csv", index=False)
    current_trade_df.to_csv(OUT / "current_code_trade_log.csv", index=False)
    diagnostics_df.to_csv(OUT / "current_code_trade_diagnostics.csv", index=False)

    exit_reason_summary = diagnostics_df.groupby(["period", "exit_reason"], dropna=False).agg(trades=("r_multiple", "size"), mean_r=("r_multiple", "mean"), median_r=("r_multiple", "median"), win_rate=("net_pnl", lambda s: float((s > 0).mean())), avg_holding_hours=("holding_hours", "mean"), avg_mfe_r=("mfe_r_to_exit", "mean"), median_capture_ratio=("capture_ratio", "median"), reached_5r_after_exit=("reached_5r_before_stop_after_exit", "sum"), reached_10r_after_exit=("reached_10r_before_stop_after_exit", "sum"), reached_20r_after_exit=("reached_20r_before_stop_after_exit", "sum")).reset_index()
    exit_reason_summary.to_csv(OUT / "current_exit_reason_summary.csv", index=False)
    pd.DataFrame([{"threshold_hours": h, "trades": int((diagnostics_df.holding_hours <= h).sum()), "share": float((diagnostics_df.holding_hours <= h).mean())} for h in [1, 2, 4, 8, 24, 48, 120]]).to_csv(OUT / "current_holding_time_summary.csv", index=False)

    notes = code_audit_notes()
    (OUT / "code_audit.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    current_exit_counts = current_trade_df.exit_reason.value_counts().to_dict()
    target_mean_r = float(current_trade_df.loc[current_trade_df.exit_reason == "target", "r_multiple"].mean())
    within_1h = float((current_trade_df.holding_hours <= 1).mean())
    within_4h = float((current_trade_df.holding_hours <= 4).mean())
    corrected = ranking.loc[ranking.policy_id == "CORRECTED_BASELINE"].iloc[0].to_dict()
    top_dev = ranking.iloc[0].to_dict()
    top_holdout = ranking.sort_values(["holdout_equal_weight_return", "holdout_trade_weighted_mean_r"], ascending=False).iloc[0].to_dict()
    ma80_diag = diagnostics_df[diagnostics_df.exit_reason == "ma80"]
    target_diag = diagnostics_df[diagnostics_df.exit_reason == "target"]
    summary = {
        "run_completed_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "markets": list(prepared.keys()),
        "exit_policy_count": len(policies),
        "current_trade_count": len(current_trade_df),
        "current_exit_counts": current_exit_counts,
        "current_target_trade_mean_r": target_mean_r,
        "current_share_closed_within_1h": within_1h,
        "current_share_closed_within_4h": within_4h,
        "ma80_reached_5r_after_exit_before_original_stop": int(ma80_diag.reached_5r_before_stop_after_exit.sum()),
        "ma80_reached_10r_after_exit_before_original_stop": int(ma80_diag.reached_10r_before_stop_after_exit.sum()),
        "target_reached_5r_after_exit_before_original_stop": int(target_diag.reached_5r_before_stop_after_exit.sum()),
        "target_reached_10r_after_exit_before_original_stop": int(target_diag.reached_10r_before_stop_after_exit.sum()),
        "corrected_baseline": corrected,
        "best_development_policy": top_dev,
        "best_holdout_descriptive_policy": top_holdout,
        "code_audit": notes,
    }
    (OUT / "exit_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    lines = [
        "EXIT / LEFT-OUT-PROFIT AUDIT",
        "=" * 40,
        f"Markets: {', '.join(prepared.keys())}",
        f"Exit policies tested: {len(policies)}",
        f"Current-code trades audited: {len(current_trade_df)}",
        f"Current exits: {current_exit_counts}",
        f"Closed within 1h: {within_1h:.1%}; within 4h: {within_4h:.1%}",
        f"Target-exit trades averaged {target_mean_r:.3f}R rather than 3R because half was generally sold earlier.",
        f"MA80 exits later reached 5R before original stop: {summary['ma80_reached_5r_after_exit_before_original_stop']}/{len(ma80_diag)}",
        f"MA80 exits later reached 10R before original stop: {summary['ma80_reached_10r_after_exit_before_original_stop']}/{len(ma80_diag)}",
        f"3R target exits later reached 5R before original stop: {summary['target_reached_5r_after_exit_before_original_stop']}/{len(target_diag)}",
        f"3R target exits later reached 10R before original stop: {summary['target_reached_10r_after_exit_before_original_stop']}/{len(target_diag)}",
        "",
        f"Corrected baseline development/validation/holdout: {corrected.get('development_equal_weight_return')}, {corrected.get('validation_equal_weight_return')}, {corrected.get('holdout_equal_weight_return')}",
        f"Best development-ranked exit policy: {top_dev.get('policy_id')}",
        f"Best holdout-descriptive exit policy: {top_holdout.get('policy_id')}",
    ]
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
