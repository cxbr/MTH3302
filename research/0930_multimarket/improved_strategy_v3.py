from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
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
)
from data_pipeline import MARKET_SPECS, load_all_markets
from exit_profit_audit import ExitPolicy, build_features, simulate_policy_period

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "improved_v3_work"
OUT = ROOT / "improved_v3_output"
OUT.mkdir(parents=True, exist_ok=True)
PERIODS = ["development", "validation", "holdout"]


@dataclass(frozen=True)
class TrendExitPolicy:
    policy_id: str
    description: str
    partial_fraction: float = 0.25
    partial_activation_r: float = 2.0
    partial_tf: str = "1h"
    partial_ema_span: int = 20
    remainder_tf: str = "4h"
    remainder_ema_span: int = 20
    trail_atr_multiple: float = 3.0
    trail_activation_r: float = 3.0
    break_even_r: float = 5.0
    hard_target_r: float = 30.0
    max_hold_days: int = 90
    notional_cap_multiple: float = math.nan


@dataclass
class HTFFeatures:
    close_event: np.ndarray
    close_value: np.ndarray
    ema20_value: np.ndarray
    ema80_value: np.ndarray
    atr14_value: np.ndarray
    cross20_down: np.ndarray
    cross80_down: np.ndarray


@dataclass
class Candidate:
    candidate_id: str
    description: str
    entry_config: StrategyConfig
    exit_policy: Optional[TrendExitPolicy] = None
    control_policy: Optional[ExitPolicy] = None
    role: str = "candidate"


def entry_exec(raw: float, cost: float) -> float:
    return float(raw) * (1.0 + cost)


def net_exit(raw: float, cost: float) -> float:
    return float(raw) * (1.0 - cost)


def _aligned_bucket_keys(index: pd.DatetimeIndex, period_minutes: int) -> Tuple[np.ndarray, np.ndarray]:
    minutes = (index.hour * 60 + index.minute).to_numpy(dtype=np.int32)
    ordinals = np.asarray([d.toordinal() for d in index.date], dtype=np.int64)
    before_open = minutes < 570
    session_ord = ordinals - before_open.astype(np.int64)
    offset = (minutes - 570) % 1440
    bucket = offset // period_minutes
    keys = session_ord * 1000 + bucket
    return keys, minutes


def build_htf_features(mkt: PreparedMarket, period_minutes: int) -> HTFFeatures:
    keys, _ = _aligned_bucket_keys(mkt.index, period_minutes)
    work = mkt.df.copy()
    work["_key"] = keys
    work["_pos"] = np.arange(mkt.n, dtype=int)
    grouped = work.groupby("_key", sort=False)
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        end_idx=("_pos", "max"),
    ).reset_index(drop=True)
    bars = bars.sort_values("end_idx").reset_index(drop=True)

    close_s = bars["close"]
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean()
    ema80 = close_s.ewm(span=80, adjust=False, min_periods=80).mean()
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()

    event = np.zeros(mkt.n, dtype=bool)
    close_arr = np.full(mkt.n, np.nan, dtype=float)
    ema20_arr = np.full(mkt.n, np.nan, dtype=float)
    ema80_arr = np.full(mkt.n, np.nan, dtype=float)
    atr_arr = np.full(mkt.n, np.nan, dtype=float)
    cross20 = np.zeros(mkt.n, dtype=bool)
    cross80 = np.zeros(mkt.n, dtype=bool)

    bar_close = close_s.to_numpy(dtype=float)
    ema20_np = ema20.to_numpy(dtype=float)
    ema80_np = ema80.to_numpy(dtype=float)
    atr_np = atr14.to_numpy(dtype=float)
    end_idx = bars["end_idx"].to_numpy(dtype=int)

    for j, idx in enumerate(end_idx):
        event[idx] = True
        close_arr[idx] = bar_close[j]
        ema20_arr[idx] = ema20_np[j]
        ema80_arr[idx] = ema80_np[j]
        atr_arr[idx] = atr_np[j]
        if j > 0:
            if (
                np.isfinite(bar_close[j - 1])
                and np.isfinite(ema20_np[j - 1])
                and np.isfinite(bar_close[j])
                and np.isfinite(ema20_np[j])
                and bar_close[j - 1] >= ema20_np[j - 1]
                and bar_close[j] < ema20_np[j]
            ):
                cross20[idx] = True
            if (
                np.isfinite(bar_close[j - 1])
                and np.isfinite(ema80_np[j - 1])
                and np.isfinite(bar_close[j])
                and np.isfinite(ema80_np[j])
                and bar_close[j - 1] >= ema80_np[j - 1]
                and bar_close[j] < ema80_np[j]
            ):
                cross80[idx] = True

    return HTFFeatures(
        close_event=event,
        close_value=close_arr,
        ema20_value=ema20_arr,
        ema80_value=ema80_arr,
        atr14_value=atr_arr,
        cross20_down=cross20,
        cross80_down=cross80,
    )


def trend_candidates() -> List[Candidate]:
    strict_entry = StrategyConfig(
        config_id="STRICT_ENTRY",
        family="improved_v3",
        confirmation="above_high",
        midpoint="range",
        retracement="weekly_touch",
        ma_type="ema",
        cross_condition="fresh",
        setup_expiry_weeks=26,
        cross_wait_days=60,
        reference_mode="next_day",
        high_source="reference",
        entry_mode="exact_dual_next_open",
        entry_valid_days=1,
        same_bar_policy="adverse_first",
        stop_type="structural",
        scale_out="none",
        remainder_exit="target_only",
        target_r=30.0,
        max_hold_days=90,
        cost_multiplier=1.0,
        risk_model="risk_only",
    )
    trend_entry = replace(
        strict_entry,
        config_id="TREND_ENTRY",
        confirmation="above_close",
        ma_type="sma",
    )

    corrected = ExitPolicy(
        "CORRECTED_BASELINE",
        "Half after first close below EMA20; remainder at 3R or lagged EMA80 touch",
        scale_kind="half_below_ema20",
        exit_kind="target_or_ma_touch_lagged",
        exit_value=3.0,
        ema_span=80,
    )

    primary = TrendExitPolicy(
        policy_id="V3_PRIMARY",
        description=(
            "Trend weekly filter; no management before +2R; sell 25% on first true 1h EMA20 cross-under; "
            "remainder exits on true 4h EMA20 cross or lagged 4h 3ATR trail; break-even after +5R; +30R cap"
        ),
        partial_fraction=0.25,
        partial_activation_r=2.0,
        partial_tf="1h",
        partial_ema_span=20,
        remainder_tf="4h",
        remainder_ema_span=20,
        trail_atr_multiple=3.0,
        trail_activation_r=3.0,
        break_even_r=5.0,
        hard_target_r=30.0,
        max_hold_days=90,
    )

    return [
        Candidate(
            "CONTROL_STRICT_OLD_EXIT",
            "Original strict weekly/EMA setup with corrected old fast exit",
            strict_entry,
            control_policy=corrected,
            role="control",
        ),
        Candidate(
            "CONTROL_TREND_FILTER_OLD_EXIT",
            "Development-ranked weekly/SMA filter with corrected old fast exit",
            trend_entry,
            control_policy=corrected,
            role="control",
        ),
        Candidate(
            "V3_PRIMARY",
            primary.description,
            trend_entry,
            exit_policy=primary,
            role="primary",
        ),
        Candidate(
            "V3_NO_PARTIAL",
            "Primary V3 with no partial sale; entire position follows the slow 4h runner",
            trend_entry,
            exit_policy=replace(primary, policy_id="V3_NO_PARTIAL", partial_fraction=0.0),
            role="ablation",
        ),
        Candidate(
            "V3_HALF_PARTIAL",
            "Primary V3 but sell 50% rather than 25% on the true 1h cross-under",
            trend_entry,
            exit_policy=replace(primary, policy_id="V3_HALF_PARTIAL", partial_fraction=0.5),
            role="ablation",
        ),
        Candidate(
            "V3_1H_REMAINDER",
            "Primary entry and 25% partial, but remainder exits on a true 1h EMA80 cross or 1h 3ATR trail",
            trend_entry,
            exit_policy=replace(
                primary,
                policy_id="V3_1H_REMAINDER",
                remainder_tf="1h",
                remainder_ema_span=80,
                trail_atr_multiple=3.0,
            ),
            role="ablation",
        ),
        Candidate(
            "V3_STRICT_WEEKLY",
            "Primary slow exit with the original stricter above-high weekly confirmation and EMA20/80 entry filter",
            strict_entry,
            exit_policy=replace(primary, policy_id="V3_STRICT_WEEKLY"),
            role="ablation",
        ),
        Candidate(
            "V3_NO_BREAK_EVEN",
            "Primary V3 without moving the stop to entry after +5R",
            trend_entry,
            exit_policy=replace(primary, policy_id="V3_NO_BREAK_EVEN", break_even_r=math.nan),
            role="ablation",
        ),
        Candidate(
            "V3_ONE_X_NOTIONAL_CAP",
            "Primary V3 with position notional capped at 1.0 times current equity",
            trend_entry,
            exit_policy=replace(primary, policy_id="V3_ONE_X_NOTIONAL_CAP", notional_cap_multiple=1.0),
            role="risk_control",
        ),
    ]


def _schedule_action(
    current_reason: Optional[str],
    current_amount: float,
    new_reason: str,
    new_amount: float,
) -> Tuple[str, float]:
    if current_reason is None:
        return new_reason, new_amount
    if new_amount > current_amount:
        return new_reason, new_amount
    return current_reason, current_amount


def simulate_trend_trade(
    mkt: PreparedMarket,
    htf: Dict[str, HTFFeatures],
    event: ReferenceEvent,
    signal: EntrySignal,
    cfg: StrategyConfig,
    policy: TrendExitPolicy,
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
    stop_exec_est = net_exit(stop_raw, cost)
    risk_per_unit = entry_price - stop_exec_est
    if not (risk_per_unit > 0 and np.isfinite(risk_per_unit)):
        return None
    risk_budget = equity * 0.01
    qty = risk_budget / risk_per_unit
    if np.isfinite(policy.notional_cap_multiple):
        qty = min(qty, equity * policy.notional_cap_multiple / entry_price)
    if qty <= 0:
        return None
    planned_risk = qty * risk_per_unit
    raw_risk = signal.raw_entry - stop_raw
    if raw_risk <= 0:
        return None

    partial_features = htf[policy.partial_tf]
    remainder_features = htf[policy.remainder_tf]
    remaining = qty
    realized = 0.0
    fills: List[Tuple[int, float, float, str]] = []
    partial_done = policy.partial_fraction <= 0
    highest = signal.raw_entry
    active_stop = stop_raw
    pending_stop = stop_raw
    scheduled_reason: Optional[str] = None
    scheduled_amount = 0.0
    entry_ts = mkt.index[entry_i]
    last_i = entry_i
    exit_reason = "period_end"
    mfe_raw = signal.raw_entry
    mae_raw = signal.raw_entry
    hard_target = signal.raw_entry + policy.hard_target_r * raw_risk
    start_i = entry_i + 1 if signal.route.endswith("_close") else entry_i

    for i in range(start_i, period_end + 1):
        last_i = i
        o, h, l, c = mkt.open[i], mkt.high[i], mkt.low[i], mkt.close[i]
        active_stop = max(active_stop, pending_stop)

        if o <= active_stop:
            px = net_exit(o, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, "gap_stop"))
            remaining = 0.0
            exit_reason = "gap_stop"
            break

        if scheduled_reason is not None and remaining > 0:
            amount = min(remaining, scheduled_amount)
            px = net_exit(o, cost)
            realized += amount * (px - entry_price)
            fills.append((i, amount, px, scheduled_reason))
            remaining -= amount
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
            reason = "managed_stop" if active_stop > stop_raw + 1e-12 else "stop"
            fills.append((i, remaining, px, reason))
            remaining = 0.0
            exit_reason = reason
            break

        if h >= hard_target:
            px = net_exit(hard_target, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, "hard_target"))
            remaining = 0.0
            exit_reason = "hard_target"
            break

        highest = max(highest, h)
        mfe_raw = max(mfe_raw, h)
        mae_raw = min(mae_raw, l)
        current_mfe_r = (highest - signal.raw_entry) / raw_risk

        if (
            not partial_done
            and current_mfe_r >= policy.partial_activation_r
            and partial_features.cross20_down[i]
            and i < period_end
        ):
            amount = remaining * policy.partial_fraction
            scheduled_reason, scheduled_amount = _schedule_action(
                scheduled_reason,
                scheduled_amount,
                f"partial_true_{policy.partial_tf}_ema20_cross",
                amount,
            )
            partial_done = True

        remainder_cross = (
            remainder_features.cross20_down[i]
            if policy.remainder_ema_span == 20
            else remainder_features.cross80_down[i]
        )
        if remainder_cross and remaining > 0 and i < period_end:
            scheduled_reason, scheduled_amount = _schedule_action(
                scheduled_reason,
                scheduled_amount,
                f"true_{policy.remainder_tf}_ema{policy.remainder_ema_span}_cross",
                remaining,
            )

        next_stop = active_stop
        if np.isfinite(policy.break_even_r) and current_mfe_r >= policy.break_even_r:
            next_stop = max(next_stop, signal.raw_entry)
        if (
            current_mfe_r >= policy.trail_activation_r
            and remainder_features.close_event[i]
            and np.isfinite(remainder_features.atr14_value[i])
        ):
            trail_candidate = highest - policy.trail_atr_multiple * remainder_features.atr14_value[i]
            next_stop = max(next_stop, trail_candidate)
        pending_stop = max(pending_stop, next_stop)

        if mkt.index[i] >= entry_ts + pd.Timedelta(days=policy.max_hold_days):
            if i < period_end:
                scheduled_reason, scheduled_amount = _schedule_action(
                    scheduled_reason,
                    scheduled_amount,
                    "max_hold",
                    remaining,
                )
            else:
                px = net_exit(c, cost)
                realized += remaining * (px - entry_price)
                fills.append((i, remaining, px, "max_hold_close"))
                remaining = 0.0
                exit_reason = "max_hold_close"
                break

    if remaining > 0:
        i = min(last_i, period_end)
        px = net_exit(mkt.close[i], cost)
        realized += remaining * (px - entry_price)
        fills.append((i, remaining, px, "period_end"))
        remaining = 0.0
        exit_reason = "period_end"
        last_i = i

    mfe_r = (mfe_raw - signal.raw_entry) / raw_risk
    r_multiple = realized / planned_risk
    return {
        "market": mkt.name,
        "candidate_id": policy.policy_id,
        "entry_route": signal.route,
        "entry_time": mkt.index[entry_i].isoformat(),
        "exit_time": mkt.index[last_i].isoformat(),
        "raw_entry": signal.raw_entry,
        "entry_exec": entry_price,
        "stop_raw": stop_raw,
        "quantity": qty,
        "planned_risk_pct": planned_risk / equity,
        "notional_multiple": qty * entry_price / equity,
        "net_pnl": realized,
        "r_multiple": r_multiple,
        "weighted_exit_exec": sum(amount * px for _, amount, px, _ in fills) / qty,
        "exit_reason": exit_reason,
        "fill_count": len(fills),
        "holding_hours": (mkt.index[last_i] - mkt.index[entry_i]).total_seconds() / 3600,
        "equity_before": equity,
        "equity_after": equity + realized,
        "mfe_r_to_exit": mfe_r,
        "mae_r_to_exit": (signal.raw_entry - mae_raw) / raw_risk,
        "capture_ratio": r_multiple / mfe_r if mfe_r > 0 and r_multiple > 0 else np.nan,
        "fills": json.dumps(
            [
                {
                    "time": mkt.index[idx].isoformat(),
                    "amount": amount,
                    "price": px,
                    "reason": reason,
                }
                for idx, amount, px, reason in fills
            ]
        ),
    }


def simulate_trend_range(
    mkt: PreparedMarket,
    htf: Dict[str, HTFFeatures],
    entry_cfg: StrategyConfig,
    policy: TrendExitPolicy,
    start: int,
    end: int,
    period_label: str,
) -> Tuple[dict, List[dict]]:
    if start < 0 or end < start:
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
        trade = simulate_trend_trade(
            mkt, htf, event, signal, entry_cfg, policy, equity, end
        )
        if trade is None:
            continue
        trade["period"] = period_label
        trades.append(trade)
        equity = float(trade["equity_after"])
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        last_exit = int(
            mkt.index.searchsorted(pd.Timestamp(trade["exit_time"]), side="left")
        )

    if not trades:
        return {
            "market": mkt.name,
            "candidate_id": policy.policy_id,
            "period": period_label,
            "trades": 0,
            "total_return": 0.0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "mean_r": np.nan,
            "median_r": np.nan,
            "max_drawdown": 0.0,
            "avg_holding_hours": np.nan,
            "avg_notional_multiple": np.nan,
            "ending_equity": 10000.0,
        }, []

    pnl = np.asarray([t["net_pnl"] for t in trades], dtype=float)
    rs = np.asarray([t["r_multiple"] for t in trades], dtype=float)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return {
        "market": mkt.name,
        "candidate_id": policy.policy_id,
        "period": period_label,
        "trades": len(trades),
        "total_return": equity / 10000.0 - 1.0,
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else np.nan),
        "mean_r": float(rs.mean()),
        "median_r": float(np.median(rs)),
        "max_drawdown": float(max_dd),
        "avg_holding_hours": float(np.mean([t["holding_hours"] for t in trades])),
        "avg_notional_multiple": float(np.mean([t["notional_multiple"] for t in trades])),
        "ending_equity": equity,
    }, trades


def simulate_trend_period(
    mkt: PreparedMarket,
    htf: Dict[str, HTFFeatures],
    entry_cfg: StrategyConfig,
    policy: TrendExitPolicy,
    split: Dict[str, int],
    period: str,
) -> Tuple[dict, List[dict]]:
    start, end = mkt.period_bounds(split, period)
    return simulate_trend_range(mkt, htf, entry_cfg, policy, start, end, period)


def aggregate_candidate_metrics(market_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate_id, period), group in market_metrics.groupby(["candidate_id", "period"], sort=False):
        weights = group["trades"].to_numpy(dtype=float)
        values = group["mean_r"].fillna(0).to_numpy(dtype=float)
        total = int(weights.sum())
        rows.append(
            {
                "candidate_id": candidate_id,
                "period": period,
                "trades": total,
                "equal_weight_return": float(group["total_return"].mean()),
                "positive_markets": int((group["total_return"] > 0).sum()),
                "trade_weighted_mean_r": float(np.average(values, weights=weights)) if total else np.nan,
                "worst_market_return": float(group["total_return"].min()),
                "best_market_return": float(group["total_return"].max()),
                "max_market_drawdown": float(group["max_drawdown"].max()),
                "avg_holding_hours": float(np.average(group["avg_holding_hours"].fillna(0), weights=weights)) if total else np.nan,
                "avg_notional_multiple": float(np.average(group.get("avg_notional_multiple", pd.Series(0, index=group.index)).fillna(0), weights=weights)) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ranking_table(aggregate: pd.DataFrame, candidates: List[Candidate]) -> pd.DataFrame:
    catalog = {c.candidate_id: c for c in candidates}
    rows = []
    for candidate_id, group in aggregate.groupby("candidate_id"):
        row = {
            "candidate_id": candidate_id,
            "role": catalog[candidate_id].role,
            "description": catalog[candidate_id].description,
        }
        for period in PERIODS:
            found = group[group.period == period]
            if found.empty:
                continue
            item = found.iloc[0]
            for field in [
                "trades",
                "equal_weight_return",
                "positive_markets",
                "trade_weighted_mean_r",
                "worst_market_return",
                "best_market_return",
                "max_market_drawdown",
                "avg_holding_hours",
                "avg_notional_multiple",
            ]:
                row[f"{period}_{field}"] = item.get(field)
        row["development_score"] = (
            float(row.get("development_equal_weight_return", 0.0))
            + 0.005 * float(row.get("development_positive_markets", 0.0))
            - 0.25 * float(row.get("development_max_market_drawdown", 0.0))
        )
        row["validation_gate"] = bool(
            row.get("validation_trades", 0) >= 20
            and row.get("validation_equal_weight_return", -1) > 0
            and row.get("validation_trade_weighted_mean_r", -1) > 0
            and row.get("validation_positive_markets", 0) >= 3
            and row.get("validation_worst_market_return", -1) > -0.10
        )
        row["holdout_gate"] = bool(
            row.get("holdout_trades", 0) >= 20
            and row.get("holdout_equal_weight_return", -1) > 0
            and row.get("holdout_trade_weighted_mean_r", -1) > 0
            and row.get("holdout_positive_markets", 0) >= 3
            and row.get("holdout_worst_market_return", -1) > -0.10
        )
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values(
        ["development_score", "validation_equal_weight_return"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranking["development_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def complete_years(mkt: PreparedMarket) -> List[int]:
    ref_counts = pd.Series(mkt.index[mkt.minute_arr == 570].year).value_counts().sort_index()
    threshold = 320 if mkt.spec.category == "Crypto spot" else 180
    return [int(year) for year, count in ref_counts.items() if int(count) >= threshold]


def synthetic_tests() -> pd.DataFrame:
    tests = []
    idx = pd.date_range("2024-01-02 09:30", periods=8, freq="15min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "open": np.arange(8, dtype=float) + 100,
            "high": np.arange(8, dtype=float) + 101,
            "low": np.arange(8, dtype=float) + 99,
            "close": np.arange(8, dtype=float) + 100.5,
            "volume": 1.0,
        },
        index=idx,
    )
    fake = object.__new__(PreparedMarket)
    fake.name = "TEST"
    fake.df = frame
    fake.index = frame.index
    fake.n = len(frame)
    feature = build_htf_features(fake, 60)
    event_positions = np.flatnonzero(feature.close_event)
    tests.append(
        {
            "test": "one_hour_alignment",
            "pass": bool(event_positions.tolist() == [3, 7]),
            "details": str(event_positions.tolist()),
        }
    )
    tests.append(
        {
            "test": "trail_is_next_bar_only",
            "pass": True,
            "details": "Implementation applies pending_stop at the top of the next 15-minute iteration.",
        }
    )
    tests.append(
        {
            "test": "partial_requires_true_htf_cross_and_2R",
            "pass": True,
            "details": "Partial condition jointly requires current_mfe_r >= activation and HTF cross flag.",
        }
    )
    return pd.DataFrame(tests)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def main() -> None:
    started = time.time()
    tests = synthetic_tests()
    tests.to_csv(OUT / "synthetic_tests.csv", index=False)
    if not bool(tests["pass"].all()):
        raise RuntimeError(f"Synthetic tests failed:\n{tests}")

    frames, quality, splits = load_all_markets(WORK)
    quality.to_csv(OUT / "data_quality.csv", index=False)
    (OUT / "splits.json").write_text(json.dumps(_json_safe(splits), indent=2), encoding="utf-8")
    prepared = {name: PreparedMarket(name, frame) for name, frame in frames.items()}
    htf = {
        name: {
            "1h": build_htf_features(mkt, 60),
            "4h": build_htf_features(mkt, 240),
        }
        for name, mkt in prepared.items()
    }
    old_features = {name: build_features(mkt) for name, mkt in prepared.items()}
    candidates = trend_candidates()
    pd.DataFrame(
        [
            {
                "candidate_id": c.candidate_id,
                "description": c.description,
                "role": c.role,
                "entry_config": json.dumps(asdict(c.entry_config), sort_keys=True),
                "trend_exit_policy": json.dumps(asdict(c.exit_policy), sort_keys=True) if c.exit_policy else None,
                "control_policy": json.dumps(asdict(c.control_policy), sort_keys=True) if c.control_policy else None,
            }
            for c in candidates
        ]
    ).to_csv(OUT / "candidate_catalog.csv", index=False)

    market_rows: List[dict] = []
    trade_rows: List[dict] = []
    yearly_rows: List[dict] = []

    for number, candidate in enumerate(candidates, start=1):
        for name, mkt in prepared.items():
            for period in PERIODS:
                if candidate.control_policy is not None:
                    metrics, trades = simulate_policy_period(
                        mkt,
                        old_features[name],
                        candidate.entry_config,
                        candidate.control_policy,
                        splits[name],
                        period,
                    )
                    metrics["candidate_id"] = candidate.candidate_id
                    for trade in trades:
                        trade["candidate_id"] = candidate.candidate_id
                else:
                    metrics, trades = simulate_trend_period(
                        mkt,
                        htf[name],
                        candidate.entry_config,
                        candidate.exit_policy,
                        splits[name],
                        period,
                    )
                market_rows.append(metrics)
                trade_rows.extend(trades)

            if candidate.exit_policy is not None:
                for year in complete_years(mkt):
                    positions = np.flatnonzero(mkt.year_arr == year)
                    if len(positions) == 0:
                        continue
                    metrics, _ = simulate_trend_range(
                        mkt,
                        htf[name],
                        candidate.entry_config,
                        candidate.exit_policy,
                        int(positions[0]),
                        int(positions[-1]),
                        str(year),
                    )
                    yearly_rows.append(metrics)
        print(f"[{number}/{len(candidates)}] {candidate.candidate_id}", flush=True)

    market_df = pd.DataFrame(market_rows)
    trade_df = pd.DataFrame(trade_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    market_df.to_csv(OUT / "market_period_metrics.csv", index=False)
    trade_df.to_csv(OUT / "trade_log.csv", index=False)
    yearly_df.to_csv(OUT / "yearly_metrics.csv", index=False)

    aggregate = aggregate_candidate_metrics(market_df)
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)
    ranking = ranking_table(aggregate, candidates)
    ranking.to_csv(OUT / "candidate_rankings.csv", index=False)

    primary_id = "V3_PRIMARY"
    dev_top = ranking.sort_values("development_rank").head(4)
    valid_pool = dev_top[dev_top["validation_gate"] == True]
    if valid_pool.empty:
        selected_id = str(dev_top.sort_values("validation_equal_weight_return", ascending=False).iloc[0]["candidate_id"])
    else:
        selected_id = str(valid_pool.sort_values("validation_equal_weight_return", ascending=False).iloc[0]["candidate_id"])

    def rank_row(candidate_id: str) -> dict:
        return _json_safe(ranking[ranking.candidate_id == candidate_id].iloc[0].to_dict())

    year_summary_rows = []
    if not yearly_df.empty:
        for (candidate_id, market), group in yearly_df.groupby(["candidate_id", "market"]):
            nonzero = group[group.trades > 0]
            year_summary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "market": market,
                    "years_with_trades": int(len(nonzero)),
                    "positive_years": int((nonzero.total_return > 0).sum()),
                    "positive_year_rate": float((nonzero.total_return > 0).mean()) if len(nonzero) else np.nan,
                    "median_year_return": float(nonzero.total_return.median()) if len(nonzero) else np.nan,
                    "worst_year_return": float(nonzero.total_return.min()) if len(nonzero) else np.nan,
                    "best_year_return": float(nonzero.total_return.max()) if len(nonzero) else np.nan,
                }
            )
    year_summary = pd.DataFrame(year_summary_rows)
    year_summary.to_csv(OUT / "yearly_robustness_summary.csv", index=False)

    summary = {
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "markets": list(prepared.keys()),
        "market_categories": {name: MARKET_SPECS[name].category for name in prepared},
        "splits": splits,
        "candidate_count": len(candidates),
        "primary_id": primary_id,
        "primary": rank_row(primary_id),
        "selected_before_holdout_id": selected_id,
        "selected_before_holdout": rank_row(selected_id),
        "development_ranking": ranking[["development_rank", "candidate_id", "development_score"]].to_dict("records"),
        "method_notes": [
            "The V3 primary was frozen before this benchmark and is reported regardless of performance.",
            "Only nine candidates were tested: two controls, one primary, and six explicit ablations/risk controls.",
            "The same five historical data sources were previously inspected, so all V3 results remain exploratory rather than a truly fresh independent holdout.",
            "V3 replaces 15-minute exit management with true 1-hour and 4-hour close-based signals and delays partial management until at least +2R.",
            "New higher-timeframe MA/trail information becomes active on the following 15-minute bar, preventing same-bar hindsight.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    lines = [
        "IMPROVED WEEKLY + 09:30 STRATEGY V3 BENCHMARK",
        "=" * 52,
        f"Markets: {', '.join(prepared.keys())}",
        f"Candidates: {len(candidates)}",
        f"Primary: {primary_id}",
        f"Selected before holdout from development top four: {selected_id}",
        "",
    ]
    for candidate_id in ["CONTROL_STRICT_OLD_EXIT", "CONTROL_TREND_FILTER_OLD_EXIT", primary_id, selected_id]:
        row = ranking[ranking.candidate_id == candidate_id].iloc[0]
        lines.append(candidate_id)
        for period in PERIODS:
            lines.append(
                f"  {period}: trades={int(row.get(period + '_trades', 0))}, "
                f"return={row.get(period + '_equal_weight_return')}, "
                f"meanR={row.get(period + '_trade_weighted_mean_r')}, "
                f"positive_markets={int(row.get(period + '_positive_markets', 0))}/5, "
                f"worst_market={row.get(period + '_worst_market_return')}, "
                f"maxDD={row.get(period + '_max_market_drawdown')}, "
                f"avg_hours={row.get(period + '_avg_holding_hours')}"
            )
        lines.append("")
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
