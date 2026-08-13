from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import blind_hot20_exit_v13 as v13
import hot20_v6_benchmark as base
import stock_v4_engine as sve
from backtest_engine import EntrySignal, PreparedMarket, ReferenceEvent

BASE_COST = 0.0005
INITIAL_EQUITY = 10_000.0


@dataclass(frozen=True)
class V14Policy(v13.V13Policy):
    profit_target_r: float = math.inf
    profit_target_fraction: float = 0.0
    secondary_target_r: float = math.inf
    lock_trigger_r: float = math.inf
    lock_stop_r: float = 0.0
    stagnation_sessions: int = 0
    stagnation_mfe_r: float = math.inf

    weekend_mode: str = "hold"
    weekend_threshold_r: float = 0.0
    allow_friday_entries: bool = True
    friday_risk_scale: float = 1.0

    max_positions: int = 4
    daily_entry_cap: int = 2
    max_open_risk: float = 0.04
    max_total_notional: float = 1.0
    max_position_notional: float = 0.35

    family: str = "v14"


def _schedule(reason: Optional[str], amount: float, new_reason: str, new_amount: float) -> Tuple[str, float]:
    if reason is None:
        return new_reason, new_amount
    if new_amount >= 0.999999:
        return new_reason, new_amount
    return reason, amount


def _session_index(mkt: PreparedMarket, bar_i: int) -> int:
    pos = mkt.date_to_pos.get(int(mkt.date_arr[bar_i]))
    return int(pos) if pos is not None else -1


def _is_friday_close_execution_bar(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 4 and ts.hour == 15 and ts.minute == 45


def _current_close_r(mkt: PreparedMarket, i: int, raw_entry: float, raw_risk: float) -> float:
    if i <= 0 or raw_risk <= 0:
        return 0.0
    return float((mkt.close[i - 1] - raw_entry) / raw_risk)


def simulate_trade(
    mkt: PreparedMarket,
    features: Dict[str, sve.TimeframeFeatures],
    event: ReferenceEvent,
    signal: EntrySignal,
    policy: V14Policy,
    period_end: int,
    cost: float = BASE_COST,
) -> Optional[dict]:
    entry_i = signal.entry_idx
    if entry_i > period_end:
        return None
    cfg = policy.variant().config()
    stop_raw = sve.compute_stop(mkt, signal, cfg)
    if not (np.isfinite(stop_raw) and 0 < stop_raw < signal.raw_entry):
        return None
    raw_risk = float(signal.raw_entry - stop_raw)
    if raw_risk <= 0:
        return None

    entry_price = sve._entry_exec(float(signal.raw_entry), cost)
    stop_exec_est = sve._net_exit(float(stop_raw), cost)
    risk_per_unit = entry_price - stop_exec_est
    if not (np.isfinite(risk_per_unit) and risk_per_unit > 0):
        return None

    time_exit_i = min(sve._session_exit_index(mkt, entry_i, policy.max_sessions), period_end)
    partial_features = features[policy.partial_tf]
    runner_features = features[policy.runner_tf]
    trail_features = features[policy.trail_tf]

    remaining = 1.0
    realized = 0.0
    fills: List[Tuple[int, float, float, str]] = []
    partial_done = policy.partial_fraction <= 0
    target_done = policy.profit_target_fraction <= 0 or not np.isfinite(policy.profit_target_r)
    highest = float(signal.raw_entry)
    mfe_raw = float(signal.raw_entry)
    mae_raw = float(signal.raw_entry)
    active_stop = float(stop_raw)
    pending_stop = float(stop_raw)
    scheduled_reason: Optional[str] = None
    scheduled_amount = 0.0
    start_i = entry_i + 1 if signal.route.endswith("_close") else entry_i
    last_i = entry_i
    exit_reason = "period_end"
    entry_session = _session_index(mkt, entry_i)
    weekends_held = 0
    weekend_reductions = 0
    weekend_gap_worst_r = 0.0

    hard_target = (
        signal.raw_entry + policy.hard_target_r * raw_risk
        if np.isfinite(policy.hard_target_r) else math.inf
    )
    first_target = (
        signal.raw_entry + policy.profit_target_r * raw_risk
        if np.isfinite(policy.profit_target_r) and policy.profit_target_fraction > 0 else math.inf
    )
    second_target = (
        signal.raw_entry + policy.secondary_target_r * raw_risk
        if np.isfinite(policy.secondary_target_r) else math.inf
    )

    for i in range(start_i, time_exit_i + 1):
        last_i = i
        ts = mkt.index[i]
        o, h, l, c = map(float, (mkt.open[i], mkt.high[i], mkt.low[i], mkt.close[i]))
        active_stop = max(active_stop, pending_stop)

        if o <= active_stop:
            px = sve._net_exit(o, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, "gap_stop"))
            weekend_gap_worst_r = min(weekend_gap_worst_r, (px - entry_price) / risk_per_unit)
            remaining = 0.0
            exit_reason = "gap_stop"
            break

        if remaining > 0 and i > entry_i and _is_friday_close_execution_bar(ts):
            previous_close_r = _current_close_r(mkt, i, signal.raw_entry, raw_risk)
            exit_all = False
            reduce_half = False
            if policy.weekend_mode == "flat_all":
                exit_all = True
            elif policy.weekend_mode == "flat_if_below":
                exit_all = previous_close_r < policy.weekend_threshold_r
            elif policy.weekend_mode == "flat_if_unprotected":
                exit_all = active_stop < signal.raw_entry and previous_close_r < policy.weekend_threshold_r
            elif policy.weekend_mode == "reduce_half_if_below":
                reduce_half = previous_close_r < policy.weekend_threshold_r
            elif policy.weekend_mode != "hold":
                raise ValueError(f"Unknown weekend mode {policy.weekend_mode}")

            if exit_all:
                px = sve._net_exit(o, cost)
                realized += remaining * (px - entry_price)
                fills.append((i, remaining, px, f"weekend_{policy.weekend_mode}"))
                remaining = 0.0
                exit_reason = f"weekend_{policy.weekend_mode}"
                break
            if reduce_half and remaining > 0.5:
                amount = remaining * 0.5
                px = sve._net_exit(o, cost)
                realized += amount * (px - entry_price)
                fills.append((i, amount, px, "weekend_reduce_half"))
                remaining -= amount
                weekend_reductions += 1
            if remaining > 0:
                weekends_held += 1

        if i == time_exit_i and i > entry_i:
            px = sve._net_exit(o, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, f"session_{policy.max_sessions}_exit"))
            remaining = 0.0
            exit_reason = f"session_{policy.max_sessions}_exit"
            break

        if policy.stagnation_sessions > 0 and i > entry_i:
            current_session = _session_index(mkt, i)
            completed = current_session - entry_session
            current_mfe_r = (highest - signal.raw_entry) / raw_risk
            first_bar_session = i == int(mkt.date_first[current_session]) if current_session >= 0 else False
            if first_bar_session and completed >= policy.stagnation_sessions and current_mfe_r < policy.stagnation_mfe_r:
                px = sve._net_exit(o, cost)
                realized += remaining * (px - entry_price)
                fills.append((i, remaining, px, "stagnation_exit"))
                remaining = 0.0
                exit_reason = "stagnation_exit"
                break

        if scheduled_reason is not None and remaining > 0:
            amount = min(remaining, scheduled_amount)
            px = sve._net_exit(o, cost)
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
            px = sve._net_exit(active_stop, cost)
            realized += remaining * (px - entry_price)
            reason = "managed_stop" if active_stop > stop_raw + 1e-12 else "stop"
            fills.append((i, remaining, px, reason))
            remaining = 0.0
            exit_reason = reason
            break

        if not target_done and h >= first_target:
            amount = min(remaining, policy.profit_target_fraction)
            px = sve._net_exit(first_target, cost)
            realized += amount * (px - entry_price)
            fills.append((i, amount, px, f"profit_target_{policy.profit_target_r:g}R"))
            remaining -= amount
            target_done = True
            if remaining <= 1e-12:
                remaining = 0.0
                exit_reason = f"profit_target_{policy.profit_target_r:g}R"
                break

        if remaining > 0 and np.isfinite(policy.secondary_target_r) and h >= second_target:
            px = sve._net_exit(second_target, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, f"secondary_target_{policy.secondary_target_r:g}R"))
            remaining = 0.0
            exit_reason = f"secondary_target_{policy.secondary_target_r:g}R"
            break

        if remaining > 0 and h >= hard_target:
            px = sve._net_exit(hard_target, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, "hard_target"))
            remaining = 0.0
            exit_reason = "hard_target"
            break

        highest = max(highest, h)
        mfe_raw = max(mfe_raw, h)
        mae_raw = min(mae_raw, l)
        current_mfe_r = (highest - signal.raw_entry) / raw_risk
        management_active = current_mfe_r >= policy.management_activation_r

        if (
            not partial_done
            and management_active
            and policy.partial_ema_span in partial_features.cross_down
            and partial_features.cross_down[policy.partial_ema_span][i]
            and i < time_exit_i
        ):
            amount = remaining * policy.partial_fraction
            scheduled_reason, scheduled_amount = _schedule(
                scheduled_reason, scheduled_amount,
                f"partial_true_{policy.partial_tf}_ema{policy.partial_ema_span}_cross",
                amount,
            )
            partial_done = True

        if (
            policy.runner_cross_enabled
            and management_active
            and policy.runner_ema_span in runner_features.cross_down
            and runner_features.cross_down[policy.runner_ema_span][i]
            and remaining > 0
            and i < time_exit_i
        ):
            scheduled_reason, scheduled_amount = _schedule(
                scheduled_reason, scheduled_amount,
                f"true_{policy.runner_tf}_ema{policy.runner_ema_span}_cross",
                remaining,
            )

        next_stop = active_stop
        if np.isfinite(policy.break_even_r) and current_mfe_r >= policy.break_even_r:
            next_stop = max(next_stop, signal.raw_entry)
        if np.isfinite(policy.lock_trigger_r) and current_mfe_r >= policy.lock_trigger_r:
            next_stop = max(next_stop, signal.raw_entry + policy.lock_stop_r * raw_risk)

        trail_active = current_mfe_r >= max(policy.management_activation_r, policy.trail_activation_r)
        if trail_active and trail_features.close_event[i]:
            if policy.trail_kind == "atr":
                atr = trail_features.atr14_value[i]
                if np.isfinite(atr):
                    next_stop = max(next_stop, highest - policy.trail_value * atr)
            elif policy.trail_kind == "pct":
                next_stop = max(next_stop, highest * (1.0 - policy.trail_value))
            elif policy.trail_kind != "none":
                raise ValueError(policy.trail_kind)
        pending_stop = max(pending_stop, next_stop)

        if i == time_exit_i == entry_i:
            px = sve._net_exit(c, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, f"session_{policy.max_sessions}_close"))
            remaining = 0.0
            exit_reason = f"session_{policy.max_sessions}_close"
            break

    if remaining > 0:
        i = min(last_i, time_exit_i, period_end)
        px = sve._net_exit(float(mkt.close[i]), cost)
        realized += remaining * (px - entry_price)
        fills.append((i, remaining, px, "period_end"))
        remaining = 0.0
        exit_reason = "period_end"
        last_i = i

    weighted_exit = sum(amount * px for _, amount, px, _ in fills)
    mfe_r = (mfe_raw - signal.raw_entry) / raw_risk
    unit_r = realized / risk_per_unit if risk_per_unit > 0 else math.nan
    return {
        "market": mkt.name,
        "candidate_id": policy.policy_id,
        "entry_route": signal.route,
        "anchor_week": event.anchor_week,
        "reference_time": mkt.index[event.ref_idx].isoformat(),
        "entry_time": mkt.index[entry_i].isoformat(),
        "exit_time": mkt.index[last_i].isoformat(),
        "raw_entry": float(signal.raw_entry),
        "entry_exec": entry_price,
        "stop_raw": float(stop_raw),
        "net_pnl": realized,
        "r_multiple": unit_r,
        "weighted_exit_exec": weighted_exit,
        "exit_reason": exit_reason,
        "fill_count": len(fills),
        "holding_hours": (mkt.index[last_i] - mkt.index[entry_i]).total_seconds() / 3600.0,
        "mfe_r_to_exit": mfe_r,
        "mae_r_to_exit": (signal.raw_entry - mae_raw) / raw_risk,
        "capture_ratio": unit_r / mfe_r if mfe_r > 0 and unit_r > 0 else math.nan,
        "weekends_held": weekends_held,
        "weekend_reductions": weekend_reductions,
        "weekend_gap_worst_r": weekend_gap_worst_r,
        "fills": json.dumps([
            {"time": mkt.index[idx].isoformat(), "amount": amount, "price": px, "reason": reason}
            for idx, amount, px, reason in fills
        ]),
    }


def passes_entry_filters(policy: V14Policy, trade: dict) -> Tuple[bool, str]:
    ok, reason = v13.passes_entry_filters(policy, trade)
    if not ok:
        return ok, reason
    entry_ts = pd.Timestamp(trade["entry_time"])
    if not policy.allow_friday_entries and entry_ts.weekday() == 4:
        return False, "friday_entry_block"
    return True, "accepted"


def generate_templates(
    ticker: str,
    mkt: PreparedMarket,
    features: dict,
    policy: V14Policy,
    period: str,
    scanner_lookup,
    period_bounds_fn,
    include_post_exit_diagnostics: bool = True,
) -> Tuple[List[dict], dict]:
    start, end = period_bounds_fn(mkt, period)
    if start < 0 or end < start:
        return [], {}
    variant = policy.variant()
    cfg = variant.config()
    templates: Dict[Tuple, dict] = {}
    rejections = defaultdict(int)
    for event in mkt.generate_events(cfg):
        if event.ref_idx < start or event.ref_idx > end:
            continue
        ref_ordinal = int(mkt.date_arr[event.ref_idx])
        hot = v13.scanner_membership(policy, ticker, ref_ordinal, scanner_lookup)
        if hot is None:
            rejections["outside_hot_list"] += 1
            continue
        signal = sve.find_variant_entry(mkt, event, variant, end)
        if signal is None or signal.entry_idx < start or signal.entry_idx > end:
            continue
        trade = simulate_trade(mkt, features, event, signal, policy, end, BASE_COST)
        if trade is None:
            continue
        ref_i, entry_i = event.ref_idx, signal.entry_idx
        atr = float(mkt.atr14[ref_i]) if np.isfinite(mkt.atr14[ref_i]) else math.nan
        ref_range = float(mkt.high[ref_i] - mkt.low[ref_i])
        close_location = float((mkt.close[ref_i] - mkt.low[ref_i]) / ref_range) if ref_range > 0 else 0.5
        stop_pct = float((signal.raw_entry - trade["stop_raw"]) / signal.raw_entry)
        entry_gap_atr = (
            max(0.0, float(signal.raw_entry - signal.selected_high)) / atr
            if np.isfinite(atr) and atr > 0 else 0.0
        )
        rank, score, universe_size = hot
        trade.update({
            "period": period,
            "policy_id": policy.policy_id,
            "policy_description": policy.description,
            "family": policy.family,
            "scanner": policy.scanner,
            "hot_rank": rank,
            "hot_score": score,
            "hot_universe_size": universe_size,
            "reference_date_ordinal": ref_ordinal,
            "entry_delay_min": float((mkt.index[entry_i] - mkt.index[ref_i]).total_seconds() / 60.0),
            "cross_age_days": float((mkt.index[entry_i] - mkt.index[event.cross_idx]).total_seconds() / 86400.0),
            "retrace_age_days": float((mkt.index[entry_i] - mkt.index[event.retrace_idx]).total_seconds() / 86400.0),
            "stop_pct": stop_pct,
            "ref_range_atr": float(ref_range / atr) if np.isfinite(atr) and atr > 0 else math.nan,
            "ref_close_location": close_location,
            "entry_gap_atr": entry_gap_atr,
            "base_cost_rate": BASE_COST,
            "entry_weekday": int(mkt.index[entry_i].weekday()),
            "friday_risk_scale": policy.friday_risk_scale,
        })
        ok, reason = passes_entry_filters(policy, trade)
        if not ok:
            rejections[reason] += 1
            continue
        if include_post_exit_diagnostics:
            trade.update(v13.post_exit_diagnostics(mkt, trade))
        key = (
            trade["market"], trade["entry_time"], trade["entry_route"],
            round(float(trade["raw_entry"]), 8), round(float(trade["stop_raw"]), 8),
        )
        templates.setdefault(key, trade)
    return sorted(
        templates.values(),
        key=lambda r: (pd.Timestamp(r["entry_time"]), int(r.get("hot_rank", 9999)), r["market"]),
    ), dict(rejections)


def adjusted_unit_values(template: dict, cost_rate: float) -> Tuple[float, float, float]:
    base_cost = float(template.get("base_cost_rate", BASE_COST))
    raw_entry = float(template["raw_entry"])
    stop_raw = float(template["stop_raw"])
    base_exit = float(template["weighted_exit_exec"])
    raw_exit = base_exit / max(1.0 - base_cost, 1e-12)
    entry_exec = raw_entry * (1.0 + cost_rate)
    exit_exec = raw_exit * (1.0 - cost_rate)
    risk_per_unit = entry_exec - stop_raw * (1.0 - cost_rate)
    pnl_per_unit = exit_exec - entry_exec
    return entry_exec, risk_per_unit, pnl_per_unit


def replay_portfolio(
    policy: V14Policy,
    templates: Sequence[dict],
    period: str,
    cost_rate: float = BASE_COST,
    keep_trades: bool = True,
) -> Tuple[dict, List[dict], dict, List[dict]]:
    candidates = sorted(templates, key=lambda r: (
        pd.Timestamp(r["entry_time"]), int(r.get("hot_rank", 9999)),
        float(r.get("entry_delay_min", math.inf)), str(r["market"]),
    ))
    equity = INITIAL_EQUITY
    peak = equity
    max_drawdown = 0.0
    open_heap: List[Tuple[int, int, dict]] = []
    open_by_ticker: Dict[str, dict] = {}
    accepted: List[dict] = []
    equity_events: List[dict] = []
    daily_entries = defaultdict(int)
    rejects = defaultdict(int)
    sequence = 0
    max_positions_seen = 0
    max_risk_seen = 0.0
    max_notional_seen = 0.0

    def settle_until(ts: pd.Timestamp) -> None:
        nonlocal equity, peak, max_drawdown
        while open_heap and pd.Timestamp(open_heap[0][2]["exit_time"]) <= ts:
            _, _, position = heapq.heappop(open_heap)
            ticker = str(position["market"])
            if open_by_ticker.get(ticker) is not position:
                continue
            equity += float(position["portfolio_pnl"])
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak if peak > 0 else 0.0)
            equity_events.append({"time": position["exit_time"], "equity": equity, "pnl": position["portfolio_pnl"]})
            del open_by_ticker[ticker]

    for template in candidates:
        entry_ts = pd.Timestamp(template["entry_time"])
        settle_until(entry_ts)
        ticker = str(template["market"])
        day_key = entry_ts.date().isoformat()
        if daily_entries[day_key] >= policy.daily_entry_cap:
            rejects["daily_entry_cap"] += 1
            continue
        if ticker in open_by_ticker:
            rejects["ticker_overlap"] += 1
            continue
        if len(open_by_ticker) >= policy.max_positions:
            rejects["position_cap"] += 1
            continue
        entry_exec, risk_per_unit, pnl_per_unit = adjusted_unit_values(template, cost_rate)
        if not (np.isfinite(risk_per_unit) and risk_per_unit > 0 and entry_exec > 0):
            rejects["invalid_unit_risk"] += 1
            continue
        risk_scale = policy.friday_risk_scale if entry_ts.weekday() == 4 else 1.0
        desired_risk_dollars = equity * policy.risk_per_trade * risk_scale
        desired_qty = math.floor(min(
            desired_risk_dollars / risk_per_unit,
            equity * policy.max_position_notional / entry_exec,
        ))
        if desired_qty < 1:
            rejects["sub_share"] += 1
            continue
        open_risk = sum(float(pos["risk_dollars"]) for pos in open_by_ticker.values())
        open_notional = sum(float(pos["notional_dollars"]) for pos in open_by_ticker.values())
        available_risk = max(0.0, equity * policy.max_open_risk - open_risk)
        available_notional = max(0.0, equity * policy.max_total_notional - open_notional)
        capacity_qty = math.floor(min(available_risk / risk_per_unit, available_notional / entry_exec))
        qty = min(desired_qty, capacity_qty)
        if qty < 1:
            rejects["portfolio_capacity"] += 1
            continue
        if qty / desired_qty < policy.min_capacity_scale:
            rejects["capacity_scale"] += 1
            continue
        risk_dollars = qty * risk_per_unit
        notional_dollars = qty * entry_exec
        pnl = qty * pnl_per_unit
        realized_r = pnl / risk_dollars if risk_dollars > 0 else math.nan
        position = dict(template)
        position.update({
            "portfolio_entry_equity": equity,
            "quantity_whole": qty,
            "entry_exec_realistic": entry_exec,
            "risk_dollars": risk_dollars,
            "notional_dollars": notional_dollars,
            "portfolio_risk_fraction": risk_dollars / equity,
            "portfolio_notional_fraction": notional_dollars / equity,
            "portfolio_pnl": pnl,
            "portfolio_r": realized_r,
            "cost_bps_per_side": cost_rate * 10000,
            "risk_scale": risk_scale,
        })
        sequence += 1
        open_by_ticker[ticker] = position
        heapq.heappush(open_heap, (pd.Timestamp(position["exit_time"]).value, sequence, position))
        accepted.append(position if keep_trades else {
            "market": ticker, "entry_time": position["entry_time"], "exit_time": position["exit_time"],
            "portfolio_pnl": pnl, "portfolio_r": realized_r,
            "holding_hours": position["holding_hours"], "exit_reason": position["exit_reason"],
        })
        daily_entries[day_key] += 1
        open_risk += risk_dollars
        open_notional += notional_dollars
        max_positions_seen = max(max_positions_seen, len(open_by_ticker))
        max_risk_seen = max(max_risk_seen, open_risk / equity if equity else 0.0)
        max_notional_seen = max(max_notional_seen, open_notional / equity if equity else 0.0)

    settle_until(pd.Timestamp.max.tz_localize("UTC"))

    if not accepted:
        return {
            "policy_id": policy.policy_id, "description": policy.description,
            "scanner": policy.scanner, "period": period, "trades": 0,
            "total_return": 0.0, "win_rate": math.nan, "profit_factor": math.nan,
            "mean_r": math.nan, "median_r": math.nan, "avg_winner_r": math.nan,
            "avg_loser_r": math.nan, "max_drawdown": 0.0, "avg_holding_hours": math.nan,
            "longest_losing_streak": 0, "positive_stocks": 0, "stocks_traded": 0,
            "ending_equity": INITIAL_EQUITY, "max_positions_seen": 0,
            "max_open_risk_fraction": 0.0, "max_open_notional_fraction": 0.0,
            "weekly_mean_return": math.nan, "weekly_median_return": math.nan,
            "weekly_positive_rate": math.nan, "weekly_ge_1pct_rate": math.nan,
            "best_week_return": math.nan, "worst_week_return": math.nan,
            "weekend_exposed_trade_rate": math.nan, "average_weekends_held": math.nan,
            "weekend_reduction_rate": math.nan, "gap_stop_rate": math.nan,
            "worst_r": math.nan, "largest_r": math.nan,
        }, [], dict(rejects), []

    pnls = np.asarray([float(r["portfolio_pnl"]) for r in accepted])
    rs = np.asarray([float(r["portfolio_r"]) for r in accepted])
    winners = rs[rs > 0]
    losers = rs[rs < 0]
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(-pnls[pnls < 0].sum())
    stock_pnl = defaultdict(float)
    for row in accepted:
        stock_pnl[str(row["market"])] += float(row["portfolio_pnl"])
    streak = 0
    longest = 0
    for row in sorted(accepted, key=lambda r: (r["exit_time"], r["entry_time"], r["market"])):
        if float(row["portfolio_pnl"]) < 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0

    weekly = defaultdict(float)
    for event in equity_events:
        week = pd.Timestamp(event["time"]).to_period("W-FRI").end_time.date().isoformat()
        weekly[week] += float(event["pnl"])
    weekly_returns = np.asarray([value / INITIAL_EQUITY for value in weekly.values()], dtype=float)

    metric = {
        "policy_id": policy.policy_id, "description": policy.description,
        "scanner": policy.scanner, "period": period, "trades": len(accepted),
        "total_return": equity / INITIAL_EQUITY - 1.0,
        "win_rate": float((pnls > 0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else math.nan),
        "mean_r": float(rs.mean()), "median_r": float(np.median(rs)),
        "avg_winner_r": float(winners.mean()) if len(winners) else math.nan,
        "avg_loser_r": float(losers.mean()) if len(losers) else math.nan,
        "max_drawdown": float(max_drawdown),
        "avg_holding_hours": float(np.mean([float(r["holding_hours"]) for r in accepted])),
        "longest_losing_streak": int(longest),
        "positive_stocks": int(sum(value > 0 for value in stock_pnl.values())),
        "stocks_traded": int(len(stock_pnl)), "ending_equity": float(equity),
        "max_positions_seen": int(max_positions_seen),
        "max_open_risk_fraction": float(max_risk_seen),
        "max_open_notional_fraction": float(max_notional_seen),
        "weekly_mean_return": float(weekly_returns.mean()) if len(weekly_returns) else math.nan,
        "weekly_median_return": float(np.median(weekly_returns)) if len(weekly_returns) else math.nan,
        "weekly_positive_rate": float((weekly_returns > 0).mean()) if len(weekly_returns) else math.nan,
        "weekly_ge_1pct_rate": float((weekly_returns >= 0.01).mean()) if len(weekly_returns) else math.nan,
        "best_week_return": float(weekly_returns.max()) if len(weekly_returns) else math.nan,
        "worst_week_return": float(weekly_returns.min()) if len(weekly_returns) else math.nan,
        "weekend_exposed_trade_rate": float(np.mean([float(r.get("weekends_held", 0)) > 0 for r in accepted])),
        "average_weekends_held": float(np.mean([float(r.get("weekends_held", 0)) for r in accepted])),
        "weekend_reduction_rate": float(np.mean([float(r.get("weekend_reductions", 0)) > 0 for r in accepted])),
        "gap_stop_rate": float(np.mean([str(r.get("exit_reason", "")) == "gap_stop" for r in accepted])),
        "worst_r": float(rs.min()), "largest_r": float(rs.max()),
    }
    return metric, accepted if keep_trades else [], dict(rejects), equity_events
