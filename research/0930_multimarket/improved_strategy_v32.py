from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import improved_strategy_v3 as v3
import improved_strategy_v31 as v31
from backtest_engine import EntrySignal, PreparedMarket, ReferenceEvent, StrategyConfig, compute_stop
from improved_strategy_v3 import HTFFeatures, TrendExitPolicy


def simulate_trend_trade_v32(
    mkt: PreparedMarket,
    htf: Dict[str, HTFFeatures],
    event: ReferenceEvent,
    signal: EntrySignal,
    cfg: StrategyConfig,
    policy: TrendExitPolicy,
    equity: float,
    period_end: int,
) -> Optional[dict]:
    """Causal trend-trade simulator used by the audited V3.2 benchmark.

    Corrections relative to the earlier research engine:
    - EMA signals are true crosses on complete higher-timeframe bars.
    - Every close-based action executes at the next 15-minute open.
    - A new break-even/trailing stop is active only from the next bar.
    - The 4-hour remainder exit is disabled until the trade has first reached
      the same +R activation threshold used for active management. This makes
      the implementation match the stated rule: no trend management before
      the trade has expanded by at least +2R.
    - Gap stops and the original/managed stop have priority over favorable
      targets when a 15-minute OHLC bar cannot reveal the intrabar path.
    """
    entry_i = signal.entry_idx
    if entry_i > period_end:
        return None
    stop_raw = compute_stop(mkt, signal, cfg)
    if not (np.isfinite(stop_raw) and stop_raw > 0 and stop_raw < signal.raw_entry):
        return None

    cost = mkt.spec.base_cost_rate
    entry_price = v3.entry_exec(signal.raw_entry, cost)
    stop_exec_est = v3.net_exit(stop_raw, cost)
    risk_per_unit = entry_price - stop_exec_est
    if not (risk_per_unit > 0 and np.isfinite(risk_per_unit)):
        return None
    risk_budget = equity * 0.01
    qty = risk_budget / risk_per_unit
    if np.isfinite(policy.notional_cap_multiple):
        qty = min(qty, equity * policy.notional_cap_multiple / entry_price)
    if not (qty > 0 and np.isfinite(qty)):
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
        o, h, l, c = map(float, (mkt.open[i], mkt.high[i], mkt.low[i], mkt.close[i]))

        # pending_stop was calculated only from completed earlier bars.
        active_stop = max(active_stop, pending_stop)

        if o <= active_stop:
            px = v3.net_exit(o, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, "gap_stop"))
            remaining = 0.0
            exit_reason = "gap_stop"
            break

        if scheduled_reason is not None and remaining > 0:
            amount = min(remaining, scheduled_amount)
            px = v3.net_exit(o, cost)
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
            px = v3.net_exit(active_stop, cost)
            realized += remaining * (px - entry_price)
            reason = "managed_stop" if active_stop > stop_raw + 1e-12 else "stop"
            fills.append((i, remaining, px, reason))
            remaining = 0.0
            exit_reason = reason
            break

        if h >= hard_target:
            px = v3.net_exit(hard_target, cost)
            realized += remaining * (px - entry_price)
            fills.append((i, remaining, px, "hard_target"))
            remaining = 0.0
            exit_reason = "hard_target"
            break

        highest = max(highest, h)
        mfe_raw = max(mfe_raw, h)
        mae_raw = min(mae_raw, l)
        current_mfe_r = (highest - signal.raw_entry) / raw_risk
        management_active = current_mfe_r >= policy.partial_activation_r

        if (
            not partial_done
            and management_active
            and partial_features.cross20_down[i]
            and i < period_end
        ):
            amount = remaining * policy.partial_fraction
            scheduled_reason, scheduled_amount = v3._schedule_action(
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
        if management_active and remainder_cross and remaining > 0 and i < period_end:
            scheduled_reason, scheduled_amount = v3._schedule_action(
                scheduled_reason,
                scheduled_amount,
                f"true_{policy.remainder_tf}_ema{policy.remainder_ema_span}_cross",
                remaining,
            )

        # Stop changes are pending until the next 15-minute bar.
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
                scheduled_reason, scheduled_amount = v3._schedule_action(
                    scheduled_reason,
                    scheduled_amount,
                    "max_hold",
                    remaining,
                )
            else:
                px = v3.net_exit(c, cost)
                realized += remaining * (px - entry_price)
                fills.append((i, remaining, px, "max_hold_close"))
                remaining = 0.0
                exit_reason = "max_hold_close"
                break

    if remaining > 0:
        i = min(last_i, period_end)
        px = v3.net_exit(float(mkt.close[i]), cost)
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


def install_patch() -> None:
    # simulate_trend_range keeps its global namespace in improved_strategy_v3,
    # so replacing this symbol patches every V3.1 range/year simulation.
    v3.simulate_trend_trade = simulate_trend_trade_v32


def main() -> None:
    install_patch()
    v31.main()


if __name__ == "__main__":
    main()
