from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

import ten_round_stock_strategy_research as research


def validate_trade_fixed(trade: dict, policy) -> List[str]:
    failures: List[str] = []
    quantity = float(trade["quantity"])
    entry = float(trade["raw_entry"])
    stop = float(trade["stop_raw"])
    entry_exec = float(trade["entry_exec"])
    equity_before = float(trade["equity_before"])
    equity_after = float(trade["equity_after"])
    pnl = float(trade["net_pnl"])
    planned = float(trade["planned_risk_pct"])
    if not quantity > 0:
        failures.append("quantity_not_positive")
    if not 0 < stop < entry:
        failures.append("invalid_stop")
    if not 0 < planned <= 0.010000001:
        failures.append("planned_risk_outside_0_1pct")
    if abs((equity_before + pnl) - equity_after) > 1e-7:
        failures.append("equity_not_reconciled")
    if pd.Timestamp(trade["exit_time"]) < pd.Timestamp(trade["entry_time"]):
        failures.append("exit_before_entry")
    notional_cap = float(getattr(policy, "notional_cap", np.inf))
    if np.isfinite(notional_cap):
        if float(trade["notional_ratio"]) > notional_cap + 1e-7:
            failures.append("notional_cap_exceeded")

    fills = json.loads(trade["fills"])
    amount = sum(float(x["amount"]) for x in fills)
    reconstructed_pnl = sum(
        float(x["amount"]) * (float(x["price"]) - entry_exec)
        for x in fills
    )
    weighted = (
        sum(float(x["amount"]) * float(x["price"]) for x in fills)
        / quantity
    )
    if abs(amount - quantity) > max(1e-8, quantity * 1e-10):
        failures.append("fill_amount_not_reconciled")
    if abs(reconstructed_pnl - pnl) > max(1e-7, abs(pnl) * 1e-10):
        failures.append("fill_pnl_not_reconciled")
    if abs(weighted - float(trade["weighted_exit_exec"])) > max(
        1e-8, abs(weighted) * 1e-10
    ):
        failures.append("weighted_exit_not_reconciled")

    # Only an exit at the original stop must have reached at least 1R adverse
    # excursion. A raised managed stop can be profitable and must not satisfy
    # this condition.
    if trade["exit_reason"] in {"stop", "gap_stop"}:
        if float(trade["mae_r_to_exit"]) < 0.999999:
            failures.append("original_stop_trade_mae_does_not_include_exit_bar")
    return failures


def validate_period_fixed(
    market,
    split: Dict[str, int],
    period: str,
    policy,
    metrics: dict,
    trades: List[dict],
) -> List[dict]:
    start, end = market.period_bounds(split, period)
    failures: List[str] = []
    previous_exit = None
    for trade in sorted(trades, key=lambda x: x["entry_time"]):
        failures.extend(validate_trade_fixed(trade, policy))
        entry_time = pd.Timestamp(trade["entry_time"])
        exit_time = pd.Timestamp(trade["exit_time"])
        entry_i = int(market.index.searchsorted(entry_time, side="left"))
        exit_i = int(market.index.searchsorted(exit_time, side="left"))
        if entry_i < start or entry_i > end or exit_i < start or exit_i > end:
            failures.append("trade_outside_period")
        if previous_exit is not None and entry_time <= previous_exit:
            failures.append("overlapping_trade")
        previous_exit = exit_time

    expected_return = (
        float(trades[-1]["equity_after"]) / 10000.0 - 1.0
        if trades
        else 0.0
    )
    if int(metrics["trades"]) != len(trades):
        failures.append("metric_trade_count_mismatch")
    if abs(float(metrics["total_return"]) - expected_return) > 1e-9:
        failures.append("metric_return_mismatch")
    if trades:
        expected_mean_r = float(
            np.mean([float(trade["r_multiple"]) for trade in trades])
        )
        if abs(float(metrics["mean_r"]) - expected_mean_r) > 1e-9:
            failures.append("metric_mean_r_mismatch")

    return [
        {
            "policy_id": policy.policy_id,
            "market": market.name,
            "period": period,
            "trades": len(trades),
            "pass": not failures,
            "failure_count": len(failures),
            "failures": ";".join(sorted(set(failures))),
        }
    ]


research.verified.validate_period = validate_period_fixed

_original_round1 = research.round1


def round1_without_unverified_percentage_targets(config, baseline):
    candidates = _original_round1(config, baseline)
    return [
        candidate
        for candidate in candidates
        if getattr(candidate.policy, "exit_kind", "") != "target_pct"
    ]


research.round1 = round1_without_unverified_percentage_targets


if __name__ == "__main__":
    research.main()
