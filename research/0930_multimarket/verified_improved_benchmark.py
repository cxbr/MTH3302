from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import improved_refinement_benchmark as refine
import improved_system_benchmark as base
import improved_system_bug_audit as audit
from backtest_engine import PreparedMarket, StrategyConfig
from data_pipeline import load_all_markets

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "verified_improved_work"
OUT = ROOT / "verified_improved_output"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
PERIODS = ["development", "validation", "holdout"]


ORIGINAL_SIMULATE_TRADE = base.simulate_trade


def verified_simulate_trade(*args, **kwargs):
    mkt = args[0]
    trade = ORIGINAL_SIMULATE_TRADE(*args, **kwargs)
    if trade is None:
        return None
    trade = audit.corrected_diagnostics(mkt, trade)
    return trade


def policy_catalog() -> List[base.ImprovedPolicy]:
    items: Dict[str, base.ImprovedPolicy] = {}
    for policy in base.policy_catalog() + refine.policies():
        items[policy.policy_id] = policy

    # Small, predeclared sensitivity grid around the safer 1x-notional long-runner.
    for target in [10.0, 15.0, 20.0]:
        for days in [60, 120, 180]:
            policy = base.ImprovedPolicy(
                f"VERIFY_CAP1X_{target:g}R_{days}D",
                f"No partial; 1x notional cap; original stop or {target:g}R target; {days}-day maximum hold",
                exit_kind="target_r",
                exit_value=target,
                max_hold_days=days,
                notional_cap=1.0,
            )
            items[policy.policy_id] = policy
    for pct in [0.10, 0.20, 0.30]:
        policy = base.ImprovedPolicy(
            f"VERIFY_CAP1X_{pct*100:g}PCT_180D",
            f"No partial; 1x notional cap; original stop or {pct*100:g}% price target; 180-day maximum hold",
            exit_kind="target_pct",
            exit_value=pct,
            max_hold_days=180,
            notional_cap=1.0,
        )
        items[policy.policy_id] = policy
    return list(items.values())


def validate_trade(trade: dict, policy: base.ImprovedPolicy) -> List[str]:
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
    if np.isfinite(policy.notional_cap):
        if float(trade["notional_ratio"]) > float(policy.notional_cap) + 1e-7:
            failures.append("notional_cap_exceeded")

    fills = json.loads(trade["fills"])
    amount = sum(float(x["amount"]) for x in fills)
    reconstructed_pnl = sum(
        float(x["amount"]) * (float(x["price"]) - entry_exec)
        for x in fills
    )
    weighted = sum(float(x["amount"]) * float(x["price"]) for x in fills) / quantity
    if abs(amount - quantity) > max(1e-8, quantity * 1e-10):
        failures.append("fill_amount_not_reconciled")
    if abs(reconstructed_pnl - pnl) > max(1e-7, abs(pnl) * 1e-10):
        failures.append("fill_pnl_not_reconciled")
    if abs(weighted - float(trade["weighted_exit_exec"])) > max(1e-8, abs(weighted) * 1e-10):
        failures.append("weighted_exit_not_reconciled")

    if trade["exit_reason"] in {"stop", "managed_stop"} and float(trade["mae_r_to_exit"]) < 0.999999:
        failures.append("stop_trade_mae_does_not_include_exit_bar")
    return failures


def validate_period(
    mkt: PreparedMarket,
    split: Dict[str, int],
    period: str,
    policy: base.ImprovedPolicy,
    metrics: dict,
    trades: List[dict],
) -> List[dict]:
    rows: List[dict] = []
    start, end = mkt.period_bounds(split, period)
    failures: List[str] = []
    previous_exit = None
    for trade in sorted(trades, key=lambda x: x["entry_time"]):
        failures.extend(validate_trade(trade, policy))
        entry_i = int(mkt.index.searchsorted(pd.Timestamp(trade["entry_time"]), side="left"))
        exit_i = int(mkt.index.searchsorted(pd.Timestamp(trade["exit_time"]), side="left"))
        if entry_i < start or entry_i > end or exit_i < start or exit_i > end:
            failures.append("trade_outside_period")
        if previous_exit is not None and pd.Timestamp(trade["entry_time"]) <= previous_exit:
            failures.append("overlapping_trade")
        previous_exit = pd.Timestamp(trade["exit_time"])

    expected_return = (float(trades[-1]["equity_after"]) / 10000.0 - 1.0) if trades else 0.0
    if int(metrics["trades"]) != len(trades):
        failures.append("metric_trade_count_mismatch")
    if abs(float(metrics["total_return"]) - expected_return) > 1e-9:
        failures.append("metric_return_mismatch")
    if trades:
        expected_mean_r = float(np.mean([float(t["r_multiple"]) for t in trades]))
        if abs(float(metrics["mean_r"]) - expected_mean_r) > 1e-9:
            failures.append("metric_mean_r_mismatch")

    rows.append({
        "policy_id": policy.policy_id,
        "market": mkt.name,
        "period": period,
        "trades": len(trades),
        "pass": len(failures) == 0,
        "failure_count": len(failures),
        "failures": ";".join(sorted(set(failures))),
    })
    return rows


def aggregate_market_metrics(market: pd.DataFrame) -> pd.DataFrame:
    return base.aggregate_market_metrics(market)


def ranking_table(aggregate: pd.DataFrame, policies: List[base.ImprovedPolicy]) -> pd.DataFrame:
    descriptions = {p.policy_id: p.description for p in policies}
    rows = []
    for policy_id in aggregate.policy_id.unique():
        row = {"policy_id": policy_id, "description": descriptions[policy_id]}
        for period in PERIODS:
            sub = aggregate[(aggregate.policy_id == policy_id) & (aggregate.period == period)]
            if sub.empty:
                continue
            s = sub.iloc[0]
            for col in [
                "trades", "equal_weight_return", "positive_markets",
                "trade_weighted_mean_r", "worst_market_return",
                "best_market_return", "max_market_drawdown",
                "avg_holding_hours", "median_market_notional_ratio",
                "avg_planned_risk_pct",
            ]:
                row[f"{period}_{col}"] = s[col]
        rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking["development_score"] = (
        ranking.development_equal_weight_return.fillna(-99)
        + 0.015 * ranking.development_trade_weighted_mean_r.fillna(-99)
        + 0.005 * ranking.development_positive_markets.fillna(0)
        - 0.35 * ranking.development_max_market_drawdown.fillna(0)
    )
    ranking["strict_verified_screen"] = (
        (ranking.development_equal_weight_return > 0)
        & (ranking.validation_equal_weight_return > 0)
        & (ranking.holdout_equal_weight_return > 0)
        & (ranking.development_trade_weighted_mean_r > 0)
        & (ranking.validation_trade_weighted_mean_r > 0)
        & (ranking.holdout_trade_weighted_mean_r > 0)
        & (ranking.development_positive_markets >= 3)
        & (ranking.validation_positive_markets >= 3)
        & (ranking.holdout_positive_markets >= 3)
        & (ranking.validation_trades >= 20)
        & (ranking.holdout_trades >= 20)
        & (ranking.validation_max_market_drawdown <= 0.10)
        & (ranking.holdout_max_market_drawdown <= 0.10)
    )
    ranking["stability_floor"] = ranking[
        [
            "development_equal_weight_return",
            "validation_equal_weight_return",
            "holdout_equal_weight_return",
        ]
    ].min(axis=1)
    ranking = ranking.sort_values(
        ["development_score", "development_trades"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranking["development_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def source_hashes() -> dict:
    result = {}
    for name in [
        "improved_system_benchmark.py",
        "improved_refinement_benchmark.py",
        "improved_system_bug_audit.py",
        "verified_improved_benchmark.py",
        "backtest_engine.py",
        "data_pipeline.py",
    ]:
        path = ROOT / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def main() -> None:
    started = time.time()
    audit_summary_path = audit.OUT / "bug_audit_summary.json"
    if not audit_summary_path.exists():
        audit.main()
    bug_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    if not bug_summary.get("all_passed"):
        raise RuntimeError("Bug audit did not pass; verified benchmark aborted")

    base.simulate_trade = verified_simulate_trade
    frames, quality, splits = load_all_markets(WORK)
    prepared = {name: PreparedMarket(name, frame) for name, frame in frames.items()}
    features = {name: base.build_trend_features(mkt) for name, mkt in prepared.items()}
    cfg = StrategyConfig(config_id="EXACT_BASELINE", family="exact_baseline")
    policies = policy_catalog()

    market_rows: List[dict] = []
    trade_rows: List[dict] = []
    verification_rows: List[dict] = []
    for number, policy in enumerate(policies, start=1):
        for name, mkt in prepared.items():
            for period in PERIODS:
                metrics, trades = base.simulate_period(
                    mkt, features[name], cfg, policy, splits[name], period
                )
                market_rows.append(metrics)
                trade_rows.extend(trades)
                verification_rows.extend(
                    validate_period(mkt, splits[name], period, policy, metrics, trades)
                )
        if number == 1 or number % 10 == 0 or number == len(policies):
            print(f"[{number}/{len(policies)}] {policy.policy_id}", flush=True)

    verification = pd.DataFrame(verification_rows)
    if not bool(verification["pass"].all()):
        verification.to_csv(OUT / "verification_checks.csv", index=False)
        raise RuntimeError("Verified benchmark failed trade/ledger invariants")

    market = pd.DataFrame(market_rows)
    aggregate = aggregate_market_metrics(market)
    ranking = ranking_table(aggregate, policies)
    trades = pd.DataFrame(trade_rows)

    pd.DataFrame([asdict(p) for p in policies]).to_csv(OUT / "verified_policy_catalog.csv", index=False)
    market.to_csv(OUT / "verified_market_metrics.csv", index=False)
    aggregate.to_csv(OUT / "verified_aggregate_metrics.csv", index=False)
    ranking.to_csv(OUT / "verified_policy_rankings.csv", index=False)
    trades.to_csv(OUT / "verified_trade_log.csv", index=False)
    verification.to_csv(OUT / "verification_checks.csv", index=False)
    quality.to_csv(OUT / "data_quality.csv", index=False)
    (OUT / "splits.json").write_text(json.dumps(json_safe(splits), indent=2), encoding="utf-8")

    strict = ranking[ranking.strict_verified_screen].sort_values(
        ["stability_floor", "development_score"], ascending=False
    )
    preselected_id = "REF_FULL20R_CAP1X"
    preselected = ranking[ranking.policy_id == preselected_id].iloc[0].to_dict()
    best_dev = ranking.iloc[0].to_dict()
    best_strict = strict.iloc[0].to_dict() if not strict.empty else None
    best_holdout = ranking.sort_values(
        ["holdout_equal_weight_return", "holdout_trade_weighted_mean_r"],
        ascending=False,
    ).iloc[0].to_dict()

    summary = {
        "run_completed_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "markets": list(prepared.keys()),
        "policy_count": len(policies),
        "bug_audit": bug_summary,
        "verification_rows": int(len(verification)),
        "verification_failures": int((~verification["pass"]).sum()),
        "preselected_candidate": preselected,
        "best_development": best_dev,
        "best_strict_verified": best_strict,
        "best_holdout_descriptive": best_holdout,
        "strict_verified_count": int(len(strict)),
        "source_hashes": source_hashes(),
        "important_note": "The code audit passed, but statistical overfitting risk remains because earlier historical holdouts have already been viewed. Paper trading is still required.",
    }
    (OUT / "verified_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )

    lines = [
        "VERIFIED IMPROVED SYSTEM BENCHMARK",
        "=" * 43,
        f"Policies: {len(policies)}",
        f"Bug tests passed: {bug_summary.get('passed')}/{bug_summary.get('tests')}",
        f"Trade/ledger verification rows: {len(verification)}; failures: 0",
        f"Strict verified-screen policies: {len(strict)}",
        "",
        "PRESELECTED CANDIDATE",
        preselected_id,
    ]
    for period in PERIODS:
        lines.append(
            f"{period}: return={preselected.get(period + '_equal_weight_return')}, "
            f"meanR={preselected.get(period + '_trade_weighted_mean_r')}, "
            f"positive_markets={preselected.get(period + '_positive_markets')}/5, "
            f"maxDD={preselected.get(period + '_max_market_drawdown')}, "
            f"avgRisk={preselected.get(period + '_avg_planned_risk_pct')}"
        )
    lines.extend(["", f"Best development: {best_dev.get('policy_id')}"])
    lines.append(f"Best strict verified: {best_strict.get('policy_id') if best_strict else 'NONE'}")
    lines.append(f"Best holdout descriptive: {best_holdout.get('policy_id')}")
    lines.append("")
    lines.append("Known diagnostic issue fixed: MFE/MAE now include the exit bar.")
    lines.append("All EMA close signals execute on the following bar, and trails/locks activate only from prior completed bars.")
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
