from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import improved_system_benchmark as base
from backtest_engine import PreparedMarket, StrategyConfig
from data_pipeline import load_all_markets

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "improved_refinement_work"
OUT = ROOT / "improved_refinement_output"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
PERIODS = ["development", "validation", "holdout"]


def policies() -> List[base.ImprovedPolicy]:
    p = [
        base.ImprovedPolicy(
            "REF_CONTROL_FULL20R_120D",
            "Control: no partial, original stop or 20R target, 120-day maximum hold",
            exit_kind="target_r", exit_value=20.0, max_hold_days=120,
        ),
        base.ImprovedPolicy(
            "REF_FULL20R_180D",
            "No partial, original stop or 20R target, 180-day maximum hold",
            exit_kind="target_r", exit_value=20.0, max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_FULL20R_BE2R",
            "20R target; move stop to break-even after 2R",
            exit_kind="target_r", exit_value=20.0,
            break_even_r=2.0, max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_FULL20R_LOCK1_AT3R",
            "20R target; break-even at 2R and lock at least 1R after 3R",
            exit_kind="target_r", exit_value=20.0,
            break_even_r=2.0, lock_trigger_r=3.0, lock_r=1.0,
            max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_FULL20R_LOCK2_AT5R",
            "20R target; break-even at 2R and lock at least 2R after 5R",
            exit_kind="target_r", exit_value=20.0,
            break_even_r=2.0, lock_trigger_r=5.0, lock_r=2.0,
            max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_FULL10R_LOCK1_AT3R",
            "10R target; break-even at 2R and lock at least 1R after 3R",
            exit_kind="target_r", exit_value=10.0,
            break_even_r=2.0, lock_trigger_r=3.0, lock_r=1.0,
            max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_TRAIL3PCT_AFTER2R_BE2R",
            "No partial; 3% trail after 2R and break-even after 2R",
            exit_kind="pct_trail", exit_value=0.03,
            activation_r=2.0, break_even_r=2.0, max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_TRAIL3PCT_AFTER2R_LOCK1",
            "No partial; 3% trail after 2R, break-even at 2R, lock 1R after 3R",
            exit_kind="pct_trail", exit_value=0.03,
            activation_r=2.0, break_even_r=2.0,
            lock_trigger_r=3.0, lock_r=1.0, max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_CROSS25_AFTER2R_REST20R",
            "Sell 25% only on a true EMA20 cross after 2R; runner to 20R; break-even at 2R and lock 1R after 3R",
            scale_kind="true_cross", scale_fraction=0.25,
            scale_activation_r=2.0, exit_kind="target_r", exit_value=20.0,
            break_even_r=2.0, lock_trigger_r=3.0, lock_r=1.0,
            max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_CROSS50_AFTER2R_REST20R",
            "Sell 50% only on a true EMA20 cross after 2R; runner to 20R; break-even at 2R and lock 1R after 3R",
            scale_kind="true_cross", scale_fraction=0.50,
            scale_activation_r=2.0, exit_kind="target_r", exit_value=20.0,
            break_even_r=2.0, lock_trigger_r=3.0, lock_r=1.0,
            max_hold_days=180,
        ),
        base.ImprovedPolicy(
            "REF_TAKE25_AT5R_REST20R",
            "Take 25% at 5R; runner to 20R; break-even at 2R and lock 1R after 3R",
            scale_kind="at_r", scale_fraction=0.25, scale_level_r=5.0,
            exit_kind="target_r", exit_value=20.0,
            break_even_r=2.0, lock_trigger_r=3.0, lock_r=1.0,
            max_hold_days=180,
        ),
    ]
    for cap in [1.0, 2.0, 3.0]:
        p.extend([
            base.ImprovedPolicy(
                f"REF_FULL20R_CAP{cap:g}X",
                f"20R target with notional capped at {cap:g}x equity",
                exit_kind="target_r", exit_value=20.0,
                max_hold_days=180, notional_cap=cap,
            ),
            base.ImprovedPolicy(
                f"REF_FULL20R_LOCK1_CAP{cap:g}X",
                f"20R target, break-even at 2R, lock 1R after 3R, cap at {cap:g}x equity",
                exit_kind="target_r", exit_value=20.0,
                break_even_r=2.0, lock_trigger_r=3.0, lock_r=1.0,
                max_hold_days=180, notional_cap=cap,
            ),
            base.ImprovedPolicy(
                f"REF_TRAIL3PCT_BE2R_CAP{cap:g}X",
                f"3% trail after 2R with break-even and notional capped at {cap:g}x equity",
                exit_kind="pct_trail", exit_value=0.03,
                activation_r=2.0, break_even_r=2.0,
                max_hold_days=180, notional_cap=cap,
            ),
            base.ImprovedPolicy(
                f"REF_CROSS25_REST20R_CAP{cap:g}X",
                f"Sell 25% on true cross after 2R; runner to 20R; lock 1R after 3R; cap at {cap:g}x equity",
                scale_kind="true_cross", scale_fraction=0.25,
                scale_activation_r=2.0, exit_kind="target_r", exit_value=20.0,
                break_even_r=2.0, lock_trigger_r=3.0, lock_r=1.0,
                max_hold_days=180, notional_cap=cap,
            ),
        ])
    return p


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


def main():
    started = time.time()
    frames, quality, splits = load_all_markets(WORK)
    prepared = {name: PreparedMarket(name, frame) for name, frame in frames.items()}
    features = {name: base.build_trend_features(mkt) for name, mkt in prepared.items()}
    cfg = StrategyConfig(config_id="EXACT_BASELINE", family="exact_baseline")
    catalog = policies()

    rows, trades = [], []
    for number, policy in enumerate(catalog, start=1):
        for name, mkt in prepared.items():
            for period in PERIODS:
                metrics, period_trades = base.simulate_period(
                    mkt, features[name], cfg, policy, splits[name], period
                )
                rows.append(metrics)
                trades.extend(period_trades)
        print(f"[{number}/{len(catalog)}] {policy.policy_id}", flush=True)

    market = pd.DataFrame(rows)
    aggregate = base.aggregate_market_metrics(market)
    ranking = base.ranking_table(aggregate, catalog)

    pd.DataFrame([asdict(p) for p in catalog]).to_csv(OUT / "refinement_policy_catalog.csv", index=False)
    market.to_csv(OUT / "refinement_market_metrics.csv", index=False)
    aggregate.to_csv(OUT / "refinement_aggregate_metrics.csv", index=False)
    ranking.to_csv(OUT / "refinement_policy_rankings.csv", index=False)
    pd.DataFrame(trades).to_csv(OUT / "refinement_trade_log.csv", index=False)
    quality.to_csv(OUT / "data_quality.csv", index=False)
    (OUT / "splits.json").write_text(json.dumps(json_safe(splits), indent=2), encoding="utf-8")

    stable = ranking[
        (ranking.development_equal_weight_return > 0)
        & (ranking.validation_equal_weight_return > 0)
        & (ranking.holdout_equal_weight_return > 0)
        & (ranking.development_positive_markets >= 3)
        & (ranking.validation_positive_markets >= 3)
        & (ranking.holdout_positive_markets >= 3)
    ].sort_values(["stability_floor", "development_score"], ascending=False)
    top_dev = ranking.iloc[0].to_dict()
    top_stable = stable.iloc[0].to_dict() if not stable.empty else None
    top_holdout = ranking.sort_values(
        ["holdout_equal_weight_return", "holdout_trade_weighted_mean_r"],
        ascending=False,
    ).iloc[0].to_dict()

    summary = {
        "run_completed_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "policy_count": len(catalog),
        "markets": list(prepared.keys()),
        "best_development": top_dev,
        "best_positive_breadth": top_stable,
        "best_holdout_descriptive": top_holdout,
        "positive_breadth_count": int(len(stable)),
        "important_note": "All results are exploratory because the historical holdout was viewed in earlier rounds.",
    }
    (OUT / "refinement_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )

    lines = [
        "IMPROVED SYSTEM REFINEMENT BENCHMARK",
        "=" * 44,
        f"Policies: {len(catalog)}",
        f"Positive-breadth policies: {len(stable)}",
        f"Best development: {top_dev.get('policy_id')}",
        f"Best positive-breadth: {top_stable.get('policy_id') if top_stable else 'NONE'}",
        f"Best holdout descriptive: {top_holdout.get('policy_id')}",
        "",
    ]
    for label, row in [("BEST DEVELOPMENT", top_dev), ("BEST POSITIVE BREADTH", top_stable), ("BEST HOLDOUT", top_holdout)]:
        if row is None:
            continue
        lines.extend([label, str(row.get("policy_id"))])
        for period in PERIODS:
            lines.append(
                f"{period}: return={row.get(period + '_equal_weight_return')}, "
                f"meanR={row.get(period + '_trade_weighted_mean_r')}, "
                f"positive_markets={row.get(period + '_positive_markets')}/5, "
                f"maxDD={row.get(period + '_max_market_drawdown')}, "
                f"risk={row.get(period + '_avg_planned_risk_pct')}"
            )
        lines.append("")
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
