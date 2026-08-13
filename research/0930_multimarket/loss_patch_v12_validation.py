from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import hot20_v6_benchmark as base
import loss_patch_v10 as v10

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "loss_patch_v12_output"
OUT.mkdir(parents=True, exist_ok=True)
v10.OUT = OUT
v10.base.OUT = OUT
base.OUT = OUT

PERIODS = v10.PERIODS


def candidate_policies():
    base_policy = v10.PatchPolicy(
        policy_id="V12_V8_CONTROL",
        description="V8 control: breakout, 0.75% risk, two entries/day",
        family="control",
        entry_mode="breakout_only",
        risk_per_trade=0.0075,
        daily_entry_cap=2,
    )
    combo = v10.make_policy(
        base_policy,
        "V12 loss patch: 0.75%-4.0% stop and trail starts at 5R",
        "v12_candidate",
        stop_min_pct=0.0075,
        stop_max_pct=0.04,
        trail_activation_r=5.0,
    )
    safe = v10.make_policy(
        base_policy,
        "V12 safe patch: 0.75%-4.0% stop, trail 5R, 0.50% risk",
        "v12_candidate",
        stop_min_pct=0.0075,
        stop_max_pct=0.04,
        trail_activation_r=5.0,
        risk_per_trade=0.005,
    )
    be2 = v10.make_policy(
        base_policy,
        "V12 BE2 patch: 0.75%-4.0% stop, trail 5R, break-even 2R",
        "v12_candidate",
        stop_min_pct=0.0075,
        stop_max_pct=0.04,
        trail_activation_r=5.0,
        break_even_r=2.0,
    )
    return [base_policy, combo, safe, be2]


def main():
    payload = json.loads((ROOT / "hot20_v6_work" / "periods.json").read_text())
    base.PERIOD_YEARS = {k: tuple(v) for k, v in payload["periods"].items()}
    prepared, features, quality, selection, controls = base.load_universe()
    scanner_lookup, scanner_table = base.build_hot_scanner(prepared, selection)
    fixed5 = tuple(t for t in v10.FIXED5_REQUESTED if t in prepared)
    universes = [
        v10.UniverseSpec("Fixed top stocks", fixed5, "none", 10000, 0.0, 0),
        v10.UniverseSpec("Broad 50-stock universe", tuple(selection), "none", 10000, 0.0, 0),
        v10.UniverseSpec("Dynamic composite top 20", tuple(selection), "composite", 20, 0.0, 30),
        v10.UniverseSpec("Dynamic momentum top 20", tuple(selection), "momentum", 20, 0.0, 30),
    ]
    policies = candidate_policies()
    templates_cache: Dict[Tuple[str,str,str], List[dict]] = {}
    metric_rows = []
    cost_rows = []
    base_trades = []
    trade_map = {}

    for policy in policies:
        sig = v10.template_signature(policy)
        print("POLICY", policy.description, flush=True)
        for universe in universes:
            for period in PERIODS:
                templates = []
                for ticker in universe.tickers:
                    key = (ticker, period, sig)
                    if key not in templates_cache:
                        templates_cache[key] = v10.generate_templates_ext(
                            ticker, prepared[ticker], features[ticker], policy, period, scanner_lookup
                        )
                    templates.extend(templates_cache[key])
                for bps in (5, 10, 15, 20):
                    metric, accepted, rejected = v10.replay_realistic(
                        policy, universe, templates, period, bps / 10000.0
                    )
                    metric["cost_bps_per_side"] = bps
                    cost_rows.append(metric)
                    if bps == 5:
                        metric_rows.append(metric)
                        trade_map[(policy.policy_id, universe.name, period)] = accepted
                        base_trades.extend(accepted)

    metric_df = pd.DataFrame(metric_rows)
    cost_df = pd.DataFrame(cost_rows)
    trades_df = pd.DataFrame(base_trades)
    metric_df.to_csv(OUT / "strategy_period_metrics.csv", index=False)
    cost_df.to_csv(OUT / "cost_stress.csv", index=False)
    trades_df.to_csv(OUT / "trade_log.csv", index=False)

    stock_rows = []
    quarter_rows = []
    loo_rows = []
    for (pid, universe_name, period), rows in trade_map.items():
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        for market, group in frame.groupby("market"):
            stock_rows.append({
                "policy_id": pid, "policy_description": group.policy_description.iloc[0],
                "universe": universe_name, "period": period, "market": market,
                "trades": len(group), "pnl": float(group.portfolio_pnl.sum()),
                "contribution_return": float(group.portfolio_pnl.sum() / 10000.0),
                "mean_r": float(group.portfolio_r.mean()),
                "win_rate": float((group.portfolio_r > 0).mean()),
                "worst_r": float(group.portfolio_r.min()), "largest_r": float(group.portfolio_r.max()),
            })
        frame["quarter"] = pd.to_datetime(frame.exit_time, utc=True).dt.to_period("Q").astype(str)
        for quarter, group in frame.groupby("quarter"):
            quarter_rows.append({
                "policy_id": pid, "policy_description": group.policy_description.iloc[0],
                "universe": universe_name, "period": period, "quarter": quarter,
                "trades": len(group), "pnl": float(group.portfolio_pnl.sum()),
                "contribution_return": float(group.portfolio_pnl.sum() / 10000.0),
            })
        total_return = float(frame.portfolio_pnl.sum() / 10000.0)
        for market, group in frame.groupby("market"):
            removed = float(group.portfolio_pnl.sum() / 10000.0)
            loo_rows.append({
                "policy_id": pid, "policy_description": group.policy_description.iloc[0],
                "universe": universe_name, "period": period, "removed_market": market,
                "full_return": total_return, "removed_contribution": removed,
                "return_without_market": total_return - removed,
            })
    pd.DataFrame(stock_rows).to_csv(OUT / "stock_contributions.csv", index=False)
    pd.DataFrame(quarter_rows).to_csv(OUT / "quarterly_contributions.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(OUT / "leave_one_stock_out.csv", index=False)

    audit_rows = [
        {"test":"whole_share_quantities","pass":bool((trades_df.quantity_whole.round()==trades_df.quantity_whole).all()),"details":f"rows={len(trades_df)}"},
        {"test":"entry_after_reference","pass":bool((pd.to_datetime(trades_df.entry_time,utc=True)>pd.to_datetime(trades_df.reference_time,utc=True)).all()),"details":"All entries occur after the reference candle."},
        {"test":"risk_limits","pass":bool((trades_df.portfolio_risk_fraction<=0.0100001).all()),"details":f"max={trades_df.portfolio_risk_fraction.max():.6f}"},
        {"test":"notional_limits","pass":bool((trades_df.portfolio_notional_fraction<=0.3500001).all()),"details":f"max={trades_df.portfolio_notional_fraction.max():.6f}"},
        {"test":"finite_metrics","pass":bool(np.isfinite(metric_df.total_return).all() and np.isfinite(metric_df.ending_equity).all()),"details":"All base-cost metrics finite."},
        {"test":"holdout_forward_not_used_for_policy_creation","pass":False,"details":"V12 candidates were motivated by V10/V11 reveal behavior; results are exploratory only."},
    ]
    pd.DataFrame(audit_rows).to_csv(OUT / "audit.csv", index=False)

    summary = {
        "policies": [v10.safe(asdict(p)) for p in policies],
        "universes": [v10.safe(asdict(u)) for u in universes],
        "periods": base.PERIOD_YEARS,
        "audit_all_mechanical_checks_pass": bool(all(bool(r["pass"]) for r in audit_rows[:-1])),
        "selection_status": "post-reveal exploratory validation",
        "limitations": [
            "The candidate combination was motivated by V10/V11 results after 2024/2025 were viewed.",
            "Cost stress does not recreate the full historical order book.",
            "Leave-one-stock-out is an attribution diagnostic and does not reallocate freed capacity.",
            "Paper trading only.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    pd.DataFrame(quality).to_csv(OUT / "data_quality.csv", index=False)
    scanner_table.to_csv(OUT / "hot_scanner_daily_features.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
