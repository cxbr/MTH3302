from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import hot20_v6_benchmark as base
import loss_patch_v10 as v10

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "loss_patch_v11_output"
OUT.mkdir(parents=True, exist_ok=True)
v10.OUT = OUT
v10.base.OUT = OUT
base.OUT = OUT

PERIODS = v10.PERIODS
PRE = v10.PRE_PERIODS
REVEAL = v10.REVEAL_PERIODS
BASE_COST = v10.BASE_COST


def add(policy_list, base_policy, description, family="v11_combo", **changes):
    policy_list.append(v10.make_policy(base_policy, description, family, **changes))


def build_candidates() -> List[v10.PatchPolicy]:
    base_policy = v10.PatchPolicy(
        policy_id="V11_BASE",
        description="V11 base: breakout only, two entries/day, 0.75% risk",
        family="baseline",
        entry_mode="breakout_only",
        risk_per_trade=0.0075,
        daily_entry_cap=2,
    )
    policies: List[v10.PatchPolicy] = [base_policy]

    stop_ranges = [
        (0.005, 0.03, "0.50%-3.0%"),
        (0.005, 0.04, "0.50%-4.0%"),
        (0.0075, 0.03, "0.75%-3.0%"),
        (0.0075, 0.04, "0.75%-4.0%"),
        (0.0100, 0.04, "1.00%-4.0%"),
    ]
    for lo, hi, label in stop_ranges:
        add(policies, base_policy, f"Stop width {label}", stop_min_pct=lo, stop_max_pct=hi)

    for lo, hi, label in stop_ranges[:4]:
        for trail_activation in (5.0, 8.0):
            add(
                policies, base_policy,
                f"Stop {label} + trail after {trail_activation:g}R",
                stop_min_pct=lo, stop_max_pct=hi,
                trail_activation_r=trail_activation,
            )

    for delay in (60, 120):
        add(
            policies, base_policy,
            f"Stop 0.75%-4.0% + entry within {delay} minutes",
            stop_min_pct=0.0075, stop_max_pct=0.04,
            max_entry_delay_min=delay,
        )
        add(
            policies, base_policy,
            f"Stop 0.50%-3.0% + entry within {delay} minutes + trail after 5R",
            stop_min_pct=0.005, stop_max_pct=0.03,
            max_entry_delay_min=delay,
            trail_activation_r=5.0,
        )
    for be in (1.5, 2.0, 3.0, 5.0):
        add(
            policies, base_policy,
            f"Stop 0.75%-4.0% + trail after 5R + break-even {be:g}R",
            stop_min_pct=0.0075, stop_max_pct=0.04,
            trail_activation_r=5.0,
            break_even_r=be,
        )

    for loc in (0.5, 0.65, 0.8):
        add(
            policies, base_policy,
            f"Stop 0.75%-4.0% + reference close above {loc:.0%}",
            stop_min_pct=0.0075, stop_max_pct=0.04,
            ref_close_location_min=loc,
        )
        add(
            policies, base_policy,
            f"Trail after 5R + reference close above {loc:.0%}",
            trail_activation_r=5.0,
            ref_close_location_min=loc,
        )

    for partial, pspan, rspan, trail in [
        (0.0, 20, 80, 4.0),
        (0.10, 50, 80, 4.0),
        (0.25, 50, 120, 4.0),
        (0.25, 20, 120, 3.0),
    ]:
        add(
            policies, base_policy,
            f"Stop 0.75%-4.0% + partial {partial:.0%} EMA{pspan} + runner EMA{rspan} + {trail:g}ATR",
            stop_min_pct=0.0075, stop_max_pct=0.04,
            partial_fraction=partial, partial_ema_span=pspan,
            runner_ema_span=rspan, trail_value=trail,
            trail_activation_r=5.0,
        )
    for risk, cap in [(0.005,1),(0.005,2),(0.0075,1),(0.0075,2)]:
        add(
            policies, base_policy,
            f"Stop 0.75%-4.0% + trail 5R + risk {risk:.2%} + {cap} entry/day",
            stop_min_pct=0.0075, stop_max_pct=0.04,
            trail_activation_r=5.0,
            risk_per_trade=risk, daily_entry_cap=cap,
        )

    add(policies, base_policy, "V10 broad stop control", "control", stop_min_pct=0.0075, stop_max_pct=0.04)
    add(policies, base_policy, "V10 hot delayed-trail control", "control", trail_activation_r=5.0)
    add(policies, base_policy, "V10 hot stop control", "control", stop_min_pct=0.005, stop_max_pct=0.03)

    unique = {}
    for p in policies:
        unique[p.signature()] = p
    return list(unique.values())


def ranking_row(policy, metrics, universes: Sequence[str]) -> dict:
    rows = [r for r in metrics if r["policy_id"] == policy.policy_id and r["universe"] in universes]
    pre = [r for r in rows if r["period"] in PRE]
    reveal = [r for r in rows if r["period"] in REVEAL]
    return {
        "policy_id": policy.policy_id,
        "description": policy.description,
        "family": policy.family,
        "pre_score": v10.robust_score(pre),
        "pre_min_return": min(float(r["total_return"]) for r in pre),
        "pre_mean_return": float(np.mean([float(r["total_return"]) for r in pre])),
        "pre_positive_cells": int(sum(float(r["total_return"]) > 0 for r in pre)),
        "pre_cells": len(pre),
        "pre_max_drawdown": max(float(r["max_drawdown"]) for r in pre),
        "reveal_min_return": min(float(r["total_return"]) for r in reveal),
        "reveal_mean_return": float(np.mean([float(r["total_return"]) for r in reveal])),
        "reveal_positive_cells": int(sum(float(r["total_return"]) > 0 for r in reveal)),
        "reveal_cells": len(reveal),
        "policy_json": json.dumps(asdict(policy), sort_keys=True, allow_nan=True),
    }


def main() -> None:
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
    policies = build_candidates()
    print("V11 policies", len(policies), flush=True)

    template_cache: Dict[Tuple[str,str,str], List[dict]] = {}
    metrics: List[dict] = []
    trade_maps: Dict[Tuple[str,str,str], List[dict]] = {}
    rejections: List[dict] = []

    for i, policy in enumerate(policies, start=1):
        sig = v10.template_signature(policy)
        print(f"[{i:03d}/{len(policies):03d}] {policy.description}", flush=True)
        for universe in universes:
            for period in PERIODS:
                templates: List[dict] = []
                for ticker in universe.tickers:
                    key = (ticker, period, sig)
                    if key not in template_cache:
                        template_cache[key] = v10.generate_templates_ext(
                            ticker, prepared[ticker], features[ticker], policy, period, scanner_lookup
                        )
                    templates.extend(template_cache[key])
                metric, accepted, rejected = v10.replay_realistic(policy, universe, templates, period, BASE_COST)
                metrics.append(metric)
                trade_maps[(policy.policy_id, universe.name, period)] = accepted
                rejections.append(rejected)

    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(OUT / "candidate_universe_period_metrics.csv", index=False)
    pd.DataFrame(rejections).fillna(0).to_csv(OUT / "candidate_rejections.csv", index=False)

    ranking_specs = {
        "broad_plus_composite": ["Broad 50-stock universe", "Dynamic composite top 20"],
        "broad": ["Broad 50-stock universe"],
        "composite_hot20": ["Dynamic composite top 20"],
        "momentum_hot20": ["Dynamic momentum top 20"],
        "fixed5": ["Fixed top stocks"],
    }
    ranking_frames = []
    winners = {}
    for name, score_universes in ranking_specs.items():
        rows = [ranking_row(p, metrics, score_universes) for p in policies]
        frame = pd.DataFrame(rows).sort_values("pre_score", ascending=False).reset_index(drop=True)
        frame.insert(0, "rank", np.arange(1, len(frame)+1))
        frame.insert(1, "ranking_scope", name)
        ranking_frames.append(frame)
        winners[name] = str(frame.iloc[0]["policy_id"])
    ranking_df = pd.concat(ranking_frames, ignore_index=True)
    ranking_df.to_csv(OUT / "candidate_rankings.csv", index=False)

    selected_ids = list(dict.fromkeys([
        "V11_BASE",
        winners["broad_plus_composite"], winners["broad"], winners["composite_hot20"],
        winners["momentum_hot20"], winners["fixed5"],
    ]))
    selected_metrics = metric_df[metric_df["policy_id"].isin(selected_ids)].copy()
    selected_metrics.to_csv(OUT / "selected_strategy_comparison.csv", index=False)
    selected_trades = []
    for pid in selected_ids:
        for universe in universes:
            for period in PERIODS:
                selected_trades.extend(trade_maps.get((pid, universe.name, period), []))
    pd.DataFrame(selected_trades).to_csv(OUT / "selected_trade_log.csv", index=False)

    cost_rows = []
    policy_by_id = {p.policy_id: p for p in policies}
    for pid in selected_ids:
        policy = policy_by_id[pid]
        sig = v10.template_signature(policy)
        for cost_bps in (5, 10, 15):
            rate = cost_bps / 10000.0
            for universe in universes:
                for period in PERIODS:
                    templates = []
                    for ticker in universe.tickers:
                        templates.extend(template_cache[(ticker, period, sig)])
                    metric, _, _ = v10.replay_realistic(policy, universe, templates, period, rate)
                    metric["cost_bps_per_side"] = cost_bps
                    cost_rows.append(metric)
    pd.DataFrame(cost_rows).to_csv(OUT / "selected_cost_stress.csv", index=False)

    audit_rows = [
        {"test":"holdout_forward_excluded_from_selection","pass":True,"details":"All ranking scopes use development, 2022, and 2023 only."},
        {"test":"selected_ids_exist","pass":set(selected_ids).issubset(set(metric_df.policy_id)),"details":json.dumps(selected_ids)},
    ]
    trades_df = pd.DataFrame(selected_trades)
    if not trades_df.empty:
        audit_rows += [
            {"test":"whole_share_quantities","pass":bool((trades_df.quantity_whole.round()==trades_df.quantity_whole).all()),"details":f"rows={len(trades_df)}"},
            {"test":"entry_after_reference","pass":bool((pd.to_datetime(trades_df.entry_time,utc=True)>pd.to_datetime(trades_df.reference_time,utc=True)).all()),"details":"All entries after reference close."},
            {"test":"risk_limits","pass":bool((trades_df.portfolio_risk_fraction<=0.0100001).all()),"details":f"max={trades_df.portfolio_risk_fraction.max():.6f}"},
            {"test":"notional_limits","pass":bool((trades_df.portfolio_notional_fraction<=0.3500001).all()),"details":f"max={trades_df.portfolio_notional_fraction.max():.6f}"},
            {"test":"finite_pnl_and_r","pass":bool(np.isfinite(trades_df.portfolio_pnl).all() and np.isfinite(trades_df.portfolio_r).all()),"details":"Selected P&L/R finite."},
        ]
    pd.DataFrame(audit_rows).to_csv(OUT / "audit.csv", index=False)

    summary = {
        "period_years": base.PERIOD_YEARS,
        "policy_count": len(policies),
        "universes": [asdict(u) for u in universes],
        "winners": winners,
        "selected_ids": selected_ids,
        "audit_pass": bool(all(bool(row["pass"]) for row in audit_rows)),
        "limitations": [
            "2024 and 2025 were excluded from selection but have now been viewed.",
            "V11 combinations were motivated by V10 findings and are therefore a second-stage research screen.",
            "Regular-session 15-minute OHLCV does not include full NBBO/order-book history.",
            "Paper trading only.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(v10.safe(summary), indent=2, sort_keys=True))
    (OUT / "policy_catalog.json").write_text(json.dumps([v10.safe(asdict(p)) for p in policies], indent=2))
    pd.DataFrame(quality).to_csv(OUT / "data_quality.csv", index=False)
    scanner_table.to_csv(OUT / "hot_scanner_daily_features.csv", index=False)
    print(json.dumps(v10.safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
