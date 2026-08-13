from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import blind_hot20_exit_v13 as v13
import hot20_v6_benchmark as base
from v14_engine import BASE_COST, V14Policy, generate_templates, replay_portfolio

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "hot20_v14_work"
RAW = WORK / "raw"
OUT = ROOT / "blind_hot20_v14_output"
OUT.mkdir(parents=True, exist_ok=True)

PERIOD_YEARS = {
    "y2020": (2020, 2020),
    "y2021": (2021, 2021),
    "y2022": (2022, 2022),
    "y2023": (2023, 2023),
}
SEARCH_PERIODS = ("y2020", "y2021", "y2022")
FINAL_PERIOD = "y2023"
FORBIDDEN_YEARS = {2024, 2025, 2026}


def safe(value):
    return v13.safe(value)


def pid(prefix: str, policy: V14Policy) -> str:
    return v13.stable_id(prefix, asdict(policy))


def with_id(policy: V14Policy, prefix: str) -> V14Policy:
    return replace(policy, policy_id=pid(prefix, policy))


def base_parent(scanner: str, close_location: float, stop_min: float, stop_max: float) -> V14Policy:
    temp = V14Policy(
        policy_id="TEMP",
        description=f"{scanner} top20 close>={close_location:.0%} stop {stop_min:.2%}-{stop_max:.2%}",
        scanner=scanner,
        top_n=20,
        min_scanner_universe=30,
        ref_close_location_min=close_location,
        stop_min_pct=stop_min,
        stop_max_pct=stop_max,
        trail_kind="none",
        break_even_r=3.0,
        partial_fraction=0.25,
        partial_ema_span=20,
        runner_cross_enabled=True,
        runner_ema_span=80,
        max_sessions=60,
        family="stage1_entry",
    )
    return with_id(temp, "E")


def stage1_policies() -> List[V14Policy]:
    policies: List[V14Policy] = []
    for scanner in ["software_heat", "consensus", "composite_v1", "impulse", "activity"]:
        for close_location in [0.50, 0.65, 0.80]:
            for stop_min in [0.005, 0.0075, 0.010, 0.0125]:
                for stop_max in [0.03, 0.04]:
                    policies.append(base_parent(scanner, close_location, stop_min, stop_max))
    policies.extend([
        replace(base_parent("software_heat", 0.65, 0.0075, 0.04), policy_id="V14_V13_BALANCED_CONTROL", description="V13 balanced control"),
        replace(base_parent("consensus", 0.65, 0.0075, 0.04), policy_id="V14_V13_HIGHWR_CONTROL", description="V13 high-WR control"),
    ])
    unique = {}
    for p in policies:
        unique.setdefault(p.trade_signature(), p)
    return list(unique.values())


def exit_profiles(parent: V14Policy) -> List[V14Policy]:
    policies: List[V14Policy] = []

    def add(label: str, **changes):
        temp = replace(parent, policy_id="TEMP", description=f"{parent.description} | {label}", family="stage2_exit", **changes)
        policies.append(with_id(temp, "X"))

    add("V13 high-WR exit", trail_kind="none", break_even_r=3.0, partial_fraction=0.25,
        runner_cross_enabled=True, runner_ema_span=80, max_sessions=60,
        profit_target_fraction=0.0, profit_target_r=math.inf, secondary_target_r=math.inf)
    for be in [1.0, 1.5, 2.0, 3.0, 5.0, math.inf]:
        add(f"EMA80 no trail BE {be}", trail_kind="none", break_even_r=be,
            partial_fraction=0.25, runner_cross_enabled=True, runner_ema_span=80,
            max_sessions=60, profit_target_fraction=0.0)
    for partial in [0.0, 0.25, 0.50, 0.75]:
        for span in [20, 50, 80]:
            add(f"partial {partial:.0%} EMA{span}; EMA80 runner; no trail",
                trail_kind="none", break_even_r=2.0, partial_fraction=partial,
                partial_ema_span=span, runner_cross_enabled=True, runner_ema_span=80,
                max_sessions=30, profit_target_fraction=0.0)

    for target in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for sessions in [2, 5, 10, 20]:
            be = math.inf if target <= 1.0 else min(1.0, target / 2.0)
            add(f"full target {target:g}R max {sessions} sessions",
                profit_target_r=target, profit_target_fraction=1.0,
                secondary_target_r=math.inf, partial_fraction=0.0,
                runner_cross_enabled=False, trail_kind="none", break_even_r=be,
                max_sessions=sessions)

    for first in [1.0, 1.5, 2.0]:
        for fraction in [0.50, 0.75]:
            for second in [2.0, 3.0, 4.0, 5.0]:
                if second <= first:
                    continue
                for sessions in [10, 20]:
                    add(f"take {fraction:.0%}@{first:g}R then {second:g}R max {sessions}",
                        profit_target_r=first, profit_target_fraction=fraction,
                        secondary_target_r=second, partial_fraction=0.0,
                        runner_cross_enabled=False, trail_kind="none",
                        break_even_r=first, max_sessions=sessions)

    for first in [1.0, 1.5, 2.0]:
        for fraction in [0.25, 0.50, 0.75]:
            for sessions in [20, 30, 60]:
                add(f"take {fraction:.0%}@{first:g}R then EMA80 max {sessions}",
                    profit_target_r=first, profit_target_fraction=fraction,
                    secondary_target_r=math.inf, partial_fraction=0.0,
                    runner_cross_enabled=True, runner_ema_span=80,
                    trail_kind="none", break_even_r=first, max_sessions=sessions)

    for stagnation_sessions in [1, 2, 3, 5]:
        for threshold in [0.5, 1.0, 1.5]:
            add(f"exit stagnant after {stagnation_sessions} sessions if MFE<{threshold:g}R",
                stagnation_sessions=stagnation_sessions, stagnation_mfe_r=threshold,
                trail_kind="none", break_even_r=2.0, partial_fraction=0.25,
                runner_cross_enabled=True, runner_ema_span=80, max_sessions=30)
    for trigger, lock in [(1.0, 0.25), (1.5, 0.5), (2.0, 0.5), (2.0, 1.0), (3.0, 1.0), (4.0, 2.0)]:
        add(f"lock {lock:g}R after {trigger:g}R",
            lock_trigger_r=trigger, lock_stop_r=lock,
            trail_kind="none", break_even_r=math.inf, partial_fraction=0.25,
            runner_cross_enabled=True, runner_ema_span=80, max_sessions=30)

    unique = {}
    for p in policies:
        unique.setdefault(p.trade_signature(), p)
    return list(unique.values())


def weekend_profiles(parent: V14Policy) -> List[V14Policy]:
    policies = [replace(parent, family="stage3_weekend")]

    def add(label: str, **changes):
        temp = replace(parent, policy_id="TEMP", description=f"{parent.description} | {label}", family="stage3_weekend", **changes)
        policies.append(with_id(temp, "W"))

    add("block Friday entries", allow_friday_entries=False)
    add("Friday risk 75%", friday_risk_scale=0.75)
    add("Friday risk 50%", friday_risk_scale=0.50)
    add("flat every Friday 15:45", weekend_mode="flat_all")
    for threshold in [0.0, 0.5, 1.0, 2.0]:
        add(f"Friday flat if R<{threshold:g}", weekend_mode="flat_if_below", weekend_threshold_r=threshold)
    for threshold in [0.5, 1.0, 2.0]:
        add(f"Friday flat if unprotected and R<{threshold:g}", weekend_mode="flat_if_unprotected", weekend_threshold_r=threshold)
    for threshold in [0.0, 0.5, 1.0]:
        add(f"Friday reduce half if R<{threshold:g}", weekend_mode="reduce_half_if_below", weekend_threshold_r=threshold)
    unique = {}
    for p in policies:
        unique.setdefault(p.trade_signature(), p)
    return list(unique.values())


def capacity_profiles(parent: V14Policy) -> List[V14Policy]:
    specs = [
        (0.005, 4, 2, 0.04, "risk 0.50%; 4 pos; 2/day"),
        (0.0075, 4, 2, 0.04, "risk 0.75%; 4 pos; 2/day"),
        (0.010, 4, 2, 0.04, "risk 1.00%; 4 pos; 2/day"),
        (0.005, 6, 4, 0.04, "risk 0.50%; 6 pos; 4/day"),
        (0.0075, 6, 4, 0.04, "risk 0.75%; 6 pos; 4/day"),
        (0.005, 8, 4, 0.04, "risk 0.50%; 8 pos; 4/day"),
        (0.0075, 8, 4, 0.06, "risk 0.75%; 8 pos; 4/day; 6% open risk"),
    ]
    policies = []
    for risk, positions, daily, open_risk, label in specs:
        temp = replace(parent, policy_id="TEMP", description=f"{parent.description} | {label}", family="stage4_capacity",
                       risk_per_trade=risk, max_positions=positions, daily_entry_cap=daily,
                       max_open_risk=open_risk)
        policies.append(with_id(temp, "C"))
    unique = {}
    for p in policies:
        payload = asdict(p)
        for key in ["policy_id", "description", "family"]:
            payload.pop(key, None)
        unique.setdefault(json.dumps(payload, sort_keys=True, allow_nan=True), p)
    return list(unique.values())


def growth_score(rows: Sequence[dict]) -> float:
    if len(rows) != len(SEARCH_PERIODS) or any(int(r["trades"]) < 15 for r in rows):
        return -1e9
    returns = np.asarray([float(r["total_return"]) for r in rows])
    if np.any(returns <= -0.25):
        return -1e9
    weights = np.asarray([max(int(r["trades"]), 1) for r in rows], dtype=float)
    wr = float(np.average([float(r["win_rate"]) for r in rows], weights=weights))
    mean_r = float(np.average([float(r["mean_r"]) for r in rows], weights=weights))
    pf = float(np.nanmean([min(float(r["profit_factor"]), 5.0) for r in rows]))
    dd = max(float(r["max_drawdown"]) for r in rows)
    log_growth = float(np.mean(np.log1p(np.clip(returns, -0.95, None))))
    min_return = float(np.min(returns))
    weekly_positive = float(np.nanmean([float(r["weekly_positive_rate"]) for r in rows]))
    return (
        4.0 * log_growth + 2.5 * min_return
        + 0.80 * (wr - 0.30) + 0.20 * mean_r + 0.05 * (pf - 1.0)
        + 0.10 * (weekly_positive - 0.50) - 1.20 * dd
    )


def high_wr_score(rows: Sequence[dict]) -> float:
    if len(rows) != len(SEARCH_PERIODS) or any(int(r["trades"]) < 15 for r in rows):
        return -1e9
    returns = np.asarray([float(r["total_return"]) for r in rows])
    if np.min(returns) <= 0:
        return -1e9
    weights = np.asarray([int(r["trades"]) for r in rows], dtype=float)
    wr = float(np.average([float(r["win_rate"]) for r in rows], weights=weights))
    mean_r = float(np.average([float(r["mean_r"]) for r in rows], weights=weights))
    pf = float(np.nanmean([min(float(r["profit_factor"]), 5.0) for r in rows]))
    dd = max(float(r["max_drawdown"]) for r in rows)
    if mean_r <= 0 or pf <= 1.05:
        return -1e9
    return 2.0 * wr + 1.5 * float(np.min(returns)) + 0.3 * mean_r + 0.05 * pf - dd


def score_policy(policy: V14Policy, metric_rows: Sequence[dict]) -> dict:
    rows = [r for r in metric_rows if r["policy_id"] == policy.policy_id and r["period"] in SEARCH_PERIODS]
    weights = np.asarray([max(int(r["trades"]), 1) for r in rows], dtype=float)
    returns = [float(r["total_return"]) for r in rows]
    return {
        "policy_id": policy.policy_id,
        "description": policy.description,
        "family": policy.family,
        "growth_score": growth_score(rows),
        "high_wr_score": high_wr_score(rows),
        "minimum_return": min(returns) if returns else math.nan,
        "average_return": float(np.mean(returns)) if returns else math.nan,
        "compound_return": float(np.prod([1.0 + r for r in returns]) - 1.0) if returns else math.nan,
        "trades": sum(int(r["trades"]) for r in rows),
        "weighted_win_rate": float(np.average([float(r["win_rate"]) for r in rows], weights=weights)) if rows else math.nan,
        "weighted_mean_r": float(np.average([float(r["mean_r"]) for r in rows], weights=weights)) if rows else math.nan,
        "average_profit_factor": float(np.nanmean([float(r["profit_factor"]) for r in rows])) if rows else math.nan,
        "maximum_drawdown": max(float(r["max_drawdown"]) for r in rows) if rows else math.nan,
        "average_weekly_positive_rate": float(np.nanmean([float(r["weekly_positive_rate"]) for r in rows])) if rows else math.nan,
        "average_weekend_exposure": float(np.nanmean([float(r["weekend_exposed_trade_rate"]) for r in rows])) if rows else math.nan,
        "worst_r": min(float(r["worst_r"]) for r in rows) if rows else math.nan,
        "policy_json": json.dumps(asdict(policy), sort_keys=True, allow_nan=True),
    }


def evaluate_policies(policies, prepared, features, selection, scanner_lookup, periods, cache, stage):
    metrics = []
    trade_map = {}
    rejection_rows = []
    for idx, policy in enumerate(policies, start=1):
        print(f"{stage} [{idx:04d}/{len(policies):04d}] {policy.description}", flush=True)
        for period in periods:
            templates = []
            entry_rejections = defaultdict(int)
            signature = policy.trade_signature()
            for ticker in selection:
                key = (ticker, period, signature)
                if key not in cache:
                    cache[key] = generate_templates(
                        ticker, prepared[ticker], features[ticker], policy, period,
                        scanner_lookup, v13.period_bounds, include_post_exit_diagnostics=True,
                    )
                rows, rej = cache[key]
                templates.extend(rows)
                for name, count in rej.items():
                    entry_rejections[name] += count
            metric, accepted, replay_rej, _ = replay_portfolio(policy, templates, period, BASE_COST, keep_trades=True)
            metrics.append(metric)
            trade_map[(policy.policy_id, period)] = accepted
            rejection_rows.append({
                "stage": stage, "policy_id": policy.policy_id, "period": period,
                **{f"entry_{k}": v for k, v in entry_rejections.items()},
                **{f"portfolio_{k}": v for k, v in replay_rej.items()},
            })
    return metrics, trade_map, rejection_rows


def unique_policies(policies: Sequence[V14Policy]) -> List[V14Policy]:
    unique = {}
    for p in policies:
        payload = asdict(p)
        for key in ["policy_id", "description", "family"]:
            payload.pop(key, None)
        unique.setdefault(json.dumps(payload, sort_keys=True, allow_nan=True), p)
    return list(unique.values())


def main() -> None:
    manifest_path = WORK / "periods.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing train-only manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train only":
        raise RuntimeError("V14 optimization requires train split only")

    v13.WORK = WORK
    v13.RAW = RAW
    v13.OUT = OUT
    v13.PERIOD_YEARS = PERIOD_YEARS
    v13.FORBIDDEN_YEARS = FORBIDDEN_YEARS
    base.WORK = WORK
    base.RAW = RAW
    base.OUT = OUT
    base.PERIOD_YEARS = PERIOD_YEARS
    base._register_market = v13.register_market

    prepared, features, quality, selection = v13.load_train_universe()
    scanner_lookup, scanner_table, scanners = v13.build_scanner_v13(prepared, selection)
    if any(int(y) in FORBIDDEN_YEARS for mkt in prepared.values() for y in np.unique(mkt.year_arr)):
        raise RuntimeError("Forbidden hidden year loaded into optimization")

    cache = {}
    all_metrics = []
    all_rejections = []
    policy_map: Dict[str, V14Policy] = {}

    s1 = stage1_policies()
    policy_map.update({p.policy_id: p for p in s1})
    s1_metrics, _, s1_rej = evaluate_policies(s1, prepared, features, selection, scanner_lookup, SEARCH_PERIODS, cache, "stage1_entry")
    all_metrics.extend(s1_metrics); all_rejections.extend(s1_rej)
    s1_rank = pd.DataFrame([score_policy(p, s1_metrics) for p in s1]).sort_values("growth_score", ascending=False).reset_index(drop=True)
    s1_rank.insert(0, "rank", np.arange(1, len(s1_rank)+1))
    s1_rank.to_csv(OUT / "stage1_entry_rankings.csv", index=False)
    parent_ids = list(dict.fromkeys(
        s1_rank.head(6).policy_id.tolist()
        + s1_rank.sort_values("high_wr_score", ascending=False).head(3).policy_id.tolist()
        + ["V14_V13_HIGHWR_CONTROL"]
    ))
    parents = [policy_map[pid] for pid in parent_ids]

    s2 = unique_policies([candidate for parent in parents for candidate in exit_profiles(parent)])
    policy_map.update({p.policy_id: p for p in s2})
    s2_metrics, _, s2_rej = evaluate_policies(s2, prepared, features, selection, scanner_lookup, SEARCH_PERIODS, cache, "stage2_exit")
    all_metrics.extend(s2_metrics); all_rejections.extend(s2_rej)
    s2_rank = pd.DataFrame([score_policy(p, s2_metrics) for p in s2]).sort_values("growth_score", ascending=False).reset_index(drop=True)
    s2_rank.insert(0, "rank", np.arange(1, len(s2_rank)+1))
    s2_rank.to_csv(OUT / "stage2_exit_rankings.csv", index=False)
    exit_ids = list(dict.fromkeys(
        s2_rank.head(8).policy_id.tolist()
        + s2_rank.sort_values("high_wr_score", ascending=False).head(5).policy_id.tolist()
    ))

    s3 = unique_policies([candidate for parent_id in exit_ids for candidate in weekend_profiles(policy_map[parent_id])])
    policy_map.update({p.policy_id: p for p in s3})
    s3_metrics, _, s3_rej = evaluate_policies(s3, prepared, features, selection, scanner_lookup, SEARCH_PERIODS, cache, "stage3_weekend")
    all_metrics.extend(s3_metrics); all_rejections.extend(s3_rej)
    s3_rank = pd.DataFrame([score_policy(p, s3_metrics) for p in s3]).sort_values("growth_score", ascending=False).reset_index(drop=True)
    s3_rank.insert(0, "rank", np.arange(1, len(s3_rank)+1))
    s3_rank.to_csv(OUT / "stage3_weekend_rankings.csv", index=False)
    weekend_ids = list(dict.fromkeys(
        s3_rank.head(6).policy_id.tolist()
        + s3_rank.sort_values("high_wr_score", ascending=False).head(4).policy_id.tolist()
    ))

    s4 = unique_policies([candidate for parent_id in weekend_ids for candidate in capacity_profiles(policy_map[parent_id])])
    policy_map.update({p.policy_id: p for p in s4})
    s4_metrics, _, s4_rej = evaluate_policies(s4, prepared, features, selection, scanner_lookup, SEARCH_PERIODS, cache, "stage4_capacity")
    all_metrics.extend(s4_metrics); all_rejections.extend(s4_rej)
    s4_rank = pd.DataFrame([score_policy(p, s4_metrics) for p in s4]).sort_values("growth_score", ascending=False).reset_index(drop=True)
    s4_rank.insert(0, "rank", np.arange(1, len(s4_rank)+1))
    s4_rank.to_csv(OUT / "stage4_capacity_rankings.csv", index=False)

    finalist_ids = list(dict.fromkeys(
        ["V14_V13_HIGHWR_CONTROL"]
        + s4_rank.head(8).policy_id.tolist()
        + s4_rank.sort_values("high_wr_score", ascending=False).head(5).policy_id.tolist()
        + s3_rank[s3_rank.description.str.contains("weekend|Friday", case=False, regex=True)].head(3).policy_id.tolist()
    ))
    finalists = [policy_map[pid] for pid in finalist_ids]
    (OUT / "pre_2023_frozen_finalists.json").write_text(json.dumps([safe(asdict(p)) for p in finalists], indent=2), encoding="utf-8")

    final_metrics, _, final_rej = evaluate_policies(finalists, prepared, features, selection, scanner_lookup, [FINAL_PERIOD], cache, "frozen_2023")
    all_metrics.extend(final_metrics); all_rejections.extend(final_rej)

    final_rank_rows = []
    for policy in finalists:
        rows = [r for r in all_metrics if r["policy_id"] == policy.policy_id and r["period"] in (*SEARCH_PERIODS, FINAL_PERIOD)]
        returns = np.asarray([float(r["total_return"]) for r in rows])
        weights = np.asarray([max(int(r["trades"]), 1) for r in rows], dtype=float)
        wr = float(np.average([float(r["win_rate"]) for r in rows], weights=weights))
        mean_r = float(np.average([float(r["mean_r"]) for r in rows], weights=weights))
        dd = max(float(r["max_drawdown"]) for r in rows)
        pf = float(np.nanmean([min(float(r["profit_factor"]), 5.0) for r in rows]))
        compound = float(np.prod(1.0 + returns) - 1.0)
        score = (
            4.0 * float(np.mean(np.log1p(np.clip(returns, -0.95, None))))
            + 2.5 * float(np.min(returns)) + 0.9 * (wr - 0.30)
            + 0.2 * mean_r + 0.05 * (pf - 1.0) - 1.2 * dd
        )
        final_rank_rows.append({
            "policy_id": policy.policy_id, "description": policy.description,
            "final_score": score, "compound_return_2020_2023": compound,
            "minimum_annual_return": float(np.min(returns)),
            "average_annual_return": float(np.mean(returns)),
            "weighted_win_rate": wr, "weighted_mean_r": mean_r,
            "average_profit_factor": pf, "maximum_drawdown": dd,
            "all_four_positive": bool(np.all(returns > 0)),
            "policy_json": json.dumps(asdict(policy), sort_keys=True, allow_nan=True),
        })
    final_rank = pd.DataFrame(final_rank_rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    final_rank.insert(0, "rank", np.arange(1, len(final_rank)+1))
    final_rank.to_csv(OUT / "final_pre2024_rankings.csv", index=False)

    eligible = final_rank[final_rank.all_four_positive]
    if eligible.empty:
        eligible = final_rank
    growth_candidates = eligible[eligible.weighted_win_rate >= 0.35]
    if growth_candidates.empty:
        growth_candidates = eligible
    primary_id = str(growth_candidates.sort_values("final_score", ascending=False).iloc[0].policy_id)
    high_wr_id = str(eligible.sort_values("weighted_win_rate", ascending=False).iloc[0].policy_id)
    weekend_candidates = eligible[eligible.description.str.contains("weekend|Friday", case=False, regex=True)]
    weekend_id = str(weekend_candidates.sort_values("final_score", ascending=False).iloc[0].policy_id) if not weekend_candidates.empty else primary_id
    selected_ids = list(dict.fromkeys(["V14_V13_HIGHWR_CONTROL", primary_id, high_wr_id, weekend_id]))
    selected = [policy_map[pid] for pid in selected_ids]
    (OUT / "frozen_hidden_policies.json").write_text(json.dumps([safe(asdict(p)) for p in selected], indent=2), encoding="utf-8")

    pre_trade_rows = []
    for policy in selected:
        for period in (*SEARCH_PERIODS, FINAL_PERIOD):
            signature = policy.trade_signature()
            templates = []
            for ticker in selection:
                key = (ticker, period, signature)
                if key not in cache:
                    cache[key] = generate_templates(ticker, prepared[ticker], features[ticker], policy, period, scanner_lookup, v13.period_bounds, True)
                templates.extend(cache[key][0])
            _, accepted, _, _ = replay_portfolio(policy, templates, period, BASE_COST, keep_trades=True)
            pre_trade_rows.extend(accepted)
    trade_df = pd.DataFrame(pre_trade_rows)
    trade_df.to_csv(OUT / "pre2024_selected_trade_log.csv", index=False)
    if not trade_df.empty:
        weekend_diag = trade_df.groupby(["policy_id", "period"], dropna=False).agg(
            trades=("portfolio_r", "size"),
            win_rate=("portfolio_r", lambda s: float((s > 0).mean())),
            mean_r=("portfolio_r", "mean"),
            weekend_exposed_rate=("weekends_held", lambda s: float((s > 0).mean())),
            average_weekends=("weekends_held", "mean"),
            gap_stop_rate=("exit_reason", lambda s: float((s == "gap_stop").mean())),
            worst_r=("portfolio_r", "min"),
            largest_r=("portfolio_r", "max"),
        ).reset_index()
        weekend_diag.to_csv(OUT / "pre2024_weekend_risk_diagnostics.csv", index=False)

    cost_rows = []
    for policy in selected:
        for period in (*SEARCH_PERIODS, FINAL_PERIOD):
            signature = policy.trade_signature()
            templates = []
            for ticker in selection:
                key = (ticker, period, signature)
                if key not in cache:
                    cache[key] = generate_templates(ticker, prepared[ticker], features[ticker], policy, period, scanner_lookup, v13.period_bounds, True)
                templates.extend(cache[key][0])
            for bps in [5, 10, 15]:
                metric, _, _, _ = replay_portfolio(policy, templates, period, bps / 10000.0, keep_trades=False)
                metric["cost_bps_per_side"] = bps
                cost_rows.append(metric)
    pd.DataFrame(cost_rows).to_csv(OUT / "pre2024_cost_stress.csv", index=False)

    pd.DataFrame(all_metrics).to_csv(OUT / "all_pre2024_period_metrics.csv", index=False)
    pd.DataFrame(all_rejections).fillna(0).to_csv(OUT / "all_pre2024_rejections.csv", index=False)
    scanner_table.to_parquet(OUT / "scanner_daily_features_pre2024.parquet", index=False)

    audit_rows = [
        {"test":"train_split_only","pass":manifest.get("source_split") == "train only","details":manifest.get("source_bounds")},
        {"test":"no_hidden_year_rows","pass":all(not set(np.unique(mkt.year_arr)).intersection(FORBIDDEN_YEARS) for mkt in prepared.values()),"details":"2024-2026 excluded from optimization"},
        {"test":"2023_finalists_frozen_before_validation","pass":True,"details":finalist_ids},
        {"test":"hidden_policies_written_after_2023_only","pass":True,"details":selected_ids},
        {"test":"whole_share_quantities","pass":bool(trade_df.empty or (trade_df.quantity_whole.round() == trade_df.quantity_whole).all()),"details":f"rows={len(trade_df)}"},
        {"test":"entry_after_reference","pass":bool(trade_df.empty or (pd.to_datetime(trade_df.entry_time, utc=True) > pd.to_datetime(trade_df.reference_time, utc=True)).all()),"details":"All entries after reference"},
        {"test":"risk_limit","pass":bool(trade_df.empty or (trade_df.portfolio_risk_fraction <= 0.0100001).all()),"details":f"max={trade_df.portfolio_risk_fraction.max() if not trade_df.empty else 0}"},
        {"test":"notional_limit","pass":bool(trade_df.empty or (trade_df.portfolio_notional_fraction <= 0.3500001).all()),"details":f"max={trade_df.portfolio_notional_fraction.max() if not trade_df.empty else 0}"},
    ]
    pd.DataFrame(audit_rows).to_csv(OUT / "pre2024_audit.csv", index=False)

    summary = {
        "periods": PERIOD_YEARS, "search_periods": SEARCH_PERIODS,
        "final_validation": FINAL_PERIOD, "forbidden_hidden_years": sorted(FORBIDDEN_YEARS),
        "stock_count": len(selection), "stage1_count": len(s1), "stage2_count": len(s2),
        "stage3_count": len(s3), "stage4_count": len(s4),
        "pre_2023_finalist_ids": finalist_ids, "primary_policy_id": primary_id,
        "high_win_rate_policy_id": high_wr_id, "weekend_policy_id": weekend_id,
        "hidden_selected_ids": selected_ids, "hidden_policy_file": "frozen_hidden_policies.json",
        "optimization_objective": "maximize robust geometric account growth with win-rate, mean-R, weekly-positive-rate and drawdown terms; no rare-runner preservation constraint",
        "mechanical_audit_pass": bool(all(bool(r["pass"]) for r in audit_rows)),
    }
    (OUT / "summary.json").write_text(json.dumps(safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "policy_catalog.json").write_text(json.dumps([safe(asdict(p)) for p in policy_map.values()], indent=2), encoding="utf-8")
    print(json.dumps(safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
