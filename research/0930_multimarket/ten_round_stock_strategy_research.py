from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import improved_system_benchmark as base
import improved_system_bug_audit as audit_core
import verified_improved_benchmark as verified
from backtest_engine import PreparedMarket, StrategyConfig
from stock_heavy_pipeline import load_stock_heavy_markets

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "ten_round_stock_work"
OUT = ROOT / "ten_round_stock_output"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
PERIODS = ["development", "validation", "holdout"]


ORIGINAL_SIMULATE_TRADE = base.simulate_trade


def verified_simulate_trade(*args, **kwargs):
    market = args[0]
    trade = ORIGINAL_SIMULATE_TRADE(*args, **kwargs)
    if trade is None:
        return None
    return audit_core.corrected_diagnostics(market, trade)


base.simulate_trade = verified_simulate_trade


@dataclass(frozen=True)
class Candidate:
    round_number: int
    round_name: str
    candidate_id: str
    description: str
    config: StrategyConfig
    policy: base.ImprovedPolicy
    parent_id: str = ""
    selection_eligible: bool = True


@dataclass
class CandidateResult:
    candidate: Candidate
    aggregate: dict
    market_rows: List[dict]
    trade_rows: List[dict]
    verification_rows: List[dict]


def slug(text: str) -> str:
    keep = []
    for char in text.upper():
        keep.append(char if char.isalnum() else "_")
    return "_".join(filter(None, "".join(keep).split("_")))[:120]


def policy_key(policy: base.ImprovedPolicy) -> str:
    data = asdict(policy)
    data.pop("policy_id", None)
    data.pop("description", None)
    return json.dumps(data, sort_keys=True, default=str)


def config_key(config: StrategyConfig) -> str:
    data = asdict(config)
    data.pop("config_id", None)
    data.pop("family", None)
    return json.dumps(data, sort_keys=True, default=str)


def clone_policy(policy: base.ImprovedPolicy, policy_id: str, description: str, **changes):
    return replace(
        policy,
        policy_id=policy_id,
        description=description,
        **changes,
    )


def make_policy(policy_id: str, description: str, **kwargs):
    valid = {f.name for f in fields(base.ImprovedPolicy)}
    cleaned = {key: value for key, value in kwargs.items() if key in valid}
    return base.ImprovedPolicy(policy_id, description, **cleaned)


def clone_config(config: StrategyConfig, config_id: str, **changes):
    return replace(config, config_id=config_id, **changes)


def aggregate_candidate(candidate: Candidate, rows: pd.DataFrame, market_count: int) -> dict:
    result = {
        "round_number": candidate.round_number,
        "round_name": candidate.round_name,
        "candidate_id": candidate.candidate_id,
        "description": candidate.description,
        "parent_id": candidate.parent_id,
        "selection_eligible": candidate.selection_eligible,
        "config_id": candidate.config.config_id,
        "policy_id": candidate.policy.policy_id,
        "config_json": json.dumps(asdict(candidate.config), sort_keys=True, default=str),
        "policy_json": json.dumps(asdict(candidate.policy), sort_keys=True, default=str),
    }
    for period in PERIODS:
        sub = rows[rows.period == period].copy()
        trades = int(sub.trades.sum())
        mean_r = (
            float(np.average(sub.mean_r.fillna(0.0), weights=sub.trades))
            if trades > 0
            else np.nan
        )
        hold_weights = sub.trades.clip(lower=1)
        result.update(
            {
                f"{period}_trades": trades,
                f"{period}_equal_weight_return": float(sub.total_return.mean()),
                f"{period}_median_market_return": float(sub.total_return.median()),
                f"{period}_positive_markets": int((sub.total_return > 0).sum()),
                f"{period}_trade_weighted_mean_r": mean_r,
                f"{period}_worst_market_return": float(sub.total_return.min()),
                f"{period}_best_market_return": float(sub.total_return.max()),
                f"{period}_max_market_drawdown": float(sub.max_drawdown.max()),
                f"{period}_avg_holding_hours": float(
                    np.average(sub.avg_holding_hours.fillna(0.0), weights=hold_weights)
                ),
            }
        )

    def score(period: str) -> float:
        ret = result[f"{period}_equal_weight_return"]
        median_ret = result[f"{period}_median_market_return"]
        mean_r = result[f"{period}_trade_weighted_mean_r"]
        breadth = result[f"{period}_positive_markets"] / market_count
        worst = result[f"{period}_worst_market_return"]
        best = result[f"{period}_best_market_return"]
        drawdown = result[f"{period}_max_market_drawdown"]
        trades = result[f"{period}_trades"]
        sparse_penalty = max(0.0, (market_count * 4 - trades) / (market_count * 4)) * 0.03
        concentration = max(0.0, best - max(0.05, 4.0 * abs(median_ret)))
        return float(
            ret
            + 0.020 * (0.0 if not np.isfinite(mean_r) else mean_r)
            + 0.020 * breadth
            + 0.30 * worst
            + 0.10 * median_ret
            - 0.35 * drawdown
            - 0.12 * concentration
            - sparse_penalty
        )

    result["development_score"] = score("development")
    result["validation_score"] = score("validation")
    result["holdout_score_descriptive"] = score("holdout")
    result["development_breadth"] = result["development_positive_markets"] / market_count
    result["validation_breadth"] = result["validation_positive_markets"] / market_count
    result["holdout_breadth"] = result["holdout_positive_markets"] / market_count
    return result


def evaluate_candidate(
    candidate: Candidate,
    prepared: Dict[str, PreparedMarket],
    features: Dict[str, base.TrendFeatures],
    splits: Dict[str, Dict[str, int]],
) -> CandidateResult:
    market_rows: List[dict] = []
    trade_rows: List[dict] = []
    verification_rows: List[dict] = []
    for market_name, market in prepared.items():
        for period in PERIODS:
            metrics, trades = base.simulate_period(
                market,
                features[market_name],
                candidate.config,
                candidate.policy,
                splits[market_name],
                period,
            )
            metrics.update(
                {
                    "round_number": candidate.round_number,
                    "round_name": candidate.round_name,
                    "candidate_id": candidate.candidate_id,
                    "candidate_description": candidate.description,
                    "config_id": candidate.config.config_id,
                    "policy_id": candidate.policy.policy_id,
                }
            )
            market_rows.append(metrics)
            for trade in trades:
                trade.update(
                    {
                        "round_number": candidate.round_number,
                        "round_name": candidate.round_name,
                        "candidate_id": candidate.candidate_id,
                        "candidate_description": candidate.description,
                        "period": period,
                    }
                )
            trade_rows.extend(trades)
            checks = verified.validate_period(
                market,
                splits[market_name],
                period,
                candidate.policy,
                metrics,
                trades,
            )
            for check in checks:
                check.update(
                    {
                        "round_number": candidate.round_number,
                        "candidate_id": candidate.candidate_id,
                    }
                )
            verification_rows.extend(checks)
    verification = pd.DataFrame(verification_rows)
    if not verification.empty and not bool(verification["pass"].all()):
        failures = verification[~verification["pass"]]
        raise RuntimeError(
            f"Candidate {candidate.candidate_id} failed ledger verification: "
            f"{failures.to_dict(orient='records')[:3]}"
        )
    market_frame = pd.DataFrame(market_rows)
    aggregate = aggregate_candidate(candidate, market_frame, len(prepared))
    return CandidateResult(
        candidate=candidate,
        aggregate=aggregate,
        market_rows=market_rows,
        trade_rows=trade_rows,
        verification_rows=verification_rows,
    )


def dedupe(candidates: Iterable[Candidate]) -> List[Candidate]:
    seen = set()
    output = []
    for candidate in candidates:
        key = (config_key(candidate.config), policy_key(candidate.policy))
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def make_candidate(
    round_number: int,
    round_name: str,
    description: str,
    config: StrategyConfig,
    policy: base.ImprovedPolicy,
    parent: str = "",
    eligible: bool = True,
) -> Candidate:
    digest = hashlib.sha1(
        (config_key(config) + policy_key(policy)).encode("utf-8")
    ).hexdigest()[:10]
    candidate_id = f"R{round_number:02d}_{slug(description)[:70]}_{digest}"
    return Candidate(
        round_number=round_number,
        round_name=round_name,
        candidate_id=candidate_id,
        description=description,
        config=config,
        policy=policy,
        parent_id=parent,
        selection_eligible=eligible,
    )


def run_round(
    round_number: int,
    round_name: str,
    candidates: Sequence[Candidate],
    prepared: Dict[str, PreparedMarket],
    features: Dict[str, base.TrendFeatures],
    splits: Dict[str, Dict[str, int]],
) -> Tuple[CandidateResult, List[CandidateResult], List[CandidateResult]]:
    candidates = dedupe(candidates)
    print(
        f"\n{'=' * 80}\nROUND {round_number}: {round_name} — {len(candidates)} candidates\n{'=' * 80}",
        flush=True,
    )
    results = []
    for index, candidate in enumerate(candidates, start=1):
        result = evaluate_candidate(candidate, prepared, features, splits)
        results.append(result)
        agg = result.aggregate
        print(
            f"[{index:02d}/{len(candidates):02d}] {candidate.candidate_id}: "
            f"dev={agg['development_equal_weight_return']:+.2%}, "
            f"breadth={agg['development_positive_markets']}/{len(prepared)}, "
            f"meanR={agg['development_trade_weighted_mean_r']:+.3f}, "
            f"score={agg['development_score']:+.4f}",
            flush=True,
        )
    eligible = [r for r in results if r.candidate.selection_eligible]
    eligible.sort(
        key=lambda r: (
            r.aggregate["development_score"],
            r.aggregate["development_positive_markets"],
            r.aggregate["development_equal_weight_return"],
        ),
        reverse=True,
    )
    if not eligible:
        raise RuntimeError(f"Round {round_number} has no selection-eligible candidates")
    champion = eligible[0]
    top = eligible[: min(3, len(eligible))]
    print(
        f"ROUND {round_number} CHAMPION: {champion.candidate.candidate_id}\n"
        f"  {champion.candidate.description}\n"
        f"  development return={champion.aggregate['development_equal_weight_return']:+.2%}, "
        f"positive markets={champion.aggregate['development_positive_markets']}/{len(prepared)}, "
        f"worst market={champion.aggregate['development_worst_market_return']:+.2%}, "
        f"max DD={champion.aggregate['development_max_market_drawdown']:.2%}",
        flush=True,
    )
    return champion, top, results


def round1(config: StrategyConfig, baseline: base.ImprovedPolicy) -> List[Candidate]:
    name = "Holding horizon and target shape"
    policies = [baseline]
    for target in [5.0, 10.0, 15.0, 20.0, 30.0]:
        policies.append(
            make_policy(
                f"R1_FULL_{target:g}R",
                f"No partial exit; original stop or {target:g}R target",
                exit_kind="target_r",
                exit_value=target,
                max_hold_days=180,
            )
        )
    for target in [0.05, 0.10, 0.15, 0.20, 0.30]:
        policies.append(
            make_policy(
                f"R1_FULL_{target*100:g}PCT",
                f"No partial exit; original stop or {target*100:g}% price target",
                exit_kind="target_pct",
                exit_value=target,
                max_hold_days=180,
            )
        )
    for pct in [0.03, 0.05, 0.075, 0.10]:
        policies.append(
            make_policy(
                f"R1_TRAIL_{pct*100:g}PCT_AFTER2R",
                f"No partial; {pct*100:g}% trailing stop activated after +2R",
                exit_kind="pct_trail",
                exit_value=pct,
                activation_r=2.0,
                max_hold_days=180,
            )
        )
    for span in [20, 50]:
        policies.append(
            make_policy(
                f"R1_DAILY_EMA{span}_AFTER2R",
                f"No partial; after +2R exit next open following a daily close below EMA{span}",
                exit_kind="daily_ema",
                daily_ema_span=span,
                activation_r=2.0,
                max_hold_days=180,
            )
        )
    for days in [20, 60, 120, 180]:
        policies.append(
            make_policy(
                f"R1_TIME_{days}D",
                f"No partial; original stop or {days}-day time exit",
                exit_kind="time_only",
                max_hold_days=days,
            )
        )
    return [
        make_candidate(1, name, p.description, config, p) for p in policies
    ]


def round2(champion: CandidateResult) -> List[Candidate]:
    number, name = 2, "Delayed partial exits"
    base_policy = champion.candidate.policy
    config = champion.candidate.config
    candidates = [
        make_candidate(number, name, "Keep round-1 winner with no new scale-out", config, base_policy, champion.candidate.candidate_id)
    ]
    for fraction in [0.25, 0.50]:
        for activation in [1.0, 2.0, 3.0, 5.0]:
            p = clone_policy(
                base_policy,
                f"R2_CROSS_{fraction:g}_{activation:g}R",
                f"Sell {fraction:.0%} only on a true EMA20 cross-under after +{activation:g}R",
                scale_kind="true_cross",
                scale_fraction=fraction,
                scale_activation_r=activation,
            )
            candidates.append(
                make_candidate(number, name, p.description, config, p, champion.candidate.candidate_id)
            )
    for fraction in [0.25, 0.50]:
        for level in [3.0, 5.0, 10.0]:
            p = clone_policy(
                base_policy,
                f"R2_TAKE_{fraction:g}_AT_{level:g}R",
                f"Take {fraction:.0%} at +{level:g}R and keep the remainder under the round-1 exit",
                scale_kind="at_r",
                scale_fraction=fraction,
                scale_level_r=level,
            )
            candidates.append(
                make_candidate(number, name, p.description, config, p, champion.candidate.candidate_id)
            )
    return candidates


def round3(champion: CandidateResult) -> List[Candidate]:
    number, name = 3, "Break-even and profit locks"
    base_policy = champion.candidate.policy
    config = champion.candidate.config
    candidates = [
        make_candidate(number, name, "Keep round-2 winner unchanged", config, base_policy, champion.candidate.candidate_id)
    ]
    for trigger in [1.0, 2.0, 3.0, 5.0]:
        p = clone_policy(
            base_policy,
            f"R3_BE_{trigger:g}R",
            f"Move the active stop to break-even only after +{trigger:g}R",
            break_even_r=trigger,
        )
        candidates.append(make_candidate(number, name, p.description, config, p, champion.candidate.candidate_id))
    locks = [(2.0, 0.5), (3.0, 1.0), (5.0, 2.0), (8.0, 3.0)]
    for trigger, lock in locks:
        p = clone_policy(
            base_policy,
            f"R3_LOCK_{lock:g}R_AT_{trigger:g}R",
            f"After +{trigger:g}R, never give back below +{lock:g}R",
            lock_trigger_r=trigger,
            lock_r=lock,
        )
        candidates.append(make_candidate(number, name, p.description, config, p, champion.candidate.candidate_id))
        p2 = clone_policy(
            base_policy,
            f"R3_BE2_LOCK_{lock:g}R_AT_{trigger:g}R",
            f"Break-even after +2R, then lock +{lock:g}R after +{trigger:g}R",
            break_even_r=2.0,
            lock_trigger_r=trigger,
            lock_r=lock,
        )
        candidates.append(make_candidate(number, name, p2.description, config, p2, champion.candidate.candidate_id))
    return candidates


def round4(champion: CandidateResult) -> List[Candidate]:
    number, name = 4, "Delayed percentage trailing stops"
    config = champion.candidate.config
    base_policy = champion.candidate.policy
    candidates = [
        make_candidate(number, name, "Keep round-3 winner unchanged", config, base_policy, champion.candidate.candidate_id)
    ]
    for pct in [0.02, 0.03, 0.05, 0.075, 0.10]:
        for activation in [2.0, 3.0, 5.0]:
            p = clone_policy(
                base_policy,
                f"R4_TRAIL_{pct*100:g}_AFTER_{activation:g}R",
                f"Use a {pct*100:g}% trailing stop only after +{activation:g}R",
                exit_kind="pct_trail",
                exit_value=pct,
                activation_r=activation,
                max_hold_days=180,
            )
            candidates.append(make_candidate(number, name, p.description, config, p, champion.candidate.candidate_id))
    for pct in [0.03, 0.05, 0.075]:
        p = clone_policy(
            base_policy,
            f"R4_TRAIL_{pct*100:g}_BE2",
            f"Break-even at +2R, then {pct*100:g}% trail after +3R",
            exit_kind="pct_trail",
            exit_value=pct,
            activation_r=3.0,
            break_even_r=2.0,
            max_hold_days=180,
        )
        candidates.append(make_candidate(number, name, p.description, config, p, champion.candidate.candidate_id))
    return candidates


def round5(champion: CandidateResult) -> List[Candidate]:
    number, name = 5, "Daily trend exits"
    config = champion.candidate.config
    base_policy = champion.candidate.policy
    candidates = [
        make_candidate(number, name, "Keep round-4 winner unchanged", config, base_policy, champion.candidate.candidate_id)
    ]
    for span in [20, 50]:
        for activation in [0.0, 1.0, 2.0, 3.0, 5.0]:
            p = clone_policy(
                base_policy,
                f"R5_DAILY_EMA{span}_A{activation:g}",
                f"Exit next open after a daily close below EMA{span}, active after +{activation:g}R",
                exit_kind="daily_ema",
                daily_ema_span=span,
                activation_r=activation,
                max_hold_days=180,
            )
            candidates.append(make_candidate(number, name, p.description, config, p, champion.candidate.candidate_id))
    # Reuse any production-tested ATR/daily policies that exist in the base catalog.
    for existing in base.policy_catalog():
        text = (existing.policy_id + " " + existing.description).lower()
        if "daily" in text or "atr" in text:
            candidates.append(
                make_candidate(
                    number,
                    name,
                    "Production-catalog comparison: " + existing.description,
                    config,
                    existing,
                    champion.candidate.candidate_id,
                )
            )
    return candidates


def round6(champion: CandidateResult) -> List[Candidate]:
    number, name = 6, "Weekly setup quality"
    policy = champion.candidate.policy
    base_cfg = champion.candidate.config
    candidates = [
        make_candidate(number, name, "Keep round-5 weekly setup unchanged", base_cfg, policy, champion.candidate.candidate_id)
    ]
    for value in ["above_high", "above_close", "above_body_top"]:
        cfg = clone_config(base_cfg, f"R6_CONFIRM_{value}", confirmation=value)
        candidates.append(make_candidate(number, name, f"Weekly confirmation: {value}", cfg, policy, champion.candidate.candidate_id))
    for value in ["range", "body"]:
        cfg = clone_config(base_cfg, f"R6_MID_{value}", midpoint=value)
        candidates.append(make_candidate(number, name, f"Weekly 50% midpoint: {value}", cfg, policy, champion.candidate.candidate_id))
    for value in ["weekly_touch", "weekly_close", "intraday_touch"]:
        cfg = clone_config(base_cfg, f"R6_RETRACE_{value}", retracement=value)
        candidates.append(make_candidate(number, name, f"Retracement event: {value}", cfg, policy, champion.candidate.candidate_id))
    for weeks in [13, 26, 52]:
        cfg = clone_config(base_cfg, f"R6_EXPIRY_{weeks}", setup_expiry_weeks=weeks)
        candidates.append(make_candidate(number, name, f"Weekly setup expires after {weeks} weeks", cfg, policy, champion.candidate.candidate_id))
    combos = [
        dict(confirmation="above_close", midpoint="range", retracement="weekly_touch"),
        dict(confirmation="above_body_top", midpoint="body", retracement="weekly_touch"),
        dict(confirmation="above_high", midpoint="range", retracement="intraday_touch"),
        dict(confirmation="above_close", midpoint="body", retracement="weekly_close"),
    ]
    for idx, changes in enumerate(combos, start=1):
        cfg = clone_config(base_cfg, f"R6_COMBO_{idx}", **changes)
        candidates.append(make_candidate(number, name, f"Weekly combination {idx}: {changes}", cfg, policy, champion.candidate.candidate_id))
    return candidates


def round7(champion: CandidateResult) -> List[Candidate]:
    number, name = 7, "20/80 cross and reference timing"
    policy = champion.candidate.policy
    base_cfg = champion.candidate.config
    candidates = [
        make_candidate(number, name, "Keep round-6 20/80 settings unchanged", base_cfg, policy, champion.candidate.candidate_id)
    ]
    for ma_type in ["ema", "sma"]:
        for condition in ["fresh", "already_or_fresh"]:
            cfg = clone_config(
                base_cfg,
                f"R7_{ma_type}_{condition}",
                ma_type=ma_type,
                cross_condition=condition,
            )
            candidates.append(make_candidate(number, name, f"{ma_type.upper()}20/80 with {condition} condition", cfg, policy, champion.candidate.candidate_id))
    for days in [30, 60, 90, 120]:
        cfg = clone_config(base_cfg, f"R7_WAIT_{days}", cross_wait_days=days)
        candidates.append(make_candidate(number, name, f"Wait at most {days} days for the 20/80 condition", cfg, policy, champion.candidate.candidate_id))
    for mode in ["next_day", "next_weekday", "same_or_next"]:
        cfg = clone_config(base_cfg, f"R7_REF_{mode}", reference_mode=mode)
        candidates.append(make_candidate(number, name, f"Reference-candle timing: {mode}", cfg, policy, champion.candidate.candidate_id))
    return candidates


def round8(champion: CandidateResult) -> List[Candidate]:
    number, name = 8, "09:30 entry route and highest-price definition"
    policy = champion.candidate.policy
    base_cfg = champion.candidate.config
    candidates = [
        make_candidate(number, name, "Keep round-7 exact 09:30 entry unchanged", base_cfg, policy, champion.candidate.candidate_id)
    ]
    for source in ["reference", "previous_day", "previous_rth", "previous_week", "since_cross"]:
        cfg = clone_config(base_cfg, f"R8_HIGH_{source}", high_source=source)
        candidates.append(make_candidate(number, name, f"Buy threshold uses {source} high", cfg, policy, champion.candidate.candidate_id))
    for mode in [
        "exact_dual_next_open",
        "exact_dual_reclaim_close",
        "breakout_only",
        "reclaim_only",
        "reclaim_touch_close",
        "dual_keep_both",
        "upper80_recovery",
        "breakout_then_upper80_recovery",
        "upper80_limit",
        "deep80_limit",
    ]:
        cfg = clone_config(base_cfg, f"R8_ENTRY_{mode}", entry_mode=mode)
        candidates.append(make_candidate(number, name, f"09:30 entry interpretation: {mode}", cfg, policy, champion.candidate.candidate_id))
    for days in [1, 3, 5, 7]:
        cfg = clone_config(base_cfg, f"R8_VALID_{days}", entry_valid_days=days)
        candidates.append(make_candidate(number, name, f"Keep the 09:30 signal valid for {days} trading day(s)", cfg, policy, champion.candidate.candidate_id))
    for policy_name in ["adverse_first", "breakout_first"]:
        cfg = clone_config(base_cfg, f"R8_SAMEBAR_{policy_name}", same_bar_policy=policy_name)
        candidates.append(make_candidate(number, name, f"Ambiguous same-bar ordering: {policy_name}", cfg, policy, champion.candidate.candidate_id))
    return candidates


def round9(champion: CandidateResult) -> List[Candidate]:
    number, name = 9, "Stop placement and exposure caps"
    base_cfg = champion.candidate.config
    base_policy = champion.candidate.policy
    candidates = [
        make_candidate(number, name, "Keep round-8 stop and sizing unchanged", base_cfg, base_policy, champion.candidate.candidate_id)
    ]
    for stop in ["structural", "structural_atr", "prior20_atr", "atr1_5"]:
        cfg = clone_config(base_cfg, f"R9_STOP_{stop}", stop_type=stop)
        candidates.append(make_candidate(number, name, f"Initial stop definition: {stop}", cfg, base_policy, champion.candidate.candidate_id))
    for cap in [1.0, 2.0, 3.0]:
        policy = clone_policy(
            base_policy,
            f"R9_CAP_{cap:g}X",
            f"Limit position notional to {cap:g} times account equity",
            notional_cap=cap,
        )
        candidates.append(make_candidate(number, name, policy.description, base_cfg, policy, champion.candidate.candidate_id))
    for multiplier in [0.5, 2.0]:
        cfg = clone_config(base_cfg, f"R9_COST_{multiplier:g}", cost_multiplier=multiplier)
        candidates.append(
            make_candidate(
                number,
                name,
                f"Cost sensitivity only: {multiplier:g}x modeled transaction cost",
                cfg,
                base_policy,
                champion.candidate.candidate_id,
                eligible=False,
            )
        )
    return candidates


def round10(
    policy_pool: Sequence[CandidateResult],
    config_pool: Sequence[CandidateResult],
    latest: CandidateResult,
) -> List[Candidate]:
    number, name = 10, "Recombine finalists and freeze the system"
    candidates = [
        make_candidate(number, name, "Keep round-9 champion unchanged", latest.candidate.config, latest.candidate.policy, latest.candidate.candidate_id)
    ]
    policies = []
    configs = []
    for result in policy_pool:
        if policy_key(result.candidate.policy) not in {policy_key(p) for p in policies}:
            policies.append(result.candidate.policy)
    for result in config_pool:
        if config_key(result.candidate.config) not in {config_key(c) for c in configs}:
            configs.append(result.candidate.config)
    policies = policies[:4]
    configs = configs[:4]
    for p_index, policy in enumerate(policies, start=1):
        for c_index, config in enumerate(configs, start=1):
            candidates.append(
                make_candidate(
                    number,
                    name,
                    f"Final recombination: exit package {p_index} with setup package {c_index}",
                    clone_config(config, f"R10_CFG_{c_index}"),
                    clone_policy(policy, f"R10_POLICY_{p_index}", policy.description),
                    latest.candidate.candidate_id,
                )
            )
    return candidates


def leave_one_market_out(market_metrics: pd.DataFrame, candidate_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for candidate_id in candidate_ids:
        for period in ["validation", "holdout"]:
            sub = market_metrics[
                (market_metrics.candidate_id == candidate_id)
                & (market_metrics.period == period)
            ]
            for excluded in sorted(sub.market.unique()):
                kept = sub[sub.market != excluded]
                trades = int(kept.trades.sum())
                mean_r = (
                    float(np.average(kept.mean_r.fillna(0), weights=kept.trades))
                    if trades else np.nan
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "period": period,
                        "excluded_market": excluded,
                        "markets_remaining": int(len(kept)),
                        "trades": trades,
                        "equal_weight_return": float(kept.total_return.mean()),
                        "positive_markets": int((kept.total_return > 0).sum()),
                        "trade_weighted_mean_r": mean_r,
                        "worst_market_return": float(kept.total_return.min()),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_mean_r(trades: pd.DataFrame, candidate_id: str, period: str, iterations: int = 10000) -> dict:
    subset = trades[(trades.candidate_id == candidate_id) & (trades.period == period)]
    values = subset.r_multiple.to_numpy(dtype=float)
    if len(values) == 0:
        return {"candidate_id": candidate_id, "period": period, "trades": 0}
    rng = np.random.default_rng(20260810 + len(values))
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "candidate_id": candidate_id,
        "period": period,
        "trades": int(len(values)),
        "mean_r": float(values.mean()),
        "median_r": float(np.median(values)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "bootstrap_probability_positive": float((means > 0).mean()),
    }


def main() -> None:
    started = time.time()
    frames, quality, splits, rejected = load_stock_heavy_markets(WORK)
    prepared = {name: PreparedMarket(name, frame) for name, frame in frames.items()}
    trend_features = {name: base.build_trend_features(market) for name, market in prepared.items()}
    print(f"Stock-heavy universe accepted: {list(prepared)}", flush=True)

    exact_config = StrategyConfig(
        config_id="STOCK_EXACT_BASE",
        family="stock_ten_round",
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
        scale_out="half_below20",
        remainder_exit="ma80_or_target",
        target_r=3.0,
        max_hold_days=60,
        cost_multiplier=1.0,
        risk_model="risk_only",
    )
    catalog = {policy.policy_id: policy for policy in base.policy_catalog()}
    baseline = catalog.get("CORRECTED_BASELINE")
    if baseline is None:
        baseline = make_policy(
            "CORRECTED_BASELINE_FALLBACK",
            "Corrected baseline: true EMA20 cross partial and 3R runner",
            scale_kind="true_cross",
            scale_fraction=0.50,
            scale_activation_r=0.0,
            exit_kind="target_r",
            exit_value=3.0,
            max_hold_days=60,
        )

    all_results: List[CandidateResult] = []
    round_champions: List[CandidateResult] = []
    round_top: Dict[int, List[CandidateResult]] = {}

    champion, top, results = run_round(1, "Holding horizon and target shape", round1(exact_config, baseline), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[1] = top

    champion, top, results = run_round(2, "Delayed partial exits", round2(champion), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[2] = top

    champion, top, results = run_round(3, "Break-even and profit locks", round3(champion), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[3] = top

    champion, top, results = run_round(4, "Delayed percentage trailing stops", round4(champion), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[4] = top

    champion, top, results = run_round(5, "Daily trend exits", round5(champion), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[5] = top

    champion, top, results = run_round(6, "Weekly setup quality", round6(champion), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[6] = top

    champion, top, results = run_round(7, "20/80 cross and reference timing", round7(champion), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[7] = top

    champion, top, results = run_round(8, "09:30 entry route and highest-price definition", round8(champion), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[8] = top

    champion, top, results = run_round(9, "Stop placement and exposure caps", round9(champion), prepared, trend_features, splits)
    all_results.extend(results); round_champions.append(champion); round_top[9] = top

    policy_pool = round_top[1] + round_top[2] + round_top[3] + round_top[4] + round_top[5] + round_top[9]
    policy_pool = sorted(policy_pool, key=lambda r: r.aggregate["development_score"], reverse=True)
    config_pool = round_top[6] + round_top[7] + round_top[8]
    config_pool = sorted(config_pool, key=lambda r: r.aggregate["development_score"], reverse=True)

    champion10, top10, results = run_round(
        10,
        "Recombine finalists and freeze the system",
        round10(policy_pool, config_pool, champion),
        prepared,
        trend_features,
        splits,
    )
    all_results.extend(results); round_champions.append(champion10); round_top[10] = top10

    ranking = pd.DataFrame([result.aggregate for result in all_results])
    ranking = ranking.sort_values(
        ["round_number", "development_score"], ascending=[True, False]
    ).reset_index(drop=True)
    market_metrics = pd.DataFrame(
        [row for result in all_results for row in result.market_rows]
    )
    trades = pd.DataFrame(
        [row for result in all_results for row in result.trade_rows]
    )
    verification = pd.DataFrame(
        [row for result in all_results for row in result.verification_rows]
    )

    round_summary = pd.DataFrame(
        [
            {
                **result.aggregate,
                "round_champion": True,
            }
            for result in round_champions
        ]
    )

    # Final selection: use development to create the shortlist, validation to choose,
    # and do not use holdout until the candidate is frozen.
    round10_frame = ranking[
        (ranking.round_number == 10) & ranking.selection_eligible
    ].sort_values("development_score", ascending=False)
    dev_shortlist = round10_frame.head(min(5, len(round10_frame))).copy()
    eligible_validation = dev_shortlist[
        (dev_shortlist.validation_trades >= len(prepared) * 3)
        & (dev_shortlist.validation_positive_markets >= max(3, math.ceil(len(prepared) / 2)))
    ]
    if eligible_validation.empty:
        eligible_validation = dev_shortlist
    selected_row = eligible_validation.sort_values(
        ["validation_score", "validation_positive_markets", "validation_equal_weight_return"],
        ascending=False,
    ).iloc[0]
    selected_id = str(selected_row.candidate_id)

    baseline_id = all_results[0].candidate.candidate_id
    finalist_ids = list(dict.fromkeys([baseline_id] + [r.candidate.candidate_id for r in round_champions] + [selected_id]))
    finalist_ranking = ranking[ranking.candidate_id.isin(finalist_ids)].copy()
    final_market = market_metrics[market_metrics.candidate_id.isin(finalist_ids)].copy()
    loo = leave_one_market_out(market_metrics, finalist_ids)
    bootstrap = pd.DataFrame(
        [
            bootstrap_mean_r(trades, selected_id, period)
            for period in ["validation", "holdout"]
        ]
    )

    ranking.to_csv(OUT / "all_candidate_rankings.csv", index=False)
    round_summary.to_csv(OUT / "round_champions.csv", index=False)
    finalist_ranking.to_csv(OUT / "finalist_rankings.csv", index=False)
    market_metrics.to_csv(OUT / "all_market_metrics.csv", index=False)
    final_market.to_csv(OUT / "finalist_market_metrics.csv", index=False)
    trades.to_csv(OUT / "all_trade_log.csv", index=False)
    verification.to_csv(OUT / "verification_checks.csv", index=False)
    loo.to_csv(OUT / "leave_one_market_out.csv", index=False)
    bootstrap.to_csv(OUT / "selected_bootstrap.csv", index=False)
    quality.to_csv(OUT / "stock_data_quality.csv", index=False)
    rejected.to_csv(OUT / "stock_data_rejected.csv", index=False)

    selected = ranking[ranking.candidate_id == selected_id].iloc[0].to_dict()
    baseline_row = ranking[ranking.candidate_id == baseline_id].iloc[0].to_dict()
    summary = {
        "run_completed_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "rounds_completed": 10,
        "candidate_count": int(len(ranking)),
        "markets": list(prepared.keys()),
        "market_count": len(prepared),
        "rejected_markets": rejected.to_dict(orient="records"),
        "baseline_candidate_id": baseline_id,
        "selected_candidate_id": selected_id,
        "selection_process": "Development ranked the round-10 shortlist; validation selected the frozen candidate; holdout was descriptive only.",
        "baseline": baseline_row,
        "selected": selected,
        "round_champions": round_summary.to_dict(orient="records"),
        "bug_and_ledger_checks": {
            "verification_rows": int(len(verification)),
            "verification_failures": int((~verification["pass"]).sum()) if len(verification) else 0,
            "mfe_mae_exit_bar_fix": True,
        },
        "statistical_warning": "This remains exploratory because multiple strategy families were tested and prior historical results had already been viewed. A fresh forward paper-trading sample is required.",
    }
    (OUT / "ten_round_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "TEN-ROUND STOCK-HEAVY STRATEGY RESEARCH",
        "=" * 48,
        f"Markets accepted: {', '.join(prepared.keys())}",
        f"Rounds completed: 10",
        f"Unique candidates evaluated: {len(ranking)}",
        f"Ledger verification failures: {summary['bug_and_ledger_checks']['verification_failures']}",
        "",
        "ROUND CHAMPIONS",
    ]
    for _, row in round_summary.sort_values("round_number").iterrows():
        lines.append(
            f"Round {int(row.round_number)} — {row.round_name}: {row.candidate_id}\n"
            f"  {row.description}\n"
            f"  development={row.development_equal_weight_return:+.2%}, "
            f"validation={row.validation_equal_weight_return:+.2%}, "
            f"holdout={row.holdout_equal_weight_return:+.2%}, "
            f"holdout breadth={int(row.holdout_positive_markets)}/{len(prepared)}"
        )
    lines.extend(
        [
            "",
            "FROZEN FINAL CANDIDATE",
            selected_id,
            str(selected.get("description")),
            f"Development return: {selected.get('development_equal_weight_return'):+.2%}",
            f"Validation return: {selected.get('validation_equal_weight_return'):+.2%}",
            f"Holdout return: {selected.get('holdout_equal_weight_return'):+.2%}",
            f"Holdout positive markets: {selected.get('holdout_positive_markets')}/{len(prepared)}",
            f"Holdout mean R: {selected.get('holdout_trade_weighted_mean_r'):+.3f}",
            f"Holdout worst market: {selected.get('holdout_worst_market_return'):+.2%}",
            f"Holdout maximum market drawdown: {selected.get('holdout_max_market_drawdown'):.2%}",
            "",
            "IMPORTANT",
            "Holdout performance did not select the candidate. The experiment is still exploratory and should remain paper-trading research only.",
        ]
    )
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
