from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_engine import PreparedMarket, StrategyConfig
from data_pipeline import MARKET_SPECS, load_all_markets
from exit_profit_audit import ExitPolicy, build_features, simulate_policy_period
from improved_strategy_v3 import (
    Candidate,
    HTFFeatures,
    TrendExitPolicy,
    aggregate_candidate_metrics,
    complete_years,
    ranking_table,
    simulate_trend_range,
)

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "improved_v31_work"
OUT = ROOT / "improved_v31_output"
OUT.mkdir(parents=True, exist_ok=True)
PERIODS = ["development", "validation", "holdout"]


def aligned_bucket_keys(index: pd.DatetimeIndex, period_minutes: int) -> np.ndarray:
    minutes = (index.hour * 60 + index.minute).to_numpy(dtype=np.int32)
    ordinals = np.asarray([d.toordinal() for d in index.date], dtype=np.int64)
    before_open = minutes < 570
    session_ord = ordinals - before_open.astype(np.int64)
    offset = (minutes - 570) % 1440
    return session_ord * 1000 + offset // period_minutes


def build_complete_htf_features(mkt: PreparedMarket, period_minutes: int) -> HTFFeatures:
    """Build causal higher-timeframe features from complete 15-minute buckets only.

    A 1-hour bar must contain exactly four 15-minute candles and a 4-hour bar
    exactly sixteen. This deliberately excludes shortened end-of-session bars,
    data gaps, and maintenance-break buckets from EMA/ATR calculations.
    """
    expected = period_minutes // 15
    work = mkt.df.copy()
    work["_key"] = aligned_bucket_keys(mkt.index, period_minutes)
    work["_pos"] = np.arange(mkt.n, dtype=int)
    bars = (
        work.groupby("_key", sort=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            end_idx=("_pos", "max"),
            count=("close", "size"),
        )
        .reset_index(drop=True)
    )
    bars = bars[bars["count"] == expected].sort_values("end_idx").reset_index(drop=True)

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

    closes = close_s.to_numpy(dtype=float)
    ema20_np = ema20.to_numpy(dtype=float)
    ema80_np = ema80.to_numpy(dtype=float)
    atr_np = atr14.to_numpy(dtype=float)
    ends = bars["end_idx"].to_numpy(dtype=int)

    for j, idx in enumerate(ends):
        event[idx] = True
        close_arr[idx] = closes[j]
        ema20_arr[idx] = ema20_np[j]
        ema80_arr[idx] = ema80_np[j]
        atr_arr[idx] = atr_np[j]
        if j == 0:
            continue
        if (
            np.isfinite(closes[j - 1])
            and np.isfinite(ema20_np[j - 1])
            and np.isfinite(closes[j])
            and np.isfinite(ema20_np[j])
            and closes[j - 1] >= ema20_np[j - 1]
            and closes[j] < ema20_np[j]
        ):
            cross20[idx] = True
        if (
            np.isfinite(closes[j - 1])
            and np.isfinite(ema80_np[j - 1])
            and np.isfinite(closes[j])
            and np.isfinite(ema80_np[j])
            and closes[j - 1] >= ema80_np[j - 1]
            and closes[j] < ema80_np[j]
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


def strict_entry_config() -> StrategyConfig:
    return StrategyConfig(
        config_id="STRICT_ENTRY",
        family="improved_v31",
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


def trend_entry_config() -> StrategyConfig:
    return replace(
        strict_entry_config(),
        config_id="TREND_ENTRY",
        confirmation="above_close",
        ma_type="sma",
    )


def base_exit(policy_id: str, cap: float, description: str) -> TrendExitPolicy:
    return TrendExitPolicy(
        policy_id=policy_id,
        description=description,
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
        notional_cap_multiple=cap,
    )


def candidate_catalog() -> List[Candidate]:
    strict = strict_entry_config()
    trend = trend_entry_config()
    corrected = ExitPolicy(
        "CORRECTED_BASELINE",
        "Half after first close below EMA20; remainder at 3R or lagged EMA80 touch",
        scale_kind="half_below_ema20",
        exit_kind="target_or_ma_touch_lagged",
        exit_value=3.0,
        ema_span=80,
    )
    primary = base_exit(
        "V31_PRIMARY",
        1.0,
        "Original strict weekly setup; 1x notional cap; no management before +2R; sell 25% on true 1h EMA20 cross; runner uses true 4h EMA20 cross or lagged 3ATR trail; break-even after +5R; +30R cap",
    )
    return [
        Candidate(
            "CONTROL_STRICT_OLD_EXIT",
            "Original strict weekly setup with corrected old fast exit",
            strict,
            control_policy=corrected,
            role="control",
        ),
        Candidate(
            "CONTROL_TREND_FILTER_OLD_EXIT",
            "Development-ranked weekly/SMA filter with corrected old fast exit",
            trend,
            control_policy=corrected,
            role="control",
        ),
        Candidate(
            "V31_PRIMARY",
            primary.description,
            strict,
            exit_policy=primary,
            role="primary",
        ),
        Candidate(
            "V31_TREND_FILTER",
            "Primary cash-capped slow exit with the exploratory above-close/SMA weekly filter",
            trend,
            exit_policy=replace(primary, policy_id="V31_TREND_FILTER"),
            role="performance_variant",
        ),
        Candidate(
            "V31_PRIMARY_2X_CAP",
            "Primary strict setup with a 2x notional cap",
            strict,
            exit_policy=replace(primary, policy_id="V31_PRIMARY_2X_CAP", notional_cap_multiple=2.0),
            role="risk_ablation",
        ),
        Candidate(
            "V31_PRIMARY_UNCAPPED",
            "Primary strict setup with risk-based sizing and no notional cap; research comparison only",
            strict,
            exit_policy=replace(primary, policy_id="V31_PRIMARY_UNCAPPED", notional_cap_multiple=math.nan),
            role="research_only",
        ),
        Candidate(
            "V31_NO_PARTIAL",
            "Primary strict setup with no partial sale; whole position follows the 4h runner",
            strict,
            exit_policy=replace(primary, policy_id="V31_NO_PARTIAL", partial_fraction=0.0),
            role="exit_ablation",
        ),
        Candidate(
            "V31_4H_EMA80",
            "Primary strict setup, but the runner requires a slower true 4h EMA80 cross",
            strict,
            exit_policy=replace(primary, policy_id="V31_4H_EMA80", remainder_ema_span=80),
            role="exit_ablation",
        ),
        Candidate(
            "V31_TRAIL_AFTER_5R",
            "Primary strict setup, but the 4h 3ATR trail is delayed until +5R",
            strict,
            exit_policy=replace(primary, policy_id="V31_TRAIL_AFTER_5R", trail_activation_r=5.0),
            role="exit_ablation",
        ),
    ]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def main() -> None:
    started = time.time()
    frames, quality, splits = load_all_markets(WORK)
    quality.to_csv(OUT / "data_quality.csv", index=False)
    (OUT / "splits.json").write_text(json.dumps(json_safe(splits), indent=2), encoding="utf-8")
    prepared = {name: PreparedMarket(name, frame) for name, frame in frames.items()}
    htf = {
        name: {
            "1h": build_complete_htf_features(mkt, 60),
            "4h": build_complete_htf_features(mkt, 240),
        }
        for name, mkt in prepared.items()
    }
    old_features = {name: build_features(mkt) for name, mkt in prepared.items()}
    candidates = candidate_catalog()
    pd.DataFrame(
        [
            {
                "candidate_id": c.candidate_id,
                "role": c.role,
                "description": c.description,
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
                    start, end = mkt.period_bounds(splits[name], period)
                    metrics, trades = simulate_trend_range(
                        mkt,
                        htf[name],
                        candidate.entry_config,
                        candidate.exit_policy,
                        start,
                        end,
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

    yearly_summary_rows = []
    if not yearly_df.empty:
        for (candidate_id, market), group in yearly_df.groupby(["candidate_id", "market"]):
            nonzero = group[group.trades > 0]
            yearly_summary_rows.append(
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
    yearly_summary = pd.DataFrame(yearly_summary_rows)
    yearly_summary.to_csv(OUT / "yearly_robustness_summary.csv", index=False)

    primary = ranking[ranking.candidate_id == "V31_PRIMARY"].iloc[0].to_dict()
    summary = {
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "markets": list(prepared.keys()),
        "market_categories": {name: MARKET_SPECS[name].category for name in prepared},
        "candidate_count": len(candidates),
        "primary_id": "V31_PRIMARY",
        "primary": json_safe(primary),
        "method_notes": [
            "V31_PRIMARY was frozen before this corrected benchmark and is reported regardless of performance.",
            "The primary preserves the original strict weekly/EMA entry setup and uses a 1x notional cap to avoid hidden leverage on multi-day holds.",
            "Higher-timeframe EMA and ATR calculations use only complete 1h/4h buckets; shortened session bars and missing-data buckets are excluded.",
            "All close-based signals execute on the next 15-minute open, and newly calculated stops become active only on the next bar.",
            "This remains exploratory because the same historical sources were already inspected in prior research passes.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    lines = [
        "IMPROVED WEEKLY + 09:30 STRATEGY V3.1 BENCHMARK",
        "=" * 54,
        f"Markets: {', '.join(prepared.keys())}",
        f"Candidates: {len(candidates)}",
        "Primary: V31_PRIMARY",
        "",
    ]
    for candidate_id in [
        "CONTROL_STRICT_OLD_EXIT",
        "V31_PRIMARY",
        "V31_TREND_FILTER",
        "V31_PRIMARY_2X_CAP",
        "V31_PRIMARY_UNCAPPED",
    ]:
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
                f"avg_hours={row.get(period + '_avg_holding_hours')}, "
                f"avg_notional={row.get(period + '_avg_notional_multiple')}"
            )
        lines.append("")
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
