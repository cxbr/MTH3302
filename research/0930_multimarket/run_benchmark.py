from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine import (
    PreparedMarket,
    aggregate_metrics,
    catalog,
    rank_and_select,
    run_synthetic_tests,
    simulate_period,
)
from data_pipeline import MARKET_SPECS, load_all_markets

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
PERIODS = ["development", "validation", "holdout"]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def row_for(aggregate: pd.DataFrame, config_id: str, period: str) -> dict:
    found = aggregate[
        (aggregate.config_id == config_id) & (aggregate.period == period)
    ]
    return _json_safe(found.iloc[0].to_dict()) if not found.empty else {}


def main() -> None:
    started = time.time()
    tests = run_synthetic_tests()
    tests.to_csv(OUT / "synthetic_tests.csv", index=False)
    if not bool(tests["pass"].all()):
        raise RuntimeError(f"Synthetic tests failed:\n{tests}")

    print("Loading and validating five market histories...", flush=True)
    frames, quality, splits = load_all_markets(WORK)
    quality.to_csv(OUT / "data_quality.csv", index=False)
    (OUT / "splits.json").write_text(
        json.dumps(_json_safe(splits), indent=2), encoding="utf-8"
    )

    prepared = {
        name: PreparedMarket(name, frame) for name, frame in frames.items()
    }
    configs = catalog()
    config_df = pd.DataFrame([asdict(config) for config in configs])
    config_df.to_csv(OUT / "config_catalog.csv", index=False)
    print(
        f"Benchmarking {len(configs)} unique configurations across "
        f"{len(prepared)} markets and {len(PERIODS)} periods...",
        flush=True,
    )

    market_rows = []
    exact_trades = []
    for config_number, config in enumerate(configs, start=1):
        config_started = time.time()
        for name, market in prepared.items():
            split = splits[name]
            for period in PERIODS:
                metrics, trades = simulate_period(
                    market, config, split, period
                )
                market_rows.append(metrics)
                if config.config_id == "EXACT_BASELINE":
                    exact_trades.extend(trades)
        if (
            config_number == 1
            or config_number % 10 == 0
            or config_number == len(configs)
        ):
            elapsed = time.time() - started
            print(
                f"[{config_number:>3}/{len(configs)}] {config.config_id} "
                f"completed in {time.time() - config_started:.1f}s; "
                f"total {elapsed / 60:.1f}m",
                flush=True,
            )

    market_metrics = pd.DataFrame(market_rows)
    market_metrics.to_csv(OUT / "market_metrics.csv", index=False)
    pd.DataFrame(exact_trades).to_csv(
        OUT / "exact_trade_log.csv", index=False
    )
    aggregate = aggregate_metrics(market_metrics)
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)

    ranking, selected_id = rank_and_select(aggregate, configs)
    ranking = ranking.merge(
        config_df, on=["config_id", "family"], how="left"
    )
    ranking.to_csv(OUT / "configuration_rankings.csv", index=False)

    selected_config = next(
        config for config in configs if config.config_id == selected_id
    )
    selected_trades = []
    selected_market_rows = []
    for name, market in prepared.items():
        for period in PERIODS:
            metrics, trades = simulate_period(
                market, selected_config, splits[name], period
            )
            selected_market_rows.append(metrics)
            selected_trades.extend(trades)
    pd.DataFrame(selected_trades).to_csv(
        OUT / "selected_trade_log.csv", index=False
    )
    pd.DataFrame(selected_market_rows).to_csv(
        OUT / "selected_market_metrics.csv", index=False
    )

    exact_market = market_metrics[
        market_metrics.config_id == "EXACT_BASELINE"
    ].copy()
    exact_market.to_csv(OUT / "exact_market_metrics.csv", index=False)

    top_development = ranking.sort_values("development_rank").head(25)
    top_development.to_csv(
        OUT / "top25_development_ranked.csv", index=False
    )
    top_holdout = ranking.sort_values(
        [
            "holdout_equal_weight_return",
            "holdout_trade_weighted_mean_r",
            "holdout_trades",
        ],
        ascending=[False, False, False],
    ).head(25)
    top_holdout.to_csv(
        OUT / "top25_holdout_descriptive.csv", index=False
    )

    family_rows = []
    for family, group in ranking.groupby("family"):
        best = group.sort_values("development_rank").iloc[0]
        family_rows.append(
            {
                "family": family,
                "configurations": len(group),
                "best_development_config": best["config_id"],
                "best_development_rank": int(best["development_rank"]),
                "development_return": best.get(
                    "development_equal_weight_return"
                ),
                "validation_return": best.get(
                    "validation_equal_weight_return"
                ),
                "holdout_return": best.get("holdout_equal_weight_return"),
                "holdout_trades": best.get("holdout_trades"),
                "holdout_mean_r": best.get(
                    "holdout_trade_weighted_mean_r"
                ),
            }
        )
    pd.DataFrame(family_rows).to_csv(
        OUT / "family_comparison.csv", index=False
    )

    exact_summary = {
        period: row_for(aggregate, "EXACT_BASELINE", period)
        for period in PERIODS
    }
    selected_summary = {
        period: row_for(aggregate, selected_id, period)
        for period in PERIODS
    }
    robust = ranking[ranking["robustness_gate_pass"] == True].sort_values(
        "development_rank"
    )
    summary = {
        "run_completed_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "markets": list(prepared.keys()),
        "market_categories": {
            name: MARKET_SPECS[name].category for name in prepared
        },
        "market_splits": splits,
        "configuration_count": len(configs),
        "market_period_simulations": len(market_metrics),
        "exact_baseline_config": asdict(
            next(
                config
                for config in configs
                if config.config_id == "EXACT_BASELINE"
            )
        ),
        "exact_baseline": exact_summary,
        "selected_before_holdout_id": selected_id,
        "selected_before_holdout_config": asdict(selected_config),
        "selected_before_holdout": selected_summary,
        "robustness_gate_pass_count": int(len(robust)),
        "robustness_gate_pass_ids": robust["config_id"].tolist(),
        "best_development_id": str(
            ranking.sort_values("development_rank").iloc[0]["config_id"]
        ),
        "best_holdout_descriptive_id": str(
            top_holdout.iloc[0]["config_id"]
        ),
        "synthetic_tests_passed": bool(tests["pass"].all()),
        "method_notes": [
            "Configurations were ranked using development data only.",
            "The reported selected candidate was chosen from the development top 20 using validation data; holdout was not used for selection.",
            "The robustness gate requires at least 20 trades in validation and holdout, positive aggregate return and mean R in both, and at least three positive markets in both.",
            "NQ is an actual E-mini Nasdaq-100 futures price series, but fractional normalized units were used for cross-market risk comparison rather than executable whole-contract sizing.",
            "Stocks and SPY use regular-session bars. NQ and BTC use their available continuous sessions while the reference candle remains 09:30 local time.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )

    lines = [
        "09:30 MONTREAL MULTI-MARKET BACKTEST",
        "=" * 44,
        f"Markets: {', '.join(summary['markets'])}",
        f"Configurations: {len(configs)}",
        f"Robustness-gate passes: {len(robust)}",
        "",
        "EXACT BASELINE",
    ]
    for period in PERIODS:
        result = exact_summary[period]
        lines.append(
            f"{period}: trades={result.get('trades')}, "
            f"equal-weight return={result.get('equal_weight_return')}, "
            f"mean R={result.get('trade_weighted_mean_r')}, "
            f"positive markets={result.get('positive_markets')}/5"
        )
    lines.extend(["", f"SELECTED BEFORE HOLDOUT: {selected_id}"])
    for period in PERIODS:
        result = selected_summary[period]
        lines.append(
            f"{period}: trades={result.get('trades')}, "
            f"equal-weight return={result.get('equal_weight_return')}, "
            f"mean R={result.get('trade_weighted_mean_r')}, "
            f"positive markets={result.get('positive_markets')}/5"
        )
    (OUT / "RESULTS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
