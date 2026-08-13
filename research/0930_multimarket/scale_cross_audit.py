from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

import exit_profit_audit as audit
from backtest_engine import PreparedMarket, StrategyConfig, simulate_period
from data_pipeline import load_all_markets


def masked_cross_features(mkt: PreparedMarket, base: audit.PolicyFeatures) -> audit.PolicyFeatures:
    actual = base.ema[20]
    cross = np.zeros(mkt.n, dtype=bool)
    cross[1:] = (
        np.isfinite(actual[:-1])
        & np.isfinite(actual[1:])
        & (mkt.close[:-1] >= actual[:-1])
        & (mkt.close[1:] < actual[1:])
    )
    ema = dict(base.ema)
    ema[20] = np.where(cross, actual, np.nan)
    return audit.PolicyFeatures(ema=ema)


def policies():
    return [
        audit.ExitPolicy(
            "CROSS20_CORRECTED_BASELINE",
            "Sell half only on a true close cross below EMA20; remainder 3R or lagged EMA80 touch",
            scale_kind="half_below_ema20",
            exit_kind="target_or_ma_touch_lagged",
            exit_value=3.0,
            ema_span=80,
        ),
        audit.ExitPolicy(
            "CROSS20_REST_EMA80_CLOSE",
            "Sell half only on a true EMA20 cross; remainder next-open exit after close below EMA80",
            scale_kind="half_below_ema20",
            exit_kind="ma_close",
            ema_span=80,
            max_hold_days=120,
        ),
        audit.ExitPolicy(
            "CROSS20_REST_EMA160_CLOSE",
            "Sell half only on a true EMA20 cross; remainder next-open exit after close below EMA160",
            scale_kind="half_below_ema20",
            exit_kind="ma_close",
            ema_span=160,
            max_hold_days=120,
        ),
        audit.ExitPolicy(
            "CROSS20_REST_EMA320_CLOSE",
            "Sell half only on a true EMA20 cross; remainder next-open exit after close below EMA320",
            scale_kind="half_below_ema20",
            exit_kind="ma_close",
            ema_span=320,
            max_hold_days=120,
        ),
        *[
            audit.ExitPolicy(
                f"CROSS20_REST_{target}R",
                f"Sell half only on a true EMA20 cross; remainder held for +{target}R",
                scale_kind="half_below_ema20",
                exit_kind="target_r",
                exit_value=float(target),
                max_hold_days=120,
            )
            for target in [3, 5, 10, 20]
        ],
        audit.ExitPolicy(
            "FULL_TRUE_CROSS_BELOW_EMA20",
            "Exit the full position next open only after a true close cross below EMA20",
            scale_kind="full_below_ema20",
            exit_kind="time_only",
            max_hold_days=120,
        ),
    ]


def main():
    frames, _, splits = load_all_markets(audit.WORK)
    prepared = {name: PreparedMarket(name, frame) for name, frame in frames.items()}
    entry_cfg = StrategyConfig(config_id="EXACT_BASELINE", family="exact_baseline")
    rows, trades, entry_state_rows = [], [], []

    for name, mkt in prepared.items():
        base_features = audit.build_features(mkt)
        cross_features = masked_cross_features(mkt, base_features)
        for period in audit.PERIODS:
            _, current_trades = simulate_period(mkt, entry_cfg, splits[name], period)
            for trade in current_trades:
                i = int(mkt.index.searchsorted(pd.Timestamp(trade["entry_time"]), side="left"))
                ema = mkt.ema20[i]
                prev_ema = mkt.ema20[i - 1] if i > 0 else np.nan
                entry_state_rows.append(
                    {
                        "market": name,
                        "period": period,
                        "entry_route": trade["entry_route"],
                        "entry_time": trade["entry_time"],
                        "entry_close_below_ema20": bool(np.isfinite(ema) and mkt.close[i] < ema),
                        "entry_is_cross_below_ema20": bool(
                            i > 0
                            and np.isfinite(ema)
                            and np.isfinite(prev_ema)
                            and mkt.close[i - 1] >= prev_ema
                            and mkt.close[i] < ema
                        ),
                    }
                )
        for policy in policies():
            for period in audit.PERIODS:
                metrics, policy_trades = audit.simulate_policy_period(
                    mkt, cross_features, entry_cfg, policy, splits[name], period
                )
                rows.append(metrics)
                trades.extend(policy_trades)

    market = pd.DataFrame(rows)
    aggregate = audit.aggregate_policy_metrics(market)
    descriptions = {p.policy_id: p.description for p in policies()}
    ranking_rows = []
    for policy_id in aggregate.policy_id.unique():
        row = {"policy_id": policy_id, "description": descriptions[policy_id]}
        for period in audit.PERIODS:
            s = aggregate[(aggregate.policy_id == policy_id) & (aggregate.period == period)].iloc[0]
            for col in [
                "trades",
                "equal_weight_return",
                "positive_markets",
                "trade_weighted_mean_r",
                "worst_market_return",
                "max_market_drawdown",
                "avg_holding_hours",
            ]:
                row[f"{period}_{col}"] = s[col]
        ranking_rows.append(row)
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["development_equal_weight_return", "development_trade_weighted_mean_r"],
        ascending=False,
    )
    ranking["development_rank"] = np.arange(1, len(ranking) + 1)

    out = audit.OUT
    pd.DataFrame([asdict(p) for p in policies()]).to_csv(out / "cross20_policy_catalog.csv", index=False)
    market.to_csv(out / "cross20_market_metrics.csv", index=False)
    aggregate.to_csv(out / "cross20_aggregate_metrics.csv", index=False)
    ranking.to_csv(out / "cross20_policy_rankings.csv", index=False)
    pd.DataFrame(trades).to_csv(out / "cross20_trade_log.csv", index=False)
    state = pd.DataFrame(entry_state_rows)
    state.to_csv(out / "entry_ema20_state.csv", index=False)
    state_summary = (
        state.groupby(["period", "entry_route"])
        .agg(
            trades=("entry_time", "size"),
            entered_below_ema20=("entry_close_below_ema20", "sum"),
            entered_on_true_cross_below=("entry_is_cross_below_ema20", "sum"),
        )
        .reset_index()
    )
    state_summary["share_entered_below_ema20"] = state_summary.entered_below_ema20 / state_summary.trades
    state_summary.to_csv(out / "entry_ema20_state_summary.csv", index=False)

    summary = {
        "policy_count": len(policies()),
        "best_development": ranking.iloc[0].to_dict(),
        "best_holdout_descriptive": ranking.sort_values(
            ["holdout_equal_weight_return", "holdout_trade_weighted_mean_r"],
            ascending=False,
        ).iloc[0].to_dict(),
        "entry_state": state_summary.to_dict(orient="records"),
    }
    (out / "cross20_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(ranking.to_string(index=False), flush=True)
    print(state_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
