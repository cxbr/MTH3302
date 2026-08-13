from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest_engine import EntrySignal, PreparedMarket, ReferenceEvent, StrategyConfig, find_entry
from improved_strategy_v3 import HTFFeatures, simulate_trend_trade
from improved_strategy_v31 import (
    OUT,
    base_exit,
    build_complete_htf_features,
    candidate_catalog,
    strict_entry_config,
)

LOCAL_TZ = "America/New_York"


def result(name: str, passed: bool, details: str) -> dict:
    return {"test": name, "pass": bool(passed), "details": details}


def make_frame(index: pd.DatetimeIndex, closes: np.ndarray | None = None) -> pd.DataFrame:
    if closes is None:
        closes = np.linspace(100.0, 101.0, len(index))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.25,
            "low": closes - 0.25,
            "close": closes,
            "volume": 1.0,
        },
        index=index,
    )


def blank_htf(n: int) -> Dict[str, HTFFeatures]:
    def one() -> HTFFeatures:
        return HTFFeatures(
            close_event=np.zeros(n, dtype=bool),
            close_value=np.full(n, np.nan),
            ema20_value=np.full(n, np.nan),
            ema80_value=np.full(n, np.nan),
            atr14_value=np.full(n, np.nan),
            cross20_down=np.zeros(n, dtype=bool),
            cross80_down=np.zeros(n, dtype=bool),
        )
    return {"1h": one(), "4h": one()}


def make_market_for_trade() -> PreparedMarket:
    idx = pd.date_range("2024-01-02 09:30", periods=40, freq="15min", tz=LOCAL_TZ)
    closes = np.full(len(idx), 100.5)
    frame = make_frame(idx, closes)
    return PreparedMarket("AAPL", frame)


def entry_tests() -> List[dict]:
    rows: List[dict] = []
    idx = pd.date_range("2024-01-02 09:30", periods=10, freq="15min", tz=LOCAL_TZ)
    df = pd.DataFrame(
        {
            "open": [100, 100, 101, 102, 103, 103, 103, 103, 103, 103],
            "high": [102, 101, 103, 104, 104, 104, 104, 104, 104, 104],
            "low": [99, 99.5, 100, 101, 102, 102, 102, 102, 102, 102],
            "close": [101, 100.5, 102.5, 103, 103, 103, 103, 103, 103, 103],
            "volume": 1.0,
        },
        index=idx,
    )
    event = ReferenceEvent("x", 0, 0, 0, 0, 0, 100.0)
    cfg = strict_entry_config()
    market = PreparedMarket("AAPL", df)
    signal = find_entry(market, event, cfg, 9)
    rows.append(result(
        "entry_breakout_after_reference",
        bool(signal and signal.route == "breakout" and signal.entry_idx == 2 and signal.structural_stop == 99.0),
        repr(signal),
    ))

    df2 = df.copy()
    df2.loc[idx[1], ["open", "high", "low", "close"]] = [100, 101, 98, 98.5]
    df2.loc[idx[2], ["open", "high", "low", "close"]] = [98.5, 100.5, 98, 100]
    signal2 = find_entry(PreparedMarket("AAPL", df2), event, cfg, 9)
    rows.append(result(
        "entry_strict_reclaim_next_open",
        bool(signal2 and "reclaim" in signal2.route and signal2.entry_idx == 3 and signal2.structural_stop == 98.0),
        repr(signal2),
    ))

    df3 = df.copy()
    df3.loc[idx[1], ["open", "high", "low", "close"]] = [100, 103, 98, 99]
    signal3 = find_entry(PreparedMarket("AAPL", df3), event, cfg, 9)
    rows.append(result(
        "entry_same_bar_adverse_first",
        bool(signal3 is None or signal3.entry_idx > 1),
        repr(signal3),
    ))
    return rows


def htf_tests() -> List[dict]:
    rows: List[dict] = []
    idx = pd.date_range("2024-01-02 09:30", periods=26, freq="15min", tz=LOCAL_TZ)
    market = PreparedMarket("AAPL", make_frame(idx))
    f1 = build_complete_htf_features(market, 60)
    f4 = build_complete_htf_features(market, 240)
    rows.append(result(
        "complete_1h_buckets_only",
        np.flatnonzero(f1.close_event).tolist() == [3, 7, 11, 15, 19, 23],
        str(np.flatnonzero(f1.close_event).tolist()),
    ))
    rows.append(result(
        "complete_4h_buckets_only",
        np.flatnonzero(f4.close_event).tolist() == [15],
        str(np.flatnonzero(f4.close_event).tolist()),
    ))

    gap_frame = make_frame(idx).drop(idx[2])
    gap_market = PreparedMarket("AAPL", gap_frame)
    gap_features = build_complete_htf_features(gap_market, 60)
    first_event_time = gap_market.index[np.flatnonzero(gap_features.close_event)[0]] if gap_features.close_event.any() else None
    rows.append(result(
        "missing_candle_invalidates_htf_bucket",
        first_event_time == pd.Timestamp("2024-01-02 11:15", tz=LOCAL_TZ),
        str(first_event_time),
    ))

    idx_long = pd.date_range("2024-01-01 09:30", periods=200, freq="15min", tz=LOCAL_TZ)
    base_frame = make_frame(idx_long, 100 + np.sin(np.arange(200) / 7.0))
    base_market = PreparedMarket("AAPL", base_frame)
    original = build_complete_htf_features(base_market, 60)
    mutated_frame = base_frame.copy()
    mutated_frame.iloc[150:, mutated_frame.columns.get_loc("close")] += 50
    mutated_frame.iloc[150:, mutated_frame.columns.get_loc("open")] += 50
    mutated_frame.iloc[150:, mutated_frame.columns.get_loc("high")] += 50
    mutated_frame.iloc[150:, mutated_frame.columns.get_loc("low")] += 50
    mutated = build_complete_htf_features(PreparedMarket("AAPL", mutated_frame), 60)
    causal = (
        np.array_equal(original.close_event[:150], mutated.close_event[:150])
        and np.allclose(original.ema20_value[:150], mutated.ema20_value[:150], equal_nan=True)
        and np.allclose(original.atr14_value[:150], mutated.atr14_value[:150], equal_nan=True)
        and np.array_equal(original.cross20_down[:150], mutated.cross20_down[:150])
    )
    rows.append(result("htf_no_future_leakage", causal, "Future bars 150+ mutated by +50."))

    hourly_closes = np.array([100.0] * 20 + [102.0, 103.0, 104.0, 105.0, 90.0])
    closes = np.repeat(hourly_closes, 4)
    idx_cross = pd.date_range("2024-01-01 09:30", periods=len(closes), freq="15min", tz=LOCAL_TZ)
    cross_features = build_complete_htf_features(PreparedMarket("AAPL", make_frame(idx_cross, closes)), 60)
    cross_positions = np.flatnonzero(cross_features.cross20_down)
    valid = len(cross_positions) >= 1
    if valid:
        i = int(cross_positions[-1])
        previous_events = np.flatnonzero(cross_features.close_event[:i])
        j = int(previous_events[-1])
        valid = (
            cross_features.close_value[j] >= cross_features.ema20_value[j]
            and cross_features.close_value[i] < cross_features.ema20_value[i]
        )
    rows.append(result("true_ema20_cross_definition", valid, str(cross_positions.tolist())))
    return rows


def simulation_tests() -> List[dict]:
    rows: List[dict] = []
    event = ReferenceEvent("x", 0, 0, 0, 0, 0, 100.0)
    cfg = strict_entry_config()

    market = make_market_for_trade()
    signal = EntrySignal("breakout", 10, 10, 100.0, 99.5, 100.0)
    htf = blank_htf(market.n)
    policy = base_exit("TEST_CAP", 1.0, "test")
    trade = simulate_trend_trade(market, htf, event, signal, cfg, policy, 10000.0, 20)
    rows.append(result(
        "risk_and_1x_notional_cap",
        bool(trade and trade["planned_risk_pct"] <= 0.0100000001 and trade["notional_multiple"] <= 1.0000000001),
        json.dumps({k: trade.get(k) for k in ["planned_risk_pct", "notional_multiple", "quantity"]}) if trade else "None",
    ))

    adverse_market = make_market_for_trade()
    adverse_market.df.iloc[10, adverse_market.df.columns.get_loc("open")] = 100.0
    adverse_market.df.iloc[10, adverse_market.df.columns.get_loc("high")] = 140.0
    adverse_market.df.iloc[10, adverse_market.df.columns.get_loc("low")] = 98.0
    adverse_market.df.iloc[10, adverse_market.df.columns.get_loc("close")] = 130.0
    adverse_market = PreparedMarket("AAPL", adverse_market.df)
    adverse_policy = replace(policy, hard_target_r=30.0)
    adverse = simulate_trend_trade(adverse_market, blank_htf(adverse_market.n), event, signal, cfg, adverse_policy, 10000.0, 20)
    rows.append(result(
        "stop_before_target_same_bar",
        bool(adverse and adverse["exit_reason"] == "stop" and adverse["r_multiple"] < 0),
        repr(adverse),
    ))

    trail_market = make_market_for_trade()
    frame = trail_market.df.copy()
    frame.iloc[10, frame.columns.get_loc("open")] = 100.0
    frame.iloc[10, frame.columns.get_loc("high")] = 105.0
    frame.iloc[10, frame.columns.get_loc("low")] = 100.0
    frame.iloc[10, frame.columns.get_loc("close")] = 104.0
    frame.iloc[11, frame.columns.get_loc("open")] = 105.0
    frame.iloc[11, frame.columns.get_loc("high")] = 106.0
    frame.iloc[11, frame.columns.get_loc("low")] = 104.5
    frame.iloc[11, frame.columns.get_loc("close")] = 105.5
    trail_market = PreparedMarket("AAPL", frame)
    trail_htf = blank_htf(trail_market.n)
    trail_htf["4h"].close_event[10] = True
    trail_htf["4h"].atr14_value[10] = 1.0
    trail_policy = replace(
        policy,
        partial_fraction=0.0,
        trail_atr_multiple=1.0,
        trail_activation_r=2.0,
        break_even_r=math.nan,
        hard_target_r=100.0,
    )
    trail_trade = simulate_trend_trade(trail_market, trail_htf, event, EntrySignal("breakout", 10, 10, 100.0, 99.0, 100.0), cfg, trail_policy, 10000.0, 11)
    rows.append(result(
        "new_trail_not_triggered_same_bar",
        bool(trail_trade and pd.Timestamp(trail_trade["exit_time"]) == trail_market.index[11]),
        repr(trail_trade),
    ))

    monotonic_market = make_market_for_trade()
    frame2 = monotonic_market.df.copy()
    frame2.iloc[10, frame2.columns.get_loc("open")] = 100.0
    frame2.iloc[10, frame2.columns.get_loc("high")] = 105.0
    frame2.iloc[10, frame2.columns.get_loc("low")] = 100.0
    frame2.iloc[10, frame2.columns.get_loc("close")] = 104.0
    frame2.iloc[11, frame2.columns.get_loc("open")] = 105.0
    frame2.iloc[11, frame2.columns.get_loc("high")] = 105.2
    frame2.iloc[11, frame2.columns.get_loc("low")] = 104.5
    frame2.iloc[11, frame2.columns.get_loc("close")] = 105.0
    frame2.iloc[12, frame2.columns.get_loc("open")] = 105.0
    frame2.iloc[12, frame2.columns.get_loc("high")] = 105.0
    frame2.iloc[12, frame2.columns.get_loc("low")] = 103.8
    frame2.iloc[12, frame2.columns.get_loc("close")] = 104.0
    monotonic_market = PreparedMarket("AAPL", frame2)
    monotonic_htf = blank_htf(monotonic_market.n)
    monotonic_htf["4h"].close_event[10] = True
    monotonic_htf["4h"].atr14_value[10] = 1.0
    monotonic_htf["4h"].close_event[11] = True
    monotonic_htf["4h"].atr14_value[11] = 5.0
    monotonic = simulate_trend_trade(monotonic_market, monotonic_htf, event, EntrySignal("breakout", 10, 10, 100.0, 99.0, 100.0), cfg, trail_policy, 10000.0, 13)
    monotonic_fill = json.loads(monotonic["fills"])[-1] if monotonic else {}
    rows.append(result(
        "managed_stop_never_moves_down",
        bool(monotonic and monotonic["exit_reason"] == "managed_stop" and abs(float(monotonic_fill.get("price", 0)) - (104.0 * (1 - market.spec.base_cost_rate))) < 0.1),
        repr(monotonic),
    ))

    partial_market = make_market_for_trade()
    frame3 = partial_market.df.copy()
    frame3.iloc[10, frame3.columns.get_loc("high")] = 101.0
    frame3.iloc[11, frame3.columns.get_loc("high")] = 103.0
    frame3.iloc[13, frame3.columns.get_loc("open")] = 102.5
    partial_market = PreparedMarket("AAPL", frame3)
    partial_htf = blank_htf(partial_market.n)
    partial_htf["1h"].cross20_down[10] = True
    partial_htf["1h"].cross20_down[11] = True
    partial_policy = replace(
        policy,
        trail_activation_r=100.0,
        break_even_r=math.nan,
        hard_target_r=100.0,
        max_hold_days=90,
    )
    partial_trade = simulate_trend_trade(partial_market, partial_htf, event, EntrySignal("breakout", 10, 10, 100.0, 99.0, 100.0), cfg, partial_policy, 10000.0, 13)
    fills = json.loads(partial_trade["fills"]) if partial_trade else []
    partial_fills = [f for f in fills if "partial_true" in f["reason"]]
    rows.append(result(
        "partial_requires_2r_and_executes_next_open_once",
        bool(len(partial_fills) == 1 and pd.Timestamp(partial_fills[0]["time"]) == partial_market.index[12]),
        json.dumps(partial_fills),
    ))
    return rows


def pre_audit() -> pd.DataFrame:
    rows = entry_tests() + htf_tests() + simulation_tests()
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pre_backtest_audit.csv", index=False)
    if not bool(df["pass"].all()):
        raise RuntimeError("Pre-backtest audit failed:\n" + df.to_string(index=False))
    return df


def post_audit() -> pd.DataFrame:
    rows: List[dict] = []
    trade_path = OUT / "trade_log.csv"
    metric_path = OUT / "market_period_metrics.csv"
    ranking_path = OUT / "candidate_rankings.csv"
    if not (trade_path.exists() and metric_path.exists() and ranking_path.exists()):
        raise FileNotFoundError("Benchmark outputs are missing")

    trades = pd.read_csv(trade_path)
    metrics = pd.read_csv(metric_path)
    ranking = pd.read_csv(ranking_path)
    catalog = {c.candidate_id: c for c in candidate_catalog()}

    rows.append(result("all_candidates_present", set(catalog) == set(ranking.candidate_id), str(sorted(ranking.candidate_id))))
    rows.append(result("all_market_period_cells_present", len(metrics) == len(catalog) * 5 * 3, str(len(metrics))))
    rows.append(result("risk_never_above_1pct", bool((trades.planned_risk_pct <= 0.0100000001).all()), str(trades.planned_risk_pct.max())))
    rows.append(result("chronology_valid", bool((pd.to_datetime(trades.exit_time) >= pd.to_datetime(trades.entry_time)).all()), "entry <= exit"))
    rows.append(result("equity_reconciles", bool(np.allclose(trades.equity_after, trades.equity_before + trades.net_pnl, atol=1e-7)), "equity_after = equity_before + net_pnl"))

    cap_ok = True
    cap_details = []
    for candidate_id, candidate in catalog.items():
        if candidate.exit_policy is None or not np.isfinite(candidate.exit_policy.notional_cap_multiple):
            continue
        subset = trades[trades.candidate_id == candidate_id]
        observed = float(subset.notional_multiple.max()) if len(subset) else 0.0
        cap_ok &= observed <= candidate.exit_policy.notional_cap_multiple + 1e-9
        cap_details.append(f"{candidate_id}:{observed:.6f}<={candidate.exit_policy.notional_cap_multiple}")
    rows.append(result("notional_caps_respected", cap_ok, "; ".join(cap_details)))

    overlap_ok = True
    overlap_details = []
    for (candidate_id, market, period), group in trades.groupby(["candidate_id", "market", "period"]):
        ordered = group.assign(
            _entry=pd.to_datetime(group.entry_time),
            _exit=pd.to_datetime(group.exit_time),
        ).sort_values("_entry")
        previous_exit = None
        for item in ordered.itertuples():
            if previous_exit is not None and item._entry <= previous_exit:
                overlap_ok = False
                overlap_details.append(f"{candidate_id}/{market}/{period}")
                break
            previous_exit = item._exit
    rows.append(result("no_overlapping_positions", overlap_ok, str(overlap_details[:10])))

    fills_ok = True
    weighted_ok = True
    for trade in trades.itertuples():
        fills = json.loads(trade.fills)
        amount = sum(float(f["amount"]) for f in fills)
        if not math.isclose(amount, float(trade.quantity), rel_tol=1e-8, abs_tol=1e-8):
            fills_ok = False
            break
        weighted = sum(float(f["amount"]) * float(f["price"]) for f in fills) / float(trade.quantity)
        if not math.isclose(weighted, float(trade.weighted_exit_exec), rel_tol=1e-8, abs_tol=1e-8):
            weighted_ok = False
            break
    rows.append(result("fill_quantities_reconcile", fills_ok, "sum(fill amounts) = quantity"))
    rows.append(result("weighted_exit_reconciles", weighted_ok, "weighted fill price matches ledger"))

    numeric_cols = ["total_return", "max_drawdown", "ending_equity"]
    finite = np.isfinite(metrics[numeric_cols].to_numpy(dtype=float)).all()
    rows.append(result("market_metrics_finite", bool(finite), str(numeric_cols)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "post_backtest_audit.csv", index=False)
    (OUT / "audit_summary.json").write_text(
        json.dumps(
            {
                "all_passed": bool(df["pass"].all()),
                "tests": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not bool(df["pass"].all()):
        raise RuntimeError("Post-backtest audit failed:\n" + df.to_string(index=False))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
    args = parser.parse_args()
    df = post_audit() if args.post else pre_audit()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
