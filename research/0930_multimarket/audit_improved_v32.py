from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from typing import List

import numpy as np
import pandas as pd

import audit_improved_v31 as audit
import audit_improved_v31_fixed as fixed
import improved_strategy_v3 as v3
from backtest_engine import EntrySignal, PreparedMarket, ReferenceEvent
from improved_strategy_v31 import OUT, base_exit, candidate_catalog, strict_entry_config
from improved_strategy_v32 import install_patch, simulate_trend_trade_v32


def additional_management_gate_tests():
    rows = []
    event = ReferenceEvent("x", 0, 0, 0, 0, 0, 100.0)
    cfg = strict_entry_config()
    signal = EntrySignal("breakout", 10, 10, 100.0, 99.0, 100.0)

    def make_market(high_at_entry: float) -> PreparedMarket:
        idx = pd.date_range("2024-01-02 09:30", periods=40, freq="15min", tz=audit.LOCAL_TZ)
        close = np.full(len(idx), 100.5, dtype=float)
        frame = audit.make_frame(idx, close)
        frame.iloc[10, frame.columns.get_loc("open")] = 100.0
        frame.iloc[10, frame.columns.get_loc("high")] = high_at_entry
        frame.iloc[10, frame.columns.get_loc("low")] = 99.5
        frame.iloc[10, frame.columns.get_loc("close")] = min(high_at_entry - 0.1, 101.5)
        frame.iloc[11, frame.columns.get_loc("open")] = 100.8
        frame.iloc[11, frame.columns.get_loc("high")] = 101.0
        frame.iloc[11, frame.columns.get_loc("low")] = 100.2
        frame.iloc[11, frame.columns.get_loc("close")] = 100.6
        frame.iloc[12, frame.columns.get_loc("open")] = 100.7
        frame.iloc[12, frame.columns.get_loc("high")] = 100.9
        frame.iloc[12, frame.columns.get_loc("low")] = 100.1
        frame.iloc[12, frame.columns.get_loc("close")] = 100.5
        return PreparedMarket("AAPL", frame)

    policy = replace(
        base_exit("GATE_TEST", 1.0, "management gate test"),
        partial_fraction=0.0,
        trail_activation_r=100.0,
        break_even_r=math.nan,
        hard_target_r=100.0,
    )

    below = make_market(101.5)
    below_htf = audit.blank_htf(below.n)
    below_htf["4h"].cross20_down[10] = True
    trade_below = simulate_trend_trade_v32(
        below, below_htf, event, signal, cfg, policy, 10000.0, 12
    )
    rows.append(audit.result(
        "remainder_cross_ignored_before_2r",
        bool(
            trade_below
            and trade_below["exit_reason"] == "period_end"
            and pd.Timestamp(trade_below["exit_time"]) == below.index[12]
        ),
        repr(trade_below),
    ))

    above = make_market(103.0)
    above_htf = audit.blank_htf(above.n)
    above_htf["4h"].cross20_down[10] = True
    trade_above = simulate_trend_trade_v32(
        above, above_htf, event, signal, cfg, policy, 10000.0, 12
    )
    fills = json.loads(trade_above["fills"]) if trade_above else []
    rows.append(audit.result(
        "remainder_cross_after_2r_executes_next_open",
        bool(
            trade_above
            and trade_above["exit_reason"] == "true_4h_ema20_cross"
            and pd.Timestamp(trade_above["exit_time"]) == above.index[11]
            and fills[-1]["reason"] == "true_4h_ema20_cross"
        ),
        repr(trade_above),
    ))
    return rows


def post_audit_v32() -> pd.DataFrame:
    """Independent audit of the completed benchmark and every trade ledger.

    This implementation deliberately avoids pandas namedtuple fields beginning
    with underscores and normalizes all market timestamps to UTC for comparison.
    It does not alter strategy results; it only verifies them.
    """
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

    rows.append(audit.result(
        "all_candidates_present",
        set(catalog) == set(ranking.candidate_id),
        str(sorted(ranking.candidate_id)),
    ))
    rows.append(audit.result(
        "all_market_period_cells_present",
        len(metrics) == len(catalog) * 5 * 3,
        f"observed={len(metrics)}, expected={len(catalog) * 5 * 3}",
    ))

    numeric_trade_cols = [
        "quantity", "planned_risk_pct", "net_pnl", "r_multiple",
        "weighted_exit_exec", "equity_before", "equity_after",
        "notional_multiple",
    ]
    trade_numeric = trades[numeric_trade_cols].to_numpy(dtype=float)
    rows.append(audit.result(
        "trade_ledger_values_finite",
        bool(np.isfinite(trade_numeric).all()),
        str(numeric_trade_cols),
    ))
    rows.append(audit.result(
        "positive_quantities",
        bool((trades.quantity > 0).all()),
        f"minimum={trades.quantity.min()}",
    ))
    rows.append(audit.result(
        "valid_initial_stops",
        bool((trades.stop_raw > 0).all() and (trades.stop_raw < trades.raw_entry).all()),
        "0 < stop < raw entry",
    ))
    rows.append(audit.result(
        "risk_never_above_1pct",
        bool((trades.planned_risk_pct <= 0.0100000001).all()),
        f"maximum={trades.planned_risk_pct.max()}",
    ))

    entry_ts = pd.to_datetime(trades.entry_time, utc=True)
    exit_ts = pd.to_datetime(trades.exit_time, utc=True)
    rows.append(audit.result(
        "chronology_valid",
        bool((exit_ts >= entry_ts).all()),
        "entry <= exit after UTC normalization",
    ))
    rows.append(audit.result(
        "equity_reconciles",
        bool(np.allclose(
            trades.equity_after,
            trades.equity_before + trades.net_pnl,
            atol=1e-7,
            rtol=1e-10,
        )),
        "equity_after = equity_before + net_pnl",
    ))

    cap_ok = True
    cap_details = []
    for candidate_id, candidate in catalog.items():
        if candidate.exit_policy is None or not np.isfinite(candidate.exit_policy.notional_cap_multiple):
            continue
        subset = trades[trades.candidate_id == candidate_id]
        observed = float(subset.notional_multiple.max()) if len(subset) else 0.0
        permitted = float(candidate.exit_policy.notional_cap_multiple)
        cap_ok &= observed <= permitted + 1e-9
        cap_details.append(f"{candidate_id}:{observed:.8f}<={permitted}")
    rows.append(audit.result(
        "notional_caps_respected",
        cap_ok,
        "; ".join(cap_details),
    ))

    overlap_ok = True
    overlap_details = []
    working = trades.copy()
    working["entry_utc"] = entry_ts
    working["exit_utc"] = exit_ts
    for (candidate_id, market, period), group in working.groupby(
        ["candidate_id", "market", "period"], sort=False
    ):
        ordered = group.sort_values("entry_utc")
        previous_exit = None
        for current_entry, current_exit in zip(ordered.entry_utc, ordered.exit_utc):
            if previous_exit is not None and current_entry <= previous_exit:
                overlap_ok = False
                overlap_details.append(f"{candidate_id}/{market}/{period}")
                break
            previous_exit = current_exit
    rows.append(audit.result(
        "no_overlapping_positions",
        overlap_ok,
        str(overlap_details[:10]),
    ))

    fills_ok = True
    weighted_ok = True
    fill_details = ""
    for trade in trades.itertuples(index=False):
        fills = json.loads(trade.fills)
        total_amount = sum(float(fill["amount"]) for fill in fills)
        if not math.isclose(
            total_amount,
            float(trade.quantity),
            rel_tol=1e-8,
            abs_tol=1e-8,
        ):
            fills_ok = False
            fill_details = f"{trade.candidate_id}/{trade.market}: amount mismatch"
            break
        weighted = sum(
            float(fill["amount"]) * float(fill["price"]) for fill in fills
        ) / float(trade.quantity)
        if not math.isclose(
            weighted,
            float(trade.weighted_exit_exec),
            rel_tol=1e-8,
            abs_tol=1e-8,
        ):
            weighted_ok = False
            fill_details = f"{trade.candidate_id}/{trade.market}: weighted price mismatch"
            break
    rows.append(audit.result(
        "fill_quantities_reconcile",
        fills_ok,
        fill_details or "sum(fill amounts) = quantity",
    ))
    rows.append(audit.result(
        "weighted_exit_reconciles",
        weighted_ok,
        fill_details or "weighted fill price matches ledger",
    ))

    metric_cols = [
        "total_return", "max_drawdown", "ending_equity", "trades"
    ]
    finite_metrics = np.isfinite(metrics[metric_cols].to_numpy(dtype=float)).all()
    rows.append(audit.result(
        "market_metrics_finite",
        bool(finite_metrics),
        str(metric_cols),
    ))
    rows.append(audit.result(
        "exit_reasons_recorded",
        bool(trades.exit_reason.notna().all() and (trades.exit_reason.astype(str).str.len() > 0).all()),
        str(sorted(trades.exit_reason.astype(str).unique())),
    ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "post_backtest_audit.csv", index=False)
    summary = {
        "all_passed": bool(df["pass"].all()),
        "tests": rows,
    }
    (OUT / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not summary["all_passed"]:
        raise RuntimeError("Post-backtest audit failed:\n" + df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
    args = parser.parse_args()

    install_patch()
    v3.simulate_trend_trade = simulate_trend_trade_v32
    audit.simulate_trend_trade = simulate_trend_trade_v32
    audit.entry_tests = fixed.fixed_entry_tests

    original_simulation_tests = audit.simulation_tests
    audit.simulation_tests = lambda: original_simulation_tests() + additional_management_gate_tests()

    df = post_audit_v32() if args.post else audit.pre_audit()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
