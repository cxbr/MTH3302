from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

import audit_improved_v31 as audit
from audit_improved_v31_fixed import fixed_entry_tests
from improved_strategy_v31 import candidate_catalog


def fixed_post_audit() -> pd.DataFrame:
    rows = []
    trade_path = audit.OUT / "trade_log.csv"
    metric_path = audit.OUT / "market_period_metrics.csv"
    ranking_path = audit.OUT / "candidate_rankings.csv"
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
        str(len(metrics)),
    ))
    rows.append(audit.result(
        "risk_never_above_1pct",
        bool((trades.planned_risk_pct <= 0.0100000001).all()),
        str(trades.planned_risk_pct.max()),
    ))

    entry_dt = pd.to_datetime(trades.entry_time, utc=True)
    exit_dt = pd.to_datetime(trades.exit_time, utc=True)
    rows.append(audit.result(
        "chronology_valid",
        bool((exit_dt >= entry_dt).all()),
        "entry <= exit after UTC normalization",
    ))
    rows.append(audit.result(
        "equity_reconciles",
        bool(np.allclose(trades.equity_after, trades.equity_before + trades.net_pnl, atol=1e-7)),
        "equity_after = equity_before + net_pnl",
    ))

    cap_ok = True
    cap_details = []
    for candidate_id, candidate in catalog.items():
        if candidate.exit_policy is None or not np.isfinite(candidate.exit_policy.notional_cap_multiple):
            continue
        subset = trades[trades.candidate_id == candidate_id]
        observed = float(subset.notional_multiple.max()) if len(subset) else 0.0
        cap_ok &= observed <= candidate.exit_policy.notional_cap_multiple + 1e-9
        cap_details.append(
            f"{candidate_id}:{observed:.6f}<={candidate.exit_policy.notional_cap_multiple}"
        )
    rows.append(audit.result("notional_caps_respected", cap_ok, "; ".join(cap_details)))

    overlap_ok = True
    overlap_details = []
    audit_frame = trades.assign(_entry_utc=entry_dt, _exit_utc=exit_dt)
    for (candidate_id, market, period), group in audit_frame.groupby(
        ["candidate_id", "market", "period"]
    ):
        ordered = group.sort_values("_entry_utc")
        entries = ordered._entry_utc.to_numpy()
        exits = ordered._exit_utc.to_numpy()
        if len(entries) > 1 and not bool(np.all(entries[1:] > exits[:-1])):
            overlap_ok = False
            overlap_details.append(f"{candidate_id}/{market}/{period}")
    rows.append(audit.result("no_overlapping_positions", overlap_ok, str(overlap_details[:10])))

    duplicate_cols = ["candidate_id", "market", "period", "entry_time", "exit_time"]
    rows.append(audit.result(
        "no_duplicate_trade_records",
        not bool(trades.duplicated(duplicate_cols).any()),
        str(int(trades.duplicated(duplicate_cols).sum())),
    ))

    fills_ok = True
    weighted_ok = True
    invalid_fill_trade = None
    for trade in trades.itertuples():
        fills = json.loads(trade.fills)
        total_amount = sum(float(f["amount"]) for f in fills)
        if not math.isclose(total_amount, float(trade.quantity), rel_tol=1e-8, abs_tol=1e-8):
            fills_ok = False
            invalid_fill_trade = f"{trade.candidate_id}/{trade.market}/{trade.entry_time}"
            break
        weighted = sum(float(f["amount"]) * float(f["price"]) for f in fills) / float(trade.quantity)
        if not math.isclose(weighted, float(trade.weighted_exit_exec), rel_tol=1e-8, abs_tol=1e-8):
            weighted_ok = False
            invalid_fill_trade = f"{trade.candidate_id}/{trade.market}/{trade.entry_time}"
            break
    rows.append(audit.result("fill_quantities_reconcile", fills_ok, str(invalid_fill_trade)))
    rows.append(audit.result("weighted_exit_reconciles", weighted_ok, str(invalid_fill_trade)))

    numeric_cols = ["total_return", "max_drawdown", "ending_equity"]
    rows.append(audit.result(
        "market_metrics_finite",
        bool(np.isfinite(metrics[numeric_cols].to_numpy(dtype=float)).all()),
        str(numeric_cols),
    ))

    primary = trades[trades.candidate_id == "V31_PRIMARY"]
    rows.append(audit.result(
        "primary_is_cash_capped",
        bool(len(primary) and primary.notional_multiple.max() <= 1.0000000001),
        str(primary.notional_multiple.max() if len(primary) else None),
    ))

    allowed_reasons = {
        "gap_stop", "stop", "managed_stop", "hard_target", "max_hold_close",
        "period_end", "partial_true_1h_ema20_cross", "true_4h_ema20_cross",
        "true_4h_ema80_cross", "CORRECTED_BASELINE", "target",
        "lagged_ema80_touch", "half_below_ema20", "full_below_ema20",
        "close_below_ema80", "time_exit", "time_exit_close",
    }
    observed_reasons = set()
    for text in trades.fills.dropna():
        for fill in json.loads(text):
            observed_reasons.add(str(fill["reason"]))
    unknown = sorted(observed_reasons - allowed_reasons)
    rows.append(audit.result("exit_reasons_are_known", len(unknown) == 0, str(unknown)))

    df = pd.DataFrame(rows)
    df.to_csv(audit.OUT / "post_backtest_audit.csv", index=False)
    (audit.OUT / "audit_summary.json").write_text(
        json.dumps({"all_passed": bool(df["pass"].all()), "tests": rows}, indent=2),
        encoding="utf-8",
    )
    if not bool(df["pass"].all()):
        raise RuntimeError("Post-backtest audit failed:\n" + df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
    args = parser.parse_args()
    audit.entry_tests = fixed_entry_tests
    df = fixed_post_audit() if args.post else audit.pre_audit()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
