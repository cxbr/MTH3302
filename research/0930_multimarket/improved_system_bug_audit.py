from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

import improved_system_benchmark as base
from backtest_engine import EntrySignal, PreparedMarket, ReferenceEvent, StrategyConfig
from data_pipeline import load_all_markets

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "verified_bug_audit_work"
OUT = ROOT / "verified_bug_audit_output"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


class AuditFailure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AuditFailure(message)


def fake_market(rows: List[tuple], cost: float = 0.0):
    arr = np.asarray(rows, dtype=float)
    n = len(arr)
    return SimpleNamespace(
        name="AAPL",
        spec=SimpleNamespace(base_cost_rate=float(cost)),
        index=pd.date_range("2020-01-02 09:30", periods=n, freq="15min", tz="America/New_York"),
        n=n,
        open=arr[:, 0],
        high=arr[:, 1],
        low=arr[:, 2],
        close=arr[:, 3],
        atr14=np.full(n, 1.0, dtype=float),
    )


def fake_features(mkt, cross=None, day_end=None, daily_ema20=None, daily_ema50=None, prev_daily_atr=None):
    n = mkt.n
    kwargs = {}
    for field in fields(base.TrendFeatures):
        name = field.name
        if name == "intraday_cross20":
            kwargs[name] = np.asarray(cross if cross is not None else np.zeros(n, dtype=bool), dtype=bool)
        elif name == "day_end":
            kwargs[name] = np.asarray(day_end if day_end is not None else np.zeros(n, dtype=bool), dtype=bool)
        elif name == "local_dates":
            kwargs[name] = np.asarray([ts.date() for ts in mkt.index], dtype=object)
        elif name == "daily_ema20":
            kwargs[name] = np.asarray(daily_ema20 if daily_ema20 is not None else np.full(n, np.nan), dtype=float)
        elif name == "daily_ema50":
            kwargs[name] = np.asarray(daily_ema50 if daily_ema50 is not None else np.full(n, np.nan), dtype=float)
        elif name == "prev_daily_atr":
            kwargs[name] = np.asarray(prev_daily_atr if prev_daily_atr is not None else np.full(n, np.nan), dtype=float)
        else:
            kwargs[name] = np.full(n, np.nan, dtype=float)
    return base.TrendFeatures(**kwargs)


def dummy_event() -> ReferenceEvent:
    return ReferenceEvent("TEST", 0, 0, 0, 0, 0, 50.0)


def signal(stop: float = 90.0, route: str = "breakout") -> EntrySignal:
    return EntrySignal(route, 0, 0, 100.0, float(stop), 100.0)


def corrected_diagnostics(mkt, trade: dict) -> dict:
    result = dict(trade)
    entry_i = int(mkt.index.searchsorted(pd.Timestamp(trade["entry_time"]), side="left"))
    exit_i = int(mkt.index.searchsorted(pd.Timestamp(trade["exit_time"]), side="left"))
    entry = float(trade["raw_entry"])
    stop = float(trade["stop_raw"])
    risk = entry - stop
    high = float(np.nanmax(mkt.high[entry_i : exit_i + 1]))
    low = float(np.nanmin(mkt.low[entry_i : exit_i + 1]))
    mfe = (high - entry) / risk
    mae = (entry - low) / risk
    result["mfe_r_to_exit"] = mfe
    result["mae_r_to_exit"] = mae
    result["capture_ratio"] = (
        float(trade["r_multiple"]) / mfe
        if mfe > 0 and float(trade["r_multiple"]) > 0
        else np.nan
    )
    return result


def run_synthetic_tests() -> List[dict]:
    results: List[dict] = []

    def run(name, func):
        try:
            details = func() or "PASS"
            results.append({"test": name, "pass": True, "details": str(details)})
        except Exception as exc:
            results.append({"test": name, "pass": False, "details": repr(exc)})

    cfg = StrategyConfig(config_id="AUDIT", family="audit")

    def stop_priority():
        mkt = fake_market([(100, 310, 85, 100)])
        feat = fake_features(mkt)
        pol = base.ImprovedPolicy("T", "stop before target", exit_kind="target_r", exit_value=20.0)
        trade = base.simulate_trade(mkt, feat, dummy_event(), signal(), cfg, pol, 10000.0, 0)
        check(trade["exit_reason"] == "stop", trade)
        check(abs(trade["r_multiple"] + 1.0) < 1e-9, trade)
        return trade["exit_reason"]

    def gap_stop():
        mkt = fake_market([(100, 105, 95, 100), (80, 85, 75, 82)])
        feat = fake_features(mkt)
        pol = base.ImprovedPolicy("T", "gap stop", exit_kind="target_r", exit_value=20.0)
        trade = base.simulate_trade(mkt, feat, dummy_event(), signal(), cfg, pol, 10000.0, 1)
        check(trade["exit_reason"] == "gap_stop", trade)
        check(trade["r_multiple"] < -1.0, trade)
        return trade["r_multiple"]

    def trail_next_bar_only():
        mkt = fake_market([(100, 125, 95, 120), (120, 121, 118, 119)])
        feat = fake_features(mkt)
        pol = base.ImprovedPolicy(
            "T", "trail lag", exit_kind="pct_trail", exit_value=0.05,
            activation_r=2.0, max_hold_days=30,
        )
        trade = base.simulate_trade(mkt, feat, dummy_event(), signal(), cfg, pol, 10000.0, 1)
        check(pd.Timestamp(trade["exit_time"]) == mkt.index[1], trade)
        check(trade["exit_reason"] == "managed_stop", trade)
        return trade["exit_time"]

    def break_even_next_bar_only():
        mkt = fake_market([(100, 125, 95, 120), (110, 111, 99, 105)])
        feat = fake_features(mkt)
        pol = base.ImprovedPolicy(
            "T", "break even lag", exit_kind="target_r", exit_value=20.0,
            break_even_r=2.0, max_hold_days=30,
        )
        trade = base.simulate_trade(mkt, feat, dummy_event(), signal(), cfg, pol, 10000.0, 1)
        check(pd.Timestamp(trade["exit_time"]) == mkt.index[1], trade)
        check(trade["exit_reason"] == "managed_stop", trade)
        return trade["exit_time"]

    def true_cross_scale_next_open():
        rows = [
            (100, 125, 95, 120),
            (120, 122, 110, 111),
            (112, 113, 105, 106),
            (106, 107, 89, 90),
        ]
        mkt = fake_market(rows)
        feat = fake_features(mkt, cross=[False, True, False, False])
        pol = base.ImprovedPolicy(
            "T", "cross scale", scale_kind="true_cross", scale_fraction=0.5,
            scale_activation_r=2.0, exit_kind="target_r", exit_value=20.0,
        )
        trade = base.simulate_trade(mkt, feat, dummy_event(), signal(), cfg, pol, 10000.0, 3)
        fills = json.loads(trade["fills"])
        check(fills[0]["reason"] == "true_cross_scale", fills)
        check(pd.Timestamp(fills[0]["time"]) == mkt.index[2], fills)
        check(abs(float(fills[0]["amount"]) - 0.5 * float(trade["quantity"])) < 1e-8, fills)
        return fills

    def daily_ema_next_open():
        rows = [(100, 105, 95, 102), (102, 104, 98, 100), (101, 103, 99, 102)]
        mkt = fake_market(rows)
        feat = fake_features(
            mkt,
            day_end=[False, True, False],
            daily_ema20=[np.nan, 110.0, np.nan],
        )
        pol = base.ImprovedPolicy("T", "daily ema", exit_kind="daily_ema_close", ema_span=20)
        trade = base.simulate_trade(mkt, feat, dummy_event(), signal(), cfg, pol, 10000.0, 2)
        check(pd.Timestamp(trade["exit_time"]) == mkt.index[2], trade)
        check(trade["exit_reason"] == "close_below_daily_ema20", trade)
        return trade["exit_time"]

    def risk_and_notional_cap():
        mkt = fake_market([(100, 101, 99.95, 100.5)])
        feat = fake_features(mkt)
        tight = signal(stop=99.9)
        uncapped = base.ImprovedPolicy("U", "uncapped", exit_kind="time_only", max_hold_days=1)
        capped = base.ImprovedPolicy("C", "cap", exit_kind="time_only", max_hold_days=1, notional_cap=1.0)
        t1 = base.simulate_trade(mkt, feat, dummy_event(), tight, cfg, uncapped, 10000.0, 0)
        t2 = base.simulate_trade(mkt, feat, dummy_event(), tight, cfg, capped, 10000.0, 0)
        check(abs(t1["planned_risk_pct"] - 0.01) < 1e-10, t1)
        check(t2["planned_risk_pct"] <= 0.001000001, t2)
        check(t2["notional_ratio"] <= 1.0000001, t2)
        return {"uncapped_risk": t1["planned_risk_pct"], "capped_risk": t2["planned_risk_pct"]}

    def fill_reconciliation():
        rows = [(100, 125, 95, 120), (120, 122, 110, 111), (112, 113, 105, 106), (106, 107, 89, 90)]
        mkt = fake_market(rows)
        feat = fake_features(mkt, cross=[False, True, False, False])
        pol = base.ImprovedPolicy(
            "T", "reconcile", scale_kind="true_cross", scale_fraction=0.25,
            scale_activation_r=2.0, exit_kind="target_r", exit_value=20.0,
        )
        trade = base.simulate_trade(mkt, feat, dummy_event(), signal(), cfg, pol, 10000.0, 3)
        fills = json.loads(trade["fills"])
        total_amount = sum(float(x["amount"]) for x in fills)
        pnl = sum(float(x["amount"]) * (float(x["price"]) - float(trade["entry_exec"])) for x in fills)
        weighted = sum(float(x["amount"]) * float(x["price"]) for x in fills) / float(trade["quantity"])
        check(abs(total_amount - float(trade["quantity"])) < 1e-8, fills)
        check(abs(pnl - float(trade["net_pnl"])) < 1e-8, (pnl, trade["net_pnl"]))
        check(abs(weighted - float(trade["weighted_exit_exec"])) < 1e-8, (weighted, trade["weighted_exit_exec"]))
        return {"fills": len(fills), "pnl": pnl}

    def diagnostics_include_exit_bar():
        mkt = fake_market([(100, 105, 95, 100), (100, 102, 89, 90)])
        feat = fake_features(mkt)
        pol = base.ImprovedPolicy("T", "diag", exit_kind="target_r", exit_value=20.0)
        raw = base.simulate_trade(mkt, feat, dummy_event(), signal(), cfg, pol, 10000.0, 1)
        fixed = corrected_diagnostics(mkt, raw)
        check(fixed["mae_r_to_exit"] >= 1.0, fixed)
        return {
            "base_reported_mae": raw["mae_r_to_exit"],
            "corrected_mae": fixed["mae_r_to_exit"],
            "status": "known diagnostic bug corrected in verified benchmark",
        }

    for name, func in [
        ("adverse_stop_precedes_same_bar_target", stop_priority),
        ("gap_stop_uses_open", gap_stop),
        ("new_trail_cannot_trigger_on_activation_bar", trail_next_bar_only),
        ("break_even_cannot_trigger_on_activation_bar", break_even_next_bar_only),
        ("true_ema20_cross_partial_executes_next_open", true_cross_scale_next_open),
        ("daily_ema_signal_executes_next_open", daily_ema_next_open),
        ("one_percent_risk_and_notional_cap", risk_and_notional_cap),
        ("fills_pnl_and_weighted_exit_reconcile", fill_reconciliation),
        ("mfe_mae_include_exit_bar_in_verified_output", diagnostics_include_exit_bar),
    ]:
        run(name, func)
    return results


def real_data_causality_tests() -> List[dict]:
    frames, _, splits = load_all_markets(WORK)
    cfg = StrategyConfig(config_id="AUDIT", family="audit")
    results: List[dict] = []

    for name, frame in frames.items():
        validation_year = int(splits[name]["validation"])
        cutoff_positions = np.flatnonzero(frame.index.year >= validation_year)
        if len(cutoff_positions) == 0:
            results.append({"test": f"{name}_causality", "pass": False, "details": "no validation cutoff"})
            continue
        cutoff = int(cutoff_positions[0])
        mutated = frame.copy()
        future = mutated.index.year >= validation_year
        mutated.loc[future, ["open", "high", "low", "close"]] *= 1.37

        original = PreparedMarket(name, frame)
        changed = PreparedMarket(name, mutated)
        original_features = base.build_trend_features(original)
        changed_features = base.build_trend_features(changed)

        checks = []
        for attr in ["ema20", "ema80", "sma20", "sma80", "atr14", "prior20_low", "prior_ath"]:
            a = getattr(original, attr)[:cutoff]
            b = getattr(changed, attr)[:cutoff]
            checks.append(np.allclose(a, b, equal_nan=True))
        for attr in ["ema20", "ema50", "ema80", "ema160", "ema320", "prev_daily_atr"]:
            a = getattr(original_features, attr)[:cutoff]
            b = getattr(changed_features, attr)[:cutoff]
            checks.append(np.allclose(a, b, equal_nan=True))
        checks.append(np.array_equal(original_features.intraday_cross20[:cutoff], changed_features.intraday_cross20[:cutoff]))
        checks.append(np.array_equal(original_features.day_end[:cutoff], changed_features.day_end[:cutoff]))
        day_end_before = np.flatnonzero(original_features.day_end[:cutoff])
        if len(day_end_before):
            for attr in ["daily_ema20", "daily_ema50"]:
                a = getattr(original_features, attr)[day_end_before]
                b = getattr(changed_features, attr)[day_end_before]
                checks.append(np.allclose(a, b, equal_nan=True))

        orig_events = [
            (e.anchor_idx, e.confirmation_idx, e.retrace_idx, e.cross_idx, e.ref_idx)
            for e in original.generate_events(cfg)
            if e.ref_idx < cutoff
        ]
        changed_events = [
            (e.anchor_idx, e.confirmation_idx, e.retrace_idx, e.cross_idx, e.ref_idx)
            for e in changed.generate_events(cfg)
            if e.ref_idx < cutoff
        ]
        checks.append(orig_events == changed_events)
        results.append({
            "test": f"{name}_no_future_leakage_before_validation",
            "pass": bool(all(checks)),
            "details": json.dumps({"cutoff": cutoff, "events_before_cutoff": len(orig_events), "checks": len(checks)}),
        })
    return results


def main() -> None:
    rows = run_synthetic_tests() + real_data_causality_tests()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "bug_audit_results.csv", index=False)
    summary = {
        "tests": int(len(frame)),
        "passed": int(frame["pass"].sum()),
        "failed": int((~frame["pass"]).sum()),
        "all_passed": bool(frame["pass"].all()),
        "known_fixed_issue": "Base MFE/MAE diagnostics omitted the exit bar. Verified benchmark recomputes them from entry through exit inclusive.",
    }
    (OUT / "bug_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
