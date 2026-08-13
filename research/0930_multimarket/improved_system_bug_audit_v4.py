from __future__ import annotations

import json
from dataclasses import fields
from typing import List

import numpy as np
import pandas as pd

import improved_system_benchmark as base
import improved_system_bug_audit as audit1
import improved_system_bug_audit_v3 as audit3
from backtest_engine import PreparedMarket, StrategyConfig
from data_pipeline import load_all_markets

WORK = audit1.WORK
OUT = audit1.OUT


def daily_ema_test() -> dict:
    try:
        mkt = audit3.fake_market_v3(
            [
                (100, 105, 95, 102),
                (102, 104, 98, 100),
                (101, 103, 99, 102),
            ]
        )
        features = audit3.fake_features_v3(
            mkt,
            day_end=[False, True, False],
            daily_ema20=[np.nan, 110.0, np.nan],
        )
        policy = base.ImprovedPolicy(
            "T",
            "daily EMA close signal",
            exit_kind="daily_ema_close",
            daily_ema_span=20,
            max_hold_days=30,
        )
        cfg = StrategyConfig(config_id="AUDIT", family="audit")
        trade = base.simulate_trade(
            mkt,
            features,
            audit1.dummy_event(),
            audit1.signal(),
            cfg,
            policy,
            10000.0,
            2,
        )
        audit1.check(pd.Timestamp(trade["exit_time"]) == mkt.index[2], trade)
        audit1.check(trade["exit_reason"] == "close_below_daily_ema20", trade)
        return {
            "test": "daily_ema_signal_executes_next_open",
            "pass": True,
            "details": trade["exit_time"],
        }
    except Exception as exc:
        return {
            "test": "daily_ema_signal_executes_next_open",
            "pass": False,
            "details": repr(exc),
        }


def compare_bar_prefix(a, b, cutoff: int) -> bool:
    aa = np.asarray(a)[:cutoff]
    bb = np.asarray(b)[:cutoff]
    if aa.dtype.kind in "bOUS" or bb.dtype.kind in "bOUS":
        return bool(np.array_equal(aa, bb))
    return bool(np.allclose(aa, bb, equal_nan=True))


def compare_daily_prefix(a, b, daily_cutoff: int, cutoff_date) -> bool:
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a.keys()) != set(b.keys()):
            return False
        return all(
            compare_daily_prefix(a[key], b[key], daily_cutoff, cutoff_date)
            for key in a
        )
    if isinstance(a, pd.DataFrame):
        if not isinstance(b, pd.DataFrame):
            return False
        try:
            mask_a = np.asarray([pd.Timestamp(x).date() < cutoff_date for x in a.index])
            mask_b = np.asarray([pd.Timestamp(x).date() < cutoff_date for x in b.index])
            aa = a.loc[mask_a]
            bb = b.loc[mask_b]
            return bool(
                aa.index.equals(bb.index)
                and aa.columns.equals(bb.columns)
                and np.allclose(aa.to_numpy(), bb.to_numpy(), equal_nan=True)
            )
        except Exception:
            return compare_bar_prefix(a.to_numpy(), b.to_numpy(), daily_cutoff)
    if isinstance(a, pd.Series):
        if not isinstance(b, pd.Series):
            return False
        try:
            mask_a = np.asarray([pd.Timestamp(x).date() < cutoff_date for x in a.index])
            mask_b = np.asarray([pd.Timestamp(x).date() < cutoff_date for x in b.index])
            aa = a.loc[mask_a]
            bb = b.loc[mask_b]
            return bool(
                aa.index.equals(bb.index)
                and np.allclose(aa.to_numpy(), bb.to_numpy(), equal_nan=True)
            )
        except Exception:
            return compare_bar_prefix(a.to_numpy(), b.to_numpy(), daily_cutoff)
    return compare_bar_prefix(a, b, daily_cutoff)


def describe_value(value) -> dict:
    result = {"type": type(value).__name__}
    if isinstance(value, dict):
        result["keys"] = [str(k) for k in value.keys()]
        result["values"] = {
            str(k): describe_value(v) for k, v in value.items()
        }
    else:
        try:
            result["length"] = int(len(value))
        except Exception:
            pass
        if isinstance(value, (pd.Series, pd.DataFrame)) and len(value):
            result["index_type"] = type(value.index).__name__
            result["index_first"] = str(value.index[0])
            result["index_last"] = str(value.index[-1])
    return result


def causality_tests() -> List[dict]:
    frames, _, splits = load_all_markets(WORK)
    cfg = StrategyConfig(config_id="AUDIT", family="audit")
    results: List[dict] = []

    per_bar_fields = {
        "prior_daily_ema_by_bar",
        "prior_daily_atr_by_bar",
        "is_day_last_bar",
        "current_daily_close_by_bar",
        "current_daily_ema_by_bar",
    }
    daily_fields = {"daily_ema", "daily_atr14"}

    for market_name, frame in frames.items():
        validation_year = int(splits[market_name]["validation"])
        cutoff_positions = np.flatnonzero(frame.index.year >= validation_year)
        if not len(cutoff_positions):
            results.append(
                {
                    "test": f"{market_name}_no_future_leakage_before_validation",
                    "pass": False,
                    "details": "no validation cutoff",
                }
            )
            continue

        cutoff = int(cutoff_positions[0])
        cutoff_date = frame.index[cutoff].date()
        dates = pd.Index(frame.index.date).unique()
        daily_cutoff = int(sum(d < cutoff_date for d in dates))

        mutated = frame.copy()
        future = mutated.index.year >= validation_year
        mutated.loc[future, ["open", "high", "low", "close"]] *= 1.37

        original = PreparedMarket(market_name, frame)
        changed = PreparedMarket(market_name, mutated)
        original_features = base.build_trend_features(original)
        changed_features = base.build_trend_features(changed)

        named_checks = {}
        for attr in [
            "ema20",
            "ema80",
            "sma20",
            "sma80",
            "atr14",
            "prior20_low",
            "prior_ath",
        ]:
            named_checks[f"market_{attr}"] = compare_bar_prefix(
                getattr(original, attr), getattr(changed, attr), cutoff
            )

        metadata = {}
        for field in fields(base.TrendFeatures):
            field_name = field.name
            a = getattr(original_features, field_name)
            b = getattr(changed_features, field_name)
            metadata[field_name] = describe_value(a)
            if field_name in per_bar_fields:
                if isinstance(a, dict):
                    ok = (
                        isinstance(b, dict)
                        and set(a.keys()) == set(b.keys())
                        and all(
                            compare_bar_prefix(a[key], b[key], cutoff)
                            for key in a
                        )
                    )
                else:
                    ok = compare_bar_prefix(a, b, cutoff)
            elif field_name in daily_fields:
                ok = compare_daily_prefix(
                    a, b, daily_cutoff, cutoff_date
                )
            else:
                ok = False
            named_checks[f"feature_{field_name}"] = bool(ok)

        original_events = [
            (e.anchor_idx, e.confirmation_idx, e.retrace_idx, e.cross_idx, e.ref_idx)
            for e in original.generate_events(cfg)
            if e.ref_idx < cutoff
        ]
        changed_events = [
            (e.anchor_idx, e.confirmation_idx, e.retrace_idx, e.cross_idx, e.ref_idx)
            for e in changed.generate_events(cfg)
            if e.ref_idx < cutoff
        ]
        named_checks["events_before_cutoff"] = original_events == changed_events

        failed = [key for key, value in named_checks.items() if not value]
        results.append(
            {
                "test": f"{market_name}_no_future_leakage_before_validation",
                "pass": not failed,
                "details": json.dumps(
                    {
                        "bar_cutoff": cutoff,
                        "daily_cutoff": daily_cutoff,
                        "cutoff_date": str(cutoff_date),
                        "events_before_cutoff": len(original_events),
                        "failed_checks": failed,
                        "feature_metadata": metadata,
                    }
                ),
            }
        )
    return results


def main() -> None:
    audit1.fake_market = audit3.fake_market_v3
    audit1.fake_features = audit3.fake_features_v3
    synthetic = [
        row
        for row in audit1.run_synthetic_tests()
        if row["test"] != "daily_ema_signal_executes_next_open"
    ]
    rows = synthetic + [daily_ema_test()] + causality_tests()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "bug_audit_results.csv", index=False)
    summary = {
        "tests": int(len(frame)),
        "passed": int(frame["pass"].sum()),
        "failed": int((~frame["pass"]).sum()),
        "all_passed": bool(frame["pass"].all()),
        "trend_feature_fields": [f.name for f in fields(base.TrendFeatures)],
        "known_fixed_issue": (
            "The base MFE/MAE diagnostic omitted the exit bar. "
            "The verified benchmark recomputes MFE/MAE from entry through exit inclusive."
        ),
    }
    (OUT / "bug_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
