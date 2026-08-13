from __future__ import annotations

import json
from dataclasses import fields
from typing import List

import numpy as np
import pandas as pd

import improved_system_benchmark as base
import improved_system_bug_audit as audit1
from backtest_engine import PreparedMarket, StrategyConfig
from data_pipeline import load_all_markets

WORK = audit1.WORK
OUT = audit1.OUT


def fake_features_v2(
    mkt,
    cross=None,
    day_end=None,
    daily_ema20=None,
    daily_ema50=None,
    prev_daily_atr=None,
):
    n = mkt.n
    cross_arr = np.asarray(
        cross if cross is not None else np.zeros(n, dtype=bool), dtype=bool
    )
    day_end_arr = np.asarray(
        day_end if day_end is not None else np.zeros(n, dtype=bool), dtype=bool
    )
    d20 = np.asarray(
        daily_ema20 if daily_ema20 is not None else np.full(n, np.nan), dtype=float
    )
    d50 = np.asarray(
        daily_ema50 if daily_ema50 is not None else np.full(n, np.nan), dtype=float
    )
    prev_atr = np.asarray(
        prev_daily_atr if prev_daily_atr is not None else np.full(n, np.nan), dtype=float
    )
    intraday_ema = {
        span: np.full(n, np.nan, dtype=float)
        for span in [20, 50, 80, 160, 320]
    }
    daily_ema = {20: d20, 50: d50}

    kwargs = {}
    for field in fields(base.TrendFeatures):
        name = field.name
        lower = name.lower()
        if "cross" in lower:
            kwargs[name] = cross_arr
        elif lower == "day_end" or ("day" in lower and "end" in lower):
            kwargs[name] = day_end_arr
        elif "date" in lower:
            kwargs[name] = np.asarray([ts.date() for ts in mkt.index], dtype=object)
        elif "daily" in lower and "ema" in lower:
            kwargs[name] = daily_ema
        elif "ema" in lower:
            kwargs[name] = intraday_ema
        elif "atr" in lower:
            kwargs[name] = prev_atr
        else:
            kwargs[name] = np.full(n, np.nan, dtype=float)
    return base.TrendFeatures(**kwargs)


def compare_feature_value(a, b, cutoff: int) -> bool:
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a.keys()) != set(b.keys()):
            return False
        return all(compare_feature_value(a[key], b[key], cutoff) for key in a)
    aa = np.asarray(a)[:cutoff]
    bb = np.asarray(b)[:cutoff]
    if aa.dtype.kind in "bOUS" or bb.dtype.kind in "bOUS":
        return bool(np.array_equal(aa, bb))
    return bool(np.allclose(aa, bb, equal_nan=True))


def real_data_causality_tests_v2() -> List[dict]:
    frames, _, splits = load_all_markets(WORK)
    cfg = StrategyConfig(config_id="AUDIT", family="audit")
    results: List[dict] = []

    for name, frame in frames.items():
        validation_year = int(splits[name]["validation"])
        cutoff_positions = np.flatnonzero(frame.index.year >= validation_year)
        if len(cutoff_positions) == 0:
            results.append(
                {
                    "test": f"{name}_no_future_leakage_before_validation",
                    "pass": False,
                    "details": "no validation cutoff",
                }
            )
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
        for attr in [
            "ema20",
            "ema80",
            "sma20",
            "sma80",
            "atr14",
            "prior20_low",
            "prior_ath",
        ]:
            checks.append(
                np.allclose(
                    getattr(original, attr)[:cutoff],
                    getattr(changed, attr)[:cutoff],
                    equal_nan=True,
                )
            )

        for field in fields(base.TrendFeatures):
            checks.append(
                compare_feature_value(
                    getattr(original_features, field.name),
                    getattr(changed_features, field.name),
                    cutoff,
                )
            )

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
        results.append(
            {
                "test": f"{name}_no_future_leakage_before_validation",
                "pass": bool(all(checks)),
                "details": json.dumps(
                    {
                        "cutoff": cutoff,
                        "events_before_cutoff": len(orig_events),
                        "feature_fields": [f.name for f in fields(base.TrendFeatures)],
                        "checks": len(checks),
                    }
                ),
            }
        )
    return results


def main() -> None:
    # Patch the original synthetic harness so it constructs the actual TrendFeatures schema.
    audit1.fake_features = fake_features_v2
    rows = audit1.run_synthetic_tests() + real_data_causality_tests_v2()
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
