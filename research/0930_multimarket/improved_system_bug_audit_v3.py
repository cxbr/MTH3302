from __future__ import annotations

import json
from dataclasses import fields
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd

import improved_system_benchmark as base
import improved_system_bug_audit as audit1
from backtest_engine import PreparedMarket, StrategyConfig
from data_pipeline import load_all_markets

WORK = audit1.WORK
OUT = audit1.OUT


def fake_market_v3(rows: List[tuple], cost: float = 0.0):
    arr = np.asarray(rows, dtype=float)
    n = len(arr)
    return SimpleNamespace(
        name="AAPL",
        spec=SimpleNamespace(base_cost_rate=float(cost)),
        index=pd.date_range(
            "2020-01-02 09:30", periods=n, freq="15min", tz="America/New_York"
        ),
        n=n,
        open=arr[:, 0],
        high=arr[:, 1],
        low=arr[:, 2],
        close=arr[:, 3],
        atr14=np.full(n, 1.0, dtype=float),
        # The synthetic cross tests use closes 120 -> 111, so a flat 115 EMA
        # creates exactly one true cross-under at the second bar.
        ema20=np.full(n, 115.0, dtype=float),
        ema80=np.full(n, 100.0, dtype=float),
    )


def fake_features_v3(
    mkt,
    cross=None,
    day_end=None,
    daily_ema20=None,
    daily_ema50=None,
    prev_daily_atr=None,
):
    n = mkt.n
    day_last = np.asarray(
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
    empty_daily_index = pd.Index([], dtype=object)
    empty_daily = pd.Series([], index=empty_daily_index, dtype=float)

    kwargs = {}
    for field in fields(base.TrendFeatures):
        name = field.name
        if name == "daily_ema":
            kwargs[name] = {20: empty_daily.copy(), 50: empty_daily.copy()}
        elif name == "prior_daily_ema_by_bar":
            kwargs[name] = {20: d20.copy(), 50: d50.copy()}
        elif name == "daily_atr14":
            kwargs[name] = empty_daily.copy()
        elif name == "prior_daily_atr_by_bar":
            kwargs[name] = prev_atr
        elif name == "is_day_last_bar":
            kwargs[name] = day_last
        elif name == "current_daily_close_by_bar":
            kwargs[name] = np.asarray(mkt.close, dtype=float)
        elif name == "current_daily_ema_by_bar":
            kwargs[name] = {20: d20.copy(), 50: d50.copy()}
        else:
            raise RuntimeError(f"Unhandled TrendFeatures field: {name}")
    return base.TrendFeatures(**kwargs)


def compare_array_prefix(a, b, cutoff: int) -> bool:
    aa = np.asarray(a)[:cutoff]
    bb = np.asarray(b)[:cutoff]
    if aa.dtype.kind in "bOUS" or bb.dtype.kind in "bOUS":
        return bool(np.array_equal(aa, bb))
    return bool(np.allclose(aa, bb, equal_nan=True))


def index_value_before_cutoff(value, cutoff_date):
    try:
        return pd.Timestamp(value).date() < cutoff_date
    except Exception:
        try:
            return value < cutoff_date
        except Exception:
            return False


def compare_daily_object(a, b, cutoff_date) -> bool:
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a.keys()) != set(b.keys()):
            return False
        return all(compare_daily_object(a[key], b[key], cutoff_date) for key in a)
    if isinstance(a, pd.DataFrame):
        if not isinstance(b, pd.DataFrame):
            return False
        mask_a = [index_value_before_cutoff(x, cutoff_date) for x in a.index]
        mask_b = [index_value_before_cutoff(x, cutoff_date) for x in b.index]
        aa = a.loc[mask_a]
        bb = b.loc[mask_b]
        return bool(aa.index.equals(bb.index) and np.allclose(aa.to_numpy(), bb.to_numpy(), equal_nan=True))
    if isinstance(a, pd.Series):
        if not isinstance(b, pd.Series):
            return False
        mask_a = [index_value_before_cutoff(x, cutoff_date) for x in a.index]
        mask_b = [index_value_before_cutoff(x, cutoff_date) for x in b.index]
        aa = a.loc[mask_a]
        bb = b.loc[mask_b]
        return bool(aa.index.equals(bb.index) and np.allclose(aa.to_numpy(), bb.to_numpy(), equal_nan=True))
    return bool(np.allclose(np.asarray(a), np.asarray(b), equal_nan=True))


def real_data_causality_tests_v3() -> List[dict]:
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
        cutoff_date = frame.index[cutoff].date()
        mutated = frame.copy()
        future = mutated.index.year >= validation_year
        mutated.loc[future, ["open", "high", "low", "close"]] *= 1.37

        original = PreparedMarket(name, frame)
        changed = PreparedMarket(name, mutated)
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
            named_checks[f"market_{attr}"] = compare_array_prefix(
                getattr(original, attr), getattr(changed, attr), cutoff
            )

        for field in fields(base.TrendFeatures):
            name_field = field.name
            a = getattr(original_features, name_field)
            b = getattr(changed_features, name_field)
            if name_field in per_bar_fields:
                if isinstance(a, dict):
                    ok = isinstance(b, dict) and set(a.keys()) == set(b.keys())
                    if ok:
                        ok = all(compare_array_prefix(a[key], b[key], cutoff) for key in a)
                else:
                    ok = compare_array_prefix(a, b, cutoff)
            elif name_field in daily_fields:
                ok = compare_daily_object(a, b, cutoff_date)
            else:
                ok = False
            named_checks[f"feature_{name_field}"] = bool(ok)

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
        named_checks["events_before_cutoff"] = orig_events == changed_events

        failed = [key for key, value in named_checks.items() if not value]
        results.append(
            {
                "test": f"{name}_no_future_leakage_before_validation",
                "pass": len(failed) == 0,
                "details": json.dumps(
                    {
                        "cutoff": cutoff,
                        "cutoff_date": str(cutoff_date),
                        "events_before_cutoff": len(orig_events),
                        "feature_fields": [f.name for f in fields(base.TrendFeatures)],
                        "failed_checks": failed,
                    }
                ),
            }
        )
    return results


def main() -> None:
    audit1.fake_market = fake_market_v3
    audit1.fake_features = fake_features_v3
    rows = audit1.run_synthetic_tests() + real_data_causality_tests_v3()
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
