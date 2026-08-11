from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pandas as pd

import improved_system_benchmark as base
import improved_system_bug_audit as audit1
import improved_system_bug_audit_v3 as audit3
import improved_system_bug_audit_v4 as audit4
from backtest_engine import EntrySignal, PreparedMarket, ReferenceEvent, StrategyConfig

OUT = audit1.OUT


def actual_daily_ema_test() -> dict:
    try:
        dates = pd.bdate_range("2020-01-02", periods=22)
        index = []
        rows = []
        for day_number, day in enumerate(dates):
            for clock in ["09:30", "15:45"]:
                index.append(
                    pd.Timestamp(
                        f"{day.date()} {clock}", tz="America/New_York"
                    )
                )
                if day_number < 20:
                    o, h, l, c = 120.0, 121.0, 119.0, 120.0
                elif day_number == 20:
                    o, h, l, c = 100.0, 103.0, 99.0, 100.0
                else:
                    o, h, l, c = 101.0, 103.0, 99.0, 102.0
                rows.append((o, h, l, c, 1000.0))

        frame = pd.DataFrame(
            rows,
            index=pd.DatetimeIndex(index),
            columns=["open", "high", "low", "close", "volume"],
        )
        market = PreparedMarket("AAPL", frame)
        features = base.build_trend_features(market)
        entry_i = 40
        signal = EntrySignal(
            "breakout", entry_i, entry_i, 100.0, 90.0, 100.0
        )
        event = ReferenceEvent(
            "TEST", 0, 0, 0, 0, entry_i - 1, 100.0
        )
        cfg = StrategyConfig(config_id="AUDIT", family="audit")
        policy = base.ImprovedPolicy(
            "T",
            "daily EMA close signal",
            exit_kind="daily_ema",
            daily_ema_span=20,
            activation_r=0.0,
            max_hold_days=30,
        )
        trade = base.simulate_trade(
            market,
            features,
            event,
            signal,
            cfg,
            policy,
            10000.0,
            market.n - 1,
        )
        expected_exit = market.index[42]
        audit1.check(
            pd.Timestamp(trade["exit_time"]) == expected_exit, trade
        )
        audit1.check(
            trade["exit_reason"] == "daily_close_below_ema20", trade
        )
        return {
            "test": "daily_ema_signal_executes_next_open",
            "pass": True,
            "details": json.dumps(
                {
                    "exit_time": trade["exit_time"],
                    "exit_reason": trade["exit_reason"],
                    "daily_ema20_at_signal": float(
                        features.current_daily_ema_by_bar[20][41]
                    ),
                    "daily_close_at_signal": float(
                        features.current_daily_close_by_bar[41]
                    ),
                }
            ),
        }
    except Exception as exc:
        return {
            "test": "daily_ema_signal_executes_next_open",
            "pass": False,
            "details": repr(exc),
        }


def main() -> None:
    audit1.fake_market = audit3.fake_market_v3
    audit1.fake_features = audit3.fake_features_v3
    synthetic = [
        row
        for row in audit1.run_synthetic_tests()
        if row["test"] != "daily_ema_signal_executes_next_open"
    ]
    rows = synthetic + [actual_daily_ema_test()] + audit4.causality_tests()
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
