from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd

import audit_improved_v31 as audit
import audit_improved_v31_fixed as fixed
import improved_strategy_v3 as v3
from backtest_engine import EntrySignal, PreparedMarket, ReferenceEvent
from improved_strategy_v31 import base_exit, strict_entry_config
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

    df = audit.post_audit() if args.post else audit.pre_audit()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
