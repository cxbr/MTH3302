from __future__ import annotations

import argparse

import pandas as pd

import audit_improved_v31 as audit
from backtest_engine import PreparedMarket, ReferenceEvent, find_entry
from improved_strategy_v31 import strict_entry_config


# The original audit fixture mixed integer and decimal literals. Pandas 3
# correctly rejects writing decimals into integer columns. This wrapper keeps
# the production strategy unchanged and fixes only the synthetic audit data so
# that every OHLC column is explicitly float64 before any mutation.
def fixed_entry_tests():
    rows = []
    idx = pd.date_range("2024-01-02 09:30", periods=10, freq="15min", tz=audit.LOCAL_TZ)
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            "high": [102.0, 101.0, 103.0, 104.0, 104.0, 104.0, 104.0, 104.0, 104.0, 104.0],
            "low": [99.0, 99.5, 100.0, 101.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0],
            "close": [101.0, 100.5, 102.5, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            "volume": [1.0] * 10,
        },
        index=idx,
        dtype="float64",
    )
    event = ReferenceEvent("x", 0, 0, 0, 0, 0, 100.0)
    cfg = strict_entry_config()
    signal = find_entry(PreparedMarket("AAPL", df), event, cfg, 9)
    rows.append(audit.result(
        "entry_breakout_after_reference",
        bool(signal and signal.route == "breakout" and signal.entry_idx == 2 and signal.structural_stop == 99.0),
        repr(signal),
    ))

    df2 = df.copy()
    df2.loc[idx[1], ["open", "high", "low", "close"]] = [100.0, 101.0, 98.0, 98.5]
    df2.loc[idx[2], ["open", "high", "low", "close"]] = [98.5, 100.5, 98.0, 100.0]
    signal2 = find_entry(PreparedMarket("AAPL", df2), event, cfg, 9)
    rows.append(audit.result(
        "entry_strict_reclaim_next_open",
        bool(signal2 and "reclaim" in signal2.route and signal2.entry_idx == 3 and signal2.structural_stop == 98.0),
        repr(signal2),
    ))

    df3 = df.copy()
    df3.loc[idx[1], ["open", "high", "low", "close"]] = [100.0, 103.0, 98.0, 99.0]
    signal3 = find_entry(PreparedMarket("AAPL", df3), event, cfg, 9)
    rows.append(audit.result(
        "entry_same_bar_adverse_first",
        bool(signal3 is None or signal3.entry_idx > 1),
        repr(signal3),
    ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
    args = parser.parse_args()
    audit.entry_tests = fixed_entry_tests
    df = audit.post_audit() if args.post else audit.pre_audit()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
