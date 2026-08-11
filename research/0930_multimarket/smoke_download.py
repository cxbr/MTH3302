from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "smoke_output"
OUT.mkdir(parents=True, exist_ok=True)

TV_MARKETS = {
    "AAPL": "NASDAQ:AAPL",
    "SPY": "AMEX:SPY",
    "ES1": "CME_MINI:ES1!",
    "NQ1": "CME_MINI:NQ1!",
}


def download_tv_one(symbol: str, start: str, end: str, output: str) -> None:
    from pytradingview import TVclient

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    client = TVclient()
    chart = client.chart
    chart.set_up_chart()

    # Request large batches to make long-history runs practical.
    real_fetch_more = chart.fetch_more
    chart.fetch_more = lambda number=100: real_fetch_more(5000)

    chart.set_market(
        symbol,
        {
            "timeframe": "15",
            "currency": "USD",
            "range": 5000,
        },
    )
    chart.on_symbol_loaded(lambda _: print(f"loaded {symbol}: {chart.get_infos.get('description', '')}"))
    client.on_connected(lambda _: chart.download_data(start=start_dt, end=end_dt, filename=output))
    client.create_connection()


def download_binance(symbol: str, start: str, end: str, output: Path) -> None:
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "15m", "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 15 * 60 * 1000
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for row in rows:
            w.writerow([int(row[0]) // 1000, row[1], row[2], row[3], row[4], row[5]])


def inspect_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        return {"path": str(path), "rows": 0}
    for col in ["time", "open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"{path}: missing {col}; columns={df.columns.tolist()}")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    ts = pd.to_datetime(df["time"], unit="s", utc=True)
    diffs = ts.diff().dropna().dt.total_seconds()
    return {
        "path": str(path),
        "rows": int(len(df)),
        "start_utc": ts.iloc[0].isoformat(),
        "end_utc": ts.iloc[-1].isoformat(),
        "median_step_seconds": float(diffs.median()) if len(diffs) else None,
        "duplicate_times": int(pd.read_csv(path)["time"].duplicated().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-one", action="store_true")
    parser.add_argument("--symbol")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.download_one:
        download_tv_one(args.symbol, args.start, args.end, args.output)
        return

    start = "2026-01-05"
    end = "2026-01-12"
    metadata = []
    failures = []

    for name, symbol in TV_MARKETS.items():
        path = OUT / f"{name}.csv"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--download-one",
            "--symbol",
            symbol,
            "--start",
            start,
            "--end",
            end,
            "--output",
            str(path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600)
            metadata.append({"market": name, "symbol": symbol, **inspect_csv(path)})
        except Exception as exc:
            failures.append({"market": name, "symbol": symbol, "error": repr(exc)})

    btc_path = OUT / "BTCUSDT.csv"
    try:
        download_binance("BTCUSDT", start, end, btc_path)
        metadata.append({"market": "BTCUSDT", "symbol": "BINANCE:BTCUSDT", **inspect_csv(btc_path)})
    except Exception as exc:
        failures.append({"market": "BTCUSDT", "symbol": "BINANCE:BTCUSDT", "error": repr(exc)})

    result = {"metadata": metadata, "failures": failures}
    (OUT / "smoke_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
