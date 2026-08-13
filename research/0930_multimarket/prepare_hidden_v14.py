from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-raw", default="hot20_v14_work/raw")
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-work", default="hot20_v14_full")
    parser.add_argument("--dataset-sha", default="unknown")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_raw = (root / args.train_raw).resolve()
    val_path = Path(args.val).resolve()
    test_path = Path(args.test).resolve()
    work = (root / args.output_work).resolve()
    raw = work / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True, exist_ok=True)
    if not train_raw.exists() or not val_path.exists() or not test_path.exists():
        raise RuntimeError("Missing train raw or hidden parquet")

    con = duckdb.connect(str(work / "prepare_hidden.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute("SET TimeZone='America/New_York'")
    con.execute("PRAGMA memory_limit='9GB'")
    (work / "duck_tmp").mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA temp_directory='{}'".format(q(work / "duck_tmp")))

    hidden = work / "hidden_15m.parquet"
    con.execute(
        f"""
        COPY (
          SELECT
            symbol,
            time_bucket(INTERVAL '15 minutes', datetime) AS bucket,
            arg_min(open, datetime) AS open,
            max(high) AS high,
            min(low) AS low,
            arg_max(close, datetime) AS close,
            sum(volume) AS volume
          FROM read_parquet(['{q(val_path)}', '{q(test_path)}'])
          WHERE datetime >= TIMESTAMPTZ '2024-01-01 00:00:00-05:00'
          GROUP BY symbol, bucket
          ORDER BY symbol, bucket
        ) TO '{q(hidden)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    bounds = con.execute(
        f"SELECT min(bucket), max(bucket), count(*), count(DISTINCT symbol) FROM read_parquet('{q(hidden)}')"
    ).fetchone()
    if str(bounds[0]) < "2024-01-01" or str(bounds[1]) >= "2027-01-01":
        raise RuntimeError(f"Unexpected hidden bounds {bounds}")

    symbols = [row[0] for row in con.execute(
        f"SELECT DISTINCT symbol FROM read_parquet('{q(hidden)}') ORDER BY symbol"
    ).fetchall()]
    stats = []
    for idx, symbol in enumerate(symbols, start=1):
        train_file = train_raw / f"{symbol}.csv.gz"
        if not train_file.exists():
            print(f"SKIP {symbol}: no train raw", flush=True)
            continue
        output = raw / f"{symbol}.csv.gz"
        con.execute(
            f"""
            COPY (
              SELECT timestamp, open, high, low, close, volume
              FROM read_csv_auto('{q(train_file)}', header=true)
              UNION ALL
              SELECT epoch_ms(bucket) AS timestamp, open, high, low, close, volume
              FROM read_parquet('{q(hidden)}')
              WHERE symbol = ?
              ORDER BY timestamp
            ) TO '{q(output)}' (HEADER TRUE, FORMAT CSV, COMPRESSION GZIP)
            """,
            [symbol],
        )
        row = con.execute(
            f"SELECT count(*), min(bucket), max(bucket), count(DISTINCT CAST(bucket AS DATE)) "
            f"FROM read_parquet('{q(hidden)}') WHERE symbol = ?",
            [symbol],
        ).fetchone()
        stats.append({
            "ticker": symbol,
            "hidden_rows": int(row[0]),
            "hidden_start": str(row[1]),
            "hidden_end": str(row[2]),
            "hidden_days": int(row[3]),
            "combined_bytes": output.stat().st_size,
        })
        print(f"[{idx:02d}/{len(symbols):02d}] {symbol}: hidden {row[0]:,} bars {row[1]} -> {row[2]}", flush=True)
    con.close()

    payload = {
        "dataset": "twelvedata/financial-world-model",
        "dataset_revision": args.dataset_sha,
        "source_files": {"validation": str(val_path), "test": str(test_path)},
        "hidden_bounds": {"min": str(bounds[0]), "max": str(bounds[1])},
        "hidden_rows_15m": int(bounds[2]),
        "hidden_symbol_count": int(bounds[3]),
        "combined_symbols": len(stats),
        "periods": {
            "blind_2024": [2024, 2024],
            "blind_2025": [2025, 2025],
            "blind_2026_ytd": [2026, 2026],
            "blind_continuous_2024_2026": [2024, 2026],
        },
        "privacy_boundary": "Hidden trade details are used internally for simulation but are not written to the result artifact.",
        "symbol_stats": stats,
    }
    (work / "hidden_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "symbol_stats"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
