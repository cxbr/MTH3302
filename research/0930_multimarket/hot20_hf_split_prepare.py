from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="hot20_hf_split_output")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_15m.parquet"
    stats_path = output_dir / f"{args.split}_stats.json"

    if not input_path.exists() or input_path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Invalid input {input_path}")

    con = duckdb.connect(str(output_dir / f"{args.split}.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute("SET TimeZone='America/New_York'")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA temp_directory='{}'".format(q(output_dir / "duck_tmp")))
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
            sum(volume) AS volume,
            count(*) AS minute_count
          FROM read_parquet('{q(input_path)}')
          WHERE datetime >= TIMESTAMPTZ '2018-01-01 00:00:00 America/New_York'
            AND datetime <  TIMESTAMPTZ '2026-01-01 00:00:00 America/New_York'
            AND (EXTRACT(hour FROM datetime) * 60 + EXTRACT(minute FROM datetime)) >= 570
            AND (EXTRACT(hour FROM datetime) * 60 + EXTRACT(minute FROM datetime)) < 960
          GROUP BY symbol, bucket
          HAVING count(*) >= 10
          ORDER BY symbol, bucket
        ) TO '{q(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    stats = con.execute(
        f"""
        SELECT count(*) AS row_count,
               count(DISTINCT symbol) AS symbol_count,
               min(bucket) AS min_bucket,
               max(bucket) AS max_bucket,
               min(minute_count) AS min_minutes,
               max(minute_count) AS max_minutes
        FROM read_parquet('{q(output_path)}')
        """
    ).fetchone()
    con.close()
    payload = {
        "split": args.split,
        "input_bytes": input_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
        "row_count": int(stats[0]),
        "symbol_count": int(stats[1]),
        "min_bucket": str(stats[2]),
        "max_bucket": str(stats[3]),
        "min_minutes": int(stats[4]),
        "max_minutes": int(stats[5]),
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
