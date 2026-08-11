from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="hot20_hf_aggregated")
    parser.add_argument("--work-dir", default="hot20_v6_work")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = (root / args.input_dir).resolve()
    work = (root / args.work_dir).resolve()
    raw = work / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True, exist_ok=True)

    files = [input_dir / f"{split}_15m.parquet" for split in ["train", "val", "test"]]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing aggregated files: {missing}")

    file_list = ",".join("'{}'".format(q(path)) for path in files)
    con = duckdb.connect(str(work / "finalize.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute("SET TimeZone='America/New_York'")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA temp_directory='{}'".format(q(work / "duck_tmp")))
    symbols = [row[0] for row in con.execute(
        f"SELECT DISTINCT symbol FROM read_parquet([{file_list}], union_by_name=true) ORDER BY symbol"
    ).fetchall()]
    stats = []
    for index, symbol in enumerate(symbols, start=1):
        output = raw / f"{symbol}.csv.gz"
        con.execute(
            f"""
            COPY (
              SELECT
                epoch_ms(bucket) AS timestamp,
                arg_min(open, bucket) AS open,
                max(high) AS high,
                min(low) AS low,
                arg_max(close, bucket) AS close,
                sum(volume) AS volume
              FROM read_parquet([{file_list}], union_by_name=true)
              WHERE symbol = ?
              GROUP BY bucket
              ORDER BY bucket
            ) TO '{q(output)}' (HEADER TRUE, FORMAT CSV, COMPRESSION GZIP)
            """,
            [symbol],
        )
        row = con.execute(
            f"""
            SELECT count(DISTINCT bucket), min(bucket), max(bucket), count(DISTINCT CAST(bucket AS DATE))
            FROM read_parquet([{file_list}], union_by_name=true)
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchone()
        stats.append({
            "ticker": symbol,
            "rows": int(row[0]),
            "start": str(row[1]),
            "end": str(row[2]),
            "days": int(row[3]),
            "bytes": output.stat().st_size,
        })
        print(f"[{index:02d}/{len(symbols):02d}] {symbol}: {row[0]:,} bars {row[1]} -> {row[2]}", flush=True)
    con.close()

    coverage: dict[str, dict[int, int]] = {}
    for path in sorted(raw.glob("*.csv.gz")):
        frame = pd.read_csv(path, usecols=["timestamp"])
        dt = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        counts = pd.Series(dt.dt.date).groupby(dt.dt.year).nunique()
        coverage[path.name.replace(".csv.gz", "")] = {int(k): int(v) for k, v in counts.items()}

    years = sorted({year for item in coverage.values() for year in item})
    usable_counts = {year: sum(days.get(year, 0) >= 120 for days in coverage.values()) for year in years}
    common = [year for year in years if usable_counts[year] >= 35]
    preferred = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    if all(year in common for year in preferred):
        periods = {
            "development": [2019, 2021],
            "inner_validation": [2022, 2022],
            "outer_validation": [2023, 2023],
            "holdout": [2024, 2024],
            "forward": [2025, 2025],
        }
    elif len(common) >= 6:
        chosen = common[-6:]
        periods = {
            "development": [chosen[0], chosen[1]],
            "inner_validation": [chosen[2], chosen[2]],
            "outer_validation": [chosen[3], chosen[3]],
            "holdout": [chosen[4], chosen[4]],
            "forward": [chosen[5], chosen[5]],
        }
    elif len(common) >= 5:
        chosen = common[-5:]
        periods = {
            "development": [chosen[0], chosen[0]],
            "inner_validation": [chosen[1], chosen[1]],
            "outer_validation": [chosen[2], chosen[2]],
            "holdout": [chosen[3], chosen[3]],
            "forward": [chosen[4], chosen[4]],
        }
    else:
        raise RuntimeError(f"Need at least five common years; got common={common}, counts={usable_counts}")

    payload = {
        "periods": periods,
        "common_years": common,
        "usable_stock_count_by_year": usable_counts,
        "coverage": coverage,
        "symbol_stats": stats,
        "symbol_count": len(stats),
        "source": "twelvedata/financial-world-model via Hugging Face",
        "regular_session_only": True,
        "aggregation": "1-minute to 15-minute OHLCV with >=10 source minutes per bar",
    }
    (work / "periods.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (work / "hf_prepare_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"coverage", "symbol_stats"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
