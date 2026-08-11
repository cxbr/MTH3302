from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "hot20_v6_work"
RAW = WORK / "raw"
PARTS = WORK / "hf_parts"
CACHE = WORK / "hf_cache"
OUT = ROOT / "hot20_v6_output"
for path in [RAW, PARTS, CACHE, OUT]:
    path.mkdir(parents=True, exist_ok=True)

REPO_ID = "twelvedata/financial-world-model"
FILES = [
    ("train", "bars_1min/train.parquet"),
    ("val", "bars_1min/val.parquet"),
    ("test", "bars_1min/test.parquet"),
]


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def download_and_partition(split: str, filename: str) -> dict:
    print(f"DOWNLOAD {filename}", flush=True)
    local = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=CACHE,
        )
    )
    print(f"DOWNLOADED {local} bytes={local.stat().st_size:,}", flush=True)
    split_dir = PARTS / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(WORK / "prepare.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA temp_directory='{}'".format(q(WORK / "duck_tmp")))
    copy_sql = f"""
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
      FROM read_parquet('{q(local)}')
      WHERE datetime >= TIMESTAMPTZ '2018-01-01 00:00:00 America/New_York'
        AND datetime <  TIMESTAMPTZ '2026-01-01 00:00:00 America/New_York'
        AND CAST(datetime AS TIME) >= TIME '09:30:00'
        AND CAST(datetime AS TIME) <  TIME '16:00:00'
      GROUP BY symbol, bucket
      HAVING count(*) >= 10
      ORDER BY symbol, bucket
    ) TO '{q(split_dir)}' (
      FORMAT PARQUET,
      PARTITION_BY (symbol),
      COMPRESSION ZSTD,
      OVERWRITE_OR_IGNORE TRUE
    )
    """
    con.execute(copy_sql)
    stats = con.execute(
        f"""
        SELECT count(*) rows, count(DISTINCT symbol) symbols,
               min(datetime) min_datetime, max(datetime) max_datetime
        FROM read_parquet('{q(local)}')
        """
    ).fetchone()
    con.close()
    print(f"PARTITIONED {split}: rows={stats[0]:,} symbols={stats[1]} {stats[2]} -> {stats[3]}", flush=True)
    try:
        local.unlink()
    except OSError:
        pass
    gc.collect()
    return {
        "split": split,
        "filename": filename,
        "rows": int(stats[0]),
        "symbols": int(stats[1]),
        "min_datetime": str(stats[2]),
        "max_datetime": str(stats[3]),
    }


def combine_symbols() -> list[dict]:
    symbol_names = set()
    for split, _ in FILES:
        for child in (PARTS / split).glob("symbol=*"):
            symbol_names.add(child.name.split("=", 1)[1])
    print(f"COMBINE {len(symbol_names)} symbols", flush=True)

    con = duckdb.connect(str(WORK / "combine.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA temp_directory='{}'".format(q(WORK / "duck_tmp")))
    rows = []
    for index, symbol in enumerate(sorted(symbol_names), start=1):
        files = []
        for split, _ in FILES:
            files.extend(sorted((PARTS / split / f"symbol={symbol}").glob("*.parquet")))
        if not files:
            continue
        file_list = ",".join("'{}'".format(q(path)) for path in files)
        output = RAW / f"{symbol}.csv.gz"
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
              GROUP BY bucket
              ORDER BY bucket
            ) TO '{q(output)}' (HEADER TRUE, FORMAT CSV, COMPRESSION GZIP)
            """
        )
        stat = con.execute(
            f"""
            SELECT count(*) rows, min(bucket), max(bucket), count(DISTINCT CAST(bucket AS DATE)) days
            FROM read_parquet([{file_list}], union_by_name=true)
            """
        ).fetchone()
        rows.append({
            "ticker": symbol,
            "rows": int(stat[0]),
            "start": str(stat[1]),
            "end": str(stat[2]),
            "days": int(stat[3]),
            "output_bytes": output.stat().st_size,
        })
        print(f"[{index:02d}/{len(symbol_names):02d}] {symbol}: {stat[0]:,} bars {stat[1]} -> {stat[2]}", flush=True)
    con.close()
    return rows


def choose_periods() -> dict:
    coverage: dict[str, dict[int, int]] = {}
    for path in sorted(RAW.glob("*.csv.gz")):
        frame = pd.read_csv(path, usecols=["timestamp"])
        dt = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        counts = pd.Series(dt.dt.date).groupby(dt.dt.year).nunique()
        coverage[path.stem.replace(".csv", "")] = {int(k): int(v) for k, v in counts.items()}

    years = sorted({year for item in coverage.values() for year in item})
    usable_counts = {year: sum(days.get(year, 0) >= 120 for days in coverage.values()) for year in years}
    common = [year for year in years if usable_counts[year] >= 35]
    if len(common) < 5:
        raise RuntimeError(f"Need five years with >=35 stocks and 120 days; got {common} counts={usable_counts}")
    chosen = common[-5:]
    periods = {
        "development": [chosen[0], chosen[0]],
        "inner_validation": [chosen[1], chosen[1]],
        "outer_validation": [chosen[2], chosen[2]],
        "holdout": [chosen[3], chosen[3]],
        "forward": [chosen[4], chosen[4]],
    }
    payload = {
        "periods": periods,
        "common_years": common,
        "chosen_years": chosen,
        "usable_stock_count_by_year": usable_counts,
        "coverage": coverage,
    }
    (WORK / "periods.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload | {"coverage": "omitted"}, indent=2), flush=True)
    return payload


def main() -> None:
    shutil.rmtree(RAW, ignore_errors=True)
    RAW.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(PARTS, ignore_errors=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    download_stats = [download_and_partition(split, filename) for split, filename in FILES]
    symbol_stats = combine_symbols()
    periods = choose_periods()
    manifest = {
        "source": REPO_ID,
        "files": download_stats,
        "symbols": symbol_stats,
        "symbol_count": len(symbol_stats),
        "periods": periods["periods"],
        "regular_session_only": True,
        "aggregation": "1-minute to causal 15-minute OHLCV",
    }
    (WORK / "hf_prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.rmtree(PARTS, ignore_errors=True)
    shutil.rmtree(CACHE, ignore_errors=True)
    print(json.dumps({k: v for k, v in manifest.items() if k != "symbols"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
