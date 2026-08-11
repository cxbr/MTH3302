from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

LOCAL_TZ = "America/New_York"


@dataclass(frozen=True)
class MarketSpec:
    name: str
    category: str
    source: str
    source_url: str
    timestamp_tz: str
    regular_hours_only: bool
    week_ending: str
    base_cost_rate: float


MARKET_SPECS: Dict[str, MarketSpec] = {
    "AAPL": MarketSpec(
        name="AAPL",
        category="Stock",
        source="Kaggle: johnkd/aapl-historical-intraday-dataset",
        source_url="https://www.kaggle.com/datasets/johnkd/aapl-historical-intraday-dataset",
        timestamp_tz=LOCAL_TZ,
        regular_hours_only=True,
        week_ending="W-FRI",
        base_cost_rate=0.00015,
    ),
    "MSFT": MarketSpec(
        name="MSFT",
        category="Stock",
        source="Kaggle: yug201/msft-price-dataset-all-timeframe",
        source_url="https://www.kaggle.com/datasets/yug201/msft-price-dataset-all-timeframe",
        timestamp_tz="UTC",
        regular_hours_only=True,
        week_ending="W-FRI",
        base_cost_rate=0.00015,
    ),
    "SPY": MarketSpec(
        name="SPY",
        category="ETF",
        source="Kaggle: rockinbrock/spy-1-minute-data",
        source_url="https://www.kaggle.com/datasets/rockinbrock/spy-1-minute-data",
        timestamp_tz=LOCAL_TZ,
        regular_hours_only=True,
        week_ending="W-FRI",
        base_cost_rate=0.00015,
    ),
    "NQ": MarketSpec(
        name="NQ",
        category="CME E-mini Nasdaq-100 futures",
        source="Kaggle: tgtanalytics/nq-futures-1min-bar-2022-2025",
        source_url="https://www.kaggle.com/datasets/tgtanalytics/nq-futures-1min-bar-2022-2025",
        timestamp_tz=LOCAL_TZ,
        regular_hours_only=False,
        week_ending="W-FRI",
        base_cost_rate=0.00004,
    ),
    "BTCUSDT": MarketSpec(
        name="BTCUSDT",
        category="Crypto spot",
        source="Binance public 15-minute klines",
        source_url="https://data.binance.vision/",
        timestamp_tz="UTC",
        regular_hours_only=False,
        week_ending="W-SUN",
        base_cost_rate=0.0008,
    ),
}

KAGGLE_HANDLES = {
    "AAPL": "johnkd/aapl-historical-intraday-dataset",
    "MSFT": "yug201/msft-price-dataset-all-timeframe",
    "SPY": "rockinbrock/spy-1-minute-data",
    "NQ": "tgtanalytics/nq-futures-1min-bar-2022-2025",
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _localize_naive(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if index.tz is not None:
        return index.tz_convert(tz)
    try:
        return index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        return index.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")


def _normalize_frame(
    frame: pd.DataFrame,
    timestamp_col: str,
    timestamp_tz: str,
    regular_hours_only: bool,
    already_15m: bool = False,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    timestamp_col = timestamp_col.lower()
    if timestamp_col not in frame.columns:
        raise ValueError(f"timestamp column {timestamp_col!r} not in {frame.columns.tolist()}")

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in frame.columns:
            raise ValueError(f"missing {col}; columns={frame.columns.tolist()}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    frame = frame.loc[timestamps.notna(), ["open", "high", "low", "close", "volume"]]
    timestamps = pd.DatetimeIndex(timestamps[timestamps.notna()])
    timestamps = _localize_naive(timestamps, timestamp_tz).tz_convert(LOCAL_TZ)
    valid = ~timestamps.isna()
    frame = frame.loc[valid].copy()
    frame.index = timestamps[valid]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    frame = frame.dropna(subset=required)
    frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)]
    frame = frame[
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    ]

    if regular_hours_only:
        minutes = frame.index.hour * 60 + frame.index.minute
        frame = frame[(minutes >= 570) & (minutes < 960)]

    if already_15m:
        out = frame.copy()
        out = out[(out.index.minute % 15) == 0]
    else:
        out = frame.resample("15min", origin="start_day", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        out = out.dropna(subset=["open", "high", "low", "close"])

    if regular_hours_only:
        minutes = out.index.hour * 60 + out.index.minute
        out = out[(minutes >= 570) & (minutes <= 945)]

    return out.astype(
        {"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"}
    )


def _load_aapl(root: Path) -> Tuple[pd.DataFrame, Path]:
    path = next(root.glob("AAPL_1min.txt"))
    frame = pd.read_csv(
        path,
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
        low_memory=False,
    )
    return _normalize_frame(frame, "datetime", LOCAL_TZ, True), path


def _load_spy(root: Path) -> Tuple[pd.DataFrame, Path]:
    path = next(root.glob("spy_1min_2008_2021_cleaned.csv"))
    frame = pd.read_csv(path, low_memory=False)
    return _normalize_frame(frame, "date", LOCAL_TZ, True), path


def _load_nq(root: Path) -> Tuple[pd.DataFrame, Path]:
    path = next(root.glob("Dataset_NQ_1min_2022_2025.csv"))
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.rename(columns={"timestamp ET": "datetime"})
    return _normalize_frame(frame, "datetime", LOCAL_TZ, False), path


def _load_msft(root: Path) -> Tuple[pd.DataFrame, Path]:
    path = next(root.glob("MSFT_1min.csv"))
    frame = pd.read_csv(path, low_memory=False)
    return _normalize_frame(frame, "datetime", "UTC", True), path


def _fetch_binance_15m(start: str, end: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        raw = pd.read_csv(cache_path)
        return _normalize_frame(raw, "datetime", "UTC", False, already_15m=True)

    session = requests.Session()
    endpoints = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    cursor = start_ms
    rows = []
    endpoint_idx = 0
    while cursor < end_ms:
        endpoint = endpoints[endpoint_idx]
        try:
            response = session.get(
                endpoint,
                params={
                    "symbol": "BTCUSDT",
                    "interval": "15m",
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=45,
            )
            response.raise_for_status()
            batch = response.json()
        except Exception:
            if endpoint_idx + 1 < len(endpoints):
                endpoint_idx += 1
                continue
            raise
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 15 * 60 * 1000
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance")
        cursor = next_cursor
        if len(rows) % 50000 < 1000:
            print(
                f"BTCUSDT downloaded {len(rows):,} bars through {pd.to_datetime(cursor, unit='ms', utc=True)}",
                flush=True,
            )
        time.sleep(0.03)

    raw = pd.DataFrame(
        {
            "datetime": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cache_path, index=False)
    return _normalize_frame(raw, "datetime", "UTC", False, already_15m=True)


def _choose_complete_years(frame: pd.DataFrame, category: str) -> Dict[str, int]:
    ref = frame[(frame.index.hour == 9) & (frame.index.minute == 30)]
    counts = pd.Series(ref.index.year).value_counts().sort_index()
    threshold = 320 if category == "Crypto spot" else 180
    complete = [int(year) for year, count in counts.items() if int(count) >= threshold]
    if len(complete) < 3:
        complete = sorted([int(x) for x in counts.nlargest(min(3, len(counts))).index])
    if len(complete) < 3:
        raise RuntimeError(f"not enough complete years: {counts.to_dict()}")
    holdout = complete[-1]
    validation = complete[-2]
    return {
        "development_start": int(frame.index.year.min()),
        "development_end": validation - 1,
        "validation": validation,
        "holdout": holdout,
    }


def _quality_row(name: str, frame: pd.DataFrame, source_path: Path | None, spec: MarketSpec) -> dict:
    local_dates = pd.Index(frame.index.date)
    ref_count = int(((frame.index.hour == 9) & (frame.index.minute == 30)).sum())
    days = int(local_dates.nunique())
    ohlc_bad = int(
        (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    return {
        "market": name,
        "category": spec.category,
        "source": spec.source,
        "source_url": spec.source_url,
        "source_file": str(source_path) if source_path else "Binance API cache",
        "source_sha256": _sha256(source_path) if source_path and source_path.exists() else None,
        "rows_15m": int(len(frame)),
        "start_local": frame.index.min().isoformat(),
        "end_local": frame.index.max().isoformat(),
        "local_dates": days,
        "reference_0930_bars": ref_count,
        "reference_coverage_pct": float(100 * ref_count / days) if days else 0.0,
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "ohlc_violations": ohlc_bad,
        "median_step_minutes": float(frame.index.to_series().diff().dropna().dt.total_seconds().median() / 60),
        "regular_hours_only": spec.regular_hours_only,
        "week_ending": spec.week_ending,
        "base_cost_rate_per_fill": spec.base_cost_rate,
    }


def load_all_markets(work_dir: Path) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, Dict[str, int]]]:
    import kagglehub

    work_dir.mkdir(parents=True, exist_ok=True)
    roots = {name: Path(kagglehub.dataset_download(handle)) for name, handle in KAGGLE_HANDLES.items()}

    loaders = {
        "AAPL": _load_aapl,
        "MSFT": _load_msft,
        "SPY": _load_spy,
        "NQ": _load_nq,
    }
    frames: Dict[str, pd.DataFrame] = {}
    quality = []
    splits: Dict[str, Dict[str, int]] = {}

    for name, loader in loaders.items():
        frame, source_path = loader(roots[name])
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if frame.empty:
            raise RuntimeError(f"{name}: no usable 15-minute rows")
        frames[name] = frame
        spec = MARKET_SPECS[name]
        quality.append(_quality_row(name, frame, source_path, spec))
        splits[name] = _choose_complete_years(frame, spec.category)
        print(
            f"{name}: {len(frame):,} 15m bars, {frame.index.min()} -> {frame.index.max()}, split={splits[name]}",
            flush=True,
        )

    btc_cache = work_dir / "BTCUSDT_15m_2018_2025.csv"
    btc = _fetch_binance_15m("2018-01-01", "2025-12-31 23:59:59", btc_cache)
    frames["BTCUSDT"] = btc
    spec = MARKET_SPECS["BTCUSDT"]
    quality.append(_quality_row("BTCUSDT", btc, btc_cache, spec))
    splits["BTCUSDT"] = _choose_complete_years(btc, spec.category)
    print(
        f"BTCUSDT: {len(btc):,} 15m bars, {btc.index.min()} -> {btc.index.max()}, split={splits['BTCUSDT']}",
        flush=True,
    )

    quality_df = pd.DataFrame(quality)
    (work_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return frames, quality_df, splits
