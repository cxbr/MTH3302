from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from data_pipeline import LOCAL_TZ, MARKET_SPECS, MarketSpec


STOCK_SOURCES = {
    "AAPL": {
        "handle": "johnkd/aapl-historical-intraday-dataset",
        "kind": "aapl_first_rate",
        "description": "FirstRateData-derived AAPL one-minute bars, 2010-2019",
    },
    "SPY": {
        "handle": "rockinbrock/spy-1-minute-data",
        "kind": "spy",
        "description": "SPY one-minute bars, 2008-2021",
    },
    "AMZN": {
        "handle": "manognap2505/equiml",
        "kind": "equiml",
        "description": "EQUIML AMZN minute bars",
    },
    "GOOGL": {
        "handle": "manognap2505/equiml",
        "kind": "equiml",
        "description": "EQUIML GOOGL minute bars",
    },
    "META": {
        "handle": "manognap2505/equiml",
        "kind": "equiml",
        "description": "EQUIML META minute bars",
    },
    "MSFT": {
        "handle": "manognap2505/equiml",
        "kind": "equiml",
        "description": "EQUIML MSFT minute bars",
    },
    "NVDA": {
        "handle": "manognap2505/equiml",
        "kind": "equiml",
        "description": "EQUIML NVDA minute bars",
    },
    "TSLA": {
        "handle": "poivronjaune/tsla-ohlcv-minute",
        "kind": "tsla_parquet",
        "description": "TSLA minute OHLCV parquet archive",
    },
}


for ticker, source in STOCK_SOURCES.items():
    if ticker not in MARKET_SPECS:
        MARKET_SPECS[ticker] = MarketSpec(
            name=ticker,
            category="ETF" if ticker == "SPY" else "Stock",
            source=source["description"],
            source_url=f"https://www.kaggle.com/datasets/{source['handle']}",
            timestamp_tz=LOCAL_TZ,
            regular_hours_only=True,
            week_ending="W-FRI",
            base_cost_rate=0.00015,
        )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rename_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    mapping = {}
    for col in frame.columns:
        lower = str(col).strip().lower()
        if lower in {"date", "datetime", "timestamp", "time"}:
            mapping[col] = "datetime"
        elif lower in {"open", "1. open", "o"}:
            mapping[col] = "open"
        elif lower in {"high", "2. high", "h"}:
            mapping[col] = "high"
        elif lower in {"low", "3. low", "l"}:
            mapping[col] = "low"
        elif lower in {"close", "4. close", "c", "adj close", "adj_close"}:
            if "close" not in mapping.values():
                mapping[col] = "close"
        elif lower in {"volume", "5. volume", "v"}:
            mapping[col] = "volume"
    frame = frame.rename(columns=mapping)
    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}; available={frame.columns.tolist()}")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    return frame[["datetime", "open", "high", "low", "close", "volume"]]


def _parse_and_localize(
    frame: pd.DataFrame,
    timezone_mode: str,
) -> pd.DataFrame:
    frame = _rename_ohlcv(frame)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    timestamp = pd.to_datetime(frame["datetime"], errors="coerce", utc=False)
    mask = timestamp.notna()
    frame = frame.loc[mask, ["open", "high", "low", "close", "volume"]].copy()
    idx = pd.DatetimeIndex(timestamp[mask])

    if idx.tz is not None:
        idx = idx.tz_convert(LOCAL_TZ)
    elif timezone_mode == "UTC":
        idx = idx.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward").tz_convert(LOCAL_TZ)
    elif timezone_mode == "NY":
        idx = idx.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
    elif timezone_mode.startswith("SHIFT_NY_"):
        hours = float(timezone_mode.split("_")[-1])
        idx = (idx - pd.Timedelta(hours=hours)).tz_localize(
            LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        raise ValueError(timezone_mode)

    valid = ~idx.isna()
    frame = frame.loc[valid].copy()
    frame.index = idx[valid]
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[
        (frame["open"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
    ]
    frame = frame[
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    ]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _rth_15m(frame: pd.DataFrame) -> pd.DataFrame:
    minutes = frame.index.hour * 60 + frame.index.minute
    frame = frame[(minutes >= 570) & (minutes < 960)]
    out = frame.resample(
        "15min", origin="start_day", label="left", closed="left"
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"])
    minutes = out.index.hour * 60 + out.index.minute
    out = out[(minutes >= 570) & (minutes <= 945)]
    return out.astype(float)


def _score_timezone(frame: pd.DataFrame) -> Tuple[float, dict]:
    if frame.empty:
        return -1.0, {}
    unique_dates = pd.Index(frame.index.date).nunique()
    ref_count = int(((frame.index.hour == 9) & (frame.index.minute == 30)).sum())
    coverage = ref_count / unique_dates if unique_dates else 0.0
    bars_per_date = pd.Series(frame.index.date).value_counts()
    complete = float((bars_per_date >= 24).mean()) if len(bars_per_date) else 0.0
    score = coverage + 0.25 * complete
    return score, {
        "reference_coverage": coverage,
        "complete_session_fraction": complete,
        "dates": int(unique_dates),
        "rows": int(len(frame)),
    }


def _best_timezone(raw: pd.DataFrame, modes: Iterable[str]) -> Tuple[pd.DataFrame, str, dict]:
    best = None
    for mode in modes:
        try:
            candidate = _rth_15m(_parse_and_localize(raw, mode))
            score, details = _score_timezone(candidate)
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, candidate, mode, details)
    if best is None:
        raise RuntimeError("No timezone interpretation produced usable bars")
    return best[1], best[2], best[3]


def _split_adjust(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    """Apply simple backward split adjustment using large overnight integer ratios."""
    out = frame.copy().sort_index()
    dates = pd.Index(out.index.date)
    unique_dates = dates.unique()
    first_indices = []
    last_indices = []
    for date in unique_dates:
        positions = np.flatnonzero(dates == date)
        first_indices.append(int(positions[0]))
        last_indices.append(int(positions[-1]))

    known = np.array(
        [0.05, 0.10, 0.20, 0.25, 1 / 3, 0.50, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0],
        dtype=float,
    )
    events = []
    prices = out[["open", "high", "low", "close"]].to_numpy(dtype=float)
    volume = out["volume"].to_numpy(dtype=float)
    for day_pos in range(1, len(unique_dates)):
        first_i = first_indices[day_pos]
        prev_i = last_indices[day_pos - 1]
        ratio = prices[first_i, 0] / prices[prev_i, 3]
        nearest = float(known[np.argmin(np.abs(np.log(known) - np.log(ratio)))])
        relative_error = abs(ratio / nearest - 1.0)
        if (ratio < 0.60 or ratio > 1.70) and relative_error <= 0.12:
            factor = nearest
            prices[:first_i, :] *= factor
            if factor > 0:
                volume[:first_i] /= factor
            events.append(
                {
                    "date": str(unique_dates[day_pos]),
                    "observed_ratio": float(ratio),
                    "applied_factor": float(factor),
                    "relative_error": float(relative_error),
                }
            )
    out[["open", "high", "low", "close"]] = prices
    out["volume"] = volume
    return out, events


def _load_aapl(root: Path) -> Tuple[pd.DataFrame, List[Path], str, dict]:
    path = next(root.rglob("AAPL_1min.txt"))
    raw = pd.read_csv(
        path,
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
        low_memory=False,
    )
    frame, mode, details = _best_timezone(raw, ["NY", "UTC"])
    return frame, [path], mode, details


def _load_spy(root: Path) -> Tuple[pd.DataFrame, List[Path], str, dict]:
    path = next(root.rglob("spy_1min_2008_2021_cleaned.csv"))
    raw = pd.read_csv(path, low_memory=False)
    frame, mode, details = _best_timezone(raw, ["NY", "UTC"])
    return frame, [path], mode, details


def _load_equiml(root: Path, ticker: str) -> Tuple[pd.DataFrame, List[Path], str, dict]:
    paths = [
        root / "train" / f"{ticker}_1min.csv",
        root / "test" / f"{ticker}_1min_test.csv",
    ]
    frames = [pd.read_csv(path, low_memory=False) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError(f"No EQUIML files for {ticker}")
    raw = pd.concat(frames, ignore_index=True)
    modes = ["UTC", "NY"] + [f"SHIFT_NY_{h}" for h in range(3, 8)]
    frame, mode, details = _best_timezone(raw, modes)
    return frame, [p for p in paths if p.exists()], mode, details


def _load_tsla(root: Path) -> Tuple[pd.DataFrame, List[Path], str, dict]:
    path = next(root.rglob("*.parquet"))
    raw = pd.read_parquet(path)
    if not any(str(c).lower() in {"date", "datetime", "timestamp", "time"} for c in raw.columns):
        raw = raw.reset_index()
        first = raw.columns[0]
        raw = raw.rename(columns={first: "datetime"})
    modes = ["UTC", "NY"] + [f"SHIFT_NY_{h}" for h in range(3, 8)]
    frame, mode, details = _best_timezone(raw, modes)
    return frame, [path], mode, details


def _complete_years(frame: pd.DataFrame) -> Tuple[List[int], Dict[str, int]]:
    reference = frame[(frame.index.hour == 9) & (frame.index.minute == 30)]
    counts = pd.Series(reference.index.year).value_counts().sort_index()
    complete = [int(year) for year, count in counts.items() if int(count) >= 180]
    if len(complete) < 3:
        complete = sorted([int(x) for x in counts.nlargest(min(3, len(counts))).index])
    if len(complete) < 3:
        raise RuntimeError(f"Fewer than three usable years: {counts.to_dict()}")
    validation = complete[-2]
    holdout = complete[-1]
    split = {
        "development_start": complete[0],
        "development_end": validation - 1,
        "validation": validation,
        "holdout": holdout,
    }
    return complete, split


def _quality_row(
    ticker: str,
    frame: pd.DataFrame,
    paths: List[Path],
    timezone_mode: str,
    timezone_details: dict,
    split_events: List[dict],
    complete_years: List[int],
    split: dict,
) -> dict:
    dates = pd.Index(frame.index.date)
    unique_dates = int(dates.nunique())
    ref_count = int(((frame.index.hour == 9) & (frame.index.minute == 30)).sum())
    bars_by_day = pd.Series(dates).value_counts()
    return {
        "market": ticker,
        "category": MARKET_SPECS[ticker].category,
        "source": MARKET_SPECS[ticker].source,
        "source_url": MARKET_SPECS[ticker].source_url,
        "files": ";".join(str(p) for p in paths),
        "file_sha256": ";".join(_sha256(p) for p in paths),
        "timezone_mode": timezone_mode,
        "rows_15m": int(len(frame)),
        "start_local": frame.index.min().isoformat(),
        "end_local": frame.index.max().isoformat(),
        "local_dates": unique_dates,
        "reference_0930_bars": ref_count,
        "reference_coverage_pct": 100.0 * ref_count / unique_dates if unique_dates else 0.0,
        "median_bars_per_day": float(bars_by_day.median()) if len(bars_by_day) else 0.0,
        "complete_session_fraction": float((bars_by_day >= 24).mean()) if len(bars_by_day) else 0.0,
        "complete_years": ",".join(map(str, complete_years)),
        "split_events": json.dumps(split_events),
        **{f"split_{key}": value for key, value in split.items()},
        **{f"tz_{key}": value for key, value in timezone_details.items()},
    }


def load_stock_heavy_markets(work_dir: Path):
    import kagglehub

    work_dir.mkdir(parents=True, exist_ok=True)
    roots: Dict[str, Path] = {}
    for source in STOCK_SOURCES.values():
        handle = source["handle"]
        if handle not in roots:
            roots[handle] = Path(kagglehub.dataset_download(handle))

    frames: Dict[str, pd.DataFrame] = {}
    splits: Dict[str, Dict[str, int]] = {}
    quality_rows = []
    rejected = []

    for ticker, source in STOCK_SOURCES.items():
        try:
            root = roots[source["handle"]]
            if source["kind"] == "aapl_first_rate":
                frame, paths, mode, tz_details = _load_aapl(root)
            elif source["kind"] == "spy":
                frame, paths, mode, tz_details = _load_spy(root)
            elif source["kind"] == "equiml":
                frame, paths, mode, tz_details = _load_equiml(root, ticker)
            elif source["kind"] == "tsla_parquet":
                frame, paths, mode, tz_details = _load_tsla(root)
            else:
                raise ValueError(source["kind"])

            frame, split_events = _split_adjust(frame)
            complete_years, split = _complete_years(frame)
            quality = _quality_row(
                ticker,
                frame,
                paths,
                mode,
                tz_details,
                split_events,
                complete_years,
                split,
            )
            if quality["reference_coverage_pct"] < 75.0:
                raise RuntimeError(
                    f"09:30 coverage only {quality['reference_coverage_pct']:.1f}%"
                )
            if quality["complete_session_fraction"] < 0.65:
                raise RuntimeError(
                    f"complete-session fraction only {quality['complete_session_fraction']:.2f}"
                )
            frames[ticker] = frame
            splits[ticker] = split
            quality_rows.append(quality)
            print(
                f"{ticker}: {len(frame):,} bars, {frame.index.min()} -> {frame.index.max()}, "
                f"09:30={quality['reference_coverage_pct']:.1f}%, split={split}, timezone={mode}",
                flush=True,
            )
        except Exception as exc:
            rejected.append({"market": ticker, "error": repr(exc)})
            print(f"REJECTED {ticker}: {exc!r}", flush=True)

    if len(frames) < 5:
        raise RuntimeError(
            f"Only {len(frames)} stock markets passed quality gates; rejected={rejected}"
        )

    quality = pd.DataFrame(quality_rows)
    rejected_frame = pd.DataFrame(rejected)
    quality.to_csv(work_dir / "stock_data_quality.csv", index=False)
    rejected_frame.to_csv(work_dir / "stock_data_rejected.csv", index=False)
    (work_dir / "stock_splits.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )
    return frames, quality, splits, rejected_frame
