from __future__ import annotations

import json
import os
from pathlib import Path

import kagglehub
import pandas as pd

OUT = Path(__file__).resolve().parent / "kaggle_smoke_output"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "AAPL": "johnkd/aapl-historical-intraday-dataset",
    "SPY": "rockinbrock/spy-1-minute-data",
    "NQ": "tgtanalytics/nq-futures-1min-bar-2022-2025",
    "MSFT": "yug201/msft-price-dataset-all-timeframe",
}


def inspect_file(path: Path) -> dict:
    item = {"path": str(path), "size_bytes": path.stat().st_size}
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            df = pd.read_csv(path, nrows=5)
            item["columns"] = [str(c) for c in df.columns]
            item["sample"] = df.head(3).astype(str).to_dict(orient="records")
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path).head(5)
            item["columns"] = [str(c) for c in df.columns]
            item["sample"] = df.head(3).astype(str).to_dict(orient="records")
    except Exception as exc:
        item["inspect_error"] = repr(exc)
    return item


def main() -> None:
    result = {"datasets": {}, "failures": []}
    for market, handle in DATASETS.items():
        print(f"Downloading {market}: {handle}", flush=True)
        try:
            root = Path(kagglehub.dataset_download(handle))
            files = [p for p in root.rglob("*") if p.is_file()]
            files.sort(key=lambda p: p.stat().st_size, reverse=True)
            result["datasets"][market] = {
                "handle": handle,
                "root": str(root),
                "total_files": len(files),
                "total_bytes": int(sum(p.stat().st_size for p in files)),
                "files": [inspect_file(p) for p in files[:20]],
            }
        except Exception as exc:
            result["failures"].append({"market": market, "handle": handle, "error": repr(exc)})
    (OUT / "kaggle_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
