from __future__ import annotations

import json
from pathlib import Path

import kagglehub
import pandas as pd

HANDLES = {
    "AAPL_YUG": "yug201/aapl-price-dataset-all-timeframe",
    "MSFT_YUG": "yug201/msft-price-dataset-all-timeframe",
    "NVDA_YUG": "yug201/nvda-price-dataset-all-timeframe",
    "AMZN_YUG": "yug201/amzn-price-dataset-all-timeframe",
    "TSLA_YUG": "yug201/tsla-price-dataset-all-timeframe",
    "EQUIML": "manognap2505/equiml",
}

OUT = Path(__file__).resolve().parent / "stock_data_discovery_output"
OUT.mkdir(parents=True, exist_ok=True)


def sample_csv(path: Path) -> dict:
    result = {
        "file": str(path),
        "size": path.stat().st_size,
    }
    try:
        frame = pd.read_csv(path, nrows=6, low_memory=False)
        result["columns"] = [str(c) for c in frame.columns]
        result["head"] = frame.head(3).astype(str).to_dict(orient="records")
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> None:
    report = {}
    for label, handle in HANDLES.items():
        print(f"Downloading {label}: {handle}", flush=True)
        root = Path(kagglehub.dataset_download(handle))
        files = sorted([p for p in root.rglob("*") if p.is_file()])
        csvs = [p for p in files if p.suffix.lower() in {".csv", ".txt"}]
        selected = []
        for path in csvs:
            lower = path.name.lower()
            if any(token in lower for token in ["1min", "1_min", "1m", "minute"]):
                selected.append(path)
        if not selected:
            selected = csvs[:20]
        report[label] = {
            "handle": handle,
            "root": str(root),
            "file_count": len(files),
            "files": [str(p.relative_to(root)) for p in files[:100]],
            "samples": [sample_csv(p) for p in selected[:20]],
        }
        print(json.dumps(report[label], indent=2)[:12000], flush=True)
    (OUT / "stock_data_discovery.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
