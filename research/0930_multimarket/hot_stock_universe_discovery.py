from __future__ import annotations

import json
from pathlib import Path

import kagglehub

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "AVGO", "NFLX", "INTC", "QCOM", "MU", "AMAT", "CRM", "ORCL",
    "ADBE", "CSCO", "PYPL", "SHOP", "UBER", "PLTR", "COIN", "MSTR",
    "JPM", "BAC", "WMT", "COST", "HD", "DIS", "BA", "CAT", "XOM",
    "CVX", "LLY", "UNH", "NKE", "SBUX", "TSM", "ARM", "SMCI",
    "MARA", "RIVN", "SNAP", "SQ", "ROKU", "PANW", "NOW", "TXN",
]

OUT = Path(__file__).resolve().parent / "hot_stock_discovery_output"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for ticker in TICKERS:
    handle = f"yug201/{ticker.lower()}-price-dataset-all-timeframe"
    row = {"ticker": ticker, "handle": handle, "available": False}
    try:
        root = Path(kagglehub.dataset_download(handle))
        files = [p for p in root.rglob("*") if p.is_file()]
        intraday = [
            p for p in files
            if p.suffix.lower() in {".csv", ".txt"}
            and any(token in p.name.lower() for token in ["1min", "1_min", "1m"])
        ]
        row.update({
            "available": bool(intraday),
            "root": str(root),
            "file_count": len(files),
            "intraday_files": [str(p.relative_to(root)) for p in intraday],
            "intraday_bytes": sum(p.stat().st_size for p in intraday),
        })
    except Exception as exc:
        row["error"] = repr(exc)
    rows.append(row)
    print(json.dumps(row), flush=True)

(OUT / "discovery.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
available = [row["ticker"] for row in rows if row.get("available")]
(OUT / "available_tickers.txt").write_text("\n".join(available) + "\n", encoding="utf-8")
print(json.dumps({"requested": len(rows), "available": len(available), "tickers": available}, indent=2))
