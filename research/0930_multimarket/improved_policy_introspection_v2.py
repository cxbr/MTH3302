from __future__ import annotations

import inspect
import json
from dataclasses import asdict, fields
from pathlib import Path

import improved_system_benchmark as base

OUT = Path(__file__).resolve().parent / "improved_policy_introspection_output"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    catalog = [asdict(p) for p in base.policy_catalog()]
    report = {
        "fields": [f.name for f in fields(base.ImprovedPolicy)],
        "constructor_signature": str(inspect.signature(base.ImprovedPolicy)),
        "simulate_trade_signature": str(inspect.signature(base.simulate_trade)),
        "catalog": catalog,
        "exit_kinds": sorted({str(row.get("exit_kind")) for row in catalog}),
        "scale_kinds": sorted({str(row.get("scale_kind")) for row in catalog}),
    }
    print(json.dumps(report, indent=2), flush=True)
    (OUT / "improved_policy_introspection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
