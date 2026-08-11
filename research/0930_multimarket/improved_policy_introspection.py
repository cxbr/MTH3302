from __future__ import annotations

import inspect
import json
from dataclasses import asdict, fields
from pathlib import Path

import improved_system_benchmark as base

OUT = Path(__file__).resolve().parent / "improved_policy_introspection_output"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    field_names = [f.name for f in fields(base.ImprovedPolicy)]
    catalog = [asdict(p) for p in base.policy_catalog()]
    report = {
        "fields": field_names,
        "catalog": catalog,
        "exit_kinds": sorted({str(row.get("exit_kind")) for row in catalog}),
        "scale_kinds": sorted({str(row.get("scale_kind")) for row in catalog}),
        "simulate_trade_source": inspect.getsource(base.simulate_trade),
        "policy_source": inspect.getsource(base.ImprovedPolicy),
    }
    print(json.dumps({
        "fields": report["fields"],
        "exit_kinds": report["exit_kinds"],
        "scale_kinds": report["scale_kinds"],
        "catalog": report["catalog"],
    }, indent=2), flush=True)
    print("\nSIMULATE_TRADE_SOURCE\n", report["simulate_trade_source"], flush=True)
    (OUT / "improved_policy_introspection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
