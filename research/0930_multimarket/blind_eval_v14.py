from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import blind_hot20_exit_v13 as v13
import hot20_v6_benchmark as base
from v14_engine import BASE_COST, V14Policy, generate_templates, replay_portfolio

ROOT = Path(__file__).resolve().parent
FULL_WORK = ROOT / "hot20_v14_full"
RAW = FULL_WORK / "raw"
INTERNAL = ROOT / "hot20_v14_hidden_internal"
PRE_OUT = ROOT / "blind_hot20_v14_output"
OUT = ROOT / "blind_hot20_v14_blind_output"
OUT.mkdir(parents=True, exist_ok=True)

PERIOD_YEARS = {
    "blind_2024": (2024, 2024),
    "blind_2025": (2025, 2025),
    "blind_2026_ytd": (2026, 2026),
    "blind_continuous_2024_2026": (2024, 2026),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(value):
    return v13.safe(value)


def restore_special(value):
    """Reverse the safe JSON representation used by the frozen-policy artifact."""
    if isinstance(value, str):
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        if value == "NaN":
            return math.nan
        return value
    if isinstance(value, list):
        return [restore_special(item) for item in value]
    if isinstance(value, dict):
        return {key: restore_special(item) for key, item in value.items()}
    return value


def main() -> None:
    frozen_path = PRE_OUT / "frozen_hidden_policies.json"
    manifest_path = FULL_WORK / "hidden_manifest.json"
    if not frozen_path.exists() or not manifest_path.exists():
        raise RuntimeError("Missing frozen policy file or hidden manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_payload = restore_special(
        json.loads(frozen_path.read_text(encoding="utf-8"))
    )
    policies = [V14Policy(**payload) for payload in frozen_payload]
    if not policies:
        raise RuntimeError("No frozen policies")

    shutil.rmtree(INTERNAL, ignore_errors=True)
    INTERNAL.mkdir(parents=True, exist_ok=True)
    v13.WORK = FULL_WORK
    v13.RAW = RAW
    v13.OUT = INTERNAL
    v13.PERIOD_YEARS = PERIOD_YEARS
    v13.FORBIDDEN_YEARS = set()
    base.WORK = FULL_WORK
    base.RAW = RAW
    base.OUT = INTERNAL
    base.PERIOD_YEARS = PERIOD_YEARS
    base._register_market = v13.register_market

    prepared, features, quality, selection = v13.load_train_universe()
    scanner_lookup, scanner_table, scanners = v13.build_scanner_v13(prepared, selection)
    years = sorted(set(int(y) for mkt in prepared.values() for y in np.unique(mkt.year_arr)))
    if not {2024, 2025, 2026}.issubset(set(years)):
        raise RuntimeError(f"Hidden years incomplete: {years}")

    cache: Dict[Tuple[str, str, str], Tuple[List[dict], dict]] = {}
    aggregate_rows = []
    rejection_rows = []
    for policy_idx, policy in enumerate(policies, start=1):
        print(f"BLIND policy {policy_idx}/{len(policies)}: {policy.description}", flush=True)
        for period in PERIOD_YEARS:
            templates = []
            entry_rej = defaultdict(int)
            signature = policy.trade_signature()
            for ticker in selection:
                key = (ticker, period, signature)
                if key not in cache:
                    cache[key] = generate_templates(
                        ticker, prepared[ticker], features[ticker], policy, period,
                        scanner_lookup, v13.period_bounds,
                        include_post_exit_diagnostics=False,
                    )
                rows, rejected = cache[key]
                templates.extend(rows)
                for name, count in rejected.items():
                    entry_rej[name] += count
            metric, _, replay_rej, _ = replay_portfolio(
                policy, templates, period, BASE_COST, keep_trades=False
            )
            aggregate_rows.append({
                "policy_id": policy.policy_id,
                "policy_description": policy.description,
                "period": period,
                "data_end": manifest["hidden_bounds"]["max"],
                "trades": metric["trades"],
                "total_return": metric["total_return"],
                "ending_equity": metric["ending_equity"],
                "win_rate": metric["win_rate"],
                "profit_factor": metric["profit_factor"],
                "mean_r": metric["mean_r"],
                "median_r": metric["median_r"],
                "avg_winner_r": metric["avg_winner_r"],
                "avg_loser_r": metric["avg_loser_r"],
                "max_drawdown": metric["max_drawdown"],
                "longest_losing_streak": metric["longest_losing_streak"],
                "avg_holding_hours": metric["avg_holding_hours"],
                "weekly_mean_return": metric["weekly_mean_return"],
                "weekly_median_return": metric["weekly_median_return"],
                "weekly_positive_rate": metric["weekly_positive_rate"],
                "weekly_ge_1pct_rate": metric["weekly_ge_1pct_rate"],
                "best_week_return": metric["best_week_return"],
                "worst_week_return": metric["worst_week_return"],
                "max_positions_seen": metric["max_positions_seen"],
                "max_open_risk_fraction": metric["max_open_risk_fraction"],
                "max_open_notional_fraction": metric["max_open_notional_fraction"],
            })
            rejection_rows.append({
                "policy_id": policy.policy_id,
                "period": period,
                "entry_rejections_total": int(sum(entry_rej.values())),
                "portfolio_rejections_total": int(sum(replay_rej.values())),
            })
            print(
                f"BLIND aggregate {period}: trades={metric['trades']} "
                f"return={metric['total_return']:+.4%} WR={metric['win_rate']:.2%} "
                f"DD={metric['max_drawdown']:.2%}",
                flush=True,
            )

    frame = pd.DataFrame(aggregate_rows)
    frame.to_csv(OUT / "blind_aggregate_performance.csv", index=False)
    pd.DataFrame(rejection_rows).to_csv(OUT / "blind_aggregate_rejections.csv", index=False)

    summary = {
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["dataset_revision"],
        "hidden_bounds": manifest["hidden_bounds"],
        "policy_file_sha256": sha256(frozen_path),
        "policies": [{"policy_id": p.policy_id, "description": p.description} for p in policies],
        "periods": PERIOD_YEARS,
        "aggregate_performance": [safe(row) for row in aggregate_rows],
        "blindness_rule": "No hidden trade ledger, per-stock table, timestamp, exit-reason table, or feature analysis is written.",
        "selection_rule": "The policy definitions were frozen before the hidden evaluator ran and are not modified by this script.",
        "2026_status": f"Year-to-date through {manifest['hidden_bounds']['max']}",
    }
    (OUT / "blind_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    forbidden_outputs = ["trade", "ledger", "stock", "instrument", "entry_time", "exit_time", "exit_reason"]
    written = [path.name.lower() for path in OUT.iterdir() if path.is_file()]
    audit_rows = [
        {"test": "frozen_policy_hash_recorded", "pass": True, "details": summary["policy_file_sha256"]},
        {"test": "hidden_data_starts_2024", "pass": str(manifest["hidden_bounds"]["min"]) >= "2024-01-01", "details": manifest["hidden_bounds"]},
        {"test": "2026_data_present", "pass": str(manifest["hidden_bounds"]["max"]) >= "2026-01-01", "details": manifest["hidden_bounds"]["max"]},
        {"test": "no_hidden_trade_level_output", "pass": not any(any(token in name for token in forbidden_outputs) for name in written), "details": written},
        {"test": "aggregate_only_columns", "pass": not any(col in frame.columns for col in ["market", "entry_time", "exit_time", "exit_reason", "hot_rank"]), "details": list(frame.columns)},
        {"test": "mechanical_risk_limits", "pass": bool((frame.max_open_risk_fraction <= 0.0600001).all() and (frame.max_open_notional_fraction <= 1.000001).all()), "details": {"max_risk": float(frame.max_open_risk_fraction.max()), "max_notional": float(frame.max_open_notional_fraction.max())}},
    ]
    pd.DataFrame(audit_rows).to_csv(OUT / "blind_audit.csv", index=False)
    if not all(bool(row["pass"]) for row in audit_rows):
        raise RuntimeError(f"Blind audit failed: {audit_rows}")

    shutil.rmtree(INTERNAL, ignore_errors=True)
    print(json.dumps(safe({
        "hidden_bounds": manifest["hidden_bounds"],
        "policies": summary["policies"],
        "audit_pass": True,
        "aggregate_rows": len(aggregate_rows),
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
