from pathlib import Path

path = Path("backtest_engine.py")
text = path.read_text(encoding="utf-8")

old = """    low_breached = False\n    fallback_armed = False\n    upper80 = ref_low + 0.80 * (high_level - ref_low)\n"""
new = """    low_breached = False\n    fallback_armed = False\n    reclaim_break_level = None\n    reclaim_break_stop = None\n    upper80 = ref_low + 0.80 * (high_level - ref_low)\n"""
if old not in text:
    raise RuntimeError("Could not locate entry state initialization")
text = text.replace(old, new, 1)

old = """        if mode in {\n            \"exact_dual_next_open\",\n            \"exact_dual_reclaim_close\",\n            \"dual_keep_both\",\n        }:\n"""
new = """        if mode in {\n            \"exact_dual_next_open\",\n            \"exact_dual_reclaim_close\",\n            \"exact_dual_reclaim_high_break\",\n            \"dual_keep_both\",\n        }:\n"""
if old not in text:
    raise RuntimeError("Could not locate exact dual mode set")
text = text.replace(old, new, 1)

old = """            if low_breached:\n                if mode == \"dual_keep_both\" and breakout:\n                    return EntrySignal(\n                        \"breakout_after_breach\",\n                        i,\n                        i,\n                        _fill_breakout(mkt, i, high_level),\n                        ref_low,\n                        high_level,\n                    )\n                if strict_reclaim:\n                    if mode == \"exact_dual_reclaim_close\":\n                        return EntrySignal(\n                            \"strict_reclaim_close\",\n                            i,\n                            i,\n                            float(c),\n                            float(l),\n                            high_level,\n                        )\n                    if i + 1 <= expiry and i + 1 <= period_end:\n                        return EntrySignal(\n                            \"strict_reclaim_next_open\",\n                            i,\n                            i + 1,\n                            float(mkt.open[i + 1]),\n                            float(l),\n                            high_level,\n                        )\n            continue\n"""
new = """            if low_breached:\n                if (\n                    mode == \"exact_dual_reclaim_high_break\"\n                    and reclaim_break_level is not None\n                ):\n                    reclaim_breakout = h > reclaim_break_level or o > reclaim_break_level\n                    if reclaim_breakout:\n                        return EntrySignal(\n                            \"strict_reclaim_high_break\",\n                            i,\n                            i,\n                            _fill_breakout(mkt, i, reclaim_break_level),\n                            float(reclaim_break_stop),\n                            high_level,\n                            float(reclaim_break_level),\n                        )\n                    continue\n                if mode == \"dual_keep_both\" and breakout:\n                    return EntrySignal(\n                        \"breakout_after_breach\",\n                        i,\n                        i,\n                        _fill_breakout(mkt, i, high_level),\n                        ref_low,\n                        high_level,\n                    )\n                if strict_reclaim:\n                    if mode == \"exact_dual_reclaim_close\":\n                        return EntrySignal(\n                            \"strict_reclaim_close\",\n                            i,\n                            i,\n                            float(c),\n                            float(l),\n                            high_level,\n                        )\n                    if mode == \"exact_dual_reclaim_high_break\":\n                        reclaim_break_level = float(h)\n                        reclaim_break_stop = float(l)\n                        continue\n                    if i + 1 <= expiry and i + 1 <= period_end:\n                        return EntrySignal(\n                            \"strict_reclaim_next_open\",\n                            i,\n                            i + 1,\n                            float(mkt.open[i + 1]),\n                            float(l),\n                            high_level,\n                        )\n            continue\n"""
if old not in text:
    raise RuntimeError("Could not locate exact dual entry block")
text = text.replace(old, new, 1)

old = """        \"exact_dual_next_open\",\n        \"exact_dual_reclaim_close\",\n        \"dual_keep_both\",\n"""
new = """        \"exact_dual_next_open\",\n        \"exact_dual_reclaim_close\",\n        \"exact_dual_reclaim_high_break\",\n        \"dual_keep_both\",\n"""
if old not in text:
    raise RuntimeError("Could not locate entry mode catalog")
text = text.replace(old, new, 1)

old = """    rows.append(\n        {\n            \"test\": \"strict_reclaim_after_breach\",\n            \"pass\": bool(\n                signal2\n                and \"reclaim\" in signal2.route\n                and signal2.entry_idx == 3\n                and signal2.structural_stop == 98.0\n            ),\n            \"detail\": repr(signal2),\n        }\n    )\n\n    df3 = df.copy()\n"""
new = """    rows.append(\n        {\n            \"test\": \"strict_reclaim_after_breach\",\n            \"pass\": bool(\n                signal2\n                and \"reclaim\" in signal2.route\n                and signal2.entry_idx == 3\n                and signal2.structural_stop == 98.0\n            ),\n            \"detail\": repr(signal2),\n        }\n    )\n\n    signal2_high = find_entry(\n        market2,\n        event,\n        replace(config, entry_mode=\"exact_dual_reclaim_high_break\"),\n        9,\n    )\n    rows.append(\n        {\n            \"test\": \"strict_reclaim_high_break\",\n            \"pass\": bool(\n                signal2_high\n                and signal2_high.route == \"strict_reclaim_high_break\"\n                and signal2_high.entry_idx == 3\n                and signal2_high.structural_stop == 98.0\n                and signal2_high.level == 100.5\n            ),\n            \"detail\": repr(signal2_high),\n        }\n    )\n\n    df3 = df.copy()\n"""
if old not in text:
    raise RuntimeError("Could not locate reclaim synthetic test")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("Added exact_dual_reclaim_high_break and its synthetic test")
