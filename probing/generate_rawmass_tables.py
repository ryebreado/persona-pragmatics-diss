#!/usr/bin/env python3
"""Generate tables 1–4 from raw attention mass (not content-word ratios)."""

import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

ATTN_DIR = Path("probing/results/qwen3_8b_run_04/attention")
OUT_DIR = Path("probing/results/attention_analysis/tables")

PERSONAS = ["baseline", "anti_gricean", "literal_thinker", "helpful_teacher", "pragmaticist"]
PERSONA_DISPLAY = {
    "baseline": "Baseline", "anti_gricean": "Anti-Gricean",
    "literal_thinker": "Literal Thinker", "helpful_teacher": "Helpful Teacher",
    "pragmaticist": "Pragmaticist",
}
CATEGORIES = [
    "true-conj", "true-quant", "false-conj", "false-quant",
    "underinf-conj", "underinf-quant",
]
DIRECTIONS = {
    "l2o": "last_to_outcome",
    "l2s": "last_to_statement",
    "s2o": "statement_to_outcome",
}
WINDOWS = {
    "early": list(range(0, 16)),
    "middle": list(range(19, 26)),
    "late": list(range(30, 36)),
}


def load_all():
    persona_data = {}
    for p in PERSONAS:
        with open(ATTN_DIR / f"attention_{p}.json") as f:
            persona_data[p] = json.load(f)
        print(f"Loaded {p}: {len(persona_data[p])} examples")
    return persona_data


def gather_raw(persona_data, direction_key):
    """category -> persona -> layer -> [head-averaged values across examples]."""
    field = DIRECTIONS[direction_key]
    n_layers = persona_data["baseline"][0]["n_layers"]
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for persona in PERSONAS:
        for ex in persona_data[persona]:
            cat = ex["category"]
            attn = np.array(ex[field])  # (n_layers, n_heads)
            for l in range(n_layers):
                out[cat][persona][l].append(np.mean(attn[l]))
    return out


def layer_mean(gathered, cat, persona, layer):
    vals = gathered[cat][persona].get(layer, [])
    return np.mean(vals) if vals else np.nan


def window_mean(gathered, cat, persona, window_layers):
    vals = []
    for l in window_layers:
        vals.extend(gathered[cat][persona].get(l, []))
    return np.mean(vals) if vals else np.nan


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def table1(persona_data):
    for dk in DIRECTIONS:
        gathered = gather_raw(persona_data, dk)
        rows = []
        for wname, wlayers in WINDOWS.items():
            for persona in PERSONAS:
                for cat in CATEGORIES:
                    m = window_mean(gathered, cat, persona, wlayers)
                    rows.append({
                        "window": wname,
                        "persona": PERSONA_DISPLAY[persona],
                        "category": cat,
                        "mean_attention": f"{m:.6f}" if not np.isnan(m) else "",
                    })
        write_csv(OUT_DIR / f"rawmass_table1_windowed_means_{dk}.csv",
                  ["window", "persona", "category", "mean_attention"], rows)


def table2(persona_data):
    n_layers = persona_data["baseline"][0]["n_layers"]
    layers = list(range(n_layers))
    for dk in DIRECTIONS:
        gathered = gather_raw(persona_data, dk)
        rows = []
        for persona in PERSONAS:
            for cat in CATEGORIES:
                best_layer, best_val = None, -1
                for l in layers:
                    m = layer_mean(gathered, cat, persona, l)
                    if not np.isnan(m) and m > best_val:
                        best_val = m
                        best_layer = l
                rows.append({
                    "persona": PERSONA_DISPLAY[persona],
                    "category": cat,
                    "peak_layer": best_layer if best_layer is not None else "",
                    "peak_magnitude": f"{best_val:.6f}" if best_layer is not None else "",
                })
        write_csv(OUT_DIR / f"rawmass_table2_peak_layers_{dk}.csv",
                  ["persona", "category", "peak_layer", "peak_magnitude"], rows)


def table3(persona_data):
    for dk in DIRECTIONS:
        gathered = gather_raw(persona_data, dk)
        rows = []
        for persona in PERSONAS:
            for wname, wlayers in WINDOWS.items():
                cat_means = []
                for cat in CATEGORIES:
                    m = window_mean(gathered, cat, persona, wlayers)
                    cat_means.append((cat, m))
                cat_means.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 0)
                row = {"persona": PERSONA_DISPLAY[persona], "window": wname}
                for rank, (cat, m) in enumerate(cat_means, 1):
                    row[f"rank_{rank}_category"] = cat
                    row[f"rank_{rank}_value"] = f"{m:.6f}" if not np.isnan(m) else ""
                rows.append(row)
        fieldnames = ["persona", "window"]
        for r in range(1, 7):
            fieldnames.extend([f"rank_{r}_category", f"rank_{r}_value"])
        write_csv(OUT_DIR / f"rawmass_table3_category_rankings_{dk}.csv", fieldnames, rows)


def table4(persona_data):
    for dk in DIRECTIONS:
        gathered = gather_raw(persona_data, dk)
        rows = []
        for cat in CATEGORIES:
            for wname, wlayers in WINDOWS.items():
                bl = window_mean(gathered, cat, "baseline", wlayers)
                row = {
                    "category": cat,
                    "window": wname,
                    "baseline": f"{bl:.6f}" if not np.isnan(bl) else "",
                }
                for persona in PERSONAS:
                    if persona == "baseline":
                        continue
                    pm = window_mean(gathered, cat, persona, wlayers)
                    diff = pm - bl if not (np.isnan(pm) or np.isnan(bl)) else np.nan
                    row[PERSONA_DISPLAY[persona]] = f"{diff:+.6f}" if not np.isnan(diff) else ""
                rows.append(row)
        fieldnames = ["category", "window", "baseline"] + [
            PERSONA_DISPLAY[p] for p in PERSONAS if p != "baseline"
        ]
        write_csv(OUT_DIR / f"rawmass_table4_persona_effects_{dk}.csv", fieldnames, rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    persona_data = load_all()
    print("\n=== Table 1: Windowed means ===")
    table1(persona_data)
    print("\n=== Table 2: Peak layers ===")
    table2(persona_data)
    print("\n=== Table 3: Category rankings ===")
    table3(persona_data)
    print("\n=== Table 4: Persona effects ===")
    table4(persona_data)
    print("\nDone.")


if __name__ == "__main__":
    main()
