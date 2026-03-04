#!/usr/bin/env python3
"""
Generate all attention analysis tables (CSVs) and dissertation figures from
the full-layer content-word attention data.

Usage:
    uv run python probing/generate_attention_tables_figures.py
    uv run python probing/generate_attention_tables_figures.py --input path/to/data.json
    uv run python probing/generate_attention_tables_figures.py --output-dir probing/results/attention_analysis
"""

import json
import argparse
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Constants ──────────────────────────────────────────────────────────────

PERSONAS = ["baseline", "anti_gricean", "literal_thinker", "helpful_teacher", "pragmaticist"]
PERSONA_DISPLAY = {
    "baseline": "Baseline",
    "anti_gricean": "Anti-Gricean",
    "literal_thinker": "Literal Thinker",
    "helpful_teacher": "Helpful Teacher",
    "pragmaticist": "Pragmaticist",
}
CATEGORIES = [
    "true-conj", "true-quant", "false-conj", "false-quant",
    "underinf-conj", "underinf-quant",
]
CONJ_CATS = ["true-conj", "false-conj", "underinf-conj"]
QUANT_CATS = ["true-quant", "false-quant", "underinf-quant"]
UNDERINF_CATS = ["underinf-conj", "underinf-quant"]

# Attention directions: raw and article-excluded
RAW_DIRECTIONS = ["l2o", "l2s", "s2o"]
NOSINK_DIRECTIONS = ["l2o_nosink", "l2s_nosink", "s2o_nosink"]

DIRECTION_LABELS = {
    "l2o": "L→O", "l2s": "L→S", "s2o": "S→O",
    "l2o_nosink": "L→O", "l2s_nosink": "L→S", "s2o_nosink": "S→O",
}
DIRECTION_FULL = {
    "l2o": "Last → Outcome", "l2s": "Last → Statement", "s2o": "Statement → Outcome",
    "l2o_nosink": "Last → Outcome (no articles)",
    "l2s_nosink": "Last → Statement (no articles)",
    "s2o_nosink": "Statement → Outcome (no articles)",
}

# Windows
WINDOWS = {
    "early": list(range(0, 16)),    # 0–15
    "middle": list(range(19, 26)),  # 19–25
    "late": list(range(30, 36)),    # 30–35
}

# Colors
PERSONA_COLORS = {
    "baseline": "#2196F3",
    "anti_gricean": "#F44336",
    "literal_thinker": "#FF9800",
    "helpful_teacher": "#4CAF50",
    "pragmaticist": "#9C27B0",
}
CATEGORY_COLORS = {
    "true-conj": "#228B22",
    "true-quant": "#90EE90",
    "false-conj": "#B22222",
    "false-quant": "#F08080",
    "underinf-conj": "#1E3A8A",
    "underinf-quant": "#93C5FD",
}

# Per-layer content fractions for normalization (from visualize_attention.py)
CONTENT_FRACTIONS = {
    "baseline": [1.0] * 36,
    "anti_gricean": [0.885, 0.885, 0.817, 0.737, 0.803, 0.841, 0.897, 0.423, 0.382, 0.274, 0.401, 0.48, 0.624, 0.585, 0.727, 0.631, 0.539, 0.649, 0.592, 0.527, 0.431, 0.471, 0.37, 0.496, 0.369, 0.33, 0.263, 0.232, 0.386, 0.209, 0.225, 0.283, 0.33, 0.325, 0.424, 0.432],
    "literal_thinker": [0.899, 0.881, 0.754, 0.591, 0.653, 0.674, 0.867, 0.42, 0.379, 0.287, 0.415, 0.489, 0.604, 0.585, 0.736, 0.638, 0.519, 0.619, 0.555, 0.529, 0.452, 0.484, 0.369, 0.527, 0.376, 0.318, 0.26, 0.226, 0.397, 0.211, 0.223, 0.28, 0.324, 0.308, 0.419, 0.428],
    "helpful_teacher": [0.889, 0.871, 0.741, 0.554, 0.627, 0.639, 0.864, 0.428, 0.374, 0.283, 0.407, 0.507, 0.61, 0.554, 0.703, 0.616, 0.523, 0.615, 0.54, 0.527, 0.486, 0.554, 0.375, 0.555, 0.385, 0.322, 0.29, 0.243, 0.401, 0.22, 0.23, 0.297, 0.335, 0.319, 0.42, 0.428],
    "pragmaticist": [0.882, 0.832, 0.665, 0.444, 0.521, 0.534, 0.822, 0.425, 0.4, 0.306, 0.424, 0.471, 0.601, 0.564, 0.69, 0.635, 0.539, 0.632, 0.555, 0.514, 0.449, 0.497, 0.379, 0.514, 0.39, 0.329, 0.28, 0.233, 0.374, 0.209, 0.224, 0.278, 0.317, 0.301, 0.404, 0.426],
}


# ── Data loading & aggregation ─────────────────────────────────────────────

def load_data(path):
    with open(path) as f:
        return json.load(f)


def gather(data, direction):
    """
    Return nested dict: category → persona → layer_int → [values across examples].
    """
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ex in data["results"]:
        cat = ex["category"]
        for cond, cond_data in ex["conditions"].items():
            for lk, val in cond_data[direction].items():
                layer = int(lk[1:])
                out[cat][cond][layer].append(val)
    return out


def layer_mean(gathered, cat, persona, layer):
    vals = gathered[cat][persona].get(layer, [])
    return np.mean(vals) if vals else np.nan


def layer_std(gathered, cat, persona, layer):
    vals = gathered[cat][persona].get(layer, [])
    return np.std(vals) if vals else np.nan


def layer_sem(gathered, cat, persona, layer):
    vals = gathered[cat][persona].get(layer, [])
    return np.std(vals) / np.sqrt(len(vals)) if vals else np.nan


def window_mean(gathered, cat, persona, window_layers):
    vals = []
    for l in window_layers:
        vals.extend(gathered[cat][persona].get(l, []))
    return np.mean(vals) if vals else np.nan


def all_layers(data):
    return data["meta"]["layers"]


# ── Table generators ───────────────────────────────────────────────────────

def table1_windowed_means(data, directions, out_dir, suffix=""):
    """Table 1: Windowed mean attention by persona × category × direction."""
    layers = all_layers(data)
    for direction in directions:
        gathered = gather(data, direction)
        label = DIRECTION_LABELS[direction]
        rows = []
        for window_name, window_layers in WINDOWS.items():
            for persona in PERSONAS:
                for cat in CATEGORIES:
                    m = window_mean(gathered, cat, persona, window_layers)
                    rows.append({
                        "window": window_name,
                        "persona": PERSONA_DISPLAY[persona],
                        "category": cat,
                        "mean_attention": f"{m:.4f}" if not np.isnan(m) else "",
                    })
        fname = f"table1_windowed_means_{direction}{suffix}.csv"
        path = out_dir / fname
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["window", "persona", "category", "mean_attention"])
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {path}")


def table2_peak_layers(data, directions, out_dir, suffix=""):
    """Table 2: Peak layer and peak magnitude per persona × category × direction."""
    layers = all_layers(data)
    for direction in directions:
        gathered = gather(data, direction)
        rows = []
        for persona in PERSONAS:
            for cat in CATEGORIES:
                best_layer = None
                best_val = -1
                for l in layers:
                    m = layer_mean(gathered, cat, persona, l)
                    if not np.isnan(m) and m > best_val:
                        best_val = m
                        best_layer = l
                rows.append({
                    "persona": PERSONA_DISPLAY[persona],
                    "category": cat,
                    "peak_layer": best_layer if best_layer is not None else "",
                    "peak_magnitude": f"{best_val:.4f}" if best_layer is not None else "",
                })
        fname = f"table2_peak_layers_{direction}{suffix}.csv"
        path = out_dir / fname
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["persona", "category", "peak_layer", "peak_magnitude"])
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {path}")


def table3_category_rankings(data, directions, out_dir, suffix=""):
    """Table 3: Category rankings within each window (highest to lowest)."""
    for direction in directions:
        gathered = gather(data, direction)
        rows = []
        for persona in PERSONAS:
            for window_name, window_layers in WINDOWS.items():
                cat_means = []
                for cat in CATEGORIES:
                    m = window_mean(gathered, cat, persona, window_layers)
                    cat_means.append((cat, m))
                cat_means.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 0)
                row = {
                    "persona": PERSONA_DISPLAY[persona],
                    "window": window_name,
                }
                for rank, (cat, m) in enumerate(cat_means, 1):
                    row[f"rank_{rank}_category"] = cat
                    row[f"rank_{rank}_value"] = f"{m:.4f}" if not np.isnan(m) else ""
                rows.append(row)
        fname = f"table3_category_rankings_{direction}{suffix}.csv"
        path = out_dir / fname
        fieldnames = ["persona", "window"]
        for r in range(1, 7):
            fieldnames.extend([f"rank_{r}_category", f"rank_{r}_value"])
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {path}")


def table4_persona_effects(data, directions, out_dir, suffix=""):
    """Table 4: Persona effect sizes (persona minus baseline) per category × window."""
    for direction in directions:
        gathered = gather(data, direction)
        rows = []
        for cat in CATEGORIES:
            for window_name, window_layers in WINDOWS.items():
                baseline_m = window_mean(gathered, cat, "baseline", window_layers)
                row = {
                    "category": cat,
                    "window": window_name,
                    "baseline": f"{baseline_m:.4f}" if not np.isnan(baseline_m) else "",
                }
                for persona in PERSONAS:
                    if persona == "baseline":
                        continue
                    p_m = window_mean(gathered, cat, persona, window_layers)
                    diff = p_m - baseline_m if not (np.isnan(p_m) or np.isnan(baseline_m)) else np.nan
                    row[PERSONA_DISPLAY[persona]] = f"{diff:+.4f}" if not np.isnan(diff) else ""
                rows.append(row)
        fname = f"table4_persona_effects_{direction}{suffix}.csv"
        path = out_dir / fname
        fieldnames = ["category", "window", "baseline"] + [
            PERSONA_DISPLAY[p] for p in PERSONAS if p != "baseline"
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {path}")


def generate_all_tables(data, out_dir):
    """Generate all 4 tables × 2 versions (raw + article-excluded)."""
    print("\n=== TABLES (raw) ===")
    table1_windowed_means(data, RAW_DIRECTIONS, out_dir, suffix="_raw")
    table2_peak_layers(data, RAW_DIRECTIONS, out_dir, suffix="_raw")
    table3_category_rankings(data, RAW_DIRECTIONS, out_dir, suffix="_raw")
    table4_persona_effects(data, RAW_DIRECTIONS, out_dir, suffix="_raw")

    print("\n=== TABLES (article-excluded) ===")
    table1_windowed_means(data, NOSINK_DIRECTIONS, out_dir, suffix="")
    table2_peak_layers(data, NOSINK_DIRECTIONS, out_dir, suffix="")
    table3_category_rankings(data, NOSINK_DIRECTIONS, out_dir, suffix="")
    table4_persona_effects(data, NOSINK_DIRECTIONS, out_dir, suffix="")


# ── Plotting utilities ─────────────────────────────────────────────────────

def _get_persona_curve(gathered, persona, categories, layers):
    """Get mean ± SD across specified categories at each layer."""
    means = []
    sds = []
    for l in layers:
        vals = []
        for cat in categories:
            vals.extend(gathered[cat][persona].get(l, []))
        means.append(np.mean(vals) if vals else np.nan)
        sds.append(np.std(vals) if vals else np.nan)
    return np.array(means), np.array(sds)


def _get_category_curve(gathered, cat, persona, layers):
    """Get mean ± SEM for a single category at each layer."""
    means = []
    sems = []
    for l in layers:
        vals = gathered[cat][persona].get(l, [])
        means.append(np.mean(vals) if vals else np.nan)
        sems.append(np.std(vals) / np.sqrt(len(vals)) if vals else np.nan)
    return np.array(means), np.array(sems)


def _save(fig, path_base):
    """Save as both PNG and PDF."""
    for ext in [".png", ".pdf"]:
        p = str(path_base) + ext
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path_base}.png/pdf")
    plt.close(fig)


# ── Main text figures ──────────────────────────────────────────────────────

def figure_a(data, out_dir):
    """L→O persona comparison, conj items only."""
    gathered = gather(data, "l2o")
    layers = all_layers(data)

    # Baseline mean ± SD for shading
    bl_mean, bl_sd = _get_persona_curve(gathered, "baseline", CONJ_CATS, layers)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Baseline SD shading
    ax.fill_between(layers, bl_mean - bl_sd, bl_mean + bl_sd,
                    color=PERSONA_COLORS["baseline"], alpha=0.12, label="Baseline ±1 SD")

    for persona in PERSONAS:
        m, _ = _get_persona_curve(gathered, persona, CONJ_CATS, layers)
        lw = 2.5 if persona == "baseline" else 1.8
        ls = "-" if persona in ("baseline", "anti_gricean") else "--"
        ax.plot(layers, m, color=PERSONA_COLORS[persona],
                label=PERSONA_DISPLAY[persona], linewidth=lw, linestyle=ls)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Content-word attention ratio", fontsize=12)
    ax.set_title("Last token → outcome attention by persona (conjunctive items)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0], layers[-1])
    fig.tight_layout()
    _save(fig, out_dir / "figure_a_l2o_persona_conj")


def figure_b(data, out_dir):
    """L→O category comparison, baseline only."""
    gathered = gather(data, "l2o")
    layers = all_layers(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in CATEGORIES:
        m, sem = _get_category_curve(gathered, cat, "baseline", layers)
        ax.plot(layers, m, color=CATEGORY_COLORS[cat], label=cat, linewidth=1.8)
        ax.fill_between(layers, m - sem, m + sem, color=CATEGORY_COLORS[cat], alpha=0.12)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Content-word attention ratio", fontsize=12)
    ax.set_title("Last token → outcome attention by category (baseline)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0], layers[-1])
    fig.tight_layout()
    _save(fig, out_dir / "figure_b_l2o_category_baseline")


def figure_c(data, out_dir):
    """L→S conj vs quant, baseline only, article-excluded."""
    gathered = gather(data, "l2s_nosink")
    layers = all_layers(data)

    conj_mean, conj_sd = _get_persona_curve(gathered, "baseline", CONJ_CATS, layers)
    quant_mean, quant_sd = _get_persona_curve(gathered, "baseline", QUANT_CATS, layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, conj_mean, color="#1E3A8A", label="Conjunctive categories", linewidth=2)
    ax.fill_between(layers, conj_mean - conj_sd, conj_mean + conj_sd, color="#1E3A8A", alpha=0.12)
    ax.plot(layers, quant_mean, color="#F08080", label="Quantifier categories", linewidth=2)
    ax.fill_between(layers, quant_mean - quant_sd, quant_mean + quant_sd, color="#F08080", alpha=0.12)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Content-word attention ratio", fontsize=12)
    ax.set_title("Last token → statement attention: conjunctive vs quantifier\n(baseline, articles excluded)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0], layers[-1])
    fig.tight_layout()
    _save(fig, out_dir / "figure_c_l2s_conj_quant_baseline_nosink")


def figure_d(data, out_dir):
    """S→O persona comparison, underinformative items only."""
    gathered = gather(data, "s2o")
    layers = all_layers(data)

    bl_mean, bl_sd = _get_persona_curve(gathered, "baseline", UNDERINF_CATS, layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(layers, bl_mean - bl_sd, bl_mean + bl_sd,
                    color=PERSONA_COLORS["baseline"], alpha=0.12, label="Baseline ±1 SD")

    for persona in PERSONAS:
        m, _ = _get_persona_curve(gathered, persona, UNDERINF_CATS, layers)
        lw = 2.5 if persona == "baseline" else 1.8
        ls = "-" if persona in ("baseline", "anti_gricean") else "--"
        ax.plot(layers, m, color=PERSONA_COLORS[persona],
                label=PERSONA_DISPLAY[persona], linewidth=lw, linestyle=ls)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Content-word attention ratio", fontsize=12)
    ax.set_title("Statement → outcome attention by persona (underinformative items)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0], layers[-1])
    fig.tight_layout()
    _save(fig, out_dir / "figure_d_s2o_persona_underinf")


def figure_e(data, out_dir):
    """L→O conj vs quant under Anti-Gricean only."""
    gathered = gather(data, "l2o")
    layers = all_layers(data)

    conj_mean, conj_sd = _get_persona_curve(gathered, "anti_gricean", CONJ_CATS, layers)
    quant_mean, quant_sd = _get_persona_curve(gathered, "anti_gricean", QUANT_CATS, layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, conj_mean, color="#1E3A8A", label="Conjunctive categories", linewidth=2)
    ax.fill_between(layers, conj_mean - conj_sd, conj_mean + conj_sd, color="#1E3A8A", alpha=0.12)
    ax.plot(layers, quant_mean, color="#F08080", label="Quantifier categories", linewidth=2)
    ax.fill_between(layers, quant_mean - quant_sd, quant_mean + quant_sd, color="#F08080", alpha=0.12)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Content-word attention ratio", fontsize=12)
    ax.set_title("Last token → outcome: conjunctive vs quantifier (Anti-Gricean)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0], layers[-1])
    fig.tight_layout()
    _save(fig, out_dir / "figure_e_l2o_conj_quant_anti_gricean")


def figure_f(data, out_dir):
    """L→S persona comparison, underinf-quant only, article-excluded."""
    gathered = gather(data, "l2s_nosink")
    layers = all_layers(data)

    bl_mean, bl_sd = _get_category_curve(gathered, "underinf-quant", "baseline", layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(layers, bl_mean - bl_sd, bl_mean + bl_sd,
                    color=PERSONA_COLORS["baseline"], alpha=0.12, label="Baseline ±1 SD")

    for persona in PERSONAS:
        m, _ = _get_category_curve(gathered, "underinf-quant", persona, layers)
        lw = 2.5 if persona == "baseline" else 1.8
        ls = "-" if persona in ("baseline", "anti_gricean") else "--"
        ax.plot(layers, m, color=PERSONA_COLORS[persona],
                label=PERSONA_DISPLAY[persona], linewidth=lw, linestyle=ls)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Content-word attention ratio", fontsize=12)
    ax.set_title("Last token → statement attention by persona\n(underinf-quant, articles excluded)",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0], layers[-1])
    fig.tight_layout()
    _save(fig, out_dir / "figure_f_l2s_persona_underinf_quant_nosink")


# ── Appendix: 5×3 persona comparison grid ─────────────────────────────────

def _appendix_grid(data, directions, out_dir, tag, title_suffix="", normalize=False):
    """
    5 personas (columns) × 3 directions (rows) grid.
    Each panel: 6 category lines across layers.
    """
    layers = all_layers(data)
    n_personas = len(PERSONAS)
    n_dirs = len(directions)

    fig, axes = plt.subplots(n_dirs, n_personas, figsize=(4 * n_personas, 3.2 * n_dirs),
                             sharex=True, sharey="row")

    for row, direction in enumerate(directions):
        gathered = gather(data, direction)
        for col, persona in enumerate(PERSONAS):
            ax = axes[row, col]
            for cat in CATEGORIES:
                m, sem = _get_category_curve(gathered, cat, persona, layers)
                if normalize:
                    fracs = np.array(CONTENT_FRACTIONS.get(persona, [1.0] * 36))
                    fracs = np.clip(fracs, 0.01, None)  # avoid div by zero
                    m = m / fracs
                    sem = sem / fracs
                ax.plot(layers, m, color=CATEGORY_COLORS[cat], label=cat, linewidth=1.3)
                ax.fill_between(layers, m - sem, m + sem,
                                color=CATEGORY_COLORS[cat], alpha=0.08)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(PERSONA_DISPLAY[persona], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(DIRECTION_LABELS[direction], fontsize=9)
            if row == n_dirs - 1:
                ax.set_xlabel("Layer")

    # Single legend
    handles = [Line2D([0], [0], color=CATEGORY_COLORS[c], lw=1.5) for c in CATEGORIES]
    fig.legend(handles, CATEGORIES, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=7)

    fig.suptitle(f"Attention by Persona × Direction{title_suffix}", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / f"appendix_persona_comparison_{tag}")


def generate_appendix_grids(data, out_dir):
    """Generate 5×3 grids: raw, nosink, raw-normalized, nosink-normalized."""
    print("\n=== APPENDIX GRIDS ===")
    _appendix_grid(data, RAW_DIRECTIONS, out_dir, "raw", " (raw)")
    _appendix_grid(data, NOSINK_DIRECTIONS, out_dir, "nosink", " (articles excluded)")
    _appendix_grid(data, RAW_DIRECTIONS, out_dir, "raw_normalized",
                   " (raw, content-normalized)", normalize=True)
    _appendix_grid(data, NOSINK_DIRECTIONS, out_dir, "nosink_normalized",
                   " (articles excluded, content-normalized)", normalize=True)


# ── Notable patterns ───────────────────────────────────────────────────────

def print_notable_patterns(data):
    """Print notable patterns to console."""
    layers = all_layers(data)
    print("\n" + "=" * 80)
    print("NOTABLE PATTERNS")
    print("=" * 80)

    # ── Pattern 1: Anti-Gricean early spike (L→O, conj) ──
    print("\n--- Anti-Gricean early-layer spike (L→O, conj items) ---")
    gathered_l2o = gather(data, "l2o")
    for persona in ["baseline", "anti_gricean"]:
        early_vals = []
        for l in range(0, 10):
            for cat in CONJ_CATS:
                early_vals.extend(gathered_l2o[cat][persona].get(l, []))
        late_vals = []
        for l in range(25, 36):
            for cat in CONJ_CATS:
                late_vals.extend(gathered_l2o[cat][persona].get(l, []))
        print(f"  {PERSONA_DISPLAY[persona]:20s} early(0-9)={np.mean(early_vals):.4f}  "
              f"late(25-35)={np.mean(late_vals):.4f}  "
              f"ratio={np.mean(early_vals)/max(np.mean(late_vals),1e-8):.2f}")

    # ── Pattern 2: Category ordering (L→O, baseline) ──
    print("\n--- Category ordering (L→O, baseline) ---")
    for window_name, window_layers in WINDOWS.items():
        ranking = []
        for cat in CATEGORIES:
            m = window_mean(gathered_l2o, cat, "baseline", window_layers)
            ranking.append((cat, m))
        ranking.sort(key=lambda x: -x[1])
        order = " > ".join(f"{c}({m:.4f})" for c, m in ranking)
        print(f"  {window_name:8s}: {order}")

    # ── Pattern 3: Persona convergence (S→O, underinf) ──
    print("\n--- Persona convergence (S→O, underinf items) ---")
    gathered_s2o = gather(data, "s2o")
    for window_name, window_layers in WINDOWS.items():
        vals_by_persona = {}
        for persona in PERSONAS:
            persona_vals = []
            for cat in UNDERINF_CATS:
                persona_vals.append(window_mean(gathered_s2o, cat, persona, window_layers))
            vals_by_persona[persona] = np.mean(persona_vals)
        spread = max(vals_by_persona.values()) - min(vals_by_persona.values())
        pairs = "  ".join(f"{PERSONA_DISPLAY[p][:5]}={v:.4f}" for p, v in vals_by_persona.items())
        print(f"  {window_name:8s}: spread={spread:.4f}  {pairs}")

    # ── Pattern 4: Conj/quant dissociation (L→S, baseline, nosink) ──
    print("\n--- Conj/quant dissociation (L→S, baseline, articles excluded) ---")
    gathered_l2s = gather(data, "l2s_nosink")
    for window_name, window_layers in WINDOWS.items():
        conj_vals = []
        quant_vals = []
        for l in window_layers:
            for cat in CONJ_CATS:
                conj_vals.extend(gathered_l2s[cat]["baseline"].get(l, []))
            for cat in QUANT_CATS:
                quant_vals.extend(gathered_l2s[cat]["baseline"].get(l, []))
        diff = np.mean(quant_vals) - np.mean(conj_vals)
        print(f"  {window_name:8s}: conj={np.mean(conj_vals):.4f}  quant={np.mean(quant_vals):.4f}  "
              f"diff={diff:+.4f}")

    # ── Pattern 5: Largest persona effect sizes ──
    print("\n--- Largest persona effects (top 10 by |effect|) ---")
    effects = []
    for direction in RAW_DIRECTIONS + NOSINK_DIRECTIONS:
        gathered = gather(data, direction)
        for cat in CATEGORIES:
            for window_name, window_layers in WINDOWS.items():
                bl = window_mean(gathered, cat, "baseline", window_layers)
                for persona in PERSONAS:
                    if persona == "baseline":
                        continue
                    pm = window_mean(gathered, cat, persona, window_layers)
                    diff = pm - bl
                    if not np.isnan(diff):
                        effects.append((abs(diff), diff, DIRECTION_LABELS[direction],
                                        cat, window_name, PERSONA_DISPLAY[persona],
                                        "nosink" in direction))
    effects.sort(key=lambda x: -x[0])
    for _, diff, dir_label, cat, win, persona, is_nosink in effects[:10]:
        sink_tag = " (nosink)" if is_nosink else ""
        print(f"  {dir_label}{sink_tag:10s} {cat:18s} {win:8s} {persona:16s} {diff:+.4f}")

    # ── Pattern 6: Consistent category rankings across personas ──
    print("\n--- Cross-persona ranking consistency (L→O, early window) ---")
    for persona in PERSONAS:
        ranking = []
        for cat in CATEGORIES:
            m = window_mean(gathered_l2o, cat, persona, WINDOWS["early"])
            ranking.append((cat, m))
        ranking.sort(key=lambda x: -x[1])
        order = " > ".join(f"{c}" for c, _ in ranking)
        print(f"  {PERSONA_DISPLAY[persona]:16s}: {order}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate attention tables & figures")
    parser.add_argument("--input", "-i",
                        default="probing/results/content_word_attention.json",
                        help="Input JSON from extract_content_word_attention.py")
    parser.add_argument("--output-dir", "-o",
                        default="probing/results/attention_analysis",
                        help="Output directory for CSVs and figures")
    args = parser.parse_args()

    data = load_data(args.input)
    print(f"Loaded {data['meta']['n_examples']} examples, "
          f"{len(data['meta']['layers'])} layers, "
          f"{len(data['meta']['conditions'])} conditions")

    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    appendix_dir = out_dir / "appendix"
    for d in [tables_dir, figures_dir, appendix_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Tables
    generate_all_tables(data, tables_dir)

    # Main text figures
    print("\n=== MAIN TEXT FIGURES ===")
    figure_a(data, figures_dir)
    figure_b(data, figures_dir)
    figure_c(data, figures_dir)
    figure_d(data, figures_dir)
    figure_e(data, figures_dir)
    figure_f(data, figures_dir)

    # Appendix grids
    generate_appendix_grids(data, appendix_dir)

    # Notable patterns
    print_notable_patterns(data)

    print(f"\nDone. All output in: {out_dir}")


if __name__ == "__main__":
    main()
