#!/usr/bin/env python3
"""
Connected slope charts of transfer probe accuracy across layer windows.

Version A: motivated (non-uniform) intervals
Version B: uniform 6-layer intervals

Usage:
    uv run python probing/plot_transfer_slopes.py probing/results/qwen3_8b_run_04/probes/transfer_last_token.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Condition metadata (matches existing codebase) ──────────────────────────

DISPLAY_NAMES = {
    "baseline_cv": "Baseline (CV)",
    "baseline→literal_thinker": "Literal Thinker",
    "baseline→helpful_teacher": "Helpful Teacher",
    "baseline→anti_gricean": "Anti-Gricean",
    "baseline→pragmaticist": "Pragmaticist",
}

_TAB10 = plt.cm.tab10
CONDITION_COLORS = {
    "baseline_cv": _TAB10(0),
    "baseline→literal_thinker": _TAB10(1),
    "baseline→helpful_teacher": _TAB10(2),
    "baseline→anti_gricean": _TAB10(3),
    "baseline→pragmaticist": _TAB10(4),
}

CONDITION_ORDER = [
    "baseline_cv",
    "baseline→literal_thinker",
    "baseline→helpful_teacher",
    "baseline→anti_gricean",
    "baseline→pragmaticist",
]

# ── Window definitions ───────────────────────────────────────────────────────

WINDOWS_A = [(0, 15), (16, 18), (19, 25), (26, 29), (30, 35)]
WINDOWS_B = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 35)]


# ── Helpers ──────────────────────────────────────────────────────────────────

def fold_window_means(experiment, lo, hi):
    """Mean accuracy across layers in [lo, hi], separately per CV fold."""
    layers_in_window = [l for l in experiment["layers"] if lo <= l["layer"] <= hi]
    if not layers_in_window or layers_in_window[0].get("fold_scores") is None:
        return None
    n_folds = len(layers_in_window[0]["fold_scores"])
    per_fold = []
    for k in range(n_folds):
        fold_accs = [l["fold_scores"][k] for l in layers_in_window]
        per_fold.append(np.mean(fold_accs))
    return np.array(per_fold)


def make_slope_chart(results, windows, x_positions, x_labels, title, output_path):
    """Connected dot plot with ±1 SD error bars."""
    by_name = {r["name"]: r for r in results}

    fig, ax = plt.subplots(figsize=(10, 3))

    for cond_key in CONDITION_ORDER:
        if cond_key not in by_name:
            continue
        exp = by_name[cond_key]
        means, sds = [], []
        for lo, hi in windows:
            folds = fold_window_means(exp, lo, hi)
            if folds is not None:
                means.append(np.mean(folds))
                sds.append(np.std(folds, ddof=1))
            else:
                means.append(float("nan"))
                sds.append(0.0)

        color = CONDITION_COLORS.get(cond_key, "#999999")
        label = DISPLAY_NAMES.get(cond_key, cond_key)

        ax.errorbar(
            x_positions, means, yerr=sds,
            fmt="o-", color=color, label=label,
            markersize=5, linewidth=1.5,
            capsize=3, capthick=0.8, elinewidth=0.8,
            zorder=3,
        )

    # Chance line
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Layer window")
    ax.set_ylabel("Mean Transfer Accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transfer probe slope charts")
    parser.add_argument("json_path", help="Path to transfer_last_token.json")
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    args = parser.parse_args()

    with open(args.json_path) as f:
        results = json.load(f)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.json_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Version A: motivated intervals, non-uniform spacing at midpoints
    x_a = [(lo + hi) / 2 for lo, hi in WINDOWS_A]
    labels_a = [f"{lo}\u2013{hi}" for lo, hi in WINDOWS_A]
    make_slope_chart(
        results, WINDOWS_A, x_a, labels_a,
        "Transfer Probe Accuracy (Motivated Windows)",
        str(out_dir / "transfer_slopes_motivated.png"),
    )

    # Version B: uniform intervals, evenly spaced
    x_b = list(range(len(WINDOWS_B)))
    labels_b = [f"{lo}\u2013{hi}" for lo, hi in WINDOWS_B]
    make_slope_chart(
        results, WINDOWS_B, x_b, labels_b,
        "Transfer Probe Accuracy (Uniform Windows)",
        str(out_dir / "transfer_slopes_uniform.png"),
    )


if __name__ == "__main__":
    main()
