#!/usr/bin/env python3
"""
Bar charts of mean transfer probe accuracy per persona, one image per layer window.
Uses the same color scheme as the transfer layer curve plot (matplotlib tab10 order).

Usage:
    # Generate all windows
    uv run python probing/plot_transfer_windows.py probing/results/qwen3_8b_run_04/probes/transfer_last_token.json

    # Specific windows only
    uv run python probing/plot_transfer_windows.py probing/results/qwen3_8b_run_04/probes/transfer_last_token.json \
        --windows 0-15 19-25
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DISPLAY_NAMES = {
    "baseline_cv": "Baseline (CV)",
    "baseline→anti_gricean": "Anti-Gricean",
    "baseline→literal_thinker": "Literal Thinker",
    "baseline→helpful_teacher": "Helpful Teacher",
    "baseline→pragmaticist": "Pragmaticist",
}

# Fixed colors matching persona_comparison_last_token_all.png (tab10 cycle order):
# baseline=C0(blue), literal_thinker=C1(orange), helpful_teacher=C2(green),
# anti_gricean=C3(red), pragmaticist=C4(purple)
_TAB10 = plt.cm.tab10
CONDITION_COLORS = {
    "baseline_cv": "#999999",
    "baseline→literal_thinker": _TAB10(1),
    "baseline→helpful_teacher": _TAB10(2),
    "baseline→anti_gricean": _TAB10(3),
    "baseline→pragmaticist": _TAB10(4),
}

# Canonical display order
CONDITION_ORDER = [
    "baseline_cv",
    "baseline→literal_thinker",
    "baseline→helpful_teacher",
    "baseline→anti_gricean",
    "baseline→pragmaticist",
]

DEFAULT_WINDOWS = [(0, 15), (16, 18), (19, 25), (26, 29), (30, 35)]


def load_transfer_results(path):
    with open(path) as f:
        return json.load(f)


def accs_in_window(experiment, lo, hi):
    """Per-layer accuracies for layers in [lo, hi] inclusive."""
    return [l["accuracy"] for l in experiment["layers"] if lo <= l["layer"] <= hi]


def mean_acc_in_window(experiment, lo, hi):
    """Mean accuracy for layers in [lo, hi] inclusive."""
    accs = accs_in_window(experiment, lo, hi)
    return np.mean(accs) if accs else float("nan")


def plot_single_window(results, lo, hi, output_path):
    """One bar chart for a single layer window, bars colored per condition."""
    # Sort results into canonical order
    by_name = {r["name"]: r for r in results}
    ordered = [by_name[k] for k in CONDITION_ORDER if k in by_name]

    names = [DISPLAY_NAMES.get(r["name"], r["name"]) for r in ordered]
    # Use fold-level window means for both the bar height and error bars
    fold_data = [fold_window_means(r, lo, hi) for r in ordered]
    means = [np.mean(f) if f is not None else mean_acc_in_window(r, lo, hi)
             for f, r in zip(fold_data, ordered)]
    sds = [np.std(f, ddof=1) if f is not None else 0.0 for f in fold_data]
    colors = [CONDITION_COLORS.get(r["name"], "#999999") for r in ordered]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=sds, color=colors, width=0.6,
                  capsize=4, error_kw={"linewidth": 1.2}, zorder=3)

    ax.set_ylabel("Mean Transfer Accuracy")
    ax.set_title(f"Layers {lo}\u2013{hi}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance", zorder=4)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, sd in zip(bars, sds):
        label_y = bar.get_height() + sd + 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def fold_window_means(experiment, lo, hi):
    """Average accuracy across layers in window, separately per CV fold.

    For each fold k, computes mean(acc_layer_i_fold_k) for layers in [lo, hi].
    Returns array of length n_folds.
    """
    layers_in_window = [l for l in experiment["layers"] if lo <= l["layer"] <= hi]
    if not layers_in_window or layers_in_window[0].get("fold_scores") is None:
        return None

    n_folds = len(layers_in_window[0]["fold_scores"])
    # fold_scores[k] = list of per-layer accuracies for fold k
    per_fold = []
    for k in range(n_folds):
        fold_accs = [l["fold_scores"][k] for l in layers_in_window]
        per_fold.append(np.mean(fold_accs))
    return np.array(per_fold)


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def run_ttest_vs_chance(results, windows):
    """One-sample t-tests vs chance (0.50).

    Unit of observation = mean accuracy across layers in window for each CV fold,
    giving n = n_folds (typically 5) per condition per window.
    """
    baseline = [r for r in results if r["name"] == "baseline_cv"]
    transfers = [r for r in results if r["name"] != "baseline_cv"]
    ordered = baseline + transfers

    rows = []
    for lo, hi in windows:
        for r in ordered:
            name = DISPLAY_NAMES.get(r["name"], r["name"])
            fold_means = fold_window_means(r, lo, hi)
            if fold_means is None:
                print(f"  Warning: no fold_scores for {r['name']} — skipping")
                continue
            n = len(fold_means)
            m = np.mean(fold_means)
            sd = np.std(fold_means, ddof=1)
            t, p = stats.ttest_1samp(fold_means, 0.5)
            rows.append({
                "window": f"L{lo}-{hi}",
                "condition": name,
                "n": n,
                "mean": m,
                "sd": sd,
                "t": t,
                "df": n - 1,
                "p": p,
                "sig": sig_stars(p),
            })

    # Print table
    print("\n" + "=" * 90)
    print("One-sample t-tests vs chance (0.50)")
    print("Unit of observation: fold-level mean across layers in window (n = 5 folds)")
    print("=" * 90)
    header = f"{'Window':<10} {'Condition':<20} {'M':>5} {'SD':>6} {'n':>3} {'df':>3} {'t':>7} {'p':>9} {'':>4}"
    print(header)
    print("-" * 90)
    prev_window = None
    for row in rows:
        if row["window"] != prev_window:
            if prev_window is not None:
                print()
            prev_window = row["window"]
        print(f"{row['window']:<10} {row['condition']:<20} {row['mean']:>5.2f} {row['sd']:>6.3f} {row['n']:>3d} "
              f"{row['df']:>3d} {row['t']:>7.2f} {row['p']:>9.4f} {row['sig']:>4}")
    print("=" * 90)
    print("* p<.05  ** p<.01  *** p<.001\n")

    return rows


def parse_window(s):
    """Parse '0-15' into (0, 15)."""
    lo, hi = s.split("-")
    return int(lo), int(hi)


def main():
    parser = argparse.ArgumentParser(description="Plot transfer probe accuracy per layer window")
    parser.add_argument("json_path", help="Path to transfer_last_token.json")
    parser.add_argument("--windows", nargs="+", default=None,
                        metavar="LO-HI",
                        help="Windows to plot, e.g. 0-15 19-25 (default: all five)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: alongside JSON)")
    args = parser.parse_args()

    results = load_transfer_results(args.json_path)
    print(f"Loaded {len(results)} conditions")
    for r in results:
        print(f"  {r['name']}: {len(r['layers'])} layers")

    windows = [parse_window(w) for w in args.windows] if args.windows else DEFAULT_WINDOWS

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.json_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for lo, hi in windows:
        out_path = out_dir / f"transfer_bars_L{lo}-{hi}.png"
        plot_single_window(results, lo, hi, str(out_path))

    run_ttest_vs_chance(results, windows)


if __name__ == "__main__":
    main()
