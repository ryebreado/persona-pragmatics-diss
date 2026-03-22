#!/usr/bin/env python3
"""
Visualize persona prompt variant results for robustness analysis.
Groups variants by persona type and compares accuracy across phrasings.
"""

import json
import argparse
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CATEGORY_ORDER = [
    'true-conj', 'true-quant',
    'false-conj', 'false-quant',
    'underinf-conj', 'underinf-quant',
]

CATEGORY_LABELS = {
    'true-conj': 'T-conj',
    'true-quant': 'T-quant',
    'false-conj': 'F-conj',
    'false-quant': 'F-quant',
    'underinf-conj': 'U-conj',
    'underinf-quant': 'U-quant',
}

# One color per variant
VARIANT_COLORS = {
    'original': '#2563EB',  # blue
    'a': '#DC2626',         # red
    'b': '#16A34A',         # green
    'c': '#D97706',         # amber
}

VARIANT_LABELS = {
    'original': 'Original',
    'a': 'Variant A',
    'b': 'Variant B',
    'c': 'Variant C',
}

# Persona display names
PERSONA_FAMILIES = ['anti_gricean', 'literal_thinker', 'pragmaticist', 'helpful_teacher']
PERSONA_DISPLAY = {
    'anti_gricean': 'Anti-Gricean',
    'literal_thinker': 'Literal Thinker',
    'pragmaticist': 'Pragmaticist',
    'helpful_teacher': 'Helpful Teacher',
}


def load_and_group_results(results_dir):
    """
    Load results and group by persona family + variant.
    Returns: {family: {variant: data}}, baseline_data
    """
    grouped = defaultdict(dict)
    baseline = None

    for filepath in sorted(glob(os.path.join(results_dir, "*.json"))):
        basename = os.path.basename(filepath)
        if os.path.isdir(filepath):
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        if '_baseline_' in basename:
            baseline = data
            continue

        persona_file = data.get('persona_file', '')
        persona_name = os.path.basename(persona_file) if persona_file else ''

        if not persona_name:
            continue

        # Determine family and variant
        # Variant files end with _a, _b, _c; originals are just the base name
        family = None
        variant = None
        for fam in PERSONA_FAMILIES:
            if persona_name == fam:
                family = fam
                variant = 'original'
                break
            elif persona_name.startswith(fam + '_'):
                suffix = persona_name[len(fam) + 1:]
                if suffix in ('a', 'b', 'c'):
                    family = fam
                    variant = suffix
                    break

        if family and variant:
            # If there's already an entry for this family+variant, keep the newer one
            # (later timestamp in filename = more recent run)
            grouped[family][variant] = data

    return grouped, baseline


def create_variant_comparison(results_dir, output_path=None, figsize=None,
                              categories=None):
    """
    Create grouped bar chart comparing variants within each persona family.
    """
    grouped, baseline = load_and_group_results(results_dir)

    if not grouped:
        print(f"No grouped variant results found in {results_dir}")
        return

    cats = categories or CATEGORY_ORDER
    cat_labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    families = [f for f in PERSONA_FAMILIES if f in grouped]
    n_families = len(families)
    variants = ['original', 'a', 'b', 'c']
    n_variants = len(variants)

    if figsize is None:
        figsize = (4 * n_families, 5)

    fig, axes = plt.subplots(1, n_families, figsize=figsize, sharey=True)
    if n_families == 1:
        axes = [axes]

    bar_width = 0.18
    offsets = np.arange(n_variants) - (n_variants - 1) / 2
    offsets = offsets * bar_width

    for ax, family in zip(axes, families):
        x = np.arange(len(cats))

        for i, variant in enumerate(variants):
            data = grouped[family].get(variant)
            if data is None:
                continue
            acc = data.get('accuracy_by_category', {})
            values = [acc.get(c, 0) for c in cats]
            ax.bar(x + offsets[i], values, bar_width * 0.9,
                   color=VARIANT_COLORS[variant], alpha=0.85,
                   label=VARIANT_LABELS[variant] if family == families[0] else None)

        # Baseline reference line per category
        if baseline:
            bl_acc = baseline.get('accuracy_by_category', {})
            for j, c in enumerate(cats):
                if c in bl_acc:
                    ax.hlines(bl_acc[c], j - 0.4, j + 0.4,
                              colors='black', linestyles='--', linewidth=0.8,
                              alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
        ax.set_title(PERSONA_DISPLAY.get(family, family), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.09)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Accuracy', fontsize=10)

    # Legend
    handles = [mpatches.Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
               for v in variants]
    handles.append(plt.Line2D([0], [0], color='black', linestyle='--',
                              linewidth=0.8, alpha=0.5, label='Baseline'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 0.98),
               fontsize=9)

    model_name = baseline.get('model', 'Unknown') if baseline else 'Unknown'
    fig.suptitle(f'Prompt Variant Robustness — {model_name}',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close()


def create_underinf_summary(results_dir, output_path=None):
    """
    Focused view: just underinf categories with mean + range across variants.
    One bar per persona family, error bars showing variant spread.
    """
    grouped, baseline = load_and_group_results(results_dir)
    if not grouped:
        print(f"No results found in {results_dir}")
        return

    families = [f for f in PERSONA_FAMILIES if f in grouped]
    cats = ['underinf-conj', 'underinf-quant']
    cat_labels = ['U-conj', 'U-quant']

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True)

    family_colors = {
        'anti_gricean': '#7C3AED',
        'literal_thinker': '#0891B2',
        'pragmaticist': '#059669',
        'helpful_teacher': '#EA580C',
    }

    bar_width = 0.15
    x = np.arange(len(families))

    for ax, cat, cat_label in zip(axes, cats, cat_labels):
        for i, family in enumerate(families):
            variant_accs = []
            for v in ['original', 'a', 'b', 'c']:
                data = grouped[family].get(v)
                if data:
                    acc = data.get('accuracy_by_category', {}).get(cat)
                    if acc is not None:
                        variant_accs.append(acc)

            if not variant_accs:
                continue

            mean_acc = np.mean(variant_accs)
            min_acc = min(variant_accs)
            max_acc = max(variant_accs)

            # Bar with error bars showing range
            ax.bar(i, mean_acc, 0.6, color=family_colors.get(family, '#888'),
                   alpha=0.8, edgecolor='white')
            ax.errorbar(i, mean_acc,
                        yerr=[[mean_acc - min_acc], [max_acc - mean_acc]],
                        fmt='none', ecolor='black', capsize=5, capthick=1.2)

            # Individual variant dots
            for acc in variant_accs:
                ax.scatter(i, acc, color='black', s=20, zorder=5, alpha=0.6)

        # Baseline reference
        if baseline:
            bl_val = baseline.get('accuracy_by_category', {}).get(cat)
            if bl_val is not None:
                ax.axhline(bl_val, color='black', linestyle='--',
                           linewidth=1, alpha=0.5, label='Baseline' if cat == cats[0] else None)

        ax.set_xticks(x)
        ax.set_xticklabels([PERSONA_DISPLAY.get(f, f) for f in families],
                           rotation=30, ha='right', fontsize=9)
        ax.set_title(cat_label, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.09)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Accuracy', fontsize=10)
    axes[0].legend(fontsize=9, loc='lower left')

    model_name = baseline.get('model', 'Unknown') if baseline else 'Unknown'
    fig.suptitle(f'Underinformativeness Detection Across Prompt Variants — {model_name}',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize prompt variant robustness across persona types',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('results_dir', help='Directory containing result JSON files')
    parser.add_argument('--output', '-o', help='Output file path (PNG)')
    parser.add_argument('--mode', choices=['full', 'underinf'], default='full',
                        help='"full" shows all categories; "underinf" shows summary with mean+range')
    parser.add_argument('--width', type=float, help='Figure width in inches')
    parser.add_argument('--height', type=float, help='Figure height in inches')
    parser.add_argument('--categories', nargs='+',
                        help='Specific categories to show (e.g., underinf-conj underinf-quant)')

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: {args.results_dir} is not a directory")
        return 1

    if args.mode == 'underinf':
        create_underinf_summary(args.results_dir, output_path=args.output)
    else:
        figsize = None
        if args.width and args.height:
            figsize = (args.width, args.height)
        create_variant_comparison(
            args.results_dir,
            output_path=args.output,
            figsize=figsize,
            categories=args.categories,
        )

    return 0


if __name__ == "__main__":
    exit(main())
