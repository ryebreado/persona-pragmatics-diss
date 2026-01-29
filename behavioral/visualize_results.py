#!/usr/bin/env python3
"""
Visualize persona experiment results as small multiples bar charts.
Shows accuracy by fine-grained category for each persona condition.
"""

import json
import argparse
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Color scheme: answer type (hue) × linguistic type (lightness)
# Dark = conjunctive, Light = quantifier
COLORS = {
    'true-conj': '#228B22',      # dark green
    'true-quant': '#90EE90',     # light green
    'false-conj': '#B22222',     # dark red
    'false-quant': '#F08080',    # light red
    'underinf-conj': '#1E3A8A',  # dark blue
    'underinf-quant': '#93C5FD', # light blue
}

# Extended colors for subcategories
COLORS_SUBCATEGORY = {
    'true-conj': '#228B22',           # dark green
    'true-quant-all': '#32CD32',      # lime green
    'true-quant-some': '#90EE90',     # light green (pragmatically ambiguous)
    'false-conj': '#B22222',          # dark red
    'false-quant-all': '#DC143C',     # crimson
    'false-quant-some': '#F08080',    # light red
    'underinf-conj': '#1E3A8A',       # dark blue
    'underinf-quant': '#93C5FD',      # light blue
}

# Display order for categories (grouped by answer type)
CATEGORY_ORDER = [
    'true-conj', 'true-quant',
    'false-conj', 'false-quant',
    'underinf-conj', 'underinf-quant'
]

# Display order for subcategories
SUBCATEGORY_ORDER = [
    'true-conj', 'true-quant-all', 'true-quant-some',
    'false-conj', 'false-quant-all', 'false-quant-some',
    'underinf-conj', 'underinf-quant'
]

# Shorter labels for x-axis
CATEGORY_LABELS = {
    'true-conj': 'T-conj',
    'true-quant': 'T-quant',
    'false-conj': 'F-conj',
    'false-quant': 'F-quant',
    'underinf-conj': 'U-conj',
    'underinf-quant': 'U-quant',
}

SUBCATEGORY_LABELS = {
    'true-conj': 'T-conj',
    'true-quant-all': 'T-q-all',
    'true-quant-some': 'T-q-some',
    'false-conj': 'F-conj',
    'false-quant-all': 'F-q-all',
    'false-quant-some': 'F-q-some',
    'underinf-conj': 'U-conj',
    'underinf-quant': 'U-quant',
}


def load_results_from_directory(results_dir):
    """
    Load all result files from a directory.
    Returns dict mapping persona name to results data.
    """
    results = {}
    json_files = glob(os.path.join(results_dir, "*.json"))

    for filepath in json_files:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Determine persona name from file or content
        if '_baseline_' in os.path.basename(filepath):
            persona_name = 'baseline'
        elif data.get('persona_file'):
            # Extract persona name from path like "personas/anti_gricean"
            persona_name = os.path.basename(data['persona_file'])
        else:
            # Fallback: extract from filename
            basename = os.path.basename(filepath)
            parts = basename.replace('.json', '').split('_')
            # Try to find persona name (after model name, before timestamp)
            persona_name = parts[-3] if len(parts) >= 3 else 'unknown'

        results[persona_name] = data

    return results


def compute_subcategory_accuracy(data):
    """
    Compute accuracy by subcategory from individual results.
    Falls back to category if subcategory not present.
    """
    from collections import defaultdict

    correct_by_subcat = defaultdict(int)
    total_by_subcat = defaultdict(int)

    for result in data.get('results', []):
        # Use subcategory if available, otherwise fall back to category
        subcat = result.get('subcategory', result.get('category', 'unknown'))
        total_by_subcat[subcat] += 1
        if result.get('correct'):
            correct_by_subcat[subcat] += 1

    accuracy_by_subcat = {}
    for subcat in total_by_subcat:
        if total_by_subcat[subcat] > 0:
            accuracy_by_subcat[subcat] = correct_by_subcat[subcat] / total_by_subcat[subcat]

    return accuracy_by_subcat


def create_subplot(ax, persona_name, accuracy_by_category, title_fontsize=12, use_subcategories=False):
    """
    Create a single bar chart subplot for one persona.
    """
    # Get accuracies in consistent order
    categories = []
    accuracies = []
    colors = []

    if use_subcategories:
        order = SUBCATEGORY_ORDER
        labels = SUBCATEGORY_LABELS
        color_map = COLORS_SUBCATEGORY
    else:
        order = CATEGORY_ORDER
        labels = CATEGORY_LABELS
        color_map = COLORS

    for cat in order:
        if cat in accuracy_by_category:
            categories.append(labels.get(cat, cat))
            accuracies.append(accuracy_by_category[cat])
            colors.append(color_map.get(cat, '#888888'))

    x = np.arange(len(categories))
    bars = ax.bar(x, accuracies, color=colors, edgecolor='white', linewidth=0.5)

    # Styling
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    ax.set_title(persona_name.replace('_', ' ').title(), fontsize=title_fontsize, fontweight='bold')

    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        if height < 1.0:  # Only label non-perfect scores
            ax.annotate(f'{acc:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    return bars


def create_visualization(results_dir, output_path=None, figsize=(15, 4), use_subcategories=False):
    """
    Create small multiples visualization for all personas in a results directory.
    """
    results = load_results_from_directory(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    # Sort personas: baseline first, then alphabetically
    persona_names = sorted(results.keys(), key=lambda x: (x != 'baseline', x))
    n_personas = len(persona_names)

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_personas, figsize=figsize, sharey=True)

    # Handle single persona case
    if n_personas == 1:
        axes = [axes]

    for ax, persona_name in zip(axes, persona_names):
        data = results[persona_name]
        if use_subcategories:
            accuracy_by_category = compute_subcategory_accuracy(data)
        else:
            accuracy_by_category = data.get('accuracy_by_category', {})
        create_subplot(ax, persona_name, accuracy_by_category, use_subcategories=use_subcategories)

    # Add overall title
    model_name = results[persona_names[0]].get('model', 'Unknown Model')
    subtitle = ' (Subcategories)' if use_subcategories else ''
    fig.suptitle(f'Scalar Implicature Performance by Persona{subtitle}\n{model_name}',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add legend
    if use_subcategories:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_SUBCATEGORY['true-conj'], label='True (conj)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_SUBCATEGORY['true-quant-all'], label='True (q-all)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_SUBCATEGORY['true-quant-some'], label='True (q-some)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_SUBCATEGORY['false-conj'], label='False (conj)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_SUBCATEGORY['false-quant-all'], label='False (q-all)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_SUBCATEGORY['false-quant-some'], label='False (q-some)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_SUBCATEGORY['underinf-conj'], label='Underinf (conj)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_SUBCATEGORY['underinf-quant'], label='Underinf (quant)'),
        ]
    else:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['true-conj'], label='True (conj)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['true-quant'], label='True (quant)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['false-conj'], label='False (conj)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['false-quant'], label='False (quant)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['underinf-conj'], label='Underinf (conj)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['underinf-quant'], label='Underinf (quant)'),
        ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 0.95),
               fontsize=8, title='Category', title_fontsize=9)

    plt.tight_layout()

    # Save or show
    if output_path:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize persona experiment results as small multiples bar charts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('results_dir', help='Directory containing result JSON files')
    parser.add_argument('--output', '-o', help='Output file path (PNG). If not specified, displays interactively.')
    parser.add_argument('--width', type=float, default=15, help='Figure width in inches')
    parser.add_argument('--height', type=float, default=4, help='Figure height in inches')
    parser.add_argument('--subcategories', '-s', action='store_true',
                       help='Use subcategories (e.g., true-quant-some vs true-quant-all)')

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: {args.results_dir} is not a directory")
        return 1

    create_visualization(
        args.results_dir,
        output_path=args.output,
        figsize=(args.width, args.height),
        use_subcategories=args.subcategories
    )

    return 0


if __name__ == "__main__":
    exit(main())
