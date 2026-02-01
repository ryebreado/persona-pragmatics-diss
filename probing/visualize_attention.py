#!/usr/bin/env python3
"""
Visualize attention patterns across personas with proper per-metric normalization.

For causal attention, each position only sees previous tokens, so:
- last_token → X: normalize by last_token position
- statement → outcome: normalize by mean statement position
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_attention_data(json_path):
    """Load attention analysis results."""
    with open(json_path) as f:
        return json.load(f)


def compute_category_means(data, metric_name):
    """
    Compute mean attention by category, with proper normalization factor.

    Returns:
        category_means: dict of category -> (n_layers, n_heads) array
        normalization_factors: dict of category -> scalar (visible tokens for this metric)
    """
    from collections import defaultdict

    category_data = defaultdict(list)
    category_norm_factors = defaultdict(list)

    for item in data:
        cat = item['category']
        if metric_name in item:
            arr = np.array(item[metric_name])
            category_data[cat].append(arr)

            # Compute normalization factor based on metric type
            if metric_name.startswith('statement_to_'):
                # For statement → X, normalize by mean statement position
                stmt_start, stmt_end = item['regions']['statement']
                visible_tokens = (stmt_start + stmt_end) / 2  # mean position
            else:
                # For last_token → X, normalize by last_token position
                visible_tokens = item['regions']['last_token']

            category_norm_factors[cat].append(visible_tokens)

    category_means = {}
    norm_factors = {}
    for cat in category_data:
        category_means[cat] = np.mean(category_data[cat], axis=0)
        norm_factors[cat] = np.mean(category_norm_factors[cat])

    return category_means, norm_factors


def normalize_attention(category_means, norm_factors, baseline_norm_factor):
    """
    Normalize attention values to account for different number of visible tokens.

    Scales attention so that comparisons are fair across different prompt lengths.
    """
    normalized = {}
    for cat, arr in category_means.items():
        # Scale by ratio of visible tokens to baseline
        scale = norm_factors[cat] / baseline_norm_factor
        normalized[cat] = arr * scale
    return normalized


def plot_attention_comparison(
    persona_data: dict,
    metric_name: str,
    output_path: str = None,
    normalize: bool = True,
):
    """
    Plot attention metric across personas and categories.

    Args:
        persona_data: dict of persona_name -> loaded JSON data
        metric_name: which attention metric to plot
        normalize: whether to apply per-metric normalization
    """
    # Category display order and colors
    categories = ['true-conj', 'true-quant', 'false-conj', 'false-quant',
                  'underinf-conj', 'underinf-quant']
    colors = {
        'true-conj': '#228B22', 'true-quant': '#90EE90',
        'false-conj': '#B22222', 'false-quant': '#F08080',
        'underinf-conj': '#1E3A8A', 'underinf-quant': '#93C5FD',
    }

    personas = list(persona_data.keys())
    n_personas = len(personas)

    fig, axes = plt.subplots(1, n_personas, figsize=(4 * n_personas, 4), sharey=True)
    if n_personas == 1:
        axes = [axes]

    # Get baseline normalization factor for reference
    baseline_key = 'baseline' if 'baseline' in persona_data else personas[0]
    _, baseline_norm_factors = compute_category_means(persona_data[baseline_key], metric_name)
    # Use mean across categories as reference
    baseline_ref = np.mean(list(baseline_norm_factors.values()))

    for ax, persona in zip(axes, personas):
        category_means, norm_factors = compute_category_means(persona_data[persona], metric_name)

        if normalize:
            category_means = normalize_attention(category_means, norm_factors, baseline_ref)
            title_suffix = " (normalized)"
        else:
            title_suffix = ""

        # Plot each category
        for cat in categories:
            if cat not in category_means:
                continue
            arr = category_means[cat]
            # Average over heads, plot by layer
            layer_means = arr.mean(axis=1)
            ax.plot(layer_means, label=cat, color=colors[cat], linewidth=1.5)

        ax.set_xlabel('Layer')
        ax.set_title(f"{persona.replace('_', ' ').title()}{title_suffix}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f'Attention ({metric_name})')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.suptitle(f'Attention: {metric_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_all_metrics_comparison(
    persona_data: dict,
    output_path: str = None,
    normalize: bool = True,
):
    """
    Plot all attention metrics in a grid: metrics as rows, personas as columns.
    """
    metrics = ['last_to_outcome', 'last_to_statement', 'statement_to_outcome']

    categories = ['true-conj', 'true-quant', 'false-conj', 'false-quant',
                  'underinf-conj', 'underinf-quant']
    colors = {
        'true-conj': '#228B22', 'true-quant': '#90EE90',
        'false-conj': '#B22222', 'false-quant': '#F08080',
        'underinf-conj': '#1E3A8A', 'underinf-quant': '#93C5FD',
    }

    personas = list(persona_data.keys())
    n_personas = len(personas)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, n_personas, figsize=(4 * n_personas, 3 * n_metrics),
                              sharex=True, sharey='row')

    for row, metric in enumerate(metrics):
        # Get baseline normalization factor for this metric
        baseline_key = 'baseline' if 'baseline' in persona_data else personas[0]
        _, baseline_norm_factors = compute_category_means(persona_data[baseline_key], metric)
        baseline_ref = np.mean(list(baseline_norm_factors.values())) if baseline_norm_factors else 1.0

        for col, persona in enumerate(personas):
            ax = axes[row, col] if n_metrics > 1 else axes[col]

            category_means, norm_factors = compute_category_means(persona_data[persona], metric)

            if not category_means:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            if normalize:
                category_means = normalize_attention(category_means, norm_factors, baseline_ref)

            for cat in categories:
                if cat not in category_means:
                    continue
                arr = category_means[cat]
                layer_means = arr.mean(axis=1)
                ax.plot(layer_means, label=cat, color=colors[cat], linewidth=1.5)

            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(persona.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(metric.replace('_', ' '), fontsize=9)
            if row == n_metrics - 1:
                ax.set_xlabel('Layer')

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=8)

    norm_text = " (per-metric normalized)" if normalize else ""
    plt.suptitle(f'Attention Patterns by Persona{norm_text}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize attention patterns')
    parser.add_argument('attention_dir', help='Directory containing attention JSON files')
    parser.add_argument('--output', '-o', help='Output path for figure')
    parser.add_argument('--no-normalize', action='store_true', help='Disable normalization')
    parser.add_argument('--metric', help='Single metric to plot (otherwise plots all)')

    args = parser.parse_args()

    attention_dir = Path(args.attention_dir)

    # Load all persona attention data
    persona_data = {}
    for json_file in sorted(attention_dir.glob('attention_*.json')):
        persona_name = json_file.stem.replace('attention_', '')
        print(f"Loading {persona_name}...")
        persona_data[persona_name] = load_attention_data(json_file)

    if not persona_data:
        print(f"No attention files found in {attention_dir}")
        return 1

    normalize = not args.no_normalize

    if args.metric:
        plot_attention_comparison(
            persona_data,
            args.metric,
            output_path=args.output,
            normalize=normalize,
        )
    else:
        plot_all_metrics_comparison(
            persona_data,
            output_path=args.output,
            normalize=normalize,
        )

    return 0


if __name__ == "__main__":
    exit(main())
