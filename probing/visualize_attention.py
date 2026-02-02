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


def compute_content_scaling_factors(persona_data: dict, baseline_key: str = 'baseline') -> dict:
    """
    Compute empirical scaling factors based on actual attention to content.

    Instead of assuming proportional dilution (attention * seq_len = constant),
    measure how much attention each persona actually gives to content regions.

    Returns dict of persona -> scaling factor (1.0 for baseline).
    """
    baseline_data = persona_data.get(baseline_key, list(persona_data.values())[0])

    # Compute baseline's mean attention to outcome (as reference)
    baseline_attn = []
    baseline_lens = []
    for item in baseline_data:
        if 'last_to_outcome' in item:
            arr = np.array(item['last_to_outcome'])
            baseline_attn.append(arr.mean())
            baseline_lens.append(item['seq_len'])

    baseline_product = np.mean(baseline_attn) * np.mean(baseline_lens)

    scaling_factors = {}
    for persona, data in persona_data.items():
        all_attn = []
        all_lens = []
        for item in data:
            if 'last_to_outcome' in item:
                arr = np.array(item['last_to_outcome'])
                all_attn.append(arr.mean())
                all_lens.append(item['seq_len'])

        if all_attn:
            persona_product = np.mean(all_attn) * np.mean(all_lens)
            # scaling > 1 means persona has more attention than expected
            # To normalize, we divide by this factor
            scaling_factors[persona] = persona_product / baseline_product
        else:
            scaling_factors[persona] = 1.0

    return scaling_factors


# Measured content attention fractions from full attention extraction
# These are the actual % of attention going to content (non-persona) tokens
# Single-value version (mean across layers) - kept for reference
MEASURED_CONTENT_FRACTIONS = {
    'baseline': 1.0,
    'anti_gricean': 0.508,
    'literal_thinker': 0.52,  # estimated similar to AG
    'helpful_teacher': 0.50,  # estimated
    'pragmaticist': 0.474,
}

# Per-layer content fractions (measured from model)
# Format: list of 36 values, one per layer
MEASURED_CONTENT_FRACTIONS_BY_LAYER = {
    'baseline': [1.0] * 36,
    'anti_gricean': [0.885, 0.885, 0.817, 0.737, 0.803, 0.841, 0.897, 0.423, 0.382, 0.274, 0.401, 0.48, 0.624, 0.585, 0.727, 0.631, 0.539, 0.649, 0.592, 0.527, 0.431, 0.471, 0.37, 0.496, 0.369, 0.33, 0.263, 0.232, 0.386, 0.209, 0.225, 0.283, 0.33, 0.325, 0.424, 0.432],
    'literal_thinker': [0.898, 0.903, 0.815, 0.738, 0.79, 0.828, 0.895, 0.418, 0.378, 0.29, 0.406, 0.498, 0.619, 0.564, 0.71, 0.626, 0.553, 0.642, 0.601, 0.542, 0.447, 0.46, 0.373, 0.516, 0.381, 0.331, 0.278, 0.232, 0.395, 0.219, 0.224, 0.289, 0.328, 0.321, 0.416, 0.421],
    'helpful_teacher': [0.889, 0.871, 0.741, 0.554, 0.627, 0.639, 0.864, 0.428, 0.374, 0.283, 0.407, 0.507, 0.61, 0.554, 0.703, 0.616, 0.523, 0.615, 0.54, 0.527, 0.486, 0.554, 0.375, 0.555, 0.385, 0.322, 0.29, 0.243, 0.401, 0.22, 0.23, 0.297, 0.335, 0.319, 0.42, 0.428],
    'pragmaticist': [0.882, 0.832, 0.665, 0.444, 0.521, 0.534, 0.822, 0.425, 0.4, 0.306, 0.424, 0.471, 0.601, 0.564, 0.69, 0.635, 0.539, 0.632, 0.555, 0.514, 0.449, 0.497, 0.379, 0.514, 0.39, 0.329, 0.28, 0.233, 0.374, 0.209, 0.224, 0.278, 0.317, 0.301, 0.404, 0.426],
    'soft_literalist': [0.899, 0.881, 0.754, 0.591, 0.653, 0.674, 0.867, 0.42, 0.379, 0.287, 0.415, 0.489, 0.604, 0.585, 0.736, 0.638, 0.519, 0.619, 0.555, 0.529, 0.452, 0.484, 0.369, 0.527, 0.376, 0.318, 0.26, 0.226, 0.397, 0.211, 0.223, 0.28, 0.324, 0.308, 0.419, 0.428],
}


def plot_all_metrics_comparison_content_normalized(
    persona_data: dict,
    output_path: str = None,
):
    """
    Plot attention metrics with per-layer content-conditional normalization.

    At each layer, divide by the measured fraction of attention going to content.
    This accounts for the fact that early layers (~85% to content) and late
    layers (~25% to content) have very different attention budgets for content.
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

    # Print content fraction summary
    print("Per-layer content attention (mean across early/mid/late):")
    for p in personas:
        fracs = MEASURED_CONTENT_FRACTIONS_BY_LAYER.get(p, [0.5] * 36)
        early = np.mean(fracs[:7])
        mid = np.mean(fracs[7:20])
        late = np.mean(fracs[20:])
        print(f"  {p}: early(0-6)={early*100:.0f}%, mid(7-19)={mid*100:.0f}%, late(20-35)={late*100:.0f}%")

    fig, axes = plt.subplots(n_metrics, n_personas, figsize=(4 * n_personas, 3 * n_metrics),
                              sharex=True, sharey='row')

    for row, metric in enumerate(metrics):
        for col, persona in enumerate(personas):
            ax = axes[row, col] if n_metrics > 1 else axes[col]

            category_means, _ = compute_category_means(persona_data[persona], metric)

            if not category_means:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            # Get per-layer content fractions
            content_fracs = np.array(MEASURED_CONTENT_FRACTIONS_BY_LAYER.get(persona, [0.5] * 36))

            for cat in categories:
                if cat not in category_means:
                    continue
                arr = category_means[cat]  # (n_layers, n_heads)

                # Normalize each layer by its content fraction
                # arr[layer, head] / content_fracs[layer]
                normalized = arr / content_fracs[:, np.newaxis]

                layer_means = normalized.mean(axis=1)
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

    plt.suptitle('Attention Patterns by Persona (per-layer content-conditional)',
                 fontsize=14, fontweight='bold', y=1.01)
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
    parser.add_argument('--content-normalize', action='store_true',
                        help='Use content-conditional normalization instead of seq_len')
    parser.add_argument('--metric', help='Single metric to plot (otherwise plots all)')

    args = parser.parse_args()

    attention_dir = Path(args.attention_dir)

    # Load all persona attention data
    persona_data = {}
    for json_file in sorted(attention_dir.glob('attention_*.json')):
        persona_name = json_file.stem.replace('attention_', '')
        # Skip non-persona files (e.g., attention_matrices_baseline.json)
        if 'matrices' in persona_name or 'head' in persona_name:
            continue
        print(f"Loading {persona_name}...")
        persona_data[persona_name] = load_attention_data(json_file)

    if not persona_data:
        print(f"No attention files found in {attention_dir}")
        return 1

    if args.content_normalize:
        plot_all_metrics_comparison_content_normalized(
            persona_data,
            output_path=args.output,
        )
    elif args.no_normalize:
        plot_all_metrics_comparison(
            persona_data,
            output_path=args.output,
            normalize=False,
        )
    elif args.metric:
        plot_attention_comparison(
            persona_data,
            args.metric,
            output_path=args.output,
            normalize=True,
        )
    else:
        plot_all_metrics_comparison(
            persona_data,
            output_path=args.output,
            normalize=True,
        )

    return 0


if __name__ == "__main__":
    exit(main())
