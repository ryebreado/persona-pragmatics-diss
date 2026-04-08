#!/usr/bin/env python3
"""
Poster heatmap: U-conj and U-quant accuracy across models and personas.
White = 100% correct, Red = 0% correct.
"""

import json
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Model directories and display names
MODELS = {
    'Claude Opus 4.5': 'results/claude_opus_4_5_run_01',
    'GPT-5.2': 'results/gpt_5_2_run_02',
    'Qwen 3 8B': 'results/qwen3_8b_run_04',
}

PERSONAS = ['baseline', 'anti_gricean', 'helpful_teacher', 'literal_thinker', 'pragmaticist']
PERSONA_LABELS = {
    'baseline': 'Baseline',
    'anti_gricean': 'Anti-Gricean',
    'helpful_teacher': 'Helpful Teacher',
    'literal_thinker': 'Literal Thinker',
    'pragmaticist': 'Pragmaticist',
}

VARIANT_RE = re.compile(r'^(.+)_([a-c])$')


def load_base_personas(results_dir):
    """Load base persona results (no _a/_b/_c variants) from a results directory."""
    results = {}
    for filepath in glob(os.path.join(results_dir, '*.json')):
        with open(filepath) as f:
            data = json.load(f)

        basename = os.path.basename(filepath)
        if '_baseline_' in basename or 'baseline' in basename.lower():
            persona = 'baseline'
        elif data.get('persona_file'):
            persona = os.path.basename(data['persona_file'])
        else:
            continue

        if VARIANT_RE.match(persona):
            continue

        if persona not in results:
            results[persona] = data

    return results


def get_underinf_accuracy(data):
    """Extract U-conj and U-quant accuracy from result data."""
    acc = data.get('accuracy_by_category', {})
    return acc.get('underinf-conj'), acc.get('underinf-quant')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Poster heatmap: U-conj/U-quant across models and personas')
    parser.add_argument('--output', '-o', default=None, help='Output PNG path')
    args = parser.parse_args()

    model_names = list(MODELS.keys())

    # Collect data into matrix: rows=personas, cols=model×category
    # Column order: Claude U-conj, Claude U-quant, GPT U-conj, GPT U-quant, Qwen U-conj, Qwen U-quant
    col_labels = []
    for model in model_names:
        col_labels.append(f'{model}\nU-conj')
        col_labels.append(f'{model}\nU-quant')

    n_rows = len(PERSONAS)
    n_cols = len(model_names) * 2
    matrix = np.full((n_rows, n_cols), np.nan)

    for mi, (model_name, results_dir) in enumerate(MODELS.items()):
        personas = load_base_personas(results_dir)
        for pi, persona in enumerate(PERSONAS):
            if persona in personas:
                uc, uq = get_underinf_accuracy(personas[persona])
                if uc is not None:
                    matrix[pi, mi * 2] = uc
                if uq is not None:
                    matrix[pi, mi * 2 + 1] = uq

    # Plot: white (1.0) to red (0.0)
    cmap = mcolors.LinearSegmentedColormap.from_list('white_red', ['#d62728', 'white'])
    fig, ax = plt.subplots(figsize=(7, 3.5))

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.4 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=text_color)

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, ha='center')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([PERSONA_LABELS[p] for p in PERSONAS], fontsize=9)

    # Add vertical lines to separate models
    for mi in range(1, len(model_names)):
        ax.axvline(x=mi * 2 - 0.5, color='grey', linewidth=1.5)

    ax.set_title('Underinformative Accuracy by Model and Persona', fontsize=12,
                 fontweight='bold', pad=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Accuracy', fontsize=9)

    plt.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        plt.savefig(args.output, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'Saved to {args.output}')
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    main()
