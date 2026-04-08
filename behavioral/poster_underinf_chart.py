#!/usr/bin/env python3
"""
Compact poster chart: U-conj and U-quant accuracy across 3 models and all persona conditions.
"""

import json
import os
import re
from glob import glob

import matplotlib.pyplot as plt
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

        # Skip variants
        if VARIANT_RE.match(persona):
            continue

        # Skip if we already have this persona (keep first match)
        if persona not in results:
            results[persona] = data

    return results


def get_underinf_accuracy(data):
    """Extract U-conj and U-quant accuracy from result data."""
    acc = data.get('accuracy_by_category', {})
    return acc.get('underinf-conj'), acc.get('underinf-quant')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Poster chart: U-conj/U-quant across models and personas')
    parser.add_argument('--output', '-o', default=None, help='Output PNG path')
    args = parser.parse_args()

    # Collect data: model -> persona -> (u_conj, u_quant)
    all_data = {}
    for model_name, results_dir in MODELS.items():
        personas = load_base_personas(results_dir)
        all_data[model_name] = {}
        for p in PERSONAS:
            if p in personas:
                uc, uq = get_underinf_accuracy(personas[p])
                all_data[model_name][p] = (uc, uq)
            else:
                all_data[model_name][p] = (None, None)

    # Layout: groups by model, within each group 5 persona pairs (U-conj, U-quant)
    n_personas = len(PERSONAS)
    n_models = len(MODELS)
    model_names = list(MODELS.keys())

    # Matplotlib default colors (C0-C4) to match probing figures
    PERSONA_COLORS = {
        'baseline': '#1f77b4',        # C0 blue
        'anti_gricean': '#d62728',    # C3 red
        'helpful_teacher': '#2ca02c', # C2 green
        'literal_thinker': '#ff7f0e', # C1 orange
        'pragmaticist': '#9467bd',    # C4 purple
    }

    bar_width = 0.12
    pair_gap = 0.02  # small gap between U-conj and U-quant of same persona
    persona_gap = 0.06  # gap between different personas
    group_gap = 0.7  # gap between model groups

    pair_width = 2 * bar_width + pair_gap
    group_width = n_personas * pair_width + (n_personas - 1) * persona_gap

    fig, ax = plt.subplots(figsize=(10, 4.5))

    group_centers = []
    for gi, model in enumerate(model_names):
        group_left = gi * (group_width + group_gap)
        for pi, persona in enumerate(PERSONAS):
            uc, uq = all_data[model][persona]
            offset = pi * (pair_width + persona_gap)
            x_conj = group_left + offset
            x_quant = x_conj + bar_width + pair_gap

            color = PERSONA_COLORS[persona]
            if uc is not None:
                ax.bar(x_conj, uc, bar_width, color=color, edgecolor='white', linewidth=0.5)
                if uc < 1.0:
                    ax.text(x_conj, uc + 0.02, f'{uc:.0%}', ha='center', va='bottom', fontsize=6.5)
            if uq is not None:
                ax.bar(x_quant, uq, bar_width, color=color, edgecolor='white', linewidth=0.5,
                       alpha=0.5)
                if uq < 1.0:
                    ax.text(x_quant, uq + 0.02, f'{uq:.0%}', ha='center', va='bottom', fontsize=6.5)

        group_centers.append(group_left + group_width / 2)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend at top: persona colors + U-conj/U-quant distinction
    legend_elements = []
    for persona in PERSONAS:
        color = PERSONA_COLORS[persona]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color,
                                             label=PERSONA_LABELS[persona]))
    # Add U-conj / U-quant key
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='grey', alpha=1.0,
                                         label='U-conj (solid)'))
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='grey', alpha=0.5,
                                         label='U-quant (faded)'))
    ax.legend(handles=legend_elements, fontsize=7, loc='upper center',
              ncol=4, columnspacing=0.8, handletextpad=0.4,
              frameon=True, framealpha=0.9, edgecolor='#cccccc')

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
