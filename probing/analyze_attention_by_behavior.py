#!/usr/bin/env python3
"""
Analyze whether attention patterns correlate with behavioral outcomes.

Tests: Is elevated early statement→outcome attention associated with
accepting underinformative cases (saying "yes")?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_attention_data(json_path: str) -> List[Dict]:
    """Load attention analysis results."""
    with open(json_path) as f:
        return json.load(f)


def load_behavioral_results(json_path: str) -> Dict[str, Dict]:
    """Load behavioral results, indexed by test_id."""
    with open(json_path) as f:
        data = json.load(f)

    results_by_id = {}
    for result in data.get('results', []):
        test_id = result.get('test_id')
        if test_id is not None:
            results_by_id[str(test_id)] = result

    return results_by_id


def analyze_attention_by_response(
    attention_data: List[Dict],
    behavioral_results: Dict[str, Dict],
    categories: List[str],
    metric: str = 'statement_to_outcome',
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Split attention data by behavioral response (accepted vs rejected).

    Returns:
        accepted_mean: (n_layers, n_heads) mean attention for accepted trials
        rejected_mean: (n_layers, n_heads) mean attention for rejected trials
        n_accepted: count of accepted trials
        n_rejected: count of rejected trials
    """
    accepted = []
    rejected = []

    for item in attention_data:
        if item['category'] not in categories:
            continue
        if metric not in item:
            continue

        test_id = str(item.get('test_id'))
        if test_id not in behavioral_results:
            continue

        behavior = behavioral_results[test_id]
        attn = np.array(item[metric])

        # For underinf cases: expected='no', so correct=True means rejected, correct=False means accepted
        if behavior.get('correct'):
            rejected.append(attn)  # Correctly said "no"
        else:
            accepted.append(attn)  # Incorrectly said "yes"

    accepted_mean = np.mean(accepted, axis=0) if accepted else np.zeros((36, 32))
    rejected_mean = np.mean(rejected, axis=0) if rejected else np.zeros((36, 32))

    return accepted_mean, rejected_mean, len(accepted), len(rejected)


def print_layer_comparison(
    accepted_mean: np.ndarray,
    rejected_mean: np.ndarray,
    layer_range: Tuple[int, int] = (0, 10),
    title: str = "",
):
    """Print comparison of accepted vs rejected attention by layer."""
    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Layer':<8} {'Accepted':<12} {'Rejected':<12} {'Diff (A-R)':<12}")
    print("-" * 45)

    for layer in range(layer_range[0], layer_range[1]):
        acc = accepted_mean[layer].mean()  # Average over heads
        rej = rejected_mean[layer].mean()
        diff = acc - rej
        print(f"{layer:<8} {acc:.6f}     {rej:.6f}     {diff:+.6f}")

    # Summary for early layers
    early_acc = accepted_mean[layer_range[0]:layer_range[1]].mean()
    early_rej = rejected_mean[layer_range[0]:layer_range[1]].mean()
    print("-" * 45)
    print(f"{'Mean':<8} {early_acc:.6f}     {early_rej:.6f}     {early_acc - early_rej:+.6f}")


def main():
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--attention-dir', default='probing/results/qwen3_8b_run_04/attention/')
    parser.add_argument('--behavioral-dir', default='probing/results/qwen3_8b_run_04/behavioral/')
    parser.add_argument('--output', '-o', help='Output figure path')

    args = parser.parse_args()

    attention_dir = Path(args.attention_dir)
    behavioral_dir = Path(args.behavioral_dir)

    # Personas to analyze
    personas = ['anti_gricean', 'soft_literalist', 'helpful_teacher', 'pragmaticist', 'baseline']

    results = {}

    for persona in personas:
        # Load attention
        attn_path = attention_dir / f'attention_{persona}.json'
        if not attn_path.exists():
            print(f"Skipping {persona}: no attention data")
            continue

        # Load behavioral results
        # Try to find matching behavioral file
        if persona == 'baseline':
            pattern = '*baseline*.json'
        else:
            pattern = f'*{persona}*.json'

        behavioral_files = list(behavioral_dir.glob(pattern))
        if not behavioral_files:
            print(f"Skipping {persona}: no behavioral data matching {pattern}")
            continue

        behavioral_path = behavioral_files[0]

        print(f"\n{'='*60}")
        print(f"Persona: {persona.upper()}")
        print(f"{'='*60}")
        print(f"Attention: {attn_path}")
        print(f"Behavioral: {behavioral_path}")

        attention_data = load_attention_data(attn_path)
        behavioral_results = load_behavioral_results(behavioral_path)

        # Analyze underinformative cases
        acc_mean, rej_mean, n_acc, n_rej = analyze_attention_by_response(
            attention_data,
            behavioral_results,
            categories=['underinf-conj', 'underinf-quant'],
        )

        print(f"\nUnderinformative trials: {n_acc} accepted, {n_rej} rejected")

        results[persona] = {
            'accepted_mean': acc_mean,
            'rejected_mean': rej_mean,
            'n_accepted': n_acc,
            'n_rejected': n_rej,
        }

        print_layer_comparison(
            acc_mean, rej_mean,
            layer_range=(0, 12),
            title=f"Statement→Outcome attention (Layers 0-11)",
        )

    # Create visualization
    if args.output and results:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, persona in zip(axes, personas + ['_empty']):
            if persona not in results:
                ax.set_visible(False)
                continue

            r = results[persona]
            layers = np.arange(36)

            # Average over heads
            acc_by_layer = r['accepted_mean'].mean(axis=1)
            rej_by_layer = r['rejected_mean'].mean(axis=1)

            ax.plot(layers, acc_by_layer, label=f'Accepted (n={r["n_accepted"]})',
                    color='red', linewidth=2)
            ax.plot(layers, rej_by_layer, label=f'Rejected (n={r["n_rejected"]})',
                    color='blue', linewidth=2)

            ax.set_title(persona.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Statement→Outcome Attention')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Highlight early layers
            ax.axvspan(0, 10, alpha=0.1, color='yellow')

        plt.suptitle('Statement→Outcome Attention by Response\n(Underinformative trials only)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved figure to {args.output}")
        plt.close()


if __name__ == "__main__":
    main()
