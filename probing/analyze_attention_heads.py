#!/usr/bin/env python3
"""
Analyze individual attention heads to find:
1. Heads that distinguish underinformative from true/false
2. Heads that show persona modulation
3. Heads that attend to scalar terms like "some"

Outputs rankings and saves raw attention matrices for visualization.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class HeadScore:
    layer: int
    head: int
    effect_size: float
    metric_name: str

    def __repr__(self):
        return f"L{self.layer}H{self.head}: {self.effect_size:+.4f}"


def load_attention_data(json_path: str) -> List[Dict]:
    """Load attention analysis results."""
    with open(json_path) as f:
        return json.load(f)


def get_metric_by_category(
    data: List[Dict],
    metric_name: str,
    categories: List[str],
) -> np.ndarray:
    """
    Extract metric values for specific categories.

    Returns: (n_examples, n_layers, n_heads) array
    """
    values = []
    for item in data:
        if item['category'] in categories and metric_name in item:
            values.append(np.array(item[metric_name]))

    if not values:
        return np.array([])
    return np.stack(values, axis=0)


def analyze_underinf_vs_true(data: List[Dict]) -> List[HeadScore]:
    """
    Analysis 1: Find heads that distinguish underinformative from true cases.

    Uses statement→outcome attention in baseline persona.
    """
    metric = 'statement_to_outcome'

    underinf = get_metric_by_category(data, metric, ['underinf-conj', 'underinf-quant'])
    true_cases = get_metric_by_category(data, metric, ['true-conj', 'true-quant'])

    if underinf.size == 0 or true_cases.size == 0:
        print("Warning: Missing data for underinf vs true analysis")
        return []

    # Mean across examples: (n_layers, n_heads)
    underinf_mean = underinf.mean(axis=0)
    true_mean = true_cases.mean(axis=0)

    # Effect size: difference in means
    diff = underinf_mean - true_mean

    n_layers, n_heads = diff.shape
    scores = []
    for layer in range(n_layers):
        for head in range(n_heads):
            scores.append(HeadScore(
                layer=layer,
                head=head,
                effect_size=diff[layer, head],
                metric_name='statement_to_outcome (underinf - true)',
            ))

    # Sort by absolute effect size
    scores.sort(key=lambda x: abs(x.effect_size), reverse=True)
    return scores


def analyze_persona_modulation(
    antigricean_data: List[Dict],
    pragmaticist_data: List[Dict],
) -> List[HeadScore]:
    """
    Analysis 2: Find heads that show persona modulation on underinf examples.

    Compares Anti-Gricean vs Pragmaticist on statement→outcome attention.
    """
    metric = 'statement_to_outcome'
    categories = ['underinf-conj', 'underinf-quant']

    ag = get_metric_by_category(antigricean_data, metric, categories)
    prag = get_metric_by_category(pragmaticist_data, metric, categories)

    if ag.size == 0 or prag.size == 0:
        print("Warning: Missing data for persona modulation analysis")
        return []

    ag_mean = ag.mean(axis=0)
    prag_mean = prag.mean(axis=0)

    # Effect size: Anti-Gricean - Pragmaticist
    diff = ag_mean - prag_mean

    n_layers, n_heads = diff.shape
    scores = []
    for layer in range(n_layers):
        for head in range(n_heads):
            scores.append(HeadScore(
                layer=layer,
                head=head,
                effect_size=diff[layer, head],
                metric_name='statement_to_outcome (AG - Prag, underinf only)',
            ))

    scores.sort(key=lambda x: abs(x.effect_size), reverse=True)
    return scores


def analyze_some_attention(data: List[Dict]) -> List[HeadScore]:
    """
    Analysis 3: Find heads that attend to "some" differently for true vs underinf.

    Uses last_token → "some" attention for quantifier examples.
    """
    metric = 'last_to_some'

    # Get examples that have the 'some' scalar term
    underinf_values = []
    true_values = []

    for item in data:
        if metric not in item:
            continue

        if item['category'] == 'underinf-quant':
            underinf_values.append(np.array(item[metric]))
        elif item['category'] == 'true-quant':
            # true-quant includes both 'some' and 'all' - we only want 'some'
            # Check if this item has 'some' in scalar_terms
            if 'some' in item.get('regions', {}).get('scalar_terms', {}):
                true_values.append(np.array(item[metric]))

    if not underinf_values or not true_values:
        print(f"Warning: Limited data for 'some' attention analysis")
        print(f"  underinf-quant with 'some': {len(underinf_values)}")
        print(f"  true-quant with 'some': {len(true_values)}")
        if not underinf_values or not true_values:
            return []

    underinf = np.stack(underinf_values, axis=0)
    true_cases = np.stack(true_values, axis=0)

    underinf_mean = underinf.mean(axis=0)
    true_mean = true_cases.mean(axis=0)

    diff = underinf_mean - true_mean

    n_layers, n_heads = diff.shape
    scores = []
    for layer in range(n_layers):
        for head in range(n_heads):
            scores.append(HeadScore(
                layer=layer,
                head=head,
                effect_size=diff[layer, head],
                metric_name='last_to_some (underinf - true)',
            ))

    scores.sort(key=lambda x: abs(x.effect_size), reverse=True)
    return scores


def find_examples_for_heads(
    data: List[Dict],
    heads: List[Tuple[int, int]],
    metric_name: str,
    categories: List[str],
    n_examples: int = 3,
) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Find good examples for visualizing specific heads.

    Selects examples where the head shows strongest activation.
    """
    examples_by_head = defaultdict(list)

    for item in data:
        if item['category'] not in categories:
            continue
        if metric_name not in item:
            continue

        arr = np.array(item[metric_name])

        for layer, head in heads:
            val = arr[layer, head]
            examples_by_head[(layer, head)].append({
                'test_id': item.get('test_id'),
                'category': item['category'],
                'attention_value': float(val),
                'regions': item['regions'],
            })

    # Sort by attention value and take top examples
    result = {}
    for head_key, examples in examples_by_head.items():
        examples.sort(key=lambda x: x['attention_value'], reverse=True)
        result[head_key] = examples[:n_examples]

    return result


def print_rankings(title: str, scores: List[HeadScore], top_n: int = 10):
    """Print formatted ranking of heads."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Metric: {scores[0].metric_name if scores else 'N/A'}")
    print()
    print(f"{'Rank':<6} {'Layer':<7} {'Head':<7} {'Effect Size':<12}")
    print("-" * 35)
    for i, score in enumerate(scores[:top_n], 1):
        print(f"{i:<6} {score.layer:<7} {score.head:<7} {score.effect_size:+.6f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze attention heads')
    parser.add_argument('attention_dir', help='Directory with attention JSON files')
    parser.add_argument('--output', '-o', help='Output directory for examples')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top heads to show')

    args = parser.parse_args()

    attention_dir = Path(args.attention_dir)

    # Load data
    print("Loading attention data...")
    baseline = load_attention_data(attention_dir / 'attention_baseline.json')
    anti_gricean = load_attention_data(attention_dir / 'attention_anti_gricean.json')
    pragmaticist = load_attention_data(attention_dir / 'attention_pragmaticist.json')

    print(f"  Baseline: {len(baseline)} examples")
    print(f"  Anti-Gricean: {len(anti_gricean)} examples")
    print(f"  Pragmaticist: {len(pragmaticist)} examples")

    # Analysis 1: Underinf vs True (baseline)
    scores_underinf = analyze_underinf_vs_true(baseline)
    print_rankings("Analysis 1: Heads distinguishing Underinformative vs True",
                   scores_underinf, args.top_n)

    # Analysis 2: Persona modulation
    scores_persona = analyze_persona_modulation(anti_gricean, pragmaticist)
    print_rankings("Analysis 2: Heads showing Persona Modulation (AG vs Prag, underinf)",
                   scores_persona, args.top_n)

    # Analysis 3: Attention to "some"
    scores_some = analyze_some_attention(baseline)
    print_rankings("Analysis 3: Heads attending to 'some' (underinf vs true)",
                   scores_some, args.top_n)

    # Find top heads overall (union of top 3 from each analysis)
    top_heads = set()
    for scores in [scores_underinf, scores_persona, scores_some]:
        for s in scores[:3]:
            top_heads.add((s.layer, s.head))

    print(f"\n{'='*60}")
    print(f"Top heads for visualization: {sorted(top_heads)}")
    print(f"{'='*60}")

    # Save examples for top heads
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find examples for each analysis type
        examples_underinf = find_examples_for_heads(
            baseline,
            list(top_heads),
            'statement_to_outcome',
            ['underinf-conj', 'underinf-quant'],
        )

        examples_true = find_examples_for_heads(
            baseline,
            list(top_heads),
            'statement_to_outcome',
            ['true-conj', 'true-quant'],
        )

        output_data = {
            'top_heads': [{'layer': l, 'head': h} for l, h in sorted(top_heads)],
            'analysis_1_underinf_vs_true': [
                {'layer': s.layer, 'head': s.head, 'effect_size': s.effect_size}
                for s in scores_underinf[:args.top_n]
            ],
            'analysis_2_persona_modulation': [
                {'layer': s.layer, 'head': s.head, 'effect_size': s.effect_size}
                for s in scores_persona[:args.top_n]
            ],
            'analysis_3_some_attention': [
                {'layer': s.layer, 'head': s.head, 'effect_size': s.effect_size}
                for s in scores_some[:args.top_n]
            ],
            'examples_underinf': {f"L{l}H{h}": ex for (l, h), ex in examples_underinf.items()},
            'examples_true': {f"L{l}H{h}": ex for (l, h), ex in examples_true.items()},
        }

        output_path = output_dir / 'head_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved analysis to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
