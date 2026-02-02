#!/usr/bin/env python3
"""
Measure how much attention goes to persona tokens vs content tokens.

If attention to persona is roughly constant regardless of persona length,
that indicates the dilution isn't proportional to sequence length.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_attention_data(json_path: str) -> List[Dict]:
    with open(json_path) as f:
        return json.load(f)


def estimate_persona_length(data: List[Dict], baseline_data: List[Dict]) -> int:
    """Estimate persona token count by comparing to baseline."""
    persona_lens = [item['seq_len'] for item in data]
    baseline_lens = [item['seq_len'] for item in baseline_data]
    return int(np.mean(persona_lens) - np.mean(baseline_lens))


def compute_attention_to_regions(
    data: List[Dict],
    persona_length: int,
) -> Dict[str, np.ndarray]:
    """
    Compute mean attention from last_token to different regions.

    Regions:
    - persona: tokens 0 to persona_length (if persona_length > 0)
    - content: tokens after persona (everything else)
    - outcome: the outcome span
    - statement: the statement span

    Returns attention as fraction of total (sums to 1 for each layer/head).
    """
    results = {
        'to_persona': [],
        'to_content': [],
        'to_outcome': [],
        'to_statement': [],
        'to_other_content': [],
    }

    for item in data:
        seq_len = item['seq_len']
        last_token = item['regions']['last_token']
        outcome_start, outcome_end = item['regions']['outcome']
        stmt_start, stmt_end = item['regions']['statement']

        # We need the raw attention row from last_token
        # But we only have aggregated metrics in the JSON
        # Let's compute from the metrics we have

        # Actually, we need to re-extract this. The JSON doesn't have full attention.
        # Let's compute what fraction of the sequence each region represents
        # and use the mean attention values we have.

        # For now, let's estimate based on region sizes and attention values
        pass

    return results


def analyze_attention_distribution(attention_dir: Path, baseline_data: List[Dict]):
    """Analyze attention distribution for each persona."""

    print("=" * 70)
    print("Attention Distribution Analysis: Persona vs Content Tokens")
    print("=" * 70)
    print()

    # We need to re-extract attention to measure this properly
    # The current JSON only has specific region metrics, not full attention rows

    # Let's compute an approximation:
    # - We know last_to_outcome and last_to_statement (mean attention per token in region)
    # - We can estimate attention to persona vs content by looking at how these scale

    personas = ['baseline', 'anti_gricean', 'literal_thinker', 'helpful_teacher', 'pragmaticist']

    baseline_mean_len = np.mean([item['seq_len'] for item in baseline_data])

    print(f"{'Persona':<20} {'Seq Len':<10} {'Persona Toks':<15} {'Content Toks':<15}")
    print("-" * 60)

    results = {}

    for persona in personas:
        json_path = attention_dir / f'attention_{persona}.json'
        if not json_path.exists():
            continue

        data = load_attention_data(json_path)
        mean_len = np.mean([item['seq_len'] for item in data])
        persona_toks = mean_len - baseline_mean_len if persona != 'baseline' else 0
        content_toks = baseline_mean_len

        print(f"{persona:<20} {mean_len:<10.1f} {persona_toks:<15.1f} {content_toks:<15.1f}")

        results[persona] = {
            'mean_seq_len': mean_len,
            'persona_tokens': persona_toks,
            'content_tokens': content_toks,
        }

    print()
    return results


def compute_content_conditional_attention(
    data: List[Dict],
    persona_tokens: int,
    metric_name: str,
) -> Dict[str, np.ndarray]:
    """
    Compute attention metrics conditional on content tokens only.

    The idea: instead of normalizing by total seq_len, we compute
    what fraction of "content attention" goes to each region.

    For last_token attending to the sequence:
    - Total attention sums to 1 over all previous tokens
    - If persona takes up P tokens, and we assume uniform baseline attention,
      then ~P/seq_len goes to persona
    - Content attention = 1 - (attention to persona)
    - Normalized metric = raw_metric / content_attention_fraction

    But we don't have attention to persona directly. We can estimate:
    - If attention were uniform, persona would get P/seq_len
    - But attention isn't uniform - it's concentrated on meaningful tokens
    - So persona probably gets LESS than P/seq_len

    Better approach: compute attention to outcome+statement as fraction of
    attention to all non-persona tokens.
    """
    from collections import defaultdict

    category_data = defaultdict(list)

    for item in data:
        cat = item['category']
        if metric_name not in item:
            continue

        arr = np.array(item[metric_name])
        seq_len = item['seq_len']

        # The region we're measuring attention TO
        if metric_name == 'last_to_outcome':
            region = item['regions']['outcome']
        elif metric_name == 'last_to_statement':
            region = item['regions']['statement']
        elif metric_name == 'statement_to_outcome':
            region = item['regions']['outcome']
        else:
            continue

        region_size = region[1] - region[0]
        content_size = seq_len - persona_tokens

        # Current metric is mean attention per token in region
        # Total attention to region = metric * region_size (approximately)
        # Fraction of content = region_size / content_size
        # Expected if uniform over content = region_size / content_size
        #
        # To normalize: compare to expected uniform over content only
        # Normalized = metric / (1 / content_size) = metric * content_size
        # This gives "attention relative to uniform content baseline"

        # Simpler: just scale by content_size / baseline_content_size
        # This is similar to what we did but ignoring persona tokens

        category_data[cat].append(arr)

    category_means = {}
    for cat in category_data:
        category_means[cat] = np.mean(category_data[cat], axis=0)

    return category_means


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--attention-dir', default='probing/results/qwen3_8b_run_04/attention/')

    args = parser.parse_args()
    attention_dir = Path(args.attention_dir)

    # Load baseline for reference
    baseline_data = load_attention_data(attention_dir / 'attention_baseline.json')

    # Analyze distribution
    results = analyze_attention_distribution(attention_dir, baseline_data)

    # Now let's look at actual attention values to see if they scale with length
    print("\nAttention Values by Persona (last_to_outcome, averaged over all layers/heads/examples):")
    print("=" * 70)
    print(f"{'Persona':<20} {'Raw Attn':<12} {'Seq Len':<10} {'Raw × Len':<12} {'Scaling':<10}")
    print("-" * 70)

    baseline_attn = None
    baseline_len = None

    for persona in ['baseline', 'anti_gricean', 'literal_thinker', 'helpful_teacher', 'pragmaticist']:
        json_path = attention_dir / f'attention_{persona}.json'
        if not json_path.exists():
            continue

        data = load_attention_data(json_path)

        # Compute mean attention across all examples, layers, heads
        all_attn = []
        all_lens = []
        for item in data:
            if 'last_to_outcome' in item:
                arr = np.array(item['last_to_outcome'])
                all_attn.append(arr.mean())
                all_lens.append(item['seq_len'])

        mean_attn = np.mean(all_attn)
        mean_len = np.mean(all_lens)

        if persona == 'baseline':
            baseline_attn = mean_attn
            baseline_len = mean_len
            scaling = 1.0
        else:
            # If dilution were proportional: attn * len should be constant
            # scaling = (attn * len) / (baseline_attn * baseline_len)
            scaling = (mean_attn * mean_len) / (baseline_attn * baseline_len)

        print(f"{persona:<20} {mean_attn:<12.6f} {mean_len:<10.1f} {mean_attn * mean_len:<12.4f} {scaling:<10.3f}")

    print()
    print("If dilution were proportional, 'Raw × Len' would be constant (Scaling ≈ 1.0)")
    print("If Scaling > 1.0: persona has MORE attention than expected (dilution < proportional)")
    print("If Scaling < 1.0: persona has LESS attention than expected (dilution > proportional)")


if __name__ == "__main__":
    main()
