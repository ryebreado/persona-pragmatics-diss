#!/usr/bin/env python3
"""
Measure actual proportion of attention going to persona vs content tokens.
"""

import json
import numpy as np
from pathlib import Path


def load_attention_data(json_path: str):
    with open(json_path) as f:
        return json.load(f)


def compute_attention_proportions(data, persona_tokens: int):
    """
    Estimate proportion of attention going to different regions.

    For last_token attention:
    - Total attention sums to 1 over all previous positions
    - We have mean attention per token for outcome and statement regions
    - Total to region = mean * region_size

    Returns mean proportions across all examples.
    """
    proportions = {
        'to_outcome': [],
        'to_statement': [],
        'to_persona': [],
        'to_other_content': [],
    }

    for item in data:
        seq_len = item['seq_len']
        last_token = item['regions']['last_token']

        outcome_start, outcome_end = item['regions']['outcome']
        stmt_start, stmt_end = item['regions']['statement']

        outcome_size = outcome_end - outcome_start
        stmt_size = stmt_end - stmt_start

        # Get mean attention values (averaged over layers and heads for simplicity)
        last_to_outcome = np.array(item['last_to_outcome']).mean()
        last_to_stmt = np.array(item['last_to_statement']).mean()

        # Total attention to each region
        # (mean attention per token in region) * (region size) = total attention to region
        total_to_outcome = last_to_outcome * outcome_size
        total_to_stmt = last_to_stmt * stmt_size

        # Total attention sums to 1
        # Remaining attention goes to persona + other content
        remaining = 1.0 - total_to_outcome - total_to_stmt

        # Split remaining between persona and other content
        # Persona tokens are at positions 0 to persona_tokens-1
        # Other content is everything else (excluding outcome and statement)

        content_tokens = seq_len - persona_tokens
        other_content_tokens = content_tokens - outcome_size - stmt_size

        if persona_tokens > 0 and other_content_tokens > 0:
            # Assume remaining attention is distributed proportionally to token count
            # (This is an approximation since we don't have full attention matrix)
            persona_fraction = persona_tokens / (persona_tokens + other_content_tokens)
            to_persona = remaining * persona_fraction
            to_other = remaining * (1 - persona_fraction)
        elif persona_tokens > 0:
            to_persona = remaining
            to_other = 0
        else:
            to_persona = 0
            to_other = remaining

        proportions['to_outcome'].append(total_to_outcome)
        proportions['to_statement'].append(total_to_stmt)
        proportions['to_persona'].append(to_persona)
        proportions['to_other_content'].append(to_other)

    return {k: np.mean(v) for k, v in proportions.items()}


def main():
    attention_dir = Path('probing/results/qwen3_8b_run_04/attention/')

    # Load baseline for reference
    baseline_data = load_attention_data(attention_dir / 'attention_baseline.json')
    baseline_len = np.mean([item['seq_len'] for item in baseline_data])

    print("=" * 80)
    print("Attention Proportions: What we can measure directly")
    print("=" * 80)
    print()

    # First, report raw numbers
    print("RAW ATTENTION VALUES (mean attention per token in region, averaged over layers/heads):")
    print()
    print(f"{'Persona':<18} {'Seq Len':<10} {'Persona Toks':<14} {'Attn/tok Outcome':<18} {'Attn/tok Statement':<18}")
    print("-" * 80)

    personas = ['baseline', 'anti_gricean', 'literal_thinker', 'helpful_teacher', 'pragmaticist']
    persona_results = {}

    for persona in personas:
        json_path = attention_dir / f'attention_{persona}.json'
        if not json_path.exists():
            continue

        data = load_attention_data(json_path)
        mean_len = np.mean([item['seq_len'] for item in data])
        persona_tokens = int(mean_len - baseline_len) if persona != 'baseline' else 0

        # Get mean attention per token values
        outcome_attn = np.mean([np.array(item['last_to_outcome']).mean() for item in data])
        stmt_attn = np.mean([np.array(item['last_to_statement']).mean() for item in data])

        print(f"{persona:<18} {mean_len:<10.1f} {persona_tokens:<14} {outcome_attn:<18.6f} {stmt_attn:<18.6f}")

        persona_results[persona] = {
            'seq_len': mean_len,
            'persona_tokens': persona_tokens,
            'outcome_attn_per_tok': outcome_attn,
            'stmt_attn_per_tok': stmt_attn,
        }

    print()
    print("=" * 80)
    print("TOTAL ATTENTION TO REGIONS (per-token attention × region size):")
    print("=" * 80)
    print()
    print(f"{'Persona':<18} {'Outcome Size':<14} {'Total to Outcome':<18} {'Stmt Size':<12} {'Total to Stmt':<16} {'Sum':<10}")
    print("-" * 90)

    for persona in personas:
        json_path = attention_dir / f'attention_{persona}.json'
        if not json_path.exists():
            continue

        data = load_attention_data(json_path)

        # Get region sizes
        outcome_sizes = [item['regions']['outcome'][1] - item['regions']['outcome'][0] for item in data]
        stmt_sizes = [item['regions']['statement'][1] - item['regions']['statement'][0] for item in data]

        mean_outcome_size = np.mean(outcome_sizes)
        mean_stmt_size = np.mean(stmt_sizes)

        r = persona_results[persona]
        total_outcome = r['outcome_attn_per_tok'] * mean_outcome_size
        total_stmt = r['stmt_attn_per_tok'] * mean_stmt_size

        print(f"{persona:<18} {mean_outcome_size:<14.1f} {total_outcome*100:<18.2f}% {mean_stmt_size:<12.1f} {total_stmt*100:<16.2f}% {(total_outcome+total_stmt)*100:<10.2f}%")

    print()
    print("=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print()
    print("If attention to content were constant regardless of persona length:")
    print("  - Total to Outcome should stay ~constant")
    print("  - Total to Statement should stay ~constant")
    print()
    print("If attention dilutes proportionally with sequence length:")
    print("  - Per-token attention × seq_len should be constant")
    print()

    # Check if it's proportional
    baseline = persona_results['baseline']
    baseline_product = baseline['outcome_attn_per_tok'] * baseline['seq_len']

    print(f"{'Persona':<18} {'Attn × SeqLen':<16} {'Ratio to Baseline':<20} {'Interpretation'}")
    print("-" * 80)
    for persona in personas:
        r = persona_results[persona]
        product = r['outcome_attn_per_tok'] * r['seq_len']
        ratio = product / baseline_product

        if ratio > 1.1:
            interp = "MORE attention than proportional"
        elif ratio < 0.9:
            interp = "LESS attention than proportional"
        else:
            interp = "~proportional"

        print(f"{persona:<18} {product:<16.4f} {ratio:<20.3f} {interp}")


if __name__ == "__main__":
    main()
