#!/usr/bin/env python3
"""
Extract full attention matrices for specific examples and heads.

Saves raw attention data for visualization (heatmaps, BertViz-style).
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from attention_analysis import create_prompt_with_persona, find_token_regions


def extract_attention_for_example(
    model,
    tokenizer,
    test_case: Dict,
    persona_prompt: Optional[str],
    heads: List[Tuple[int, int]],
    device: str = "mps",
) -> Dict:
    """
    Extract full attention matrices for specific heads on one example.

    Returns dict with token info and attention matrices per head.
    """
    prompt = create_prompt_with_persona(
        test_case['scenario'], test_case['items'], test_case['outcome'],
        test_case['question'], test_case['statement'], persona_prompt
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack attention: (n_layers, n_heads, seq, seq)
    attentions = torch.stack(outputs.attentions, dim=0).squeeze(1).cpu()

    regions = find_token_regions(tokenizer, test_case, persona_prompt)

    result = {
        'test_id': test_case.get('test_id'),
        'category': test_case['category'],
        'statement': test_case['statement'],
        'outcome': test_case['outcome'],
        'tokens': tokens,
        'regions': {
            'outcome': regions.outcome_span,
            'statement': regions.statement_span,
            'scalar_terms': regions.scalar_terms,
            'last_token': regions.last_token,
        },
        'heads': {},
    }

    for layer, head in heads:
        attn_matrix = attentions[layer, head].numpy()
        result['heads'][f'L{layer}H{head}'] = attn_matrix.tolist()

    return result


def select_examples(
    test_cases: List[Dict],
    categories: List[str],
    n_per_category: int = 2,
) -> List[Dict]:
    """Select a few examples from each category."""
    from collections import defaultdict

    by_category = defaultdict(list)
    for tc in test_cases:
        if tc['category'] in categories:
            by_category[tc['category']].append(tc)

    selected = []
    for cat in categories:
        selected.extend(by_category[cat][:n_per_category])

    return selected


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract attention matrices for visualization')
    parser.add_argument('--model', default='Qwen/Qwen3-8B', help='Model name')
    parser.add_argument('--device', default='mps', help='Device')
    parser.add_argument('--data', default='data/scalar_implicature_250.json', help='Test data')
    parser.add_argument('--head-analysis', help='Path to head_analysis.json for top heads')
    parser.add_argument('--heads', nargs='+', help='Heads to extract as L#H# (e.g., L18H12)')
    parser.add_argument('--persona', help='Persona file')
    parser.add_argument('--output', '-o', required=True, help='Output JSON path')
    parser.add_argument('--categories', nargs='+',
                        default=['true-quant', 'underinf-quant'],
                        help='Categories to sample from')
    parser.add_argument('--n-examples', type=int, default=2, help='Examples per category')

    args = parser.parse_args()

    # Parse heads
    heads = []
    if args.head_analysis:
        with open(args.head_analysis) as f:
            analysis = json.load(f)
        for h in analysis['top_heads']:
            heads.append((h['layer'], h['head']))
        print(f"Using top heads from analysis: {heads}")
    elif args.heads:
        for h in args.heads:
            # Parse L#H# format
            parts = h.upper().replace('L', '').split('H')
            heads.append((int(parts[0]), int(parts[1])))
        print(f"Using specified heads: {heads}")
    else:
        # Default to some interesting heads
        heads = [(18, 12), (24, 18), (3, 20)]
        print(f"Using default heads: {heads}")

    # Load data
    with open(args.data) as f:
        test_cases = json.load(f)

    examples = select_examples(test_cases, args.categories, args.n_examples)
    print(f"Selected {len(examples)} examples from categories: {args.categories}")

    # Load persona
    persona_prompt = None
    if args.persona:
        with open(args.persona) as f:
            persona_prompt = f.read().strip()
        print(f"Using persona: {args.persona}")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()

    # Extract attention
    results = []
    for i, tc in enumerate(examples):
        print(f"Processing {i+1}/{len(examples)}: {tc['category']} - {tc.get('test_id', i)}")
        result = extract_attention_for_example(
            model, tokenizer, tc, persona_prompt, heads, args.device
        )
        results.append(result)

    # Save
    output = {
        'model': args.model,
        'persona': args.persona,
        'heads': [{'layer': l, 'head': h} for l, h in heads],
        'examples': results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
