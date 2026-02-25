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


def extract_attention_layers(
    model,
    tokenizer,
    test_case: Dict,
    persona_prompt: Optional[str],
    layers: List[int],
    device: str = "mps",
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Extract full attention matrices for all heads at specified layers.

    Returns:
        attn: float16 array of shape (n_layers, n_heads, seq_len, seq_len)
        tokens: list of token strings
        meta: dict with regions, test case info, layer indices
    """
    prompt = create_prompt_with_persona(
        test_case['scenario'], test_case['items'], test_case['outcome'],
        test_case['question'], test_case['statement'], persona_prompt
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Extract only requested layers
    attn_list = []
    for layer_idx in layers:
        # Each element is (batch, n_heads, seq, seq)
        attn_list.append(outputs.attentions[layer_idx].squeeze(0).cpu())

    # (n_selected_layers, n_heads, seq, seq)
    attn = torch.stack(attn_list, dim=0).to(torch.float16).numpy()

    regions = find_token_regions(tokenizer, test_case, persona_prompt)
    meta = {
        'test_id': test_case.get('test_id'),
        'category': test_case['category'],
        'statement': test_case['statement'],
        'outcome': test_case['outcome'],
        'tokens': [str(t) for t in tokens],
        'layers': layers,
        'regions': {
            'outcome': list(regions.outcome_span),
            'statement': list(regions.statement_span),
            'scalar_terms': regions.scalar_terms,
            'last_token': regions.last_token,
        },
    }

    return attn, [str(t) for t in tokens], meta


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
    parser.add_argument('--layers', help='Comma-separated layer indices (extracts all heads per layer)')
    parser.add_argument('--test-ids', help='Comma-separated test IDs to extract (bypasses category selection)')
    parser.add_argument('--persona', help='Persona file')
    parser.add_argument('--output', '-o', required=True, help='Output path (.json or .npz)')
    parser.add_argument('--format', choices=['json', 'npz'], default='json',
                        help='Output format: json (per-head) or npz (full layer tensors)')
    parser.add_argument('--categories', nargs='+',
                        default=['true-quant', 'underinf-quant'],
                        help='Categories to sample from')
    parser.add_argument('--n-examples', type=int, default=2, help='Examples per category')

    args = parser.parse_args()

    # Parse layers (for npz format)
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]
        print(f"Extracting all heads at layers: {layers}")

    # Parse heads (for json format)
    heads = []
    if not layers:
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

    # Select examples
    if args.test_ids:
        target_ids = [int(x) for x in args.test_ids.split(',')]
        examples = [tc for tc in test_cases if tc.get('test_id') in target_ids]
        print(f"Selected {len(examples)} examples by test_id: {target_ids}")
    else:
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

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'npz' and layers:
        # NPZ format: save full layer tensors with metadata sidecar
        for i, tc in enumerate(examples):
            print(f"Processing {i+1}/{len(examples)}: {tc['category']} - {tc.get('test_id', i)}")
            attn, tokens, meta = extract_attention_layers(
                model, tokenizer, tc, persona_prompt, layers, args.device
            )

            # For single example, save directly; for multiple, suffix with test_id
            if len(examples) == 1:
                npz_path = args.output
            else:
                stem = Path(args.output).stem
                npz_path = str(Path(args.output).parent / f"{stem}_test{tc.get('test_id', i)}.npz")

            np.savez_compressed(npz_path, attention=attn)

            meta_path = npz_path.replace('.npz', '_meta.json')
            meta['model'] = args.model
            meta['persona'] = args.persona
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            size_mb = Path(npz_path).stat().st_size / 1024 / 1024
            print(f"Saved {npz_path} ({size_mb:.1f} MB) + {meta_path}")

    else:
        # Original JSON format with per-head extraction
        results = []
        for i, tc in enumerate(examples):
            print(f"Processing {i+1}/{len(examples)}: {tc['category']} - {tc.get('test_id', i)}")
            result = extract_attention_for_example(
                model, tokenizer, tc, persona_prompt, heads, args.device
            )
            results.append(result)

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
