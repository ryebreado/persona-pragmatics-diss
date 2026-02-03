#!/usr/bin/env python3
"""
Analyze attention patterns TO persona text.

Examines what parts of the persona the model attends to at decision time,
and whether this differs by persona type or stimulus category.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from attention_analysis import (
    AttentionExtractor,
    create_prompt_with_persona,
)


# Key semantic terms for each persona
PERSONA_KEYWORDS = {
    'pragmaticist': [
        'pragmatics', 'Gricean', 'informative', 'omits', 'maxim', 'quantity',
        'appropriately', 'convey', 'inadequate', 'relevant'
    ],
    'anti_gricean': [
        'truth', 'value', 'literally', 'true', 'complete', 'informative',
        'partial', 'truths', 'matter'
    ],
    'literal_thinker': [
        'true', 'false', 'accurately', 'describes', 'factual', 'accuracy',
        'completeness', 'strictly'
    ],
    'helpful_teacher': [
        'teacher', 'children', 'complete', 'whole', 'story', 'important',
        'information', 'evaluating', 'answers'
    ],
    'soft_literalist': [
        'evaluating', 'focus', 'factually', 'accurate', 'worry', 'detail',
        'true', 'good', 'enough'
    ],
}

# Semantic groupings within personas for finer analysis
PERSONA_SEMANTIC_REGIONS = {
    'pragmaticist': {
        'identity': ['expert', 'pragmatics', 'Gricean'],
        'criterion': ['informative', 'appropriately', 'convey'],
        'violation': ['omits', 'violates', 'maxim', 'quantity', 'inadequate'],
    },
    'anti_gricean': {
        'criterion': ['truth', 'value', 'literally', 'true'],
        'dismissal': ['not', 'matter', 'complete', 'informative'],
        'acceptance': ['partial', 'truths', 'still'],
    },
    'literal_thinker': {
        'criterion': ['true', 'false', 'strictly'],
        'focus': ['accurately', 'describes', 'factual', 'accuracy'],
        'dismissal': ['rather', 'than', 'completeness'],
    },
    'helpful_teacher': {
        'identity': ['teacher', 'children', 'learn'],
        'criterion': ['complete', 'whole', 'story'],
        'focus': ['important', 'information', 'evaluating'],
    },
    'soft_literalist': {
        'criterion': ['factually', 'accurate', 'true'],
        'dismissal': ['worry', 'detail', 'every'],
        'acceptance': ['good', 'enough'],
    },
}


def find_keyword_positions(tokenizer, prompt: str, keywords: List[str]) -> Dict[str, List[int]]:
    """
    Find token positions for keywords in a prompt.

    Returns dict mapping keyword -> list of token indices where it appears.
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    token_strs = [tokenizer.decode([t]).lower() for t in tokens]

    keyword_positions = {}
    for keyword in keywords:
        kw_lower = keyword.lower()
        positions = []
        for i, tok in enumerate(token_strs):
            # Check if keyword is in this token (handles subword tokenization)
            if kw_lower in tok or tok.strip() == kw_lower:
                positions.append(i)
        if positions:
            keyword_positions[keyword] = positions

    return keyword_positions


def find_persona_span(tokenizer, prompt: str, persona_text: str) -> Tuple[int, int]:
    """Find the token span for the persona portion of the prompt."""
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    # Persona is at the start - find where it ends
    persona_tokens = tokenizer.encode(persona_text, add_special_tokens=False)
    return (0, len(persona_tokens))


def analyze_persona_attention(
    extractor: AttentionExtractor,
    test_case: Dict,
    persona_name: str,
    persona_text: str,
) -> Dict:
    """
    Analyze attention to persona regions for a single test case.
    """
    prompt = create_prompt_with_persona(
        test_case['scenario'], test_case['items'], test_case['outcome'],
        test_case['question'], test_case['statement'], persona_text
    )

    # Get attention weights
    attentions = extractor.get_attention_weights(prompt)
    n_layers, n_heads, seq_len, _ = attentions.shape
    last_token = seq_len - 1

    # Find persona span
    persona_span = find_persona_span(extractor.tokenizer, prompt, persona_text)
    persona_start, persona_end = persona_span

    # Find keyword positions within persona
    keywords = PERSONA_KEYWORDS.get(persona_name, [])
    keyword_positions = find_keyword_positions(extractor.tokenizer, prompt, keywords)

    # Filter to only keywords within persona span
    keyword_positions = {
        k: [p for p in v if persona_start <= p < persona_end]
        for k, v in keyword_positions.items()
    }
    keyword_positions = {k: v for k, v in keyword_positions.items() if v}

    # Find semantic region positions
    semantic_regions = PERSONA_SEMANTIC_REGIONS.get(persona_name, {})
    region_positions = {}
    for region_name, region_keywords in semantic_regions.items():
        positions = []
        for kw in region_keywords:
            if kw in keyword_positions:
                positions.extend(keyword_positions[kw])
        if positions:
            region_positions[region_name] = sorted(set(positions))

    results = {
        'test_id': test_case.get('test_id', 0),
        'category': test_case['category'],
        'persona': persona_name,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'seq_len': seq_len,
        'persona_span': persona_span,
        'persona_length': persona_end - persona_start,
    }

    # Measure attention from last token to entire persona
    persona_attn = attentions[:, :, last_token, persona_start:persona_end]
    results['last_to_persona_mean'] = persona_attn.mean(dim=-1).numpy().tolist()
    results['last_to_persona_max'] = persona_attn.max(dim=-1).values.numpy().tolist()
    results['last_to_persona_sum'] = persona_attn.sum(dim=-1).numpy().tolist()

    # Measure attention to each keyword
    keyword_attention = {}
    for keyword, positions in keyword_positions.items():
        # Mean attention across all positions where keyword appears
        kw_attn = attentions[:, :, last_token, positions].mean(dim=-1).numpy()
        keyword_attention[keyword] = kw_attn.tolist()
    results['keyword_attention'] = keyword_attention

    # Measure attention to semantic regions
    region_attention = {}
    for region_name, positions in region_positions.items():
        if positions:
            region_attn = attentions[:, :, last_token, positions].mean(dim=-1).numpy()
            region_attention[region_name] = region_attn.tolist()
    results['region_attention'] = region_attention

    # Also track attention from different positions (not just last token)
    # Measure attention from statement region to persona
    # (Does the model's representation of the statement attend to persona instructions?)

    # Find statement span
    statement_marker = 'Someone answered: "'
    prompt_text = prompt
    statement_start_char = prompt_text.find(statement_marker) + len(statement_marker)

    # Build char-to-token mapping
    tokens = extractor.tokenizer.encode(prompt, add_special_tokens=False)
    token_strs = [extractor.tokenizer.decode([t]) for t in tokens]
    char_to_tok = []
    for tok_idx, tok_str in enumerate(token_strs):
        for _ in tok_str:
            char_to_tok.append(tok_idx)

    if statement_start_char < len(char_to_tok):
        statement_start_tok = char_to_tok[statement_start_char]
        statement_end_char = statement_start_char + len(test_case['statement'])
        statement_end_tok = char_to_tok[min(statement_end_char, len(char_to_tok) - 1)] + 1

        # Statement to persona attention
        stmt_to_persona = attentions[:, :, statement_start_tok:statement_end_tok, persona_start:persona_end]
        results['statement_to_persona_mean'] = stmt_to_persona.mean(dim=(-1, -2)).numpy().tolist()

    return results


def aggregate_by_category(results: List[Dict]) -> Dict:
    """Aggregate attention metrics by category."""
    by_category = defaultdict(list)
    for r in results:
        by_category[r['category']].append(r)

    aggregated = {}
    for category, category_results in by_category.items():
        n = len(category_results)

        # Stack the attention arrays
        persona_mean = np.array([r['last_to_persona_mean'] for r in category_results])

        aggregated[category] = {
            'n_examples': n,
            'last_to_persona_mean': {
                'mean': persona_mean.mean(axis=0).tolist(),
                'std': persona_mean.std(axis=0).tolist(),
            }
        }

        # Aggregate keyword attention
        all_keywords = set()
        for r in category_results:
            all_keywords.update(r['keyword_attention'].keys())

        keyword_agg = {}
        for kw in all_keywords:
            kw_values = [r['keyword_attention'][kw] for r in category_results if kw in r['keyword_attention']]
            if kw_values:
                kw_array = np.array(kw_values)
                keyword_agg[kw] = {
                    'mean': kw_array.mean(axis=0).tolist(),
                    'std': kw_array.std(axis=0).tolist(),
                    'n': len(kw_values),
                }
        aggregated[category]['keyword_attention'] = keyword_agg

        # Aggregate region attention
        all_regions = set()
        for r in category_results:
            all_regions.update(r['region_attention'].keys())

        region_agg = {}
        for region in all_regions:
            region_values = [r['region_attention'][region] for r in category_results if region in r['region_attention']]
            if region_values:
                region_array = np.array(region_values)
                region_agg[region] = {
                    'mean': region_array.mean(axis=0).tolist(),
                    'std': region_array.std(axis=0).tolist(),
                    'n': len(region_values),
                }
        aggregated[category]['region_attention'] = region_agg

    return aggregated


def compare_personas(results_by_persona: Dict[str, List[Dict]]) -> Dict:
    """Compare attention patterns across personas."""
    comparison = {}

    for persona, results in results_by_persona.items():
        # Overall persona attention by layer
        persona_attn = np.array([r['last_to_persona_mean'] for r in results])
        comparison[persona] = {
            'n_examples': len(results),
            'persona_attention_by_layer': persona_attn.mean(axis=0).mean(axis=1).tolist(),  # Mean across heads
            'persona_attention_overall': float(persona_attn.mean()),
        }

        # Keyword attention summary
        keyword_summary = defaultdict(list)
        for r in results:
            for kw, attn in r['keyword_attention'].items():
                # Mean across layers and heads
                keyword_summary[kw].append(np.mean(attn))

        comparison[persona]['keyword_attention_means'] = {
            kw: float(np.mean(vals)) for kw, vals in keyword_summary.items()
        }

        # Region attention summary
        region_summary = defaultdict(list)
        for r in results:
            for region, attn in r['region_attention'].items():
                region_summary[region].append(np.mean(attn))

        comparison[persona]['region_attention_means'] = {
            region: float(np.mean(vals)) for region, vals in region_summary.items()
        }

    return comparison


def run_persona_attention_analysis(
    model_name: str,
    test_cases: List[Dict],
    personas_dir: str,
    device: str = "mps",
    output_dir: Optional[str] = None,
) -> Dict:
    """Run attention analysis for all personas."""
    extractor = AttentionExtractor(model_name, device)

    personas_path = Path(personas_dir)
    persona_files = ['pragmaticist', 'anti_gricean', 'soft_literalist', 'helpful_teacher']

    all_results = {}

    for persona_name in persona_files:
        persona_file = personas_path / persona_name
        if not persona_file.exists():
            print(f"Warning: {persona_file} not found, skipping")
            continue

        persona_text = persona_file.read_text().strip()
        print(f"\n{'='*60}")
        print(f"Analyzing persona: {persona_name}")
        print(f"{'='*60}")

        results = []
        for i, test_case in enumerate(test_cases):
            print(f"  Processing {i+1}/{len(test_cases)}: {test_case['category']}")

            analysis = analyze_persona_attention(
                extractor, test_case, persona_name, persona_text
            )
            results.append(analysis)

            # Clear cache periodically
            if (i + 1) % 10 == 0:
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()

        all_results[persona_name] = results

        # Save per-persona results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            with open(output_path / f"persona_attention_{persona_name}.json", 'w') as f:
                json.dump(results, f, indent=2)

            # Also save category aggregates
            category_agg = aggregate_by_category(results)
            with open(output_path / f"persona_attention_{persona_name}_by_category.json", 'w') as f:
                json.dump(category_agg, f, indent=2)

    # Cross-persona comparison
    comparison = compare_personas(all_results)

    if output_dir:
        with open(Path(output_dir) / "persona_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)

    return {
        'per_persona': all_results,
        'comparison': comparison,
    }


def print_summary(comparison: Dict):
    """Print a summary of persona attention patterns."""
    print("\n" + "="*70)
    print("PERSONA ATTENTION SUMMARY")
    print("="*70)

    for persona, data in comparison.items():
        print(f"\n{persona.upper()}")
        print("-" * 40)
        print(f"  Overall attention to persona: {data['persona_attention_overall']:.6f}")
        print(f"  N examples: {data['n_examples']}")

        # Per-layer attention (show a few key layers)
        layer_attn = data['persona_attention_by_layer']
        n_layers = len(layer_attn)
        print(f"\n  Attention by layer (mean across heads):")
        print(f"    Early (0-3):  {np.mean(layer_attn[:4]):.6f}")
        print(f"    Mid ({n_layers//2-2}-{n_layers//2+1}):   {np.mean(layer_attn[n_layers//2-2:n_layers//2+2]):.6f}")
        print(f"    Late ({n_layers-4}-{n_layers-1}): {np.mean(layer_attn[-4:]):.6f}")

        if data['keyword_attention_means']:
            print(f"\n  Top attended keywords:")
            sorted_kw = sorted(data['keyword_attention_means'].items(), key=lambda x: -x[1])
            for kw, val in sorted_kw[:5]:
                print(f"    {kw}: {val:.6f}")

        if data['region_attention_means']:
            print(f"\n  Semantic region attention:")
            for region, val in sorted(data['region_attention_means'].items(), key=lambda x: -x[1]):
                print(f"    {region}: {val:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze attention to persona text')
    parser.add_argument('--model', default='Qwen/Qwen3-8B', help='Model name')
    parser.add_argument('--device', default='mps', help='Device')
    parser.add_argument('--data', default='data/scalar_implicature_250.json', help='Test data')
    parser.add_argument('--personas', default='personas', help='Personas directory')
    parser.add_argument('--output', default='probing/analysis/persona_attention', help='Output directory')
    parser.add_argument('--num-examples', type=int, help='Limit examples')

    args = parser.parse_args()

    with open(args.data) as f:
        test_cases = json.load(f)

    if args.num_examples:
        test_cases = test_cases[:args.num_examples]

    results = run_persona_attention_analysis(
        args.model,
        test_cases,
        args.personas,
        device=args.device,
        output_dir=args.output,
    )

    print_summary(results['comparison'])
