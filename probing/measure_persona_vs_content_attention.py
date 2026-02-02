#!/usr/bin/env python3
"""
Measure actual proportion of attention to persona vs content (all non-persona tokens).

Extracts full attention rows from model to compute exact proportions.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import from existing module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from attention_analysis import create_prompt_with_persona


def measure_attention_split(
    model,
    tokenizer,
    test_case: dict,
    persona_prompt: str | None,
    baseline_len: int,
    device: str = "mps",
) -> dict:
    """
    Measure what fraction of last_token attention goes to persona vs content.

    Returns dict with attention proportions per layer.
    """
    prompt = create_prompt_with_persona(
        test_case['scenario'], test_case['items'], test_case['outcome'],
        test_case['question'], test_case['statement'], persona_prompt
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs['input_ids'].shape[1]

    # Persona tokens are at the start
    if persona_prompt:
        persona_tokens = seq_len - baseline_len
    else:
        persona_tokens = 0

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack attention: (n_layers, n_heads, seq, seq)
    attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)

    n_layers = attentions.shape[0]
    last_token = seq_len - 1

    # Get attention from last token to all previous positions
    # Shape: (n_layers, n_heads, seq_len)
    last_attn = attentions[:, :, last_token, :last_token + 1].cpu().numpy()

    results = {
        'seq_len': seq_len,
        'persona_tokens': persona_tokens,
        'content_tokens': seq_len - persona_tokens,
        'by_layer': [],
    }

    for layer in range(n_layers):
        # Average over heads
        attn_row = last_attn[layer].mean(axis=0)  # (seq_len,)

        # Sum attention to persona vs content
        if persona_tokens > 0:
            to_persona = attn_row[:persona_tokens].sum()
            to_content = attn_row[persona_tokens:].sum()
        else:
            to_persona = 0.0
            to_content = attn_row.sum()

        results['by_layer'].append({
            'layer': layer,
            'to_persona': float(to_persona),
            'to_content': float(to_content),
        })

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--data', default='data/scalar_implicature_250.json')
    parser.add_argument('--n-examples', type=int, default=10)

    args = parser.parse_args()

    # Load test cases
    with open(args.data) as f:
        test_cases = json.load(f)

    # Sample a few examples
    test_cases = test_cases[:args.n_examples]

    # Load personas
    personas_dir = Path('personas')
    personas = {
        'baseline': None,
        'anti_gricean': (personas_dir / 'anti_gricean').read_text().strip(),
        'literal_thinker': (personas_dir / 'literal_thinker').read_text().strip(),
        'helpful_teacher': (personas_dir / 'helpful_teacher').read_text().strip(),
        'pragmaticist': (personas_dir / 'pragmaticist').read_text().strip(),
    }

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

    # Get baseline length
    baseline_prompt = create_prompt_with_persona(
        test_cases[0]['scenario'], test_cases[0]['items'], test_cases[0]['outcome'],
        test_cases[0]['question'], test_cases[0]['statement'], None
    )
    baseline_len = len(tokenizer.encode(baseline_prompt, add_special_tokens=False))

    print(f"Baseline prompt length: {baseline_len} tokens")
    print()

    # Measure for each persona
    for persona_name, persona_prompt in personas.items():
        print(f"{'='*60}")
        print(f"Persona: {persona_name.upper()}")
        print(f"{'='*60}")

        all_results = []
        for i, tc in enumerate(test_cases):
            result = measure_attention_split(
                model, tokenizer, tc, persona_prompt, baseline_len, args.device
            )
            all_results.append(result)

            if i == 0:
                print(f"Seq len: {result['seq_len']}, Persona tokens: {result['persona_tokens']}, Content tokens: {result['content_tokens']}")

        # Average across examples
        n_layers = len(all_results[0]['by_layer'])

        print()
        print(f"{'Layer':<8} {'To Persona':<14} {'To Content':<14} {'% to Content':<14}")
        print("-" * 50)

        layer_means = []
        for layer in range(n_layers):
            to_persona = np.mean([r['by_layer'][layer]['to_persona'] for r in all_results])
            to_content = np.mean([r['by_layer'][layer]['to_content'] for r in all_results])
            pct_content = to_content / (to_persona + to_content) * 100 if (to_persona + to_content) > 0 else 0

            layer_means.append({'to_persona': to_persona, 'to_content': to_content, 'pct_content': pct_content})

            if layer < 12 or layer % 6 == 0:  # Show early layers and every 6th
                print(f"{layer:<8} {to_persona:<14.4f} {to_content:<14.4f} {pct_content:<14.1f}%")

        # Overall mean
        mean_to_persona = np.mean([lm['to_persona'] for lm in layer_means])
        mean_to_content = np.mean([lm['to_content'] for lm in layer_means])
        mean_pct = mean_to_content / (mean_to_persona + mean_to_content) * 100

        print("-" * 50)
        print(f"{'MEAN':<8} {mean_to_persona:<14.4f} {mean_to_content:<14.4f} {mean_pct:<14.1f}%")
        print()

        # Print as Python list for copy-paste
        content_fracs = [lm['to_content'] for lm in layer_means]
        print(f"'{persona_name}': {[round(f, 3) for f in content_fracs]},")
        print()

        # Clear cache
        if args.device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
