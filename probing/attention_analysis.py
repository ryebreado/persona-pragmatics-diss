#!/usr/bin/env python3
"""
Attention pattern analysis for scalar implicature experiments.

Identifies token regions of interest and extracts attention patterns between them.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass


@dataclass
class TokenRegions:
    """Token positions for key regions in the prompt."""
    n_tokens: int
    outcome_span: Tuple[int, int]  # (start, end) token indices
    statement_span: Tuple[int, int]
    scalar_terms: Dict[str, int]  # term -> token index
    last_token: int


def create_prompt_with_persona(scenario, items, outcome, question, statement, persona_prompt=None):
    """Create prompt for scalar implicature evaluation with optional persona."""
    base_prompt = f"""{scenario}
{items}
{outcome}

{question}
Someone answered: "{statement}"

Is this answer correct?
- Respond "yes" if the answer is accurate and complete
- Respond "no" if the answer is wrong or incomplete

Answer (yes/no):"""

    if persona_prompt:
        return f"{persona_prompt}\n\n{base_prompt}"
    else:
        return base_prompt


def find_token_regions(tokenizer, test_case: Dict, persona_prompt: Optional[str] = None) -> TokenRegions:
    """
    Find token positions for key regions in the prompt.

    Regions:
    - Outcome: The factual statement about what happened
    - Statement: The answer being evaluated (inside quotes)
    - Scalar terms: "some", "all", "and" within the statement
    - Last token: The decision point
    """
    prompt = create_prompt_with_persona(
        test_case['scenario'], test_case['items'], test_case['outcome'],
        test_case['question'], test_case['statement'], persona_prompt
    )

    # Tokenize full prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    token_strs = [tokenizer.decode([t]) for t in tokens]

    # Build char-to-token mapping
    char_to_tok = []
    for tok_idx, tok_str in enumerate(token_strs):
        for _ in tok_str:
            char_to_tok.append(tok_idx)

    def char_span_to_token_span(start_char, end_char):
        if start_char >= len(char_to_tok) or start_char < 0:
            return (0, 0)
        end_char = min(end_char, len(char_to_tok))
        return (char_to_tok[start_char], char_to_tok[end_char - 1] + 1)

    # Find OUTCOME: it's on line 3 (index 2) for base prompt
    # With persona, need to account for extra lines
    if persona_prompt:
        persona_lines = persona_prompt.count('\n') + 2  # +2 for blank line separator
        outcome_line_idx = persona_lines + 2
    else:
        outcome_line_idx = 2

    lines = prompt.split('\n')
    if outcome_line_idx < len(lines):
        outcome_line = lines[outcome_line_idx]
        outcome_start_char = prompt.find(outcome_line)
        outcome_end_char = outcome_start_char + len(outcome_line)
        outcome_span = char_span_to_token_span(outcome_start_char, outcome_end_char)
    else:
        outcome_span = (0, 0)

    # Find STATEMENT: it's after 'Someone answered: "'
    statement_marker = 'Someone answered: "'
    statement_start_char = prompt.find(statement_marker) + len(statement_marker)
    statement_end_char = statement_start_char + len(test_case['statement'])
    statement_span = char_span_to_token_span(statement_start_char, statement_end_char)

    # Find SCALAR TERMS within statement region
    scalar_terms = {}
    statement_text = test_case['statement'].lower()
    for term in ['some', 'all', 'and']:
        term_pos = statement_text.find(term)
        if term_pos != -1:
            term_char = statement_start_char + term_pos
            if term_char < len(char_to_tok):
                scalar_terms[term] = char_to_tok[term_char]

    return TokenRegions(
        n_tokens=len(tokens),
        outcome_span=outcome_span,
        statement_span=statement_span,
        scalar_terms=scalar_terms,
        last_token=len(tokens) - 1,
    )


class AttentionExtractor:
    """Extract attention patterns from a model."""

    def __init__(self, model_name: str, device: str = "mps"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="eager",  # Need eager for attention weights
        )
        self.model.eval()
        self.device = device
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        print(f"Loaded: {self.n_layers} layers, {self.n_heads} heads")

    def get_attention_weights(self, prompt: str) -> torch.Tensor:
        """
        Get attention weights for a prompt.

        Returns:
            Tensor of shape (n_layers, n_heads, seq_len, seq_len)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Stack attention from all layers: (n_layers, batch, n_heads, seq, seq)
        attentions = torch.stack(outputs.attentions, dim=0)
        # Remove batch dim: (n_layers, n_heads, seq, seq)
        attentions = attentions.squeeze(1)

        return attentions.cpu()

    def analyze_region_attention(
        self,
        attentions: torch.Tensor,
        regions: TokenRegions,
    ) -> Dict:
        """
        Analyze attention patterns between key regions.

        Returns dict with attention statistics for key region pairs.
        """
        n_layers, n_heads, seq_len, _ = attentions.shape

        results = {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'seq_len': seq_len,
            'regions': {
                'outcome': regions.outcome_span,
                'statement': regions.statement_span,
                'scalar_terms': regions.scalar_terms,
                'last_token': regions.last_token,
            }
        }

        # Key attention patterns to analyze:
        # 1. Last token -> Outcome (does decision attend to facts?)
        # 2. Last token -> Statement (does decision attend to answer?)
        # 3. Last token -> Scalar term (does decision attend to "some"/"all"?)
        # 4. Statement -> Outcome (does answer representation attend to facts?)

        def get_attention_to_region(from_pos: int, to_span: Tuple[int, int]) -> np.ndarray:
            """Get attention from one position to a region, per layer and head."""
            start, end = to_span
            if start >= end or start < 0 or end > seq_len:
                return np.zeros((n_layers, n_heads))
            # Sum attention to all tokens in region
            attn = attentions[:, :, from_pos, start:end].sum(dim=-1).numpy()
            return attn

        def get_region_to_region_attention(from_span: Tuple[int, int], to_span: Tuple[int, int]) -> np.ndarray:
            """Get mean attention from one region to another."""
            from_start, from_end = from_span
            to_start, to_end = to_span
            if from_start >= from_end or to_start >= to_end:
                return np.zeros((n_layers, n_heads))
            # Mean over source positions, sum over target positions
            attn = attentions[:, :, from_start:from_end, to_start:to_end]
            attn = attn.sum(dim=-1).mean(dim=-1).numpy()
            return attn

        # Last token attention patterns
        results['last_to_outcome'] = get_attention_to_region(
            regions.last_token, regions.outcome_span
        ).tolist()

        results['last_to_statement'] = get_attention_to_region(
            regions.last_token, regions.statement_span
        ).tolist()

        # Attention to scalar terms
        for term, pos in regions.scalar_terms.items():
            results[f'last_to_{term}'] = attentions[:, :, regions.last_token, pos].numpy().tolist()

        # Statement to outcome attention
        results['statement_to_outcome'] = get_region_to_region_attention(
            regions.statement_span, regions.outcome_span
        ).tolist()

        return results


def run_attention_analysis(
    model_name: str,
    test_cases: List[Dict],
    persona_prompt: Optional[str] = None,
    device: str = "mps",
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Run attention analysis on a set of test cases.
    """
    extractor = AttentionExtractor(model_name, device)

    results = []
    for i, test_case in enumerate(test_cases):
        print(f"Processing {i+1}/{len(test_cases)}: {test_case['category']}")

        # Create prompt and find regions
        prompt = create_prompt_with_persona(
            test_case['scenario'], test_case['items'], test_case['outcome'],
            test_case['question'], test_case['statement'], persona_prompt
        )
        regions = find_token_regions(extractor.tokenizer, test_case, persona_prompt)

        # Get attention weights
        attentions = extractor.get_attention_weights(prompt)

        # Analyze
        analysis = extractor.analyze_region_attention(attentions, regions)
        analysis['test_id'] = test_case.get('test_id', i)
        analysis['category'] = test_case['category']

        results.append(analysis)

        # Clear cache periodically
        if (i + 1) % 10 == 0:
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Attention pattern analysis')
    parser.add_argument('--model', default='Qwen/Qwen3-8B', help='Model name')
    parser.add_argument('--device', default='mps', help='Device')
    parser.add_argument('--data', default='data/scalar_implicature_250.json', help='Test data')
    parser.add_argument('--persona', help='Persona file')
    parser.add_argument('--output', help='Output JSON path')
    parser.add_argument('--num-examples', type=int, help='Limit examples')

    args = parser.parse_args()

    with open(args.data) as f:
        test_cases = json.load(f)

    if args.num_examples:
        test_cases = test_cases[:args.num_examples]

    persona_prompt = None
    if args.persona:
        with open(args.persona) as f:
            persona_prompt = f.read().strip()

    run_attention_analysis(
        args.model,
        test_cases,
        persona_prompt=persona_prompt,
        device=args.device,
        output_path=args.output,
    )
