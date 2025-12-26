#!/usr/bin/env python3
"""
Local evaluation of scalar implicature with personas.
Supports activation tracking and runs on MPS/CUDA/CPU.
Production-ready for both test runs and full experiments.
"""

import json
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from pathlib import Path


class LocalModelWithActivations:
    """Wrapper for local model that can optionally capture activations"""

    def __init__(self, model_name: str, device: str = "mps"):
        """
        Load model and tokenizer.

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
            device: Device to use ("mps", "cuda", or "cpu")
        """
        print(f"Loading model: {model_name}")
        print(f"Target device: {device}")

        # Check device availability
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"

        self.device = device
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model loading kwargs
        model_kwargs = {
            "low_cpu_mem_usage": True
        }

        # Note: Quantization (load_in_8bit/load_in_4bit) doesn't work on Mac/MPS
        # Only use normal loading
        if device != "cpu":
            model_kwargs["dtype"] = torch.float16
            model_kwargs["device_map"] = device
        else:
            model_kwargs["dtype"] = torch.float32

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()

        # Storage for activations
        self.activations = {}
        self.hooks = []
        self.n_layers = self.model.config.num_hidden_layers

        print(f"Model loaded successfully on {self.device}")
        print(f"Model has {self.n_layers} layers")

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Register forward hooks to capture activations.

        Args:
            layer_indices: Which layers to capture (None = all layers)
        """
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))

        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                # Store the hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                # Detach and move to CPU to save GPU memory
                self.activations[f"layer_{layer_idx}"] = hidden_states.detach().cpu()
            return hook_fn

        # Clear existing hooks
        self.remove_hooks()

        # Register new hooks
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(create_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def get_yes_no_logprobs(self, prompt: str) -> Dict[str, float]:
        """
        Get log probabilities for "yes" and "no" tokens after the prompt.

        Args:
            prompt: Input prompt

        Returns:
            Dictionary with 'yes' and 'no' logprobs
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Get token IDs for yes/no
        # Try common variations of yes/no tokens
        yes_tokens = self.tokenizer.encode(" yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode(" no", add_special_tokens=False)
        yes_token_id = yes_tokens[0] if yes_tokens else None
        no_token_id = no_tokens[0] if no_tokens else None

        # Also try without leading space
        if yes_token_id is None:
            yes_tokens = self.tokenizer.encode("yes", add_special_tokens=False)
            yes_token_id = yes_tokens[0] if yes_tokens else None
        if no_token_id is None:
            no_tokens = self.tokenizer.encode("no", add_special_tokens=False)
            no_token_id = no_tokens[0] if no_tokens else None

        if yes_token_id is None or no_token_id is None:
            raise ValueError("Could not find token IDs for 'yes' and 'no'")

        # Get logits for next token
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token's logits

        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        yes_logprob = log_probs[yes_token_id].item()
        no_logprob = log_probs[no_token_id].item()

        return {
            'yes': yes_logprob,
            'no': no_logprob,
            'yes_token_id': yes_token_id,
            'no_token_id': no_token_id
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        track_activations: bool = False,
        layer_indices: Optional[List[int]] = None,
        get_logprobs: bool = False
    ) -> Dict:
        """
        Generate response from the model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            track_activations: Whether to capture activations
            layer_indices: Which layers to track (None = all)
            get_logprobs: Whether to get yes/no logprobs

        Returns:
            dict with 'text', and optionally 'activations' and 'logprobs'
        """
        # Get logprobs if requested (before generation)
        logprobs_data = None
        if get_logprobs:
            try:
                logprobs_data = self.get_yes_no_logprobs(prompt)
            except Exception as e:
                print(f"Warning: Could not get logprobs: {e}")

        # Register hooks if tracking activations
        if track_activations:
            self.register_hooks(layer_indices)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            if temperature == 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        # Decode response
        response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

        result = {
            'text': response,
            'prompt_length': prompt_length,
            'generated_length': outputs.shape[1] - prompt_length
        }

        # Include logprobs if requested
        if logprobs_data:
            result['logprobs'] = logprobs_data

        # Include activations if tracked
        if track_activations and self.activations:
            # Store activations as numpy arrays
            result['activations'] = {
                layer_name: acts.numpy() for layer_name, acts in self.activations.items()
            }
            # Clean up
            self.remove_hooks()

        return result


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


def evaluate_response(response_text: str, expected_answer: str) -> bool:
    """
    Check if the response matches expected answer.
    Uses word boundaries to avoid false positives like "no" in "Note".
    """
    import re

    response_lower = response_text.lower().strip()
    expected_lower = expected_answer.lower().strip()

    # First, check for word-boundary matches of yes/no in the first 50 chars
    # This avoids matching "no" in "Note" or "yes" in "yesterday"
    first_part = response_lower[:50]

    if expected_lower == "yes":
        # Look for "yes" as a complete word
        if re.search(r'\byes\b', first_part):
            return True
        # Also check for affirmative variations
        if re.search(r'\b(correct|true|right)\b', response_lower):
            return True
    elif expected_lower == "no":
        # Look for "no" as a complete word (not in "note", "know", etc.)
        if re.search(r'\bno\b', first_part):
            return True
        # Also check for negative variations
        if re.search(r'\b(incorrect|false|wrong)\b', response_lower):
            return True

    return False


def evaluate_with_logprobs(yes_logprob: float, no_logprob: float, expected_answer: str) -> Tuple[bool, float]:
    """
    Evaluate based on yes/no logprobs.

    Args:
        yes_logprob: Log probability of "yes" token
        no_logprob: Log probability of "no" token
        expected_answer: Expected answer ("yes" or "no")

    Returns:
        Tuple of (is_correct, confidence)
    """
    # Convert logprobs to probabilities
    yes_prob = np.exp(yes_logprob)
    no_prob = np.exp(no_logprob)

    # Normalize (they should already be normalized, but just in case)
    total = yes_prob + no_prob
    yes_prob = yes_prob / total
    no_prob = no_prob / total

    # Determine prediction based on higher probability
    prediction = "yes" if yes_prob > no_prob else "no"
    confidence = max(yes_prob, no_prob)

    is_correct = (prediction == expected_answer.lower())

    return is_correct, confidence


def run_single_test(
    model_wrapper: LocalModelWithActivations,
    test_case: Dict,
    temperature: float = 0.0,
    track_activations: bool = False,
    persona_prompt: Optional[str] = None,
    layer_indices: Optional[List[int]] = None,
    save_activations: bool = False,
    use_logprobs: bool = False
) -> Optional[Dict]:
    """
    Run a single test case.

    Args:
        model_wrapper: Model wrapper instance
        test_case: Test case dictionary
        temperature: Sampling temperature
        track_activations: Whether to track activations
        persona_prompt: Optional persona prompt
        layer_indices: Which layers to track
        save_activations: Whether to save full activations in result
        use_logprobs: Whether to use logprobs for evaluation

    Returns:
        Dictionary with test results
    """
    prompt = create_prompt_with_persona(
        test_case['scenario'],
        test_case['items'],
        test_case['outcome'],
        test_case['question'],
        test_case['statement'],
        persona_prompt
    )

    try:
        response_data = model_wrapper.generate(
            prompt,
            max_new_tokens=50,
            temperature=temperature,
            track_activations=track_activations,
            layer_indices=layer_indices,
            get_logprobs=use_logprobs
        )

        response = response_data['text']

        # Use logprobs for evaluation if available
        if use_logprobs and 'logprobs' in response_data:
            logprobs = response_data['logprobs']
            is_correct, confidence = evaluate_with_logprobs(
                logprobs['yes'],
                logprobs['no'],
                test_case['expected']
            )
        else:
            # Fall back to text-based evaluation
            is_correct = evaluate_response(response, test_case['expected'])
            confidence = None

        result = {
            'test_id': test_case.get('test_id', 'unknown'),
            'response': response,
            'expected': test_case['expected'],
            'correct': is_correct,
            'category': test_case['category']
        }

        # Include confidence if using logprobs
        if confidence is not None:
            result['confidence'] = confidence

        # Include logprobs data if available
        if use_logprobs and 'logprobs' in response_data:
            result['yes_logprob'] = response_data['logprobs']['yes']
            result['no_logprob'] = response_data['logprobs']['no']

        # Include activation info if tracked
        if track_activations:
            if save_activations and 'activations' in response_data:
                # Save full activations (can be large!)
                result['activations'] = response_data['activations']
            elif 'activations' in response_data:
                # Just save activation shapes for reference
                result['activation_shapes'] = {
                    layer: arr.shape for layer, arr in response_data['activations'].items()
                }

        return result

    except Exception as e:
        print(f"Error processing test case: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_output_filename(model_name: str, persona_file: Optional[str] = None,
                            persona_text: Optional[str] = None,
                            track_activations: bool = False) -> str:
    """Generate automatic filename based on model and persona"""
    # Clean model name
    model_clean = model_name.split('/')[-1].replace('-', '_').replace('.', '_')

    # Get persona identifier
    if persona_file:
        persona_name = os.path.splitext(os.path.basename(persona_file))[0]
    elif persona_text:
        words = persona_text.split()[:3]
        persona_name = '_'.join(w.lower().replace('.', '').replace(',', '') for w in words)
    else:
        persona_name = "baseline"

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add activation flag
    act_suffix = "_with_acts" if track_activations else ""

    filename = f"scalar_implicature_{model_clean}_{persona_name}{act_suffix}_{timestamp}.json"
    return os.path.join("results", filename)


def main():
    parser = argparse.ArgumentParser(
        description='Local evaluation with optional activation tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('test_file', help='JSON file with test cases')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct',
                       help='HuggingFace model name')
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'],
                       help='Device to use (note: large models may need cpu)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--persona', type=str, help='Persona prompt to prepend')
    parser.add_argument('--persona-file', type=str, help='File containing persona prompt')
    parser.add_argument('--track-activations', action='store_true',
                       help='Track model activations during inference')
    parser.add_argument('--save-activations', action='store_true',
                       help='Save full activation arrays (can be very large!)')
    parser.add_argument('--use-logprobs', action='store_true',
                       help='Use logprobs of yes/no tokens for evaluation (more robust for small models)')
    parser.add_argument('--layers', type=str,
                       help='Comma-separated layer indices to track (e.g., "0,5,10")')
    parser.add_argument('--num-examples', type=int, default=None,
                       help='Number of examples to run (default: all)')
    parser.add_argument('--output', help='Output file for results (auto-generated if not specified)')
    parser.add_argument('--verbose', action='store_true', help='Show full responses')

    args = parser.parse_args()

    # Load persona prompt
    persona_prompt = None
    if args.persona:
        persona_prompt = args.persona
    elif args.persona_file:
        try:
            with open(args.persona_file, 'r') as f:
                persona_prompt = f.read().strip()
        except FileNotFoundError:
            print(f"Error: Persona file not found: {args.persona_file}")
            sys.exit(1)

    # Load test cases
    try:
        with open(args.test_file, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test file not found: {args.test_file}")
        sys.exit(1)

    # Limit number of examples if specified
    if args.num_examples:
        test_cases = test_cases[:args.num_examples]
        print(f"Running {args.num_examples} example(s) for testing")

    # Parse layer indices
    layer_indices = None
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(',')]

    # Initialize model
    print("="*80)
    model_wrapper = LocalModelWithActivations(args.model, device=args.device)
    print("="*80)

    # Run tests
    results = []
    correct_by_category = {'true': 0, 'false': 0, 'underinformative': 0}
    total_by_category = {'true': 0, 'false': 0, 'underinformative': 0}

    print(f"\nRunning {len(test_cases)} scalar implicature test(s) with {args.model}...")
    if persona_prompt:
        preview = persona_prompt[:100] + ('...' if len(persona_prompt) > 100 else '')
        print(f"Using persona: {preview}")
    if args.use_logprobs:
        print("Evaluation method: Logprobs (yes/no token probabilities)")
    else:
        print("Evaluation method: Text matching")
    if args.track_activations:
        print("Activation tracking: ENABLED")
        if layer_indices:
            print(f"  Tracking layers: {layer_indices}")
        else:
            print(f"  Tracking all {model_wrapper.n_layers} layers")
        if args.save_activations:
            print("  WARNING: Saving full activations (output will be large!)")
    print("="*80)

    for i, test_case in enumerate(test_cases):
        test_id = test_case.get('test_id', i+1)
        print(f"Test {test_id}: {test_case['category']}")

        result = run_single_test(
            model_wrapper,
            test_case,
            args.temperature,
            args.track_activations,
            persona_prompt,
            layer_indices,
            args.save_activations,
            args.use_logprobs
        )

        if result:
            results.append(result)

            category = result['category']
            total_by_category[category] += 1
            if result['correct']:
                correct_by_category[category] += 1

            status = "✓" if result['correct'] else "✗"

            # Build confidence string if available
            confidence_str = ""
            if 'confidence' in result:
                confidence_str = f" (confidence: {result['confidence']:.3f})"
            elif args.use_logprobs and 'yes_logprob' in result:
                # Show raw logprobs
                confidence_str = f" (yes: {result['yes_logprob']:.2f}, no: {result['no_logprob']:.2f})"

            if args.verbose:
                print(f"  {status} Expected: {result['expected']}, Got: {result['response']}{confidence_str}")
            else:
                response_preview = result['response'][:50].replace('\n', ' ')
                print(f"  {status} Expected: {result['expected']}, Got: {response_preview}...{confidence_str}")

        print()

    # Calculate statistics
    total_correct = sum(correct_by_category.values())
    total_tests = sum(total_by_category.values())

    print("="*80)
    print("SCALAR IMPLICATURE EVALUATION RESULTS")
    if persona_prompt:
        preview = persona_prompt[:100] + ('...' if len(persona_prompt) > 100 else '')
        print(f"Persona: {preview}")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    if total_tests > 0:
        print(f"Overall Accuracy: {total_correct/total_tests:.3f} ({total_correct}/{total_tests})")
        print()

        for category in ['true', 'false', 'underinformative']:
            if total_by_category[category] > 0:
                accuracy = correct_by_category[category] / total_by_category[category]
                print(f"{category.capitalize()} statements: {accuracy:.3f} "
                      f"({correct_by_category[category]}/{total_by_category[category]})")

                # Show average confidence for this category if using logprobs
                if args.use_logprobs:
                    category_results = [r for r in results if r['category'] == category and 'confidence' in r]
                    if category_results:
                        avg_confidence = sum(r['confidence'] for r in category_results) / len(category_results)
                        print(f"  Average confidence: {avg_confidence:.3f}")

        # Overall average confidence
        if args.use_logprobs:
            confidence_results = [r for r in results if 'confidence' in r]
            if confidence_results:
                overall_confidence = sum(r['confidence'] for r in confidence_results) / len(confidence_results)
                print(f"\nOverall average confidence: {overall_confidence:.3f}")

    # Generate output filename if not specified
    if not args.output:
        args.output = generate_output_filename(
            args.model, args.persona_file, args.persona, args.track_activations
        )

    # Ensure results directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    output_data = {
        'model': args.model,
        'device': args.device,
        'temperature': args.temperature,
        'persona_prompt': persona_prompt,
        'persona_file': args.persona_file,
        'use_logprobs': args.use_logprobs,
        'track_activations': args.track_activations,
        'tracked_layers': layer_indices,
        'total_tests': total_tests,
        'total_accuracy': total_correct/total_tests if total_tests > 0 else 0,
        'accuracy_by_category': {
            cat: correct_by_category[cat]/total_by_category[cat]
            for cat in correct_by_category if total_by_category[cat] > 0
        },
        'results': results
    }

    # Add confidence statistics if using logprobs
    if args.use_logprobs:
        confidence_by_category = {}
        for category in ['true', 'false', 'underinformative']:
            category_results = [r for r in results if r['category'] == category and 'confidence' in r]
            if category_results:
                avg_confidence = sum(r['confidence'] for r in category_results) / len(category_results)
                confidence_by_category[category] = avg_confidence

        output_data['confidence_by_category'] = confidence_by_category

        confidence_results = [r for r in results if 'confidence' in r]
        if confidence_results:
            output_data['overall_confidence'] = sum(r['confidence'] for r in confidence_results) / len(confidence_results)

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
