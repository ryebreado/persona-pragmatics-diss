#!/usr/bin/env python3
import json
import argparse
import sys
import os
from datetime import datetime
from llm_client import query_llm, _get_provider

def create_prompt_with_persona(scenario, items, outcome, question, statement, persona_prompt=None):
    """
    Create a prompt for scalar implicature evaluation with optional persona.
    
    Args:
        scenario: The setup (e.g., "The monkey loves eating yummy stuff.")
        items: Available items (e.g., "There's a banana, a cake, an orange and a biscuit")
        outcome: What actually happened (e.g., "The monkey ate the orange and the biscuit")
        question: The experimenter's question (e.g., "So, what did the monkey eat?")
        statement: The statement to evaluate (e.g., "The monkey ate the biscuit")
        persona_prompt: Optional persona/system prompt to prepend
    
    Returns:
        str: Complete prompt for the LLM
    """
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

def evaluate_response(response_text, expected_answer):
    """
    Check if the LLM's response matches the expected answer.
    
    Args:
        response_text: The LLM's response
        expected_answer: Expected "yes" or "no"
    
    Returns:
        bool: True if correct, False otherwise
    """
    response_lower = response_text.lower().strip()
    expected_lower = expected_answer.lower().strip()
    
    # Handle various response formats
    if expected_lower in response_lower:
        return True
    
    # Check for common variations
    if expected_lower == "no" and any(word in response_lower for word in ["incorrect", "false", "wrong"]):
        return True
    if expected_lower == "yes" and any(word in response_lower for word in ["correct", "true", "right"]):
        return True
        
    return False

def extract_yes_no_confidence(logprobs_data, response_text):
    """
    Extract confidence scores for yes/no responses from logprobs.
    
    Args:
        logprobs_data: Logprobs data from the API
        response_text: The actual response text
    
    Returns:
        dict: Confidence information including probabilities for yes/no
    """
    import numpy as np
    
    if not logprobs_data or not logprobs_data.content:
        return None
    
    # Look for the first token that contains yes/no
    for token_data in logprobs_data.content:
        token = token_data.token.lower().strip()
        
        if 'yes' in token or 'no' in token:
            confidence_data = {
                'chosen_token': token_data.token,
                'chosen_logprob': token_data.logprob,
                'chosen_probability': np.exp(token_data.logprob),
                'alternatives': []
            }
            
            # Add top alternatives
            if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                for alt in token_data.top_logprobs:
                    confidence_data['alternatives'].append({
                        'token': alt.token,
                        'logprob': alt.logprob,
                        'probability': np.exp(alt.logprob)
                    })
            
            return confidence_data
    
    return None

def run_single_test(test_case, model, temperature=0.0, use_logprobs=False, persona_prompt=None):
    """
    Run a single scalar implicature test with optional persona.
    
    Args:
        test_case: Dictionary with test case data
        model: Model to use
        temperature: Sampling temperature
        use_logprobs: Whether to return logprobs data
        persona_prompt: Optional persona prompt to prepend
    
    Returns:
        dict: Results including correctness and optionally logprobs
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
        if use_logprobs:
            response_data = query_llm(prompt, model, return_usage=True, logprobs=True, temperature=temperature)
            response = response_data['text']
            logprobs_data = response_data.get('logprobs')
        else:
            response = query_llm(prompt, model, temperature=temperature)
            logprobs_data = None
        
        is_correct = evaluate_response(response, test_case['expected'])
        
        result = {
            'test_id': test_case.get('test_id', 'unknown'),
            'prompt': prompt,
            'response': response,
            'expected': test_case['expected'],
            'correct': is_correct,
            'category': test_case['category']
        }
        
        if use_logprobs and logprobs_data:
            result['logprobs'] = extract_yes_no_confidence(logprobs_data, response)
        
        return result
    except Exception as e:
        print(f"Error processing test case: {e}")
        return None

def get_next_run_number(model_clean, results_base="results"):
    """Find the next available run number for a model."""
    from glob import glob

    # Look for existing directories like {model}_run_01, {model}_run_02, etc.
    pattern = os.path.join(results_base, f"{model_clean}_run_*")
    existing = glob(pattern)

    if not existing:
        return 1

    # Extract run numbers and find max
    run_numbers = []
    for path in existing:
        dirname = os.path.basename(path)
        # Extract number after "_run_"
        parts = dirname.split("_run_")
        if len(parts) == 2:
            try:
                run_numbers.append(int(parts[1]))
            except ValueError:
                pass

    return max(run_numbers) + 1 if run_numbers else 1


def get_or_create_run_dir(model, results_base="results", new_run=False):
    """
    Get existing run directory or create new one.

    Args:
        model: Model name
        results_base: Base directory for results
        new_run: If True, always create a new run directory

    Returns the run directory path.
    """
    from glob import glob

    # Clean model name for directory
    model_clean = model.replace('/', '_').replace('-', '_').replace('.', '_')

    pattern = os.path.join(results_base, f"{model_clean}_run_*")
    existing_dirs = sorted(glob(pattern))

    # Reuse most recent directory unless explicitly starting a new run
    if existing_dirs and not new_run:
        return existing_dirs[-1]

    # Create new run directory
    run_num = get_next_run_number(model_clean, results_base)
    run_dir = os.path.join(results_base, f"{model_clean}_run_{run_num:02d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def generate_output_filename(model, persona_file=None, persona_text=None, new_run=False):
    """Generate automatic filename based on model and persona"""
    # Clean model name for filename
    model_clean = model.replace('/', '_').replace('-', '_').replace('.', '_')

    # Get or create run directory
    run_dir = get_or_create_run_dir(model, new_run=new_run)

    # Get persona identifier
    if persona_file:
        persona_name = os.path.splitext(os.path.basename(persona_file))[0]
    elif persona_text:
        # Use first few words of persona text
        words = persona_text.split()[:3]
        persona_name = '_'.join(w.lower().replace('.', '').replace(',', '') for w in words)
    else:
        persona_name = "baseline"

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename with scalar_implicature prefix to match existing convention
    filename = f"scalar_implicature_{model_clean}_{persona_name}_{timestamp}.json"
    return os.path.join(run_dir, filename)

def validate_api_key(model):
    """
    Validate that the required API key is set before running tests.

    Args:
        model: Model name to check

    Raises:
        SystemExit: If required API key is not set
    """
    provider = _get_provider(model)

    env_var_map = {
        'anthropic': 'ANTHROPIC_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'together': 'TOGETHERAI_API_KEY'
    }

    env_var = env_var_map.get(provider)
    if env_var and not os.getenv(env_var):
        print(f"Error: {env_var} environment variable not set", file=sys.stderr)
        print(f"Please set {env_var} before running tests with {model}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Evaluate scalar implicature understanding with persona prompts')
    parser.add_argument('test_file', help='JSON file with test cases')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--logprobs', action='store_true', help='Include logprobs and confidence analysis')
    parser.add_argument('--persona', type=str, help='Persona prompt to prepend to all test cases')
    parser.add_argument('--persona-file', type=str, help='File containing persona prompt')
    parser.add_argument('--output', help='Output file for results (auto-generated if not specified)')
    parser.add_argument('--verbose', action='store_true', help='Show full responses in output')
    parser.add_argument('--new-run', action='store_true', help='Start a new run instead of continuing the most recent one')

    args = parser.parse_args()

    # Validate API key before starting
    validate_api_key(args.model)
    
    # Load persona prompt
    persona_prompt = None
    if args.persona:
        persona_prompt = args.persona
    elif args.persona_file:
        try:
            with open(args.persona_file, 'r') as f:
                persona_prompt = f.read().strip()
        except FileNotFoundError:
            print(f"Persona file not found: {args.persona_file}")
            sys.exit(1)
    
    # Load test cases
    try:
        with open(args.test_file, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Test file not found: {args.test_file}")
        sys.exit(1)
    
    results = []
    # Track both fine-grained categories and high-level groupings
    correct_by_category = {}  # Fine-grained (e.g., 'true-conj', 'underinf-quant')
    total_by_category = {}
    # High-level groupings: true, false, underinformative
    correct_by_group = {'true': 0, 'false': 0, 'underinformative': 0}
    total_by_group = {'true': 0, 'false': 0, 'underinformative': 0}

    print(f"Running {len(test_cases)} scalar implicature tests with {args.model}...")
    if persona_prompt:
        print(f"Using persona: {persona_prompt[:100]}{'...' if len(persona_prompt) > 100 else ''}")
    print("="*80)

    # Track timing
    run_start_time = datetime.now()

    consecutive_failures = 0
    max_consecutive_failures = 3

    for i, test_case in enumerate(test_cases):
        test_id = test_case.get('test_id', i+1)
        print(f"Test {test_id}: {test_case['category']}")

        result = run_single_test(test_case, args.model, args.temperature, args.logprobs, persona_prompt)
        if result is None:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"\nError: {max_consecutive_failures} consecutive test failures. "
                      "This usually indicates an API configuration issue.", file=sys.stderr)
                print("Aborting to avoid creating empty results file.", file=sys.stderr)
                sys.exit(1)
            continue

        consecutive_failures = 0  # Reset on success
        if result:
            results.append(result)

            category = result['category']
            # Initialize category counters if not seen before
            if category not in total_by_category:
                total_by_category[category] = 0
                correct_by_category[category] = 0
            total_by_category[category] += 1
            if result['correct']:
                correct_by_category[category] += 1

            # Map to high-level group (true, false, underinformative)
            if category.startswith('true'):
                group = 'true'
            elif category.startswith('false'):
                group = 'false'
            elif category.startswith('underinf'):
                group = 'underinformative'
            else:
                group = category  # Fallback for legacy categories
                if group not in total_by_group:
                    total_by_group[group] = 0
                    correct_by_group[group] = 0
            total_by_group[group] += 1
            if result['correct']:
                correct_by_group[group] += 1
            
            status = "✓" if result['correct'] else "✗"
            confidence_str = ""
            if args.logprobs and 'logprobs' in result and result['logprobs']:
                prob = result['logprobs']['chosen_probability']
                confidence_str = f" (confidence: {prob:.3f})"
            
            if args.verbose:
                print(f"  {status} Expected: {result['expected']}, Got: {result['response']}{confidence_str}")
            else:
                response_preview = result['response'][:50].replace('\n', ' ')
                print(f"  {status} Expected: {result['expected']}, Got: {response_preview}...{confidence_str}")
        
        print()
    
    # Calculate timing
    run_end_time = datetime.now()
    run_duration = run_end_time - run_start_time
    run_duration_seconds = run_duration.total_seconds()

    # Calculate statistics
    total_correct = sum(correct_by_group.values())
    total_tests = sum(total_by_group.values())

    print("="*80)
    print("SCALAR IMPLICATURE EVALUATION RESULTS")
    if persona_prompt:
        print(f"Persona: {persona_prompt[:100]}{'...' if len(persona_prompt) > 100 else ''}")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    if total_tests > 0:
        print(f"Overall Accuracy: {total_correct/total_tests:.3f} ({total_correct}/{total_tests})")
    print()

    # High-level group results
    print("--- High-Level Results ---")
    for group in ['true', 'false', 'underinformative']:
        if total_by_group.get(group, 0) > 0:
            accuracy = correct_by_group[group] / total_by_group[group]
            print(f"{group.capitalize()} statements: {accuracy:.3f} ({correct_by_group[group]}/{total_by_group[group]})")

            if args.logprobs:
                # Calculate average confidence for this group
                group_results = [r for r in results
                                if (r['category'].startswith(group) or r['category'] == group)
                                and 'logprobs' in r and r['logprobs']]
                if group_results:
                    avg_confidence = sum(r['logprobs']['chosen_probability'] for r in group_results) / len(group_results)
                    print(f"  Average confidence: {avg_confidence:.3f}")

    # Fine-grained category results
    if len(total_by_category) > 3:  # More than just true/false/underinformative
        print()
        print("--- Fine-Grained Results ---")
        for category in sorted(total_by_category.keys()):
            if total_by_category[category] > 0:
                accuracy = correct_by_category[category] / total_by_category[category]
                print(f"  {category}: {accuracy:.3f} ({correct_by_category[category]}/{total_by_category[category]})")

    if args.logprobs:
        confidence_results = [r for r in results if 'logprobs' in r and r['logprobs']]
        if confidence_results:
            overall_confidence = sum(r['logprobs']['chosen_probability'] for r in confidence_results) / len(confidence_results)
            print(f"\nOverall average confidence: {overall_confidence:.3f}")
    
    # Generate output filename if not specified
    if not args.output:
        args.output = generate_output_filename(args.model, args.persona_file, args.persona, args.new_run)
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "results", exist_ok=True)
    
    # Save results
    output_data = {
        'model': args.model,
        'temperature': args.temperature,
        'persona_prompt': persona_prompt,
        'persona_file': args.persona_file,
        'run_duration_seconds': run_duration_seconds,
        'total_tests': total_tests,
        'total_accuracy': total_correct/total_tests if total_tests > 0 else 0,
        # High-level group accuracy (true/false/underinformative)
        'accuracy_by_group': {
            grp: correct_by_group[grp]/total_by_group[grp]
            for grp in correct_by_group if total_by_group.get(grp, 0) > 0
        },
        # Fine-grained category accuracy (e.g., true-conj, underinf-quant)
        'accuracy_by_category': {
            cat: correct_by_category[cat]/total_by_category[cat]
            for cat in correct_by_category if total_by_category[cat] > 0
        },
        'confidence_by_group': {},
        'confidence_by_category': {},
        'results': results
    }

    if args.logprobs:
        # High-level group confidence
        for group in ['true', 'false', 'underinformative']:
            group_results = [r for r in results
                           if (r['category'].startswith(group) or r['category'] == group)
                           and 'logprobs' in r and r['logprobs']]
            if group_results:
                avg_confidence = sum(r['logprobs']['chosen_probability'] for r in group_results) / len(group_results)
                output_data['confidence_by_group'][group] = avg_confidence

        # Fine-grained category confidence
        for category in total_by_category.keys():
            category_results = [r for r in results if r['category'] == category and 'logprobs' in r and r['logprobs']]
            if category_results:
                avg_confidence = sum(r['logprobs']['chosen_probability'] for r in category_results) / len(category_results)
                output_data['confidence_by_category'][category] = avg_confidence
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()