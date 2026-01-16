#!/usr/bin/env python3
import json
import argparse
import sys
import os
from glob import glob


def find_results_in_directory(results_dir):
    """
    Find all result files in a directory.
    Returns tuple of (baseline_file, persona_files).
    """
    all_files = glob(os.path.join(results_dir, "*.json"))

    baseline_file = None
    persona_files = []

    for f in all_files:
        basename = os.path.basename(f)
        if "_baseline_" in basename:
            baseline_file = f
        else:
            persona_files.append(f)

    return baseline_file, sorted(persona_files)


def load_results(file_path):
    """Load results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        return None

def extract_test_results(results):
    """Extract test results indexed by test_id"""
    test_dict = {}
    for result in results.get('results', []):
        test_id = result.get('test_id', 'unknown')
        test_dict[test_id] = {
            'response': result.get('response', '').strip(),
            'expected': result.get('expected', ''),
            'correct': result.get('correct', False),
            'category': result.get('category', ''),
            'confidence': None
        }
        
        # Extract confidence if available
        if 'logprobs' in result and result['logprobs']:
            test_dict[test_id]['confidence'] = result['logprobs'].get('chosen_probability', None)
    
    return test_dict

def compare_responses(baseline_result, persona_result, confidence_threshold=0.1):
    """Compare two test results and return differences"""
    differences = []
    
    # Check for answer differences
    baseline_correct = baseline_result.get('correct', False)
    persona_correct = persona_result.get('correct', False)
    
    if baseline_correct != persona_correct:
        differences.append({
            'type': 'answer_difference',
            'baseline_correct': baseline_correct,
            'persona_correct': persona_correct,
            'baseline_response': baseline_result.get('response', ''),
            'persona_response': persona_result.get('response', ''),
            'expected': baseline_result.get('expected', '')
        })
    
    # Check for confidence differences (if both have confidence data)
    baseline_conf = baseline_result.get('confidence')
    persona_conf = persona_result.get('confidence')
    
    if baseline_conf is not None and persona_conf is not None:
        conf_diff = abs(baseline_conf - persona_conf)
        if conf_diff >= confidence_threshold:
            differences.append({
                'type': 'confidence_difference',
                'baseline_confidence': baseline_conf,
                'persona_confidence': persona_conf,
                'difference': conf_diff,
                'baseline_response': baseline_result.get('response', ''),
                'persona_response': persona_result.get('response', '')
            })
    
    return differences

def find_baseline_file(persona_file, results_dir="results"):
    """Find the corresponding baseline file for a given persona file"""
    # Extract model name from persona filename
    # Format: scalar_implicature_MODEL_PERSONA_DATE_TIME.json
    # Example: scalar_implicature_gpt_4o_mini_literal_thinker_20251113_204421.json
    
    filename = os.path.basename(persona_file)
    parts = filename.split('_')
    
    if len(parts) < 6:  # Need at least: scalar, implicature, model, persona, date, time.json
        return None
    
    # The last two parts are always DATE and TIME.json
    # The second to last identifiable part is the persona name
    # Everything from index 2 to the persona is the model name
    
    # Find the baseline pattern by replacing the persona name with "baseline"
    # Look for "_baseline_" in existing files to identify the model
    
    # Try a different approach: look at all baseline files and find the best match
    all_baselines = glob(f"{results_dir}/scalar_implicature_*_baseline_*.json")
    
    # Extract model parts from the persona filename
    # Skip the first 2 parts (scalar_implicature) and last 2 parts (date_time.json)
    persona_parts = parts[2:-2]
    
    # Try different model lengths (in case model name has multiple parts)
    for i in range(1, len(persona_parts)):
        candidate_model = '_'.join(persona_parts[:i])
        pattern = f"{results_dir}/scalar_implicature_{candidate_model}_baseline_*.json"
        matches = glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    
    # Debug output if no match found
    print(f"Debug: Could not find baseline for: {filename}")
    print(f"Debug: Tried model candidates: {[' _'.join(persona_parts[:i]) for i in range(1, len(persona_parts))]}")
    print(f"Debug: Available baselines: {[os.path.basename(f) for f in all_baselines]}")
    
    return None

def get_persona_name(filepath):
    """Extract persona name from filename."""
    basename = os.path.basename(filepath)
    # Format: scalar_implicature_MODEL_PERSONA_DATE_TIME.json
    parts = basename.replace('.json', '').split('_')
    # Find persona by looking for known names or position
    known_personas = ['baseline', 'anti_gricean', 'helpful_teacher', 'literal_thinker', 'pragmaticist']
    for i, part in enumerate(parts):
        if part in known_personas:
            return part
        # Check for two-word personas
        if i < len(parts) - 1:
            two_word = f"{part}_{parts[i+1]}"
            if two_word in known_personas:
                return two_word
    # Fallback: assume persona is after model name, before timestamp
    if len(parts) >= 6:
        return parts[-3]  # e.g., "pragmaticist" from ..._pragmaticist_20260115_...
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description='Compare persona evaluation results with baseline')
    parser.add_argument('path', nargs='+',
                       help='Results directory OR persona result JSON files to compare')
    parser.add_argument('--baseline', help='Baseline file (auto-detected if not specified)')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                       help='Confidence difference threshold (default: 0.1 = 10%%)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output with examples')

    args = parser.parse_args()

    # Determine if input is a directory or list of files
    if len(args.path) == 1 and os.path.isdir(args.path[0]):
        # Directory mode: find baseline and persona files in directory
        results_dir = args.path[0]
        baseline_file, persona_files = find_results_in_directory(results_dir)

        if not baseline_file:
            print(f"No baseline file found in {results_dir}")
            print("Looking for files matching: *_baseline_*.json")
            sys.exit(1)

        if not persona_files:
            print(f"No persona result files found in {results_dir}")
            sys.exit(1)

        print(f"Found baseline: {os.path.basename(baseline_file)}")
        print(f"Found {len(persona_files)} persona files")
        args.baseline = baseline_file
    else:
        # File mode: treat paths as persona files
        persona_files = args.path
        results_dir = os.path.dirname(persona_files[0]) if persona_files else 'results'

    # Load baseline once if in directory mode
    baseline_data = None
    if args.baseline:
        baseline_data = load_results(args.baseline)

    # Collect all results for concise summary
    all_persona_results = []

    for persona_file in persona_files:
        # Load persona results
        persona_data = load_results(persona_file)
        if not persona_data:
            continue

        # Find or use specified baseline
        if args.baseline:
            current_baseline_file = args.baseline
            current_baseline_data = baseline_data
        else:
            current_baseline_file = find_baseline_file(persona_file, results_dir)
            if not current_baseline_file:
                print(f"Could not find baseline file for {persona_file}")
                continue
            current_baseline_data = load_results(current_baseline_file)

        if not current_baseline_data:
            continue

        # Extract test results for detailed comparison
        baseline_tests = extract_test_results(current_baseline_data)
        persona_tests = extract_test_results(persona_data)

        # Compare results
        all_differences = {}
        answer_diffs = 0

        for test_id in baseline_tests:
            if test_id not in persona_tests:
                continue

            differences = compare_responses(
                baseline_tests[test_id],
                persona_tests[test_id],
                args.confidence_threshold
            )

            if differences:
                all_differences[test_id] = differences
                for diff in differences:
                    if diff['type'] == 'answer_difference':
                        answer_diffs += 1

        # Get accuracies
        baseline_acc = current_baseline_data.get('total_accuracy', 0)
        persona_acc = persona_data.get('total_accuracy', 0)

        # Get group accuracies
        baseline_grp = current_baseline_data.get('accuracy_by_group', current_baseline_data.get('accuracy_by_category', {}))
        persona_grp = persona_data.get('accuracy_by_group', persona_data.get('accuracy_by_category', {}))

        # Store for summary
        persona_name = get_persona_name(persona_file)
        all_persona_results.append({
            'name': persona_name,
            'file': persona_file,
            'baseline_acc': baseline_acc,
            'persona_acc': persona_acc,
            'baseline_grp': baseline_grp,
            'persona_grp': persona_grp,
            'baseline_cat': current_baseline_data.get('accuracy_by_category', {}),
            'persona_cat': persona_data.get('accuracy_by_category', {}),
            'answer_diffs': answer_diffs,
            'all_differences': all_differences,
            'baseline_tests': baseline_tests,
            'persona_data': persona_data
        })

    if not all_persona_results:
        print("No results to compare.")
        return

    # Print results
    if args.verbose:
        # Verbose mode: detailed output per persona
        for result in all_persona_results:
            print(f"\n{'='*80}")
            print(f"COMPARING: {os.path.basename(result['file'])}")
            print(f"{'='*80}")

            acc_diff = result['persona_acc'] - result['baseline_acc']
            print(f"Accuracy: {result['baseline_acc']:.3f} -> {result['persona_acc']:.3f} (Δ{acc_diff:+.3f})")

            print(f"\nBy Group:")
            for group in ['true', 'false', 'underinformative']:
                if group in result['baseline_grp'] and group in result['persona_grp']:
                    base_acc = result['baseline_grp'][group]
                    pers_acc = result['persona_grp'][group]
                    diff = pers_acc - base_acc
                    print(f"  {group}: {base_acc:.3f} -> {pers_acc:.3f} (Δ{diff:+.3f})")

            # Fine-grained categories
            all_categories = set(result['baseline_cat'].keys()) | set(result['persona_cat'].keys())
            fine_grained = [c for c in all_categories if '-' in c]

            if fine_grained:
                print(f"\nBy Category:")
                for category in sorted(fine_grained):
                    if category in result['baseline_cat'] and category in result['persona_cat']:
                        base_acc = result['baseline_cat'][category]
                        pers_acc = result['persona_cat'][category]
                        diff = pers_acc - base_acc
                        print(f"  {category}: {base_acc:.3f} -> {pers_acc:.3f} (Δ{diff:+.3f})")

            # Detailed differences
            if result['all_differences']:
                print(f"\nDETAILED DIFFERENCES ({result['answer_diffs']} items):")

                for test_id in sorted(result['all_differences'].keys()):
                    differences = result['all_differences'][test_id]
                    test_category = result['baseline_tests'][test_id]['category']

                    print(f"\nTest {test_id} ({test_category}):")

                    for diff in differences:
                        if diff['type'] == 'answer_difference':
                            baseline_status = "correct" if diff['baseline_correct'] else "wrong"
                            persona_status = "correct" if diff['persona_correct'] else "wrong"
                            print(f"  Baseline: {baseline_status}, Persona: {persona_status}")
                            print(f"  Expected: {diff['expected']}")
    else:
        # Concise mode: table of deltas
        model = all_persona_results[0]['persona_data'].get('model', 'unknown')
        print(f"\nModel: {model}")
        print(f"Baseline accuracy: {all_persona_results[0]['baseline_acc']:.3f}")

        # Header
        print(f"\n{'Persona':<20} {'Acc':>6} {'Δ':>7} | {'true':>6} {'false':>6} {'underinf':>8}")
        print("-" * 65)

        for result in all_persona_results:
            name = result['name']
            acc = result['persona_acc']
            acc_diff = acc - result['baseline_acc']

            # Group deltas
            true_d = result['persona_grp'].get('true', 0) - result['baseline_grp'].get('true', 0)
            false_d = result['persona_grp'].get('false', 0) - result['baseline_grp'].get('false', 0)
            underinf_d = result['persona_grp'].get('underinformative', 0) - result['baseline_grp'].get('underinformative', 0)

            print(f"{name:<20} {acc:>6.3f} {acc_diff:>+7.3f} | {true_d:>+6.3f} {false_d:>+6.3f} {underinf_d:>+8.3f}")

if __name__ == "__main__":
    main()