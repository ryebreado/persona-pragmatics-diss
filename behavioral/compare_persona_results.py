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

def main():
    parser = argparse.ArgumentParser(description='Compare persona evaluation results with baseline')
    parser.add_argument('path', nargs='+',
                       help='Results directory OR persona result JSON files to compare')
    parser.add_argument('--baseline', help='Baseline file (auto-detected if not specified)')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                       help='Confidence difference threshold (default: 0.1 = 10%%)')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary statistics')

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

    for persona_file in persona_files:
        print(f"\n{'='*80}")
        print(f"COMPARING: {os.path.basename(persona_file)}")
        print(f"{'='*80}")
        
        # Load persona results
        persona_data = load_results(persona_file)
        if not persona_data:
            continue
        
        # Find or use specified baseline
        if args.baseline:
            baseline_file = args.baseline
        else:
            baseline_file = find_baseline_file(persona_file, results_dir)
            if not baseline_file:
                print(f"Could not find baseline file for {persona_file}")
                print(f"Looking for pattern: scalar_implicature_*_baseline_*.json")
                continue
        
        print(f"Baseline: {os.path.basename(baseline_file)}")
        
        # Load baseline results
        baseline_data = load_results(baseline_file)
        if not baseline_data:
            continue
        
        # Extract test results
        baseline_tests = extract_test_results(baseline_data)
        persona_tests = extract_test_results(persona_data)
        
        # Compare results
        all_differences = {}
        answer_diffs = 0
        confidence_diffs = 0
        
        for test_id in baseline_tests:
            if test_id not in persona_tests:
                print(f"Warning: Test {test_id} missing in persona results")
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
                    elif diff['type'] == 'confidence_difference':
                        confidence_diffs += 1
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"Model: {persona_data.get('model', 'unknown')}")
        print(f"Persona: {persona_data.get('persona_prompt', 'unknown')[:100]}...")
        print(f"Total tests: {len(baseline_tests)}")
        print(f"Answer differences: {answer_diffs}")
        print(f"Confidence differences (≥{args.confidence_threshold:.0%}): {confidence_diffs}")
        
        # Print accuracy comparison
        baseline_acc = baseline_data.get('total_accuracy', 0)
        persona_acc = persona_data.get('total_accuracy', 0)
        acc_diff = persona_acc - baseline_acc
        print(f"Accuracy: Baseline {baseline_acc:.3f} → Persona {persona_acc:.3f} (Δ{acc_diff:+.3f})")
        
        # High-level group breakdown (true/false/underinformative)
        # Try accuracy_by_group first (new format), fall back to accuracy_by_category (old format)
        baseline_grp = baseline_data.get('accuracy_by_group', baseline_data.get('accuracy_by_category', {}))
        persona_grp = persona_data.get('accuracy_by_group', persona_data.get('accuracy_by_category', {}))

        print(f"\nBy Group (high-level):")
        for group in ['true', 'false', 'underinformative']:
            if group in baseline_grp and group in persona_grp:
                base_acc = baseline_grp[group]
                pers_acc = persona_grp[group]
                diff = pers_acc - base_acc
                print(f"  {group.capitalize()}: {base_acc:.3f} → {pers_acc:.3f} (Δ{diff:+.3f})")

        # Fine-grained category breakdown (if available)
        baseline_cat = baseline_data.get('accuracy_by_category', {})
        persona_cat = persona_data.get('accuracy_by_category', {})

        # Check if we have fine-grained categories (more than just true/false/underinformative)
        all_categories = set(baseline_cat.keys()) | set(persona_cat.keys())
        fine_grained = [c for c in all_categories if '-' in c]  # e.g., 'true-conj', 'underinf-quant'

        if fine_grained:
            print(f"\nBy Category (fine-grained):")
            for category in sorted(all_categories):
                if category in baseline_cat and category in persona_cat:
                    base_acc = baseline_cat[category]
                    pers_acc = persona_cat[category]
                    diff = pers_acc - base_acc
                    print(f"  {category}: {base_acc:.3f} → {pers_acc:.3f} (Δ{diff:+.3f})")
        
        if not args.summary_only and all_differences:
            print(f"\nDETAILED DIFFERENCES:")
            
            for test_id in sorted(all_differences.keys()):
                differences = all_differences[test_id]
                test_category = baseline_tests[test_id]['category']
                
                print(f"\nTest {test_id} ({test_category}):")
                
                for diff in differences:
                    if diff['type'] == 'answer_difference':
                        baseline_status = "✓" if diff['baseline_correct'] else "✗"
                        persona_status = "✓" if diff['persona_correct'] else "✗"
                        
                        print(f"  ANSWER DIFFERENCE:")
                        print(f"    Expected: {diff['expected']}")
                        print(f"    Baseline: {baseline_status} \"{diff['baseline_response'][:50]}...\"")
                        print(f"    Persona:  {persona_status} \"{diff['persona_response'][:50]}...\"")
                    
                    elif diff['type'] == 'confidence_difference':
                        print(f"  CONFIDENCE DIFFERENCE:")
                        print(f"    Baseline: {diff['baseline_confidence']:.3f}")
                        print(f"    Persona:  {diff['persona_confidence']:.3f}")
                        print(f"    Difference: {diff['difference']:.3f} ({diff['difference']:.1%})")

if __name__ == "__main__":
    main()