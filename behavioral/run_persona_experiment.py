#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
import time
from glob import glob
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False, text=True)
        print(f"✓ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {description}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {command[0]}")
        return False

def find_persona_files(persona_dir="personas"):
    """
    Find all persona files (with or without .txt extension).
    Returns both the file path and the persona name.
    """
    if not os.path.exists(persona_dir):
        print(f"Persona directory not found: {persona_dir}")
        return []

    # Try both with and without .txt extension
    persona_files_txt = glob(os.path.join(persona_dir, "*.txt"))

    # Also find files without extension
    all_files = []
    for item in os.listdir(persona_dir):
        item_path = os.path.join(persona_dir, item)
        if os.path.isfile(item_path):
            all_files.append(item_path)

    # Remove duplicates and hidden files
    unique_files = list(set(all_files))
    persona_files = [f for f in unique_files if not os.path.basename(f).startswith('.')]

    return sorted(persona_files)


def read_persona_content(persona_file):
    """Read persona content from file"""
    try:
        with open(persona_file, 'r') as f:
            content = f.read().strip()
        return content if content else None
    except Exception as e:
        print(f"Warning: Could not read persona file {persona_file}: {e}")
        return None


def is_local_model(model):
    """Determine if model should run locally vs via API"""
    # If model has / it's likely a HuggingFace model path
    if '/' in model:
        return True
    # API models start with these prefixes
    if model.startswith(('gpt-', 'claude-', 'anthropic.')):
        return False
    # Default to local for other cases
    return True

def check_logprobs_support(model):
    """Check if model supports logprobs"""
    # Models that explicitly DON'T support logprobs
    no_logprobs_models = [
        'gpt-4o-mini',
        'gpt-4o',
        'o1',
        'o1-mini',
        'o1-preview',
    ]

    # Check if model is in the no-logprobs list
    for no_lp_model in no_logprobs_models:
        if model.startswith(no_lp_model):
            return False

    # Based on the model routing logic in llm_client.py
    if model.startswith('gpt-3.5') or model.startswith('gpt-4-') or model.startswith('meta-') or '/' in model:
        return True
    elif model.startswith('claude-'):
        return False
    else:
        # Default to False for unknown models
        return False

def find_recent_results(model, pattern_type="baseline", results_dir="results"):
    """Find the most recent result files for a model"""
    # Match the same logic as generate_output_filename - only use the last part after /
    model_clean = model.split('/')[-1].replace('-', '_').replace('.', '_')

    if pattern_type == "baseline":
        pattern = f"{results_dir}/scalar_implicature_{model_clean}_baseline_*.json"
    else:  # persona results
        pattern = f"{results_dir}/scalar_implicature_{model_clean}_*_*.json"
        # Exclude baseline files
        all_files = glob(pattern)
        return [f for f in all_files if "_baseline_" not in f]

    files = glob(pattern)
    return sorted(files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(
        description='Run complete persona experiment (local or API models)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model', help='Model to test (e.g., gpt-4o-mini, meta-llama/Llama-3.2-8B-Instruct)')
    parser.add_argument('--test-file', default='data/scalar_implicature_250.json',
                       help='Test file to use')
    parser.add_argument('--persona-dir', default='personas',
                       help='Directory containing persona files')
    parser.add_argument('--results-dir', default='results',
                       help='Directory for results')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                       help='Confidence difference threshold for comparison')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline run (use existing)')
    parser.add_argument('--skip-personas', action='store_true',
                       help='Skip persona runs (use existing)')
    parser.add_argument('--run-comparison', action='store_true',
                       help='Run comparison after evaluations (default: skip)')
    parser.add_argument('--comparison-only', action='store_true',
                       help='Only run comparisons (skip evaluation runs)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed comparison output')

    # Local model specific options
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'],
                       help='Device for local models (mps/cuda/cpu, use cpu for large models)')
    parser.add_argument('--use-logprobs', action='store_true',
                       help='Use logprobs for evaluation (auto-enabled for API models that support it)')
    parser.add_argument('--track-activations', action='store_true',
                       help='Track model activations (local models only)')
    parser.add_argument('--layers', type=str,
                       help='Comma-separated layer indices to track (local models only)')
    parser.add_argument('--track-keywords', type=str,
                       help='Comma-separated keywords to track (e.g., "some,all,and")')

    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    # Determine if using local model or API
    use_local = is_local_model(args.model)

    # Check if model supports logprobs (for API models, or if explicitly requested for local)
    if use_local:
        use_logprobs = args.use_logprobs
        evaluator_script = 'behavioral/evaluate_local_with_activations.py'
    else:
        use_logprobs = check_logprobs_support(args.model)
        evaluator_script = 'behavioral/evaluate_scalar_implicature_with_persona.py'

    experiment_start = datetime.now()
    print(f"\n{'='*80}")
    print(f"STARTING PERSONA EXPERIMENT")
    print(f"Model: {args.model}")
    print(f"Model type: {'Local' if use_local else 'API'}")
    if use_local:
        print(f"Device: {args.device}")
    print(f"Test file: {args.test_file}")
    print(f"Logprobs: {'Yes' if use_logprobs else 'No'}")
    if args.track_activations:
        print(f"Track activations: Yes")
        if args.layers:
            print(f"  Layers: {args.layers}")
    print(f"Started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    success_count = 0
    total_runs = 0
    
    # Step 1: Run baseline
    if not args.comparison_only and not args.skip_baseline:
        print(f"\nSTEP 1: Running baseline evaluation (no persona)...")

        cmd = [
            'python', evaluator_script,
            args.test_file,
            '--model', args.model,
            '--temperature', str(args.temperature)
        ]

        # Add local-specific options
        if use_local:
            cmd.extend(['--device', args.device])

            if use_logprobs:
                cmd.append('--use-logprobs')

            if args.track_activations:
                cmd.append('--track-activations')
                if args.layers:
                    cmd.extend(['--layers', args.layers])
                if args.track_keywords:
                    cmd.extend(['--track-keywords', args.track_keywords])
        else:
            # API-specific options
            if use_logprobs:
                cmd.append('--logprobs')

        if args.verbose:
            cmd.append('--verbose')

        total_runs += 1
        if run_command(cmd, f"Baseline evaluation for {args.model}"):
            success_count += 1
        else:
            print("❌ Baseline failed - cannot continue")
            sys.exit(1)
    
    # Step 2: Run persona evaluations
    if not args.comparison_only and not args.skip_personas:
        persona_files = find_persona_files(args.persona_dir)

        if not persona_files:
            print(f"No persona files found in {args.persona_dir}")
        else:
            print(f"\nSTEP 2: Running {len(persona_files)} persona evaluations...")

            for i, persona_file in enumerate(persona_files, 1):
                persona_name = os.path.splitext(os.path.basename(persona_file))[0]
                print(f"\n[{i}/{len(persona_files)}] Persona: {persona_name}")

                # Check if persona file has content
                persona_content = read_persona_content(persona_file)
                if not persona_content:
                    print(f"⚠️  Persona file {persona_name} is empty - skipping")
                    continue

                cmd = [
                    'python', evaluator_script,
                    args.test_file,
                    '--model', args.model,
                    '--temperature', str(args.temperature),
                    '--persona-file', persona_file
                ]

                # Add local-specific options
                if use_local:
                    cmd.extend(['--device', args.device])

                    if use_logprobs:
                        cmd.append('--use-logprobs')

                    if args.track_activations:
                        cmd.append('--track-activations')
                        if args.layers:
                            cmd.extend(['--layers', args.layers])
                        if args.track_keywords:
                            cmd.extend(['--track-keywords', args.track_keywords])
                else:
                    # API-specific options
                    if use_logprobs:
                        cmd.append('--logprobs')

                if args.verbose:
                    cmd.append('--verbose')

                total_runs += 1
                if run_command(cmd, f"Persona evaluation: {persona_name}"):
                    success_count += 1

                # Small delay between runs
                time.sleep(1 if not use_local else 2)
    
    # Step 3: Run comparisons (only if requested or comparison-only mode)
    if args.run_comparison or args.comparison_only:
        print(f"\nSTEP 3: Running comparisons...")

        # Find all persona result files for this model
        persona_results = find_recent_results(args.model, "persona", args.results_dir)

        if not persona_results:
            print(f"No persona results found for model {args.model}")
            print("Make sure you've run the evaluations first!")
            sys.exit(1)

        print(f"Found {len(persona_results)} persona result files to compare")

        cmd = [
            'python', 'behavioral/compare_persona_results.py',
            args.results_dir,
            '--confidence-threshold', str(args.confidence_threshold)
        ]

        if not args.verbose:
            cmd.append('--summary-only')

        total_runs += 1
        if run_command(cmd, "Persona comparisons"):
            success_count += 1
    else:
        print(f"\nSTEP 3: Skipping comparisons (use --run-comparison to enable)")
    
    # Summary
    experiment_end = datetime.now()
    duration = experiment_end - experiment_start
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Duration: {duration}")
    print(f"Successful runs: {success_count}/{total_runs}")
    
    if success_count < total_runs:
        print(f"⚠️  {total_runs - success_count} runs failed")
        sys.exit(1)
    else:
        print("🎉 All runs completed successfully!")
    
    # Show recent results
    baseline_files = find_recent_results(args.model, "baseline", args.results_dir)
    persona_files = find_recent_results(args.model, "persona", args.results_dir)
    
    if baseline_files:
        print(f"\nBaseline result: {os.path.basename(baseline_files[-1])}")
    
    if persona_files:
        print(f"Persona results ({len(persona_files)}):")
        for f in persona_files[-5:]:  # Show last 5
            print(f"  {os.path.basename(f)}")
        if len(persona_files) > 5:
            print(f"  ... and {len(persona_files) - 5} more")

if __name__ == "__main__":
    main()