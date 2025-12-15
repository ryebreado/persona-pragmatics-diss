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
    """Find all persona files"""
    if not os.path.exists(persona_dir):
        print(f"Persona directory not found: {persona_dir}")
        return []
    
    persona_files = glob(os.path.join(persona_dir, "*.txt"))
    return sorted(persona_files)

def check_logprobs_support(model):
    """Check if model supports logprobs"""
    # Based on the model routing logic in llm_client.py
    if model.startswith('gpt-') or model.startswith('meta-') or '/' in model:
        return True
    elif model.startswith('claude-'):
        return False
    else:
        # Default to False for unknown models
        return False

def find_recent_results(model, pattern_type="baseline", results_dir="results"):
    """Find the most recent result files for a model"""
    model_clean = model.replace('/', '_').replace('-', '_').replace('.', '_')
    
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
    parser = argparse.ArgumentParser(description='Run complete persona experiment for a model')
    parser.add_argument('model', help='Model to test (e.g., gpt-4o-mini, claude-sonnet-4-20250514)')
    parser.add_argument('--test-file', default='data/scalar_implicature_full.json', 
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
    parser.add_argument('--comparison-only', action='store_true',
                       help='Only run comparisons (skip evaluation runs)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed comparison output')
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Check if model supports logprobs
    use_logprobs = check_logprobs_support(args.model)
    if use_logprobs:
        print(f"Model {args.model} supports logprobs - will include confidence analysis")
    else:
        print(f"Model {args.model} does not support logprobs - answer comparison only")
    
    experiment_start = datetime.now()
    print(f"\n{'='*80}")
    print(f"STARTING PERSONA EXPERIMENT")
    print(f"Model: {args.model}")
    print(f"Test file: {args.test_file}")
    print(f"Logprobs: {'Yes' if use_logprobs else 'No'}")
    print(f"Started: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    success_count = 0
    total_runs = 0
    
    # Step 1: Run baseline
    if not args.comparison_only and not args.skip_baseline:
        print(f"\nSTEP 1: Running baseline evaluation...")
        
        cmd = [
            'python', 'evaluate_scalar_implicature_with_persona.py',
            args.test_file,
            '--model', args.model,
            '--temperature', str(args.temperature)
        ]
        
        if use_logprobs:
            cmd.append('--logprobs')
        
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
                
                cmd = [
                    'python', 'evaluate_scalar_implicature_with_persona.py',
                    args.test_file,
                    '--model', args.model,
                    '--temperature', str(args.temperature),
                    '--persona-file', persona_file
                ]
                
                if use_logprobs:
                    cmd.append('--logprobs')
                
                total_runs += 1
                if run_command(cmd, f"Persona evaluation: {persona_name}"):
                    success_count += 1
                
                # Small delay to avoid API rate limits
                time.sleep(1)
    
    # Step 3: Run comparisons
    print(f"\nSTEP 3: Running comparisons...")
    
    # Find all persona result files for this model
    persona_results = find_recent_results(args.model, "persona", args.results_dir)
    
    if not persona_results:
        print(f"No persona results found for model {args.model}")
        print("Make sure you've run the evaluations first!")
        sys.exit(1)
    
    print(f"Found {len(persona_results)} persona result files to compare")
    
    cmd = [
        'python', 'compare_persona_results.py',
        *persona_results,
        '--confidence-threshold', str(args.confidence_threshold),
        '--results-dir', args.results_dir
    ]
    
    if not args.verbose:
        cmd.append('--summary-only')
    
    total_runs += 1
    if run_command(cmd, "Persona comparisons"):
        success_count += 1
    
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