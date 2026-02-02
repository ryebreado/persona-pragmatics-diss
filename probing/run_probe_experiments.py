#!/usr/bin/env python3
"""
Run the full suite of probing experiments for the dissertation.

Experiments:
1. Baseline layer curve - Where does underinf detection emerge?
2. Persona comparison - Does Anti-Gricean change the curve?
3. Cross-condition transfer - Same representation or different?
4. Conj vs Quant split - Are they encoded differently?
5. Three-way classification - True vs false vs underinf patterns

Usage:
    # Run all experiments for a model
    uv run python probing/run_probe_experiments.py results/qwen3_8b_run_03/

    # Run specific experiment
    uv run python probing/run_probe_experiments.py results/qwen3_8b_run_03/ --experiment baseline

    # Compare token positions
    uv run python probing/run_probe_experiments.py results/qwen3_8b_run_03/ --token-position mean_pooled
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from train_probes import (
    load_activations,
    run_layer_curve,
    run_transfer_experiment,
    plot_layer_curve,
    save_results,
    ExperimentResult
)


def find_activation_files(run_dir: Path) -> Dict[str, Path]:
    """Find all activation files in a run directory."""
    files = {}
    for pt_file in run_dir.glob("*_activations.pt"):
        name = pt_file.stem.replace("_activations", "")
        # Extract persona name
        parts = name.split("_with_acts_")[0].split("_")
        # Find persona: everything after model name
        # e.g., qwen3_8b_baseline -> baseline
        # e.g., qwen3_8b_anti_gricean -> anti_gricean
        for i, part in enumerate(parts):
            if part in ['baseline', 'anti', 'helpful', 'literal', 'pragmaticist', 'soft']:
                persona = "_".join(parts[i:])
                break
        else:
            persona = "unknown"
        files[persona] = pt_file
    return files


def run_baseline_curve(
    activations_path: Path,
    output_dir: Path,
    token_position: str = "last_token"
) -> List[ExperimentResult]:
    """
    Experiment 1: Baseline layer curve for underinf detection.
    Where does the model start distinguishing underinformative statements?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Baseline Layer Curve")
    print("="*60)

    activations = load_activations(str(activations_path))
    results = []

    # Main probe: underinf vs not
    print(f"\nTask: underinf_binary ({token_position})")
    result = run_layer_curve(
        activations,
        task="underinf_binary",
        token_position=token_position
    )
    result.train_condition = activations_path.stem
    results.append(result)

    # Save and plot
    exp_name = f"baseline_underinf_{token_position}"
    save_results(results, str(output_dir / f"{exp_name}.json"))
    plot_layer_curve(
        results,
        f"Underinformative Detection ({token_position})",
        str(output_dir / f"{exp_name}.png")
    )

    return results


def run_persona_comparison(
    activation_files: Dict[str, Path],
    output_dir: Path,
    token_position: str = "last_token",
    personas: Optional[List[str]] = None
) -> List[ExperimentResult]:
    """
    Experiment 2: Compare probe accuracy across personas.
    Does Anti-Gricean suppress the underinf signal?

    Generates pairwise plots: baseline vs each persona separately.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Persona Comparison")
    print("="*60)

    if 'baseline' not in activation_files:
        print("  Baseline not found, skipping persona comparison")
        return []

    # First, get baseline results
    print(f"\nbaseline:")
    baseline_acts = load_activations(str(activation_files['baseline']))
    baseline_result = run_layer_curve(
        baseline_acts,
        task="underinf_binary",
        token_position=token_position
    )
    baseline_result.name = "baseline"
    baseline_result.train_condition = "baseline"

    all_results = [baseline_result]

    # Get personas to compare (exclude baseline)
    if personas is None:
        personas = [p for p in activation_files.keys() if p != 'baseline']

    # Run each persona and create pairwise plot
    for persona in personas:
        if persona not in activation_files:
            print(f"  Skipping {persona} (not found)")
            continue

        print(f"\n{persona}:")
        activations = load_activations(str(activation_files[persona]))

        result = run_layer_curve(
            activations,
            task="underinf_binary",
            token_position=token_position
        )
        result.name = persona
        result.train_condition = persona
        all_results.append(result)

        # Create pairwise plot: baseline vs this persona
        pairwise_results = [baseline_result, result]
        exp_name = f"baseline_vs_{persona}_{token_position}"
        save_results(pairwise_results, str(output_dir / f"{exp_name}.json"))
        plot_layer_curve(
            pairwise_results,
            f"Baseline vs {persona.replace('_', ' ').title()} ({token_position})",
            str(output_dir / f"{exp_name}.png")
        )

    # Save combined results JSON and also generate all-in-one plot
    if all_results:
        exp_name = f"persona_comparison_{token_position}"
        save_results(all_results, str(output_dir / f"{exp_name}.json"))
        plot_layer_curve(
            all_results,
            f"All Personas ({token_position})",
            str(output_dir / f"{exp_name}_all.png")
        )

    return all_results


def run_transfer_experiments(
    activation_files: Dict[str, Path],
    output_dir: Path,
    token_position: str = "last_token"
) -> List[ExperimentResult]:
    """
    Experiment 3: Cross-condition transfer.
    Train on baseline, test on Anti-Gricean.
    If transfers well → same representation, different decision.
    If fails → persona changes the representation itself.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Cross-Condition Transfer")
    print("="*60)

    if 'baseline' not in activation_files:
        print("  Baseline not found, skipping transfer experiments")
        return []

    baseline_acts = load_activations(str(activation_files['baseline']))
    results = []

    # Also run baseline→baseline for comparison (should be ~same as CV)
    print("\nBaseline → Baseline (sanity check):")
    result_baseline = run_layer_curve(
        baseline_acts,
        task="underinf_binary",
        token_position=token_position
    )
    result_baseline.name = "baseline_cv"
    result_baseline.train_condition = "baseline"
    results.append(result_baseline)

    # Transfer to each persona
    for persona, path in activation_files.items():
        if persona == 'baseline':
            continue

        print(f"\nBaseline → {persona}:")
        persona_acts = load_activations(str(path))

        result = run_transfer_experiment(
            baseline_acts, persona_acts,
            task="underinf_binary",
            token_position=token_position
        )
        result.name = f"baseline→{persona}"
        result.train_condition = "baseline"
        result.test_condition = persona
        results.append(result)

    # Plot
    if len(results) > 1:
        exp_name = f"transfer_{token_position}"
        save_results(results, str(output_dir / f"{exp_name}.json"))
        plot_layer_curve(
            results,
            f"Transfer: Baseline → Personas ({token_position})",
            str(output_dir / f"{exp_name}.png")
        )

    return results


def run_conj_quant_split(
    activations_path: Path,
    output_dir: Path,
    token_position: str = "last_token"
) -> List[ExperimentResult]:
    """
    Experiment 4: Conjunctive vs Quantifier split.
    Are underinf-conj and underinf-quant encoded in the same layer/direction?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Conj vs Quant Split")
    print("="*60)

    activations = load_activations(str(activations_path))
    results = []

    # Probe for underinf-conj
    print(f"\nTask: underinf_conj ({token_position})")
    result_conj = run_layer_curve(
        activations,
        task="underinf_conj",
        token_position=token_position
    )
    result_conj.name = "underinf_conj"
    result_conj.train_condition = activations_path.stem
    results.append(result_conj)

    # Probe for underinf-quant
    print(f"\nTask: underinf_quant ({token_position})")
    result_quant = run_layer_curve(
        activations,
        task="underinf_quant",
        token_position=token_position
    )
    result_quant.name = "underinf_quant"
    result_quant.train_condition = activations_path.stem
    results.append(result_quant)

    # Plot comparison
    exp_name = f"conj_quant_split_{token_position}"
    save_results(results, str(output_dir / f"{exp_name}.json"))
    plot_layer_curve(
        results,
        f"Conj vs Quant Detection ({token_position})",
        str(output_dir / f"{exp_name}.png")
    )

    return results


def run_three_way_classification(
    activations_path: Path,
    output_dir: Path,
    token_position: str = "last_token"
) -> List[ExperimentResult]:
    """
    Experiment 5: Three-way classification (true vs false vs underinf).
    Does underinformative pattern with true (semantically correct)
    or false (pragmatically wrong)?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Three-Way Classification")
    print("="*60)

    activations = load_activations(str(activations_path))

    print(f"\nTask: three_way ({token_position})")
    result = run_layer_curve(
        activations,
        task="three_way",
        token_position=token_position
    )
    result.train_condition = activations_path.stem

    # Save and plot
    exp_name = f"three_way_{token_position}"
    save_results([result], str(output_dir / f"{exp_name}.json"))
    plot_layer_curve(
        [result],
        f"True vs False vs Underinf ({token_position})",
        str(output_dir / f"{exp_name}.png")
    )

    return [result]


def main():
    parser = argparse.ArgumentParser(
        description='Run probing experiments for dissertation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('run_dir',
                       help='Directory containing activation .pt files')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'baseline', 'persona', 'transfer',
                               'conj_quant', 'three_way'],
                       help='Which experiment to run')
    parser.add_argument('--token-position', default='last_token',
                       help='Token position to probe')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: probing/results/<run_name>)')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return

    # Find activation files
    activation_files = find_activation_files(run_dir)
    print(f"Found activation files:")
    for persona, path in activation_files.items():
        print(f"  {persona}: {path.name}")

    if not activation_files:
        print("No activation files found!")
        return

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('probing/results') / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Get baseline path (for experiments that need a single file)
    baseline_path = activation_files.get('baseline')
    if not baseline_path and activation_files:
        baseline_path = list(activation_files.values())[0]

    # Run experiments
    all_results = {}

    if args.experiment in ['all', 'baseline']:
        all_results['baseline'] = run_baseline_curve(
            baseline_path, output_dir, args.token_position
        )

    if args.experiment in ['all', 'persona']:
        all_results['persona'] = run_persona_comparison(
            activation_files, output_dir, args.token_position
        )

    if args.experiment in ['all', 'transfer']:
        all_results['transfer'] = run_transfer_experiments(
            activation_files, output_dir, args.token_position
        )

    if args.experiment in ['all', 'conj_quant']:
        all_results['conj_quant'] = run_conj_quant_split(
            baseline_path, output_dir, args.token_position
        )

    if args.experiment in ['all', 'three_way']:
        all_results['three_way'] = run_three_way_classification(
            baseline_path, output_dir, args.token_position
        )

    # Final summary
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
