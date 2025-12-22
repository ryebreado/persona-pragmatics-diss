#!/usr/bin/env python3
"""
Train linear probes across all layers of GPT-2 to find which layers
best encode the distinction between different statement categories.
"""

import json
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from train_scalar_implicature_probes import (
    format_prompt, extract_activations, prepare_binary_data
)


def train_probe_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict:
    """
    Train probe with cross-validation to get more robust estimates.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train with cross-validation on training set
    probe = LogisticRegression(max_iter=1000, random_state=random_state, C=0.1)  # Add regularization

    # Only do CV if we have enough samples
    # For stratified CV, we need at least cv samples per class
    min_class_size = min(np.bincount(y_train))
    max_cv_folds = min(cv, min_class_size)

    if max_cv_folds >= 2:
        cv_scores = cross_val_score(probe, X_train, y_train, cv=max_cv_folds)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    else:
        cv_mean = None
        cv_std = None

    # Train on full training set
    probe.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, probe.predict(X_train))
    test_acc = accuracy_score(y_test, probe.predict(X_test))

    results = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Probe all layers of GPT-2')
    parser.add_argument('--data', default='data/scalar_implicature_full.json',
                       help='Path to data file')
    parser.add_argument('--position', default='last',
                       help='Token position to extract (last, mean, or integer)')
    parser.add_argument('--output-dir', default='probing/analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Fraction of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"Model has {n_layers} layers")

    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data) as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples")

    # Define the three binary classification tasks
    tasks = [
        ("underinformative", "true"),
        ("underinformative", "false"),
        ("true", "false")
    ]

    # Store results for all layers
    all_results = {task: [] for task in [f"{a}_vs_{b}" for a, b in tasks]}

    print("\n" + "="*80)
    print("PROBING ALL LAYERS")
    print("="*80)

    # Extract activations and train probes for each layer
    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx}/{n_layers-1}")
        print("-" * 40)

        # Extract activations for this layer
        with torch.no_grad():
            activations, categories = extract_activations(
                model, examples, layer=layer_idx, position=args.position
            )

        # Train probes for each task
        for class_a, class_b in tasks:
            task_name = f"{class_a}_vs_{class_b}"

            # Prepare data
            X, y, _ = prepare_binary_data(activations, categories, class_a, class_b)

            # Train probe with cross-validation
            results = train_probe_with_cv(
                X, y, test_size=args.test_size, random_state=args.seed
            )
            results['layer'] = layer_idx

            all_results[task_name].append(results)

            # Print concise results
            cv_str = f"{results['cv_mean']:.3f}±{results['cv_std']:.3f}" if results['cv_mean'] else "N/A"
            print(f"  {task_name:30s} Test: {results['test_accuracy']:.3f}, CV: {cv_str}")

    # Find best layers for each task
    print("\n" + "="*80)
    print("BEST LAYERS BY TASK")
    print("="*80)

    for task_name, results_list in all_results.items():
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i]['test_accuracy'])
        best_result = results_list[best_idx]

        print(f"\n{task_name}:")
        print(f"  Best layer: {best_result['layer']}")
        print(f"  Test accuracy: {best_result['test_accuracy']:.3f}")
        if best_result['cv_mean']:
            print(f"  CV accuracy: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
        print(f"  Train accuracy: {best_result['train_accuracy']:.3f}")

    # Save detailed results
    results_path = output_dir / f"layer_analysis_{args.position}.json"
    with open(results_path, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for task_name, results_list in all_results.items():
            serializable_results[task_name] = [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in r.items()}
                for r in results_list
            ]

        json.dump({
            'model': 'gpt2-small',
            'n_layers': n_layers,
            'position': args.position,
            'test_size': args.test_size,
            'tasks': serializable_results
        }, f, indent=2)

    print(f"\nSaved detailed results to {results_path}")

    # Create summary plot data
    summary_path = output_dir / f"layer_summary_{args.position}.txt"
    with open(summary_path, 'w') as f:
        f.write("Layer-by-layer test accuracy summary\n")
        f.write("="*80 + "\n\n")

        for task_name in all_results.keys():
            f.write(f"\n{task_name}\n")
            f.write("-" * 40 + "\n")
            f.write("Layer  Test_Acc  Train_Acc  CV_Mean  CV_Std\n")
            for result in all_results[task_name]:
                layer = result['layer']
                test_acc = result['test_accuracy']
                train_acc = result['train_accuracy']
                cv_mean = result['cv_mean'] if result['cv_mean'] else 0.0
                cv_std = result['cv_std'] if result['cv_std'] else 0.0
                f.write(f"{layer:5d}  {test_acc:8.3f}  {train_acc:9.3f}  {cv_mean:7.3f}  {cv_std:6.3f}\n")

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
