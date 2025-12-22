#!/usr/bin/env python3
"""
Train linear probes on GPT-2 small to classify scalar implicature examples.
Three binary classification tasks:
1. underinformative vs true
2. underinformative vs false
3. true vs false
"""

import json
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def format_prompt(ex: Dict) -> str:
    """Format example as prompt for GPT-2"""
    return f"""{ex['scenario']}
{ex['items']}
{ex['outcome']}
{ex['question']}

Statement: "{ex['statement']}"

Is this statement an accurate answer to the question? Answer yes or no."""


def extract_activations(
    model: HookedTransformer,
    examples: List[Dict],
    layer: int = -1,
    position: str = "last"
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract activations from GPT-2 for all examples.

    Args:
        model: HookedTransformer model
        examples: List of example dictionaries
        layer: Which layer to extract from (-1 for last layer)
        position: Which token position to extract ('last', 'mean', or int for specific position)

    Returns:
        activations: numpy array of shape (n_examples, hidden_dim)
        categories: list of category labels
    """
    activations = []
    categories = []

    print(f"Extracting activations from layer {layer}, position={position}...")

    for i, ex in enumerate(examples):
        prompt = format_prompt(ex)

        # Get activations using run_with_cache
        _, cache = model.run_with_cache(prompt)

        # Extract residual stream activations from specified layer
        # cache has keys like 'blocks.{layer}.hook_resid_post'
        if layer == -1:
            layer_idx = model.cfg.n_layers - 1
        else:
            layer_idx = layer

        resid_key = f"blocks.{layer_idx}.hook_resid_post"
        layer_activations = cache[resid_key]  # shape: [1, seq_len, hidden_dim]

        # Extract based on position strategy
        if position == "last":
            # Use last token's activation
            act = layer_activations[0, -1, :].cpu().numpy()
        elif position == "mean":
            # Use mean across all tokens
            act = layer_activations[0, :, :].mean(dim=0).cpu().numpy()
        else:
            # Use specific position
            act = layer_activations[0, int(position), :].cpu().numpy()

        activations.append(act)
        categories.append(ex['category'])

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(examples)} examples")

    activations = np.array(activations)
    print(f"Extracted activations shape: {activations.shape}")

    return activations, categories


def train_binary_probe(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[LogisticRegression, Dict]:
    """
    Train a binary linear probe.

    Returns:
        probe: Trained LogisticRegression model
        results: Dictionary with accuracy and other metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train logistic regression probe
    probe = LogisticRegression(max_iter=1000, random_state=random_state)
    probe.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, probe.predict(X_train))
    test_acc = accuracy_score(y_test, probe.predict(X_test))

    # Get detailed metrics
    y_pred = probe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    results = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'classification_report': report
    }

    return probe, results


def prepare_binary_data(
    activations: np.ndarray,
    categories: List[str],
    class_a: str,
    class_b: str
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Prepare data for binary classification between two classes.

    Returns:
        X: Activations for the two classes
        y: Binary labels (0 for class_a, 1 for class_b)
        indices: Original indices of selected examples
    """
    indices = [i for i, cat in enumerate(categories) if cat in [class_a, class_b]]
    X = activations[indices]
    y = np.array([0 if categories[i] == class_a else 1 for i in indices])

    return X, y, indices


def main():
    parser = argparse.ArgumentParser(description='Train linear probes on GPT-2')
    parser.add_argument('--data', default='data/scalar_implicature_full.json',
                       help='Path to data file')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer to extract activations from (-1 for last)')
    parser.add_argument('--position', default='last',
                       help='Token position to extract (last, mean, or integer)')
    parser.add_argument('--output-dir', default='probing/models',
                       help='Directory to save trained probes')
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

    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data) as f:
        examples = json.load(f)

    # Count categories
    category_counts = {}
    for ex in examples:
        cat = ex['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"Loaded {len(examples)} examples:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Extract activations
    with torch.no_grad():
        activations, categories = extract_activations(
            model, examples, layer=args.layer, position=args.position
        )

    # Define the three binary classification tasks
    tasks = [
        ("underinformative", "true"),
        ("underinformative", "false"),
        ("true", "false")
    ]

    results_summary = {}

    print("\n" + "="*80)
    print("TRAINING LINEAR PROBES")
    print("="*80)

    for class_a, class_b in tasks:
        task_name = f"{class_a}_vs_{class_b}"
        print(f"\nTask: {task_name}")
        print("-" * 40)

        # Prepare data
        X, y, indices = prepare_binary_data(activations, categories, class_a, class_b)
        print(f"Dataset size: {len(X)} ({(y==0).sum()} {class_a}, {(y==1).sum()} {class_b})")

        # Train probe
        probe, results = train_binary_probe(
            X, y, test_size=args.test_size, random_state=args.seed
        )

        # Print results
        print(f"Train accuracy: {results['train_accuracy']:.3f}")
        print(f"Test accuracy:  {results['test_accuracy']:.3f}")

        # Save probe
        probe_path = output_dir / f"probe_{task_name}_layer{args.layer}.pkl"
        with open(probe_path, 'wb') as f:
            pickle.dump({
                'probe': probe,
                'class_a': class_a,
                'class_b': class_b,
                'layer': args.layer,
                'position': args.position,
                'results': results
            }, f)
        print(f"Saved probe to {probe_path}")

        results_summary[task_name] = results

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for task_name, results in results_summary.items():
        print(f"{task_name:30s} Test Acc: {results['test_accuracy']:.3f}")

    # Save overall results
    results_path = output_dir / f"results_layer{args.layer}.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for task_name, results in results_summary.items():
            serializable_results[task_name] = {
                'train_accuracy': float(results['train_accuracy']),
                'test_accuracy': float(results['test_accuracy']),
                'train_size': int(results['train_size']),
                'test_size': int(results['test_size'])
            }

        json.dump({
            'layer': args.layer,
            'position': args.position,
            'model': 'gpt2-small',
            'tasks': serializable_results
        }, f, indent=2)

    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
