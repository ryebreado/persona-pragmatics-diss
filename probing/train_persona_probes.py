#!/usr/bin/env python3
"""
Train linear probes to classify persona identity from model activations.
Probes at each layer to determine where persona information is encoded.

Usage:
    python probing/train_persona_probes.py results/run_02_qwen3_8b --output figures/probe_accuracy.png
"""

import json
import argparse
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_activations_from_results(results_dir: str) -> Tuple[Dict[str, List], List[str], int]:
    """
    Load last-token activations from all persona result files in a directory.

    Returns:
        layer_activations: Dict mapping layer name to list of activation vectors
        labels: List of persona labels for each example
        n_layers: Number of layers found
    """
    json_files = glob(os.path.join(results_dir, "*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {results_dir}")

    layer_activations = {}  # layer_name -> list of activation vectors
    labels = []
    persona_to_label = {}  # Map persona names to integer labels
    n_layers = 0

    for filepath in sorted(json_files):
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Determine persona name
        if '_baseline_' in os.path.basename(filepath):
            persona_name = 'baseline'
        elif data.get('persona_file'):
            persona_name = os.path.basename(data['persona_file'])
        else:
            persona_name = 'unknown'

        # Assign integer label to persona
        if persona_name not in persona_to_label:
            persona_to_label[persona_name] = len(persona_to_label)

        label = persona_to_label[persona_name]

        # Check if we have activation data
        results = data.get('results', [])
        has_activations = False

        for result in results:
            if 'last_token_activations' in result:
                has_activations = True
                activations = result['last_token_activations']

                # Initialize layer lists if needed
                if not layer_activations:
                    for layer_name in activations.keys():
                        layer_activations[layer_name] = []
                    n_layers = len(activations)

                # Add activations for each layer
                for layer_name, act in activations.items():
                    # Convert list to numpy array if needed
                    if isinstance(act, list):
                        act = np.array(act)
                    layer_activations[layer_name].append(act)

                labels.append(label)

        if not has_activations:
            print(f"Warning: No activations found in {os.path.basename(filepath)}")
            print("  Run experiments with --track-activations to collect activation data")

    print(f"Loaded {len(labels)} examples from {len(persona_to_label)} personas:")
    for persona, label in sorted(persona_to_label.items(), key=lambda x: x[1]):
        count = labels.count(label)
        print(f"  [{label}] {persona}: {count} examples")

    return layer_activations, labels, n_layers, persona_to_label


def train_probe_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Train a logistic regression probe for one layer.

    Returns:
        train_accuracy, test_accuracy
    """
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train probe
    probe = LogisticRegression(max_iter=1000, random_state=random_state)
    probe.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, probe.predict(X_train))
    test_acc = accuracy_score(y_test, probe.predict(X_test))

    return train_acc, test_acc


def train_probes_all_layers(
    layer_activations: Dict[str, List],
    labels: List[int],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Train probes at each layer.

    Returns:
        Dict mapping layer name to {train_acc, test_acc}
    """
    y = np.array(labels)
    results = {}

    # Sort layers by index
    layer_names = sorted(layer_activations.keys(), key=lambda x: int(x.split('_')[1]))

    print(f"\nTraining probes for {len(layer_names)} layers...")

    for layer_name in layer_names:
        acts = layer_activations[layer_name]
        X = np.array(acts)

        train_acc, test_acc = train_probe_for_layer(X, y, test_size, random_state)
        results[layer_name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }

        layer_idx = int(layer_name.split('_')[1])
        print(f"  Layer {layer_idx:2d}: train={train_acc:.3f}, test={test_acc:.3f}")

    return results


def plot_probe_accuracy(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    title: str = "Persona Probe Accuracy by Layer"
):
    """
    Plot probe accuracy vs layer depth.
    """
    # Extract layer indices and accuracies
    layer_names = sorted(results.keys(), key=lambda x: int(x.split('_')[1]))
    layer_indices = [int(name.split('_')[1]) for name in layer_names]
    train_accs = [results[name]['train_accuracy'] for name in layer_names]
    test_accs = [results[name]['test_accuracy'] for name in layer_names]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layer_indices, train_accs, 'b-o', label='Train', alpha=0.7)
    ax.plot(layer_indices, test_accs, 'r-o', label='Test', alpha=0.7)

    # Styling
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    # Add chance level line (1/n_personas)
    n_personas = 5  # baseline + 4 personas
    ax.axhline(y=1/n_personas, color='gray', linestyle='--', alpha=0.5, label=f'Chance ({1/n_personas:.2f})')

    plt.tight_layout()

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train linear probes to classify persona from activations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('results_dir', help='Directory containing result JSON files with activations')
    parser.add_argument('--output', '-o', help='Output path for accuracy plot (PNG)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-results', help='Path to save detailed results JSON')

    args = parser.parse_args()

    # Load activations
    print(f"Loading activations from {args.results_dir}...")
    try:
        layer_activations, labels, n_layers, persona_map = load_activations_from_results(args.results_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if not layer_activations:
        print("Error: No activation data found. Run experiments with --track-activations first.")
        return 1

    # Train probes
    results = train_probes_all_layers(
        layer_activations, labels,
        test_size=args.test_size,
        random_state=args.seed
    )

    # Find best layer
    best_layer = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    best_acc = results[best_layer]['test_accuracy']
    print(f"\nBest layer: {best_layer} (test accuracy: {best_acc:.3f})")

    # Plot results
    plot_probe_accuracy(
        results,
        output_path=args.output,
        title=f"Persona Probe Accuracy by Layer\n({len(labels)} examples, {len(persona_map)} personas)"
    )

    # Save detailed results
    if args.save_results:
        output_data = {
            'n_layers': n_layers,
            'n_examples': len(labels),
            'personas': {v: k for k, v in persona_map.items()},  # label -> name
            'test_size': args.test_size,
            'seed': args.seed,
            'results': {
                layer: {
                    'layer_index': int(layer.split('_')[1]),
                    'train_accuracy': float(res['train_accuracy']),
                    'test_accuracy': float(res['test_accuracy'])
                }
                for layer, res in results.items()
            },
            'best_layer': best_layer,
            'best_test_accuracy': float(best_acc)
        }

        with open(args.save_results, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved detailed results to {args.save_results}")

    return 0


if __name__ == "__main__":
    exit(main())
