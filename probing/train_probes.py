#!/usr/bin/env python3
"""
Linear probing for scalar implicature activations.

Supports multiple experiments:
- Baseline layer curve (underinf vs not)
- Cross-condition transfer (train baseline, test persona)
- 3-way classification (true vs false vs underinf)
- Conj vs quant split

Handles different token positions (last_token, mean_pooled, future: quantifier, conj).
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


@dataclass
class ProbeResult:
    """Results from a single probe."""
    layer: int
    accuracy: float
    std: float
    n_samples: int
    labels: List[str]


@dataclass
class ExperimentResult:
    """Results from a full experiment across layers."""
    name: str
    token_position: str
    layer_results: List[ProbeResult]
    train_condition: str
    test_condition: Optional[str] = None


def load_activations(path: str) -> List[Dict]:
    """Load activations from .pt file."""
    data = torch.load(path, weights_only=True)
    return data


def get_activation_matrix(
    activations: List[Dict],
    layer: int,
    token_position: str = "last_token",
    return_mask: bool = False
) -> np.ndarray:
    """
    Extract activation matrix for a specific layer and token position.

    Args:
        activations: List of activation dicts from .pt file
        layer: Layer index
        token_position: One of 'last_token', 'mean_pooled', or 'keyword_*' positions
        return_mask: If True, return (matrix, mask) where mask indicates which examples have this position

    Returns:
        Array of shape (n_samples, hidden_dim), or tuple of (array, mask) if return_mask=True
        For keyword positions, only includes examples where that keyword was found.
    """
    acts = []
    mask = []
    for example in activations:
        if token_position in example:
            # Shape is (n_layers, hidden_dim), we want layer at index `layer`
            layer_act = example[token_position][layer].float().numpy()
            acts.append(layer_act)
            mask.append(True)
        elif token_position.startswith('keyword_'):
            # Keyword not found in this example - skip it
            mask.append(False)
        else:
            raise ValueError(f"Token position '{token_position}' not found. "
                           f"Available: {list(example.keys())}")

    if not acts:
        raise ValueError(f"No examples have token position '{token_position}'")

    if return_mask:
        return np.stack(acts), np.array(mask)
    return np.stack(acts)


def get_labels(
    activations: List[Dict],
    task: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract labels for classification task.

    Args:
        activations: List of activation dicts
        task: One of:
            - 'underinf_binary': underinf vs everything else
            - 'underinf_conj': underinf-conj vs everything else
            - 'underinf_quant': underinf-quant vs everything else
            - 'three_way': true vs false vs underinf
            - 'six_way': all six categories

    Returns:
        Tuple of (label array, label names)
    """
    categories = [ex['category'] for ex in activations]

    if task == 'underinf_binary':
        labels = [1 if 'underinf' in cat else 0 for cat in categories]
        label_names = ['not_underinf', 'underinf']

    elif task == 'underinf_conj':
        labels = [1 if cat == 'underinf-conj' else 0 for cat in categories]
        label_names = ['other', 'underinf-conj']

    elif task == 'underinf_quant':
        labels = [1 if cat == 'underinf-quant' else 0 for cat in categories]
        label_names = ['other', 'underinf-quant']

    elif task == 'three_way':
        label_map = {}
        for cat in categories:
            if cat.startswith('true'):
                label_map[cat] = 0
            elif cat.startswith('false'):
                label_map[cat] = 1
            elif cat.startswith('underinf'):
                label_map[cat] = 2
        labels = [label_map[cat] for cat in categories]
        label_names = ['true', 'false', 'underinf']

    elif task == 'six_way':
        unique_cats = sorted(set(categories))
        cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
        labels = [cat_to_idx[cat] for cat in categories]
        label_names = unique_cats

    else:
        raise ValueError(f"Unknown task: {task}")

    return np.array(labels), label_names


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    max_iter: int = 1000
) -> Tuple[float, float, LogisticRegression]:
    """
    Train logistic regression probe with cross-validation.

    Returns:
        Tuple of (mean accuracy, std, fitted model on all data)
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    clf = LogisticRegression(max_iter=max_iter, solver='lbfgs')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    # Fit on all data for later use
    clf.fit(X_scaled, y)

    return scores.mean(), scores.std(), clf


def train_and_test_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000
) -> Tuple[float, LogisticRegression]:
    """
    Train probe on one dataset, test on another (for transfer experiments).

    Returns:
        Tuple of (test accuracy, fitted model)
    """
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    clf = LogisticRegression(max_iter=max_iter, solver='lbfgs')
    clf.fit(X_train_scaled, y_train)

    # Test
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, clf


def run_layer_curve(
    activations: List[Dict],
    task: str,
    token_position: str = "last_token",
    n_folds: int = 5,
    layers: Optional[List[int]] = None
) -> ExperimentResult:
    """
    Run probing experiment across all layers.

    Args:
        activations: Loaded activation data
        task: Classification task
        token_position: Token position to use
        n_folds: Cross-validation folds
        layers: Specific layers to probe (None = all)

    Returns:
        ExperimentResult with accuracy at each layer
    """
    # Get labels for all examples
    y_all, label_names = get_labels(activations, task)

    # For keyword positions, we need to filter to examples that have the keyword
    if token_position.startswith('keyword_'):
        # Find which examples have this keyword
        mask = np.array([token_position in ex for ex in activations])
        n_with_keyword = mask.sum()
        print(f"  Note: {n_with_keyword}/{len(activations)} examples have '{token_position}'")
        y = y_all[mask]

        # Find first example with this keyword to get n_layers
        first_with_kw = next(ex for ex in activations if token_position in ex)
        n_layers = first_with_kw[token_position].shape[0]
    else:
        mask = None
        y = y_all
        n_layers = activations[0][token_position].shape[0]

    if layers is None:
        layers = list(range(n_layers))

    results = []
    for layer in layers:
        X = get_activation_matrix(activations, layer, token_position)
        acc, std, _ = train_probe(X, y, n_folds=n_folds)
        results.append(ProbeResult(
            layer=layer,
            accuracy=acc,
            std=std,
            n_samples=len(y),
            labels=label_names
        ))
        print(f"  Layer {layer:2d}: {acc:.3f} (+/- {std:.3f})")

    return ExperimentResult(
        name=f"{task}_{token_position}",
        token_position=token_position,
        layer_results=results,
        train_condition="same"
    )


def run_transfer_experiment(
    train_activations: List[Dict],
    test_activations: List[Dict],
    task: str,
    token_position: str = "last_token",
    layers: Optional[List[int]] = None
) -> ExperimentResult:
    """
    Train probes on one condition, test on another.

    Args:
        train_activations: Training condition activations
        test_activations: Test condition activations
        task: Classification task
        token_position: Token position to use
        layers: Specific layers to probe (None = all)

    Returns:
        ExperimentResult with transfer accuracy at each layer
    """
    # Get labels (should be same categories in both)
    y_train, label_names = get_labels(train_activations, task)
    y_test, _ = get_labels(test_activations, task)

    # Determine number of layers
    n_layers = train_activations[0][token_position].shape[0]
    if layers is None:
        layers = list(range(n_layers))

    results = []
    for layer in layers:
        X_train = get_activation_matrix(train_activations, layer, token_position)
        X_test = get_activation_matrix(test_activations, layer, token_position)

        acc, _ = train_and_test_probe(X_train, y_train, X_test, y_test)
        results.append(ProbeResult(
            layer=layer,
            accuracy=acc,
            std=0.0,  # No CV for transfer
            n_samples=len(y_test),
            labels=label_names
        ))
        print(f"  Layer {layer:2d}: {acc:.3f}")

    return ExperimentResult(
        name=f"{task}_{token_position}_transfer",
        token_position=token_position,
        layer_results=results,
        train_condition="train",
        test_condition="test"
    )


def plot_layer_curve(
    results: List[ExperimentResult],
    title: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """Plot layer-wise accuracy curves for one or more experiments."""
    fig, ax = plt.subplots(figsize=figsize)

    for result in results:
        layers = [r.layer for r in result.layer_results]
        accs = [r.accuracy for r in result.layer_results]
        stds = [r.std for r in result.layer_results]

        label = result.name
        if result.test_condition:
            label = f"{result.name} (train→test)"

        ax.plot(layers, accs, marker='o', label=label, linewidth=2, markersize=4)
        if any(s > 0 for s in stds):
            ax.fill_between(
                layers,
                [a - s for a, s in zip(accs, stds)],
                [a + s for a, s in zip(accs, stds)],
                alpha=0.2
            )

    # Add chance level
    n_classes = len(results[0].layer_results[0].labels)
    chance = 1.0 / n_classes
    ax.axhline(y=chance, color='gray', linestyle='--', label=f'Chance ({chance:.2f})')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    return fig


def save_results(results: List[ExperimentResult], output_path: str):
    """Save experiment results to JSON."""
    data = []
    for result in results:
        data.append({
            'name': result.name,
            'token_position': result.token_position,
            'train_condition': result.train_condition,
            'test_condition': result.test_condition,
            'layers': [
                {
                    'layer': r.layer,
                    'accuracy': r.accuracy,
                    'std': r.std,
                    'n_samples': r.n_samples,
                    'labels': r.labels
                }
                for r in result.layer_results
            ]
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train linear probes on scalar implicature activations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('activations', nargs='+',
                       help='Path(s) to activation .pt file(s). '
                            'For transfer experiments, provide train and test files.')

    # Task selection
    parser.add_argument('--task', default='underinf_binary',
                       choices=['underinf_binary', 'underinf_conj', 'underinf_quant',
                               'three_way', 'six_way'],
                       help='Classification task')

    # Token position
    parser.add_argument('--token-position', default='last_token',
                       help='Token position to probe (e.g., last_token, mean_pooled)')

    # Experiment type
    parser.add_argument('--transfer', action='store_true',
                       help='Run transfer experiment (requires 2 activation files: train, test)')

    # Layer selection
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer indices to probe (default: all)')

    # Cross-validation
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of CV folds')

    # Output
    parser.add_argument('--output-dir', type=str, default='probing/results',
                       help='Output directory for results')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (auto-generated if not specified)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')

    args = parser.parse_args()

    # Parse layers
    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load activations
    print(f"Loading activations from {args.activations[0]}...")
    activations = load_activations(args.activations[0])
    print(f"  Loaded {len(activations)} examples")
    print(f"  Token positions available: {list(activations[0].keys())}")

    n_layers = activations[0][args.token_position].shape[0]
    hidden_dim = activations[0][args.token_position].shape[1]
    print(f"  {n_layers} layers, {hidden_dim} hidden dim")

    results = []

    if args.transfer:
        # Transfer experiment
        if len(args.activations) != 2:
            parser.error("Transfer experiment requires exactly 2 activation files (train, test)")

        print(f"\nLoading test activations from {args.activations[1]}...")
        test_activations = load_activations(args.activations[1])
        print(f"  Loaded {len(test_activations)} examples")

        print(f"\nRunning transfer experiment: {args.task}")
        print(f"  Train: {Path(args.activations[0]).stem}")
        print(f"  Test:  {Path(args.activations[1]).stem}")

        result = run_transfer_experiment(
            activations, test_activations,
            task=args.task,
            token_position=args.token_position,
            layers=layers
        )
        result.train_condition = Path(args.activations[0]).stem
        result.test_condition = Path(args.activations[1]).stem
        results.append(result)

    else:
        # Standard layer curve
        print(f"\nRunning layer curve: {args.task}")
        result = run_layer_curve(
            activations,
            task=args.task,
            token_position=args.token_position,
            n_folds=args.folds,
            layers=layers
        )
        result.train_condition = Path(args.activations[0]).stem
        results.append(result)

    # Generate experiment name
    if args.name:
        exp_name = args.name
    else:
        base_name = Path(args.activations[0]).stem.replace('_activations', '')
        exp_name = f"{base_name}_{args.task}_{args.token_position}"
        if args.transfer:
            exp_name += "_transfer"

    # Save results
    json_path = output_dir / f"{exp_name}.json"
    save_results(results, str(json_path))

    # Plot
    if not args.no_plot:
        plot_path = output_dir / f"{exp_name}.png"
        title = f"{args.task} ({args.token_position})"
        if args.transfer:
            title += " - Transfer"
        plot_layer_curve(results, title, str(plot_path))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        accs = [r.accuracy for r in result.layer_results]
        best_layer = result.layer_results[np.argmax(accs)].layer
        best_acc = max(accs)
        print(f"{result.name}:")
        print(f"  Best layer: {best_layer} ({best_acc:.3f})")
        print(f"  Mean accuracy: {np.mean(accs):.3f}")


if __name__ == "__main__":
    main()
