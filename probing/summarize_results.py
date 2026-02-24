"""Print summary tables for probing results.

Usage:
    uv run python probing/summarize_results.py probing/results/qwen3_8b_run_04/probes/
    uv run python probing/summarize_results.py probing/results/qwen3_8b_run_04/probes/transfer_last_token.json
    uv run python probing/summarize_results.py probing/results/qwen3_8b_run_04/probes/ --layers 0,5,10,15,20,25,30,35
    uv run python probing/summarize_results.py probing/results/qwen3_8b_run_04/probes/persona_comparison_last_token.json --sigmoid
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def load_results(path: Path) -> dict[str, list]:
    """Load results from a file or directory. Returns {filename: [experiments]}."""
    if path.is_file():
        return {path.stem: json.loads(path.read_text())}
    elif path.is_dir():
        results = {}
        for f in sorted(path.glob("*.json")):
            results[f.stem] = json.loads(f.read_text())
        return results
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


def peak_layer(layers: list[dict]) -> tuple[int, float, float]:
    """Return (layer, accuracy, std) for the best layer."""
    best = max(layers, key=lambda l: l["accuracy"])
    return best["layer"], best["accuracy"], best["std"]


def print_peak_summary(all_results: dict[str, list]):
    """One-line-per-experiment summary showing peak accuracy."""
    print("=" * 90)
    print(f"{'File':<40} {'Experiment':<25} {'Peak Layer':>5} {'Acc':>7} {'± Std':>7} {'N':>5}")
    print("=" * 90)
    for filename, experiments in all_results.items():
        for exp in experiments:
            layer, acc, std = peak_layer(exp["layers"])
            n = exp["layers"][0].get("n_samples", "?")
            label = exp["name"]
            print(f"{filename:<40} {label:<25} {layer:>5} {acc:>7.1%} {std:>7.1%} {n:>5}")
    print()


def print_layer_table(all_results: dict[str, list], layer_subset: list[int] | None = None):
    """Per-file tables showing accuracy at each layer."""
    for filename, experiments in all_results.items():
        # Determine layers to show
        all_layers = [l["layer"] for l in experiments[0]["layers"]]
        if layer_subset:
            layers_to_show = [l for l in layer_subset if l in all_layers]
        else:
            # Default: show every 5th layer + last layer
            layers_to_show = [l for l in all_layers if l % 5 == 0]
            if all_layers[-1] not in layers_to_show:
                layers_to_show.append(all_layers[-1])

        # Build experiment names
        exp_names = [exp["name"] for exp in experiments]
        name_width = max(len(n) for n in exp_names)
        name_width = max(name_width, 10)

        print(f"── {filename} ──")

        # Header with train/test info
        for exp in experiments:
            train = exp.get("train_condition", "")
            test = exp.get("test_condition", "")
            # Shorten long condition names
            train = train.split("_activations")[0] if "_activations" in train else train
            if test:
                test = test.split("_activations")[0] if "_activations" in test else test
                print(f"  {exp['name']}: train={train} → test={test}")
            else:
                print(f"  {exp['name']}: {train} (CV)")
        print()

        # Column header
        header = f"{'Experiment':<{name_width}}"
        for layer in layers_to_show:
            header += f"  L{layer:>2}"
        header += "  | peak"
        print(header)
        print("-" * len(header))

        # Data rows
        for exp in experiments:
            layer_map = {l["layer"]: l for l in exp["layers"]}
            row = f"{exp['name']:<{name_width}}"
            for layer in layers_to_show:
                acc = layer_map[layer]["accuracy"]
                row += f" {acc:>.0%}".rjust(5)
            pl, pa, ps = peak_layer(exp["layers"])
            row += f"  | L{pl} {pa:.1%}±{ps:.1%}"
            print(row)
        print()


def sigmoid(x, floor, ceiling, k, x0):
    """4-parameter sigmoid: floor + (ceiling - floor) / (1 + exp(-k * (x - x0)))"""
    return floor + (ceiling - floor) / (1 + np.exp(-k * (x - x0)))


def fit_sigmoid(layers: list[dict]) -> dict:
    """Fit a sigmoid to layer accuracy curve. Returns fit parameters and R²."""
    x = np.array([l["layer"] for l in layers], dtype=float)
    y = np.array([l["accuracy"] for l in layers], dtype=float)

    # Initial guesses: floor=min(y), ceiling=max(y), k=1, x0=midpoint
    p0 = [y.min(), y.max(), 1.0, x[len(x) // 2]]
    bounds = (
        [0.0, 0.5, 0.01, -5],      # lower bounds
        [1.0, 1.01, 10.0, 40],      # upper bounds
    )

    try:
        popt, pcov = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_pred = sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return {
            "floor": popt[0],
            "ceiling": popt[1],
            "steepness": popt[2],
            "inflection": popt[3],
            "r2": r2,
        }
    except RuntimeError:
        return None


def print_sigmoid_table(all_results: dict[str, list]):
    """Fit sigmoid per experiment and print inflection point + steepness."""
    print("=" * 95)
    print(f"{'File':<35} {'Experiment':<20} {'Inflect':>7} {'Steep':>6} {'Floor':>6} {'Ceil':>6} {'R²':>6}")
    print("=" * 95)

    for filename, experiments in all_results.items():
        for exp in experiments:
            fit = fit_sigmoid(exp["layers"])
            label = exp["name"]
            if fit:
                print(
                    f"{filename:<35} {label:<20} "
                    f"L{fit['inflection']:>5.1f} "
                    f"{fit['steepness']:>6.2f} "
                    f"{fit['floor']:>5.1%} "
                    f"{fit['ceiling']:>5.1%} "
                    f"{fit['r2']:>6.3f}"
                )
            else:
                print(f"{filename:<35} {label:<20}  (fit failed)")
    print()
    print("Inflect = inflection point (layer where accuracy is halfway between floor and ceiling)")
    print("Steep   = steepness k (higher = sharper transition)")
    print()


def print_deviation_table(all_results: dict[str, list], baseline_name: str = "baseline"):
    """For each persona, show layers where its mean is outside baseline ±1 SD."""
    for filename, experiments in all_results.items():
        # Find baseline
        baseline = None
        personas = []
        for exp in experiments:
            if exp["name"] == baseline_name:
                baseline = exp
            else:
                personas.append(exp)
        if not baseline:
            print(f"  No '{baseline_name}' experiment found in {filename}, skipping")
            continue

        # Build baseline lookup: layer -> (acc, std)
        bl = {l["layer"]: (l["accuracy"], l["std"]) for l in baseline["layers"]}

        print(f"── {filename}: layers where persona mean is outside baseline ±1 SD ──")
        print()

        for persona in personas:
            outside = []
            for l in persona["layers"]:
                layer = l["layer"]
                p_acc = l["accuracy"]
                bl_acc, bl_std = bl[layer]
                if p_acc > bl_acc + bl_std:
                    outside.append((layer, p_acc, bl_acc, bl_std, "+"))
                elif p_acc < bl_acc - bl_std:
                    outside.append((layer, p_acc, bl_acc, bl_std, "-"))

            if outside:
                print(f"  {persona['name']}: {len(outside)} layers outside baseline ±1 SD")
                print(f"    {'Layer':>5}  {'Persona':>8}  {'Baseline':>8}  {'BL ±1SD':>15}  {'Dir':>3}")
                for layer, p_acc, bl_acc, bl_std, direction in outside:
                    lo, hi = bl_acc - bl_std, bl_acc + bl_std
                    print(
                        f"    L{layer:>3}  {p_acc:>8.1%}  {bl_acc:>8.1%}  "
                        f"[{lo:.1%}, {hi:.1%}]  {direction:>3}"
                    )
            else:
                print(f"  {persona['name']}: all layers within baseline ±1 SD")
            print()


def main():
    parser = argparse.ArgumentParser(description="Summarize probing results as tables")
    parser.add_argument("path", help="JSON file or directory of JSON files")
    parser.add_argument("--layers", help="Comma-separated layer numbers to show (e.g. 0,10,20,30)")
    parser.add_argument("--peaks-only", action="store_true", help="Only show peak accuracy summary")
    parser.add_argument("--sigmoid", action="store_true", help="Fit sigmoid curves and report inflection/steepness")
    parser.add_argument("--deviation", action="store_true",
                        help="Show layers where persona mean falls outside baseline ±1 SD")
    parser.add_argument("--baseline", default="baseline", help="Name of baseline experiment (default: baseline)")
    args = parser.parse_args()

    path = Path(args.path)
    layer_subset = [int(x) for x in args.layers.split(",")] if args.layers else None

    all_results = load_results(path)

    if args.sigmoid:
        print_sigmoid_table(all_results)
    elif args.deviation:
        print_deviation_table(all_results, args.baseline)
    else:
        print_peak_summary(all_results)
        if not args.peaks_only:
            print_layer_table(all_results, layer_subset)


if __name__ == "__main__":
    main()
