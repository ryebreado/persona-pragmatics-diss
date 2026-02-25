#!/usr/bin/env python3
"""
Produce a compact summary of conj vs quant probe accuracy across personas.

Two-stage summary:
  - Early/noisy layers (e.g. 5, 10, 15) where there's variance
  - Late/stable layers (e.g. 19, 25, 30) where accuracy is high and stable

Shows: 1) conj vs quant separation exists, 2) personas don't change it much.
"""

import json
import numpy as np
from pathlib import Path


def load_probe_results(json_path):
    """Load layer accuracies from a probe result JSON (single-experiment file)."""
    with open(json_path) as f:
        data = json.load(f)
    # Single-experiment files have a list with one entry
    entry = data[0] if isinstance(data, list) else data
    layers = {l['layer']: (l['accuracy'], l['std']) for l in entry['layers']}
    return layers


def main():
    probe_dir = Path('probing/results/qwen3_8b_run_04/probes/conj_quant')

    # Map persona label -> file prefix
    personas = {
        'Baseline':  'conj_quant_split_last_token.json',  # summary has both, handle separately
        'Anti-Gricean': 'qwen3_8b_anti_gricean_with_acts_20260129_034357',
        'Literal Thinker': 'qwen3_8b_literal_thinker_with_acts_20260202_213603',
        'Helpful Teacher': 'qwen3_8b_helpful_teacher_with_acts_20260129_035524',
        'Pragmaticist': 'qwen3_8b_pragmaticist_with_acts_20260129_041028',
    }

    # Load baseline from summary file
    with open(probe_dir / 'conj_quant_split_last_token.json') as f:
        baseline_data = json.load(f)

    results = {}  # (persona, probe_type) -> {layer: (acc, std)}

    for entry in baseline_data:
        probe_type = 'conj' if 'conj' in entry['name'] else 'quant'
        layers = {l['layer']: (l['accuracy'], l['std']) for l in entry['layers']}
        results[('Baseline', probe_type)] = layers

    # Load persona files
    for persona, prefix in personas.items():
        if persona == 'Baseline':
            continue
        for probe_type in ['conj', 'quant']:
            fpath = probe_dir / f'{prefix}_underinf_{probe_type}_last_token.json'
            if fpath.exists():
                results[(persona, probe_type)] = load_probe_results(fpath)

    # Representative layers
    repr_layers = [5, 10, 15, 19, 25, 30, 35]

    persona_order = ['Baseline', 'Anti-Gricean', 'Literal Thinker', 'Helpful Teacher', 'Pragmaticist']

    for probe_type in ['conj', 'quant']:
        print(f"\n{'='*80}")
        print(f"  UNDERINF-{probe_type.upper()} PROBE (underinf-{probe_type} vs other)")
        print(f"{'='*80}")

        # Header
        header = f"  {'Persona':<20}"
        for l in repr_layers:
            header += f"  L{l:<5}"
        print(header)
        print(f"  {'-'*72}")

        for persona in persona_order:
            key = (persona, probe_type)
            if key not in results:
                continue
            layers = results[key]
            row = f"  {persona:<20}"
            for l in repr_layers:
                if l in layers:
                    acc, std = layers[l]
                    row += f"  {acc:.3f}"
                else:
                    row += f"  {'—':>5}"
            print(row)

    # Summary statistics: mean and range across personas at each layer
    print(f"\n\n{'='*80}")
    print(f"  CROSS-PERSONA SUMMARY (mean ± range across 5 conditions)")
    print(f"{'='*80}")

    header = f"  {'Probe':<12}"
    for l in repr_layers:
        header += f"  {'L' + str(l):<12}"
    print(header)
    print(f"  {'-'*100}")

    for probe_type in ['conj', 'quant']:
        row = f"  {probe_type:<12}"
        for l in repr_layers:
            accs = []
            for persona in persona_order:
                key = (persona, probe_type)
                if key in results and l in results[key]:
                    accs.append(results[key][l][0])
            if accs:
                mean_acc = np.mean(accs)
                spread = max(accs) - min(accs)
                row += f"  {mean_acc:.3f}±{spread:.3f}"
            else:
                row += f"  {'—':>10}"
        print(row)

    # Conj vs quant difference at each layer (averaged across personas)
    print(f"\n  {'conj-quant':<12}", end="")
    for l in repr_layers:
        conj_accs = []
        quant_accs = []
        for persona in persona_order:
            if ('Baseline', 'conj') in results and l in results[('Baseline', 'conj')]:
                pass  # already included
            kc = (persona, 'conj')
            kq = (persona, 'quant')
            if kc in results and kq in results and l in results[kc] and l in results[kq]:
                conj_accs.append(results[kc][l][0])
                quant_accs.append(results[kq][l][0])
        if conj_accs and quant_accs:
            diff = np.mean(conj_accs) - np.mean(quant_accs)
            print(f"  {diff:+.3f}      ", end="")
        else:
            print(f"  {'—':>10}", end="")
    print()


if __name__ == "__main__":
    main()
