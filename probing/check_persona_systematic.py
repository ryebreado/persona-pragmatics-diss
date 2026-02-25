#!/usr/bin/env python3
"""
Check whether any persona systematically deviates from the mean
in conj/quant probe accuracy across layers.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats


def load_probe_layers(json_path):
    """Load {layer: accuracy} from probe result JSON."""
    with open(json_path) as f:
        data = json.load(f)
    entry = data[0] if isinstance(data, list) else data
    return {l['layer']: l['accuracy'] for l in entry['layers']}


def main():
    probe_dir = Path('probing/results/qwen3_8b_run_04/probes/conj_quant')

    persona_files = {
        'Baseline': None,  # handled separately
        'Anti-Gricean': 'qwen3_8b_anti_gricean_with_acts_20260129_034357',
        'Literal Thinker': 'qwen3_8b_literal_thinker_with_acts_20260202_213603',
        'Helpful Teacher': 'qwen3_8b_helpful_teacher_with_acts_20260129_035524',
        'Pragmaticist': 'qwen3_8b_pragmaticist_with_acts_20260129_041028',
    }

    # Load baseline
    with open(probe_dir / 'conj_quant_split_last_token.json') as f:
        bl = json.load(f)

    all_layers = list(range(36))

    for probe_type in ['conj', 'quant']:
        print(f"\n{'#'*70}")
        print(f"#  UNDERINF-{probe_type.upper()} PROBE — Persona deviation analysis")
        print(f"{'#'*70}")

        # Collect all persona curves
        curves = {}

        # Baseline
        for entry in bl:
            if probe_type in entry['name']:
                curves['Baseline'] = {l['layer']: l['accuracy'] for l in entry['layers']}

        # Others
        for persona, prefix in persona_files.items():
            if persona == 'Baseline':
                continue
            fpath = probe_dir / f'{prefix}_underinf_{probe_type}_last_token.json'
            if fpath.exists():
                curves[persona] = load_probe_layers(fpath)

        personas = list(curves.keys())
        n_layers = len(all_layers)

        # Build matrix: (n_personas, n_layers)
        mat = np.array([[curves[p].get(l, np.nan) for l in all_layers] for p in personas])

        # Per-layer mean across personas
        layer_means = mat.mean(axis=0)

        # Deviation from mean at each layer
        deviations = mat - layer_means[np.newaxis, :]  # (n_personas, n_layers)

        # Focus on noisy layers (0-18) where there's variance to explain
        noisy = list(range(0, 19))
        stable = list(range(19, 36))

        print(f"\n  NOISY LAYERS (0-18): Mean deviation from cross-persona mean")
        print(f"  {'Persona':<20} {'Mean dev':<10} {'SD dev':<10} {'% layers above mean'}")
        print(f"  {'-'*60}")

        for i, persona in enumerate(personas):
            devs = deviations[i, noisy]
            mean_dev = devs.mean()
            sd_dev = devs.std()
            pct_above = (devs > 0).mean() * 100
            print(f"  {persona:<20} {mean_dev:+.4f}    {sd_dev:.4f}    {pct_above:.0f}%")

        # Friedman test: is there a systematic persona effect across layers?
        # (non-parametric repeated measures)
        print(f"\n  Friedman test (are personas systematically ranked differently across layers 0-18?):")
        noisy_mat = mat[:, noisy]  # (n_personas, n_noisy_layers)
        # Friedman expects (n_observations, n_treatments) = (n_layers, n_personas)
        stat, p = stats.friedmanchisquare(*[noisy_mat[i, :] for i in range(len(personas))])
        print(f"    chi2={stat:.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")

        # Rank each persona at each layer, see if any persona's mean rank deviates
        print(f"\n  Mean rank across layers 0-18 (1=lowest accuracy, 5=highest):")
        ranks = np.zeros_like(noisy_mat)
        for j in range(len(noisy)):
            ranks[:, j] = stats.rankdata(noisy_mat[:, j])

        for i, persona in enumerate(personas):
            mean_rank = ranks[i, :].mean()
            print(f"    {persona:<20} {mean_rank:.2f}")

        # Pairwise: does any specific persona differ from the rest?
        print(f"\n  Per-persona Wilcoxon signed-rank vs group mean (layers 0-18):")
        for i, persona in enumerate(personas):
            devs = deviations[i, noisy]
            if np.all(devs == 0):
                print(f"    {persona:<20} all zero")
                continue
            try:
                stat, p = stats.wilcoxon(devs, alternative='two-sided')
                print(f"    {persona:<20} median dev={np.median(devs):+.4f}, W={stat:.0f}, p={p:.4f} {'*' if p < 0.05 else ''}")
            except ValueError as e:
                print(f"    {persona:<20} {e}")


if __name__ == "__main__":
    main()
