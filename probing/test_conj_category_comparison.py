#!/usr/bin/env python3
"""
Compare early-layer statement→outcome attention across categories in AG.
Key question: Does false-conj show the same early-layer elevation as
accepted underinf-conj?
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_attention_by_category_and_response(attention_path, behavioral_path, metric='statement_to_outcome'):
    """Load per-trial attention, grouped by (category, response)."""
    with open(attention_path) as f:
        attention_data = json.load(f)
    with open(behavioral_path) as f:
        behavioral_data = json.load(f)

    behavioral_by_id = {}
    for r in behavioral_data.get('results', []):
        tid = r.get('test_id')
        if tid is not None:
            behavioral_by_id[str(tid)] = r

    groups = {}
    for item in attention_data:
        if metric not in item:
            continue
        test_id = str(item.get('test_id'))
        if test_id not in behavioral_by_id:
            continue

        cat = item['category']
        behavior = behavioral_by_id[test_id]
        correct = behavior.get('correct')
        attn = np.array(item[metric])  # (n_layers, n_heads)

        key = (cat, 'correct' if correct else 'incorrect')
        if key not in groups:
            groups[key] = []
        groups[key].append(attn)

    # Convert to arrays
    for key in groups:
        groups[key] = np.array(groups[key])

    return groups


def print_group_summary(groups, layer_range, label):
    """Print mean attention for each group in the layer range."""
    lo, hi = layer_range
    print(f"\n  {label}")
    print(f"  {'Category':<20} {'Response':<12} {'n':<6} {'Mean attn':<12} {'SD'}")
    print(f"  {'-'*62}")

    group_means = {}
    for (cat, resp), trials in sorted(groups.items()):
        means = trials[:, lo:hi, :].mean(axis=(1, 2))  # one value per trial
        print(f"  {cat:<20} {resp:<12} {len(trials):<6} {means.mean():.6f}     {means.std():.6f}")
        group_means[(cat, resp)] = means

    return group_means


def main():
    attn_dir = Path('probing/results/qwen3_8b_run_04/attention')
    behav_dir = Path('results/qwen3_8b_run_04')

    ag_attn = attn_dir / 'attention_anti_gricean.json'
    ag_behav = behav_dir / 'qwen3_8b_anti_gricean_with_acts_20260129_034357.json'
    bl_attn = attn_dir / 'attention_baseline.json'
    bl_behav = behav_dir / 'qwen3_8b_baseline_with_acts_20260129_033222.json'

    early = (0, 9)
    mid = (12, 22)

    for persona_label, attn_path, behav_path in [
        ("ANTI-GRICEAN", ag_attn, ag_behav),
        ("BASELINE", bl_attn, bl_behav),
    ]:
        print(f"\n{'#'*70}")
        print(f"#  {persona_label}")
        print(f"{'#'*70}")

        groups = load_attention_by_category_and_response(attn_path, behav_path)

        for label, layer_range in [("Early layers 0-8", early), ("Mid layers 12-21", mid)]:
            group_means = print_group_summary(groups, layer_range, label)

        # Key comparisons for AG
        if persona_label == "ANTI-GRICEAN":
            print(f"\n  {'='*62}")
            print(f"  KEY COMPARISONS (early layers 0-8)")
            print(f"  {'='*62}")

            comparisons = [
                ("underinf-conj ACCEPTED vs false-conj CORRECT",
                 ('underinf-conj', 'incorrect'), ('false-conj', 'correct')),
                ("underinf-conj ACCEPTED vs true-conj CORRECT",
                 ('underinf-conj', 'incorrect'), ('true-conj', 'correct')),
                ("underinf-conj ACCEPTED vs underinf-quant REJECTED",
                 ('underinf-conj', 'incorrect'), ('underinf-quant', 'correct')),
                ("false-conj CORRECT vs false-quant CORRECT",
                 ('false-conj', 'correct'), ('false-quant', 'correct')),
                ("true-conj CORRECT vs true-quant CORRECT",
                 ('true-conj', 'correct'), ('true-quant', 'correct')),
            ]

            lo, hi = early
            for label, key_a, key_b in comparisons:
                if key_a not in groups or key_b not in groups:
                    print(f"\n  {label}: missing group")
                    continue

                a = groups[key_a][:, lo:hi, :].mean(axis=(1, 2))
                b = groups[key_b][:, lo:hi, :].mean(axis=(1, 2))

                t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
                pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
                d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

                print(f"\n  {label}")
                print(f"    A: {a.mean():.6f} (n={len(a)})  vs  B: {b.mean():.6f} (n={len(b)})")
                print(f"    t={t_stat:.3f}, p={t_p:.4f} {'*' if t_p < 0.05 else ''},  d={d:.3f}")


if __name__ == "__main__":
    main()
