#!/usr/bin/env python3
"""
Statistical test: Is the early-layer attention difference between accepted
and rejected underinformative trials significant in Anti-Gricean?

Tests whether elevated statement→outcome attention in layers 0-8 is
associated with behavioral response (accept vs reject).
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_per_trial_attention(attention_path, behavioral_path, categories, metric='statement_to_outcome'):
    """Load per-trial attention and split by behavioral outcome."""
    with open(attention_path) as f:
        attention_data = json.load(f)
    with open(behavioral_path) as f:
        behavioral_data = json.load(f)

    behavioral_by_id = {}
    for r in behavioral_data.get('results', []):
        tid = r.get('test_id')
        if tid is not None:
            behavioral_by_id[str(tid)] = r

    accepted_trials = []  # incorrectly said "yes" to underinf
    rejected_trials = []  # correctly said "no" to underinf

    for item in attention_data:
        if item['category'] not in categories:
            continue
        if metric not in item:
            continue

        test_id = str(item.get('test_id'))
        if test_id not in behavioral_by_id:
            continue

        behavior = behavioral_by_id[test_id]
        attn = np.array(item[metric])  # shape: (n_layers, n_heads)

        if behavior.get('correct'):
            rejected_trials.append(attn)
        else:
            accepted_trials.append(attn)

    return np.array(accepted_trials), np.array(rejected_trials)


def run_test_battery(accepted, rejected, label, layer_range):
    """Run t-test, Mann-Whitney, Cohen's d on a layer range."""
    n_acc, n_rej = len(accepted), len(rejected)
    lo, hi = layer_range

    acc_means = accepted[:, lo:hi, :].mean(axis=(1, 2))
    rej_means = rejected[:, lo:hi, :].mean(axis=(1, 2))

    t_stat, t_p = stats.ttest_ind(acc_means, rej_means, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(acc_means, rej_means, alternative='two-sided')
    pooled_std = np.sqrt((acc_means.std()**2 + rej_means.std()**2) / 2)
    cohens_d = (acc_means.mean() - rej_means.mean()) / pooled_std if pooled_std > 0 else 0

    print(f"\n  {label}:")
    print(f"    Accepted mean: {acc_means.mean():.6f} (SD={acc_means.std():.6f})")
    print(f"    Rejected mean: {rej_means.mean():.6f} (SD={rej_means.std():.6f})")
    print(f"    Difference:    {acc_means.mean() - rej_means.mean():+.6f}")
    print(f"    Welch's t-test:  t={t_stat:.3f}, p={t_p:.4f} {'*' if t_p < 0.05 else ''}")
    print(f"    Mann-Whitney U:  U={u_stat:.0f}, p={u_p:.4f} {'*' if u_p < 0.05 else ''}")
    print(f"    Cohen's d:       {cohens_d:.3f}")

    return cohens_d


def main():
    attn_dir = Path('probing/results/qwen3_8b_run_04/attention')
    behav_dir = Path('results/qwen3_8b_run_04')

    personas = {
        'anti_gricean': (
            attn_dir / 'attention_anti_gricean.json',
            behav_dir / 'qwen3_8b_anti_gricean_with_acts_20260129_034357.json',
        ),
        'literal_thinker': (
            attn_dir / 'attention_literal_thinker.json',
            behav_dir / 'qwen3_8b_literal_thinker_with_acts_20260202_213603.json',
        ),
        'baseline': (
            attn_dir / 'attention_baseline.json',
            behav_dir / 'qwen3_8b_baseline_with_acts_20260129_033222.json',
        ),
    }

    early_layers = (0, 9)   # layers 0-8 inclusive
    mid_layers = (12, 22)   # layers 12-21 (peak region)

    # Category splits to test
    category_splits = {
        'ALL underinf': ['underinf-conj', 'underinf-quant'],
        'CONJ only':    ['underinf-conj'],
        'QUANT only':   ['underinf-quant'],
    }

    for persona_name, (attn_path, behav_path) in personas.items():
        if not attn_path.exists() or not behav_path.exists():
            print(f"Skipping {persona_name}: missing data")
            continue

        print(f"\n{'#'*70}")
        print(f"#  {persona_name.upper()}")
        print(f"{'#'*70}")

        for split_name, categories in category_splits.items():
            accepted, rejected = load_per_trial_attention(
                attn_path, behav_path, categories
            )

            n_acc, n_rej = len(accepted), len(rejected)
            print(f"\n{'='*70}")
            print(f"  {split_name}  (accepted n={n_acc}, rejected n={n_rej})")
            print(f"{'='*70}")

            if n_acc < 2 or n_rej < 2:
                print("  Not enough trials in both groups for statistical test.")
                continue

            for label, layer_range in [("Early layers 0-8", early_layers), ("Mid layers 12-21", mid_layers)]:
                run_test_battery(accepted, rejected, label, layer_range)

            # Per-layer t-tests for layers 0-8
            print(f"\n  Per-layer Welch's t-tests (layers 0-8):")
            print(f"  {'Layer':<8} {'Acc mean':<12} {'Rej mean':<12} {'t':<10} {'p':<10} {'sig'}")
            print(f"  {'-'*60}")
            p_values = []
            for layer in range(early_layers[0], early_layers[1]):
                acc_layer = accepted[:, layer, :].mean(axis=1)
                rej_layer = rejected[:, layer, :].mean(axis=1)
                t_stat, t_p = stats.ttest_ind(acc_layer, rej_layer, equal_var=False)
                p_values.append(t_p)
                sig = '*' if t_p < 0.05 else ''
                print(f"  {layer:<8} {acc_layer.mean():.6f}     {rej_layer.mean():.6f}     {t_stat:<10.3f} {t_p:<10.4f} {sig}")

            bonf_threshold = 0.05 / len(p_values)
            any_bonf = any(p < bonf_threshold for p in p_values)
            print(f"\n  Bonferroni threshold (9 tests): p < {bonf_threshold:.4f}")
            print(f"  Any layer survives Bonferroni: {'Yes' if any_bonf else 'No'}")

            # Permutation test
            print(f"\n  Permutation test (layers 0-8 mean, 10000 permutations):")
            all_trials = np.concatenate([
                accepted[:, early_layers[0]:early_layers[1], :].mean(axis=(1, 2)),
                rejected[:, early_layers[0]:early_layers[1], :].mean(axis=(1, 2)),
            ])
            observed_diff = (
                accepted[:, early_layers[0]:early_layers[1], :].mean(axis=(1, 2)).mean() -
                rejected[:, early_layers[0]:early_layers[1], :].mean(axis=(1, 2)).mean()
            )
            n_perms = 10000
            rng = np.random.default_rng(42)
            perm_diffs = np.empty(n_perms)
            for i in range(n_perms):
                perm = rng.permutation(len(all_trials))
                perm_acc = all_trials[perm[:n_acc]]
                perm_rej = all_trials[perm[n_acc:]]
                perm_diffs[i] = perm_acc.mean() - perm_rej.mean()

            perm_p = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()
            print(f"    Observed diff: {observed_diff:+.6f}")
            print(f"    Permutation p: {perm_p:.4f} {'*' if perm_p < 0.05 else ''}")


if __name__ == "__main__":
    main()
