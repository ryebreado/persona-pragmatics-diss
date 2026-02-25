#!/usr/bin/env python3
"""
Test whether the mid-layer accept/reject difference survives within
conj and quant categories, or is also a confound.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_attention_by_category_and_response(attention_path, behavioral_path, metric='statement_to_outcome'):
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
        attn = np.array(item[metric])

        key = (cat, 'correct' if correct else 'incorrect')
        if key not in groups:
            groups[key] = []
        groups[key].append(attn)

    for key in groups:
        groups[key] = np.array(groups[key])
    return groups


def test_within(groups, cat, layer_range, label):
    lo, hi = layer_range
    acc_key = (cat, 'incorrect')  # accepted underinf = incorrect
    rej_key = (cat, 'correct')   # rejected underinf = correct

    acc = groups.get(acc_key)
    rej = groups.get(rej_key)

    print(f"\n  {label}: {cat}")

    if acc is None or rej is None or len(acc) < 2 or len(rej) < 2:
        n_a = len(acc) if acc is not None else 0
        n_r = len(rej) if rej is not None else 0
        print(f"    n_accepted={n_a}, n_rejected={n_r} — insufficient for test")
        if n_a >= 1 and n_r >= 1:
            a_mean = acc[:, lo:hi, :].mean() if acc is not None else float('nan')
            r_mean = rej[:, lo:hi, :].mean() if rej is not None else float('nan')
            print(f"    Accepted mean: {a_mean:.6f}, Rejected mean: {r_mean:.6f}")
        return

    a = acc[:, lo:hi, :].mean(axis=(1, 2))
    r = rej[:, lo:hi, :].mean(axis=(1, 2))

    t_stat, t_p = stats.ttest_ind(a, r, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(a, r, alternative='two-sided')
    pooled_std = np.sqrt((a.std()**2 + r.std()**2) / 2)
    d = (a.mean() - r.mean()) / pooled_std if pooled_std > 0 else 0

    print(f"    Accepted (n={len(acc)}): {a.mean():.6f} (SD={a.std():.6f})")
    print(f"    Rejected (n={len(rej)}): {r.mean():.6f} (SD={r.std():.6f})")
    print(f"    Diff: {a.mean() - r.mean():+.6f}")
    print(f"    Welch t={t_stat:.3f}, p={t_p:.4f} {'*' if t_p < 0.05 else ''},  d={d:.3f}")
    print(f"    Mann-Whitney U={u_stat:.0f}, p={u_p:.4f} {'*' if u_p < 0.05 else ''}")


def main():
    attn_dir = Path('probing/results/qwen3_8b_run_04/attention')
    behav_dir = Path('results/qwen3_8b_run_04')

    mid = (12, 22)

    personas = [
        ("ANTI-GRICEAN",
         attn_dir / 'attention_anti_gricean.json',
         behav_dir / 'qwen3_8b_anti_gricean_with_acts_20260129_034357.json'),
        ("LITERAL THINKER",
         attn_dir / 'attention_literal_thinker.json',
         behav_dir / 'qwen3_8b_literal_thinker_with_acts_20260202_213603.json'),
    ]

    for name, attn_path, behav_path in personas:
        print(f"\n{'#'*70}")
        print(f"#  {name} — Mid-layer (12-21) within-category tests")
        print(f"{'#'*70}")

        groups = load_attention_by_category_and_response(attn_path, behav_path)

        # Also show the conj vs quant structural difference for context
        conj_trials = []
        quant_trials = []
        for (cat, _), data in groups.items():
            if '-conj' in cat:
                conj_trials.append(data)
            elif '-quant' in cat:
                quant_trials.append(data)

        conj_all = np.concatenate(conj_trials)
        quant_all = np.concatenate(quant_trials)
        c = conj_all[:, mid[0]:mid[1], :].mean(axis=(1, 2))
        q = quant_all[:, mid[0]:mid[1], :].mean(axis=(1, 2))
        t_stat, t_p = stats.ttest_ind(c, q, equal_var=False)
        pooled_std = np.sqrt((c.std()**2 + q.std()**2) / 2)
        d = (c.mean() - q.mean()) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Structural: All conj ({c.mean():.6f}, n={len(conj_all)}) vs All quant ({q.mean():.6f}, n={len(quant_all)})")
        print(f"    t={t_stat:.3f}, p={t_p:.4f} {'*' if t_p < 0.05 else ''},  d={d:.3f}")

        # Within-category tests
        for cat in ['underinf-conj', 'underinf-quant']:
            test_within(groups, cat, mid, "Mid layers 12-21")


if __name__ == "__main__":
    main()
