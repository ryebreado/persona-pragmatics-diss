#!/usr/bin/env python3
"""
Plot showing that the apparent accept/reject early-layer attention difference
in Anti-Gricean is actually a Simpson's paradox: conjunctive items have
structurally higher s→o attention, and AG accepts conj / rejects quant.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
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

    for key in groups:
        groups[key] = np.array(groups[key])

    return groups


def pool_groups(groups, category_filter, response_filter=None):
    """Pool trials matching category substring and optional response."""
    trials = []
    for (cat, resp), data in groups.items():
        if category_filter not in cat:
            continue
        if response_filter and resp != response_filter:
            continue
        trials.append(data)
    if not trials:
        return None
    return np.concatenate(trials, axis=0)


def layer_curve(trials):
    """Mean attention per layer, averaged over heads. Returns (n_layers,)."""
    return trials.mean(axis=2).mean(axis=0)  # mean over heads, then over trials


def layer_curve_with_se(trials):
    """Mean and SE per layer."""
    per_trial = trials.mean(axis=2)  # (n_trials, n_layers)
    mean = per_trial.mean(axis=0)
    se = per_trial.std(axis=0) / np.sqrt(len(trials))
    return mean, se


def main():
    attn_dir = Path('probing/results/qwen3_8b_run_04/attention')
    behav_dir = Path('results/qwen3_8b_run_04')

    ag_groups = load_attention_by_category_and_response(
        attn_dir / 'attention_anti_gricean.json',
        behav_dir / 'qwen3_8b_anti_gricean_with_acts_20260129_034357.json',
    )

    layers = np.arange(36)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    # --- Panel A: Original accept/reject (misleading) ---
    ax = axes[0, 0]
    underinf_acc = pool_groups(ag_groups, 'underinf', 'incorrect')
    underinf_rej = pool_groups(ag_groups, 'underinf', 'correct')

    acc_mean, acc_se = layer_curve_with_se(underinf_acc)
    rej_mean, rej_se = layer_curve_with_se(underinf_rej)

    ax.plot(layers, acc_mean, color='red', linewidth=2, label=f'Accepted (n={len(underinf_acc)})')
    ax.fill_between(layers, acc_mean - acc_se, acc_mean + acc_se, color='red', alpha=0.15)
    ax.plot(layers, rej_mean, color='blue', linewidth=2, label=f'Rejected (n={len(underinf_rej)})')
    ax.fill_between(layers, rej_mean - rej_se, rej_mean + rej_se, color='blue', alpha=0.15)

    ax.set_title('(A) Accepted vs Rejected\n(all underinf)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.axvspan(0, 9, alpha=0.08, color='yellow')
    ax.set_ylabel('Statement→Outcome Attention')

    # --- Panel B: Conj vs Quant (the real structure) ---
    ax = axes[0, 1]
    all_conj = pool_groups(ag_groups, '-conj')
    all_quant = pool_groups(ag_groups, '-quant')

    conj_mean, conj_se = layer_curve_with_se(all_conj)
    quant_mean, quant_se = layer_curve_with_se(all_quant)

    ax.plot(layers, conj_mean, color='red', linewidth=2, label=f'All conj (n={len(all_conj)})')
    ax.fill_between(layers, conj_mean - conj_se, conj_mean + conj_se, color='red', alpha=0.15)
    ax.plot(layers, quant_mean, color='blue', linewidth=2, label=f'All quant (n={len(all_quant)})')
    ax.fill_between(layers, quant_mean - quant_se, quant_mean + quant_se, color='blue', alpha=0.15)

    ax.set_title('(B) Conj vs Quant\n(all categories)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.axvspan(0, 9, alpha=0.08, color='yellow')

    # --- Panel C: Within conj, accept vs reject ---
    ax = axes[1, 0]
    conj_acc = pool_groups(ag_groups, 'underinf-conj', 'incorrect')
    conj_rej = pool_groups(ag_groups, 'underinf-conj', 'correct')

    conj_acc_mean, conj_acc_se = layer_curve_with_se(conj_acc)
    ax.plot(layers, conj_acc_mean, color='red', linewidth=2, label=f'Accepted (n={len(conj_acc)})')
    ax.fill_between(layers, conj_acc_mean - conj_acc_se, conj_acc_mean + conj_acc_se, color='red', alpha=0.15)

    if conj_rej is not None and len(conj_rej) >= 1:
        conj_rej_mean = layer_curve(conj_rej)
        ax.plot(layers, conj_rej_mean, color='blue', linewidth=2,
                linestyle='--', label=f'Rejected (n={len(conj_rej)})')

    ax.set_title('(C) Within conj: Accept vs Reject\n(underinf-conj only)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.axvspan(0, 9, alpha=0.08, color='yellow')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Statement→Outcome Attention')

    # --- Panel D: Within quant, accept vs reject ---
    ax = axes[1, 1]
    quant_acc = pool_groups(ag_groups, 'underinf-quant', 'incorrect')
    quant_rej = pool_groups(ag_groups, 'underinf-quant', 'correct')

    quant_rej_mean, quant_rej_se = layer_curve_with_se(quant_rej)
    ax.plot(layers, quant_rej_mean, color='blue', linewidth=2, label=f'Rejected (n={len(quant_rej)})')
    ax.fill_between(layers, quant_rej_mean - quant_rej_se, quant_rej_mean + quant_rej_se, color='blue', alpha=0.15)

    if quant_acc is not None and len(quant_acc) >= 1:
        quant_acc_mean = layer_curve(quant_acc)
        ax.plot(layers, quant_acc_mean, color='red', linewidth=2,
                linestyle='--', label=f'Accepted (n={len(quant_acc)})')

    ax.set_title('(D) Within quant: Accept vs Reject\n(underinf-quant only)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.axvspan(0, 9, alpha=0.08, color='yellow')
    ax.set_xlabel('Layer')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Anti-Gricean: Early-Layer Attention Difference is a Conj/Quant Confound',
        fontsize=13, fontweight='bold', y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = attn_dir / 'attention_conj_quant_confound.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
