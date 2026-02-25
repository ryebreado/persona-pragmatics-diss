#!/usr/bin/env python3
"""
Visualize attention patterns for a single test case across conditions.

Loads npz + meta.json files from extract_attention_matrices.py and produces:
  1. Statement→Outcome heatmaps (per head, baseline vs persona)
  2. Last→Outcome bar charts
  3. Last→Statement bar charts
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_npz_with_meta(npz_path: str) -> Tuple[np.ndarray, Dict]:
    """Load attention array and metadata sidecar."""
    data = np.load(npz_path)
    attn = data['attention']  # (n_layers, n_heads, seq_len, seq_len)

    meta_path = npz_path.replace('.npz', '_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    return attn, meta


def get_region_tokens(meta: Dict, region: str) -> Tuple[List[str], Tuple[int, int]]:
    """Get token strings and span for a region."""
    span = tuple(meta['regions'][region])
    tokens = meta['tokens'][span[0]:span[1]]
    return tokens, span


def select_top_heads(
    attn: np.ndarray,
    meta: Dict,
    n_heads: int = 3,
) -> List[Tuple[int, int]]:
    """
    Auto-select top heads by statement→outcome attention mass.

    Returns list of (layer_array_idx, head_idx) tuples.
    """
    stmt_start, stmt_end = meta['regions']['statement']
    out_start, out_end = meta['regions']['outcome']

    # Mean attention from statement tokens to outcome tokens per head
    # attn shape: (n_layers, n_heads, seq, seq)
    stmt_to_out = attn[:, :, stmt_start:stmt_end, out_start:out_end]
    mass = stmt_to_out.mean(axis=(2, 3))  # (n_layers, n_heads)

    # Get top heads
    flat_indices = np.argsort(mass.ravel())[::-1][:n_heads]
    top = []
    for idx in flat_indices:
        layer_idx = idx // mass.shape[1]
        head_idx = idx % mass.shape[1]
        top.append((int(layer_idx), int(head_idx)))

    return top


def _clean_token(token: str) -> str:
    """Clean sentencepiece/BPE token for display."""
    t = token.replace('▁', ' ').replace('\u0120', ' ')
    # GPT-2 byte encoding: Ċ (U+010A) = newline byte
    t = t.replace('\u010a', '').replace('\n', '').replace('\r', '')
    return t.strip() or '.'


def _color_token_label(token: str) -> str:
    """Return color for a token based on semantic content."""
    t = _clean_token(token).lower()
    if 'green' in t:
        return '#228B22'
    if 'purple' in t:
        return '#7B2D8B'
    if t == 'and':
        return '#D2691E'
    return 'black'


def plot_statement_outcome_heatmaps(
    baseline_attn: np.ndarray,
    persona_attn: np.ndarray,
    baseline_meta: Dict,
    persona_meta: Dict,
    heads: List[Tuple[int, int]],
    output_path: str,
    persona_name: str = "Anti-Gricean",
):
    """
    Plot 1: Statement→Outcome attention heatmaps.

    Grid: rows = heads, cols = [Baseline, Persona]
    Each cell: submatrix where rows = statement tokens, cols = outcome tokens
    """
    n_heads = len(heads)
    fig, axes = plt.subplots(n_heads, 2, figsize=(14, 3.5 * n_heads),
                             squeeze=False)

    conditions = [
        ('Baseline', baseline_attn, baseline_meta),
        (persona_name, persona_attn, persona_meta),
    ]

    for row, (layer_idx, head_idx) in enumerate(heads):
        baseline_layer = baseline_meta['layers'][layer_idx]
        vmin_row, vmax_row = float('inf'), float('-inf')
        submatrices = []

        # First pass: extract submatrices and find shared vmin/vmax
        for label, attn, meta in conditions:
            stmt_start, stmt_end = meta['regions']['statement']
            out_start, out_end = meta['regions']['outcome']
            sub = attn[layer_idx, head_idx, stmt_start:stmt_end, out_start:out_end]
            submatrices.append(sub)
            vmin_row = min(vmin_row, sub.min())
            vmax_row = max(vmax_row, sub.max())

        # Second pass: plot
        for col, (label, attn, meta) in enumerate(conditions):
            ax = axes[row, col]
            sub = submatrices[col]

            stmt_tokens, _ = get_region_tokens(meta, 'statement')
            out_tokens, _ = get_region_tokens(meta, 'outcome')

            im = ax.imshow(sub, cmap='Blues', vmin=vmin_row, vmax=vmax_row,
                           aspect='auto', interpolation='nearest')

            # Token labels
            ax.set_xticks(range(len(out_tokens)))
            ax.set_xticklabels([_clean_token(t) for t in out_tokens],
                               rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(stmt_tokens)))
            ax.set_yticklabels([_clean_token(t) for t in stmt_tokens], fontsize=8)

            # Color token labels
            for tick_idx, t in enumerate(out_tokens):
                color = _color_token_label(t)
                ax.get_xticklabels()[tick_idx].set_color(color)
            for tick_idx, t in enumerate(stmt_tokens):
                color = _color_token_label(t)
                ax.get_yticklabels()[tick_idx].set_color(color)

            layer_label = baseline_meta['layers'][layer_idx]
            if row == 0:
                ax.set_title(f"{label}\nL{layer_label}H{head_idx}", fontsize=11)
            else:
                ax.set_title(f"L{layer_label}H{head_idx}", fontsize=11)

            if col == 0:
                ax.set_ylabel("Statement →", fontsize=10)
            if row == n_heads - 1:
                ax.set_xlabel("← Outcome", fontsize=10)

        # Colorbar for this row
        fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.8, pad=0.02)

    fig.suptitle(
        f"Statement → Outcome Attention (Test {baseline_meta['test_id']})",
        fontsize=14, y=1.01,
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_last_to_region_bars(
    baseline_attn: np.ndarray,
    persona_attn: np.ndarray,
    baseline_meta: Dict,
    persona_meta: Dict,
    heads: List[Tuple[int, int]],
    region: str,
    output_path: str,
    persona_name: str = "Anti-Gricean",
):
    """
    Plots 2 & 3: Last token → region bar charts.

    Grid: rows = heads, cols = [Baseline, Persona]
    """
    n_heads = len(heads)
    fig, axes = plt.subplots(n_heads, 2, figsize=(14, 3 * n_heads),
                             squeeze=False)

    conditions = [
        ('Baseline', baseline_attn, baseline_meta),
        (persona_name, persona_attn, persona_meta),
    ]

    for row, (layer_idx, head_idx) in enumerate(heads):
        # Find shared y-axis max across conditions
        ymax = 0
        bars_data = []
        for label, attn, meta in conditions:
            region_start, region_end = meta['regions'][region]
            last_tok = meta['regions']['last_token']
            weights = attn[layer_idx, head_idx, last_tok, region_start:region_end]
            tokens, _ = get_region_tokens(meta, region)
            bars_data.append((weights, tokens))
            ymax = max(ymax, weights.max())

        for col, (label, attn, meta) in enumerate(conditions):
            ax = axes[row, col]
            weights, tokens = bars_data[col]
            clean_labels = [_clean_token(t) for t in tokens]

            # Bar colors based on token content
            colors = []
            for t in tokens:
                tl = t.strip().lower()
                if 'green' in tl:
                    colors.append('#228B22')
                elif 'purple' in tl:
                    colors.append('#7B2D8B')
                elif tl in ('and', '▁and'):
                    colors.append('#D2691E')
                elif region == 'outcome':
                    colors.append('#93C5FD')
                else:
                    colors.append('#6495ED')

            ax.bar(range(len(weights)), weights, color=colors,
                   edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(clean_labels)))
            ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, ymax * 1.15)
            ax.grid(True, alpha=0.3, axis='y')

            layer_label = baseline_meta['layers'][layer_idx]
            if row == 0:
                ax.set_title(f"{label}\nL{layer_label}H{head_idx}", fontsize=11)
            else:
                ax.set_title(f"L{layer_label}H{head_idx}", fontsize=11)

            if col == 0:
                ax.set_ylabel("Attention weight", fontsize=9)

    region_display = "Outcome" if region == "outcome" else "Statement"
    fig.suptitle(
        f"Last Token → {region_display} Attention (Test {baseline_meta['test_id']})",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize attention for a single test case across conditions'
    )
    parser.add_argument('--baseline', required=True, help='Baseline .npz file')
    parser.add_argument('--persona', required=True, help='Persona .npz file')
    parser.add_argument('--persona-name', default='Anti-Gricean', help='Display name for persona')
    parser.add_argument('--heads', nargs='+',
                        help='Heads as L#H# (layer_array_idx, head). Auto-selects top 3 if omitted.')
    parser.add_argument('--n-heads', type=int, default=3, help='Number of auto-selected heads')
    parser.add_argument('--plot', nargs='+', default=['all'],
                        choices=['all', 'heatmap', 'last_outcome', 'last_statement'],
                        help='Which plots to generate')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: same as baseline)')

    args = parser.parse_args()

    # Load data
    baseline_attn, baseline_meta = load_npz_with_meta(args.baseline)
    persona_attn, persona_meta = load_npz_with_meta(args.persona)

    print(f"Baseline: {baseline_attn.shape} ({baseline_meta['category']}, "
          f"test {baseline_meta['test_id']})")
    print(f"Persona:  {persona_attn.shape} ({persona_meta['category']}, "
          f"test {persona_meta['test_id']})")

    # Determine heads
    if args.heads:
        heads = []
        for h in args.heads:
            parts = h.upper().replace('L', '').split('H')
            heads.append((int(parts[0]), int(parts[1])))
    else:
        heads = select_top_heads(baseline_attn, baseline_meta, args.n_heads)
        layer_labels = [f"L{baseline_meta['layers'][li]}H{hi}" for li, hi in heads]
        print(f"Auto-selected top {args.n_heads} heads (by stmt→outcome mass): {layer_labels}")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.baseline).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    test_id = baseline_meta['test_id']
    plots = args.plot
    if 'all' in plots:
        plots = ['heatmap', 'last_outcome', 'last_statement']

    if 'heatmap' in plots:
        plot_statement_outcome_heatmaps(
            baseline_attn, persona_attn,
            baseline_meta, persona_meta,
            heads,
            str(out_dir / f"test{test_id}_stmt_outcome_heatmap.png"),
            persona_name=args.persona_name,
        )

    if 'last_outcome' in plots:
        plot_last_to_region_bars(
            baseline_attn, persona_attn,
            baseline_meta, persona_meta,
            heads, 'outcome',
            str(out_dir / f"test{test_id}_last_outcome_bars.png"),
            persona_name=args.persona_name,
        )

    if 'last_statement' in plots:
        plot_last_to_region_bars(
            baseline_attn, persona_attn,
            baseline_meta, persona_meta,
            heads, 'statement',
            str(out_dir / f"test{test_id}_last_statement_bars.png"),
            persona_name=args.persona_name,
        )


if __name__ == "__main__":
    main()
