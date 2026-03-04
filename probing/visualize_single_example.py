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
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from visualize_attention import MEASURED_CONTENT_FRACTIONS_BY_LAYER


def load_npz_with_meta(npz_path: str) -> Tuple[np.ndarray, Dict]:
    """Load attention array and metadata sidecar."""
    data = np.load(npz_path)
    attn = data['attention']  # (n_layers, n_heads, seq_len, seq_len)

    meta_path = npz_path.replace('.npz', '_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    return attn, meta


def _content_fraction(meta: Dict, actual_layer: int) -> float:
    """Get per-layer content fraction for this condition."""
    persona_field = meta.get('persona')
    if persona_field is None:
        key = 'baseline'
    else:
        # "personas/anti_gricean" -> "anti_gricean"
        key = persona_field.split('/')[-1]
    fracs = MEASURED_CONTENT_FRACTIONS_BY_LAYER.get(key, [1.0] * 36)
    if actual_layer < len(fracs):
        return fracs[actual_layer]
    return 1.0


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


def select_top_heads_last_token(
    attn: np.ndarray,
    meta: Dict,
    target_region: str,
    n_heads: int = 3,
    max_layer: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Auto-select top heads by last_token→region attention mass.

    Args:
        max_layer: If set, only consider layers <= this value (actual layer number,
                   not array index).

    Returns list of (layer_array_idx, head_idx) tuples.
    """
    tgt_start, tgt_end = meta['regions'][target_region]
    last_tok = meta['regions']['last_token']

    # attn shape: (n_layers, n_heads, seq, seq)
    last_to_tgt = attn[:, :, last_tok, tgt_start:tgt_end]
    n_tgt = tgt_end - tgt_start
    mass = last_to_tgt.mean(axis=2)  # (n_layers, n_heads)

    # Mask out layers beyond max_layer
    if max_layer is not None:
        for li, actual_layer in enumerate(meta['layers']):
            if actual_layer > max_layer:
                mass[li, :] = 0

    # Filter out single-token-dominated heads (>80% mass on one token)
    if n_tgt > 1:
        tgt_sum = last_to_tgt.sum(axis=2)
        tgt_sum[tgt_sum == 0] = 1.0
        max_frac = last_to_tgt.max(axis=2) / tgt_sum
        mass[max_frac > 0.8] = 0

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


SINK_WORDS = {'the', 'a', 'an'}


def _is_sink_token(token: str) -> bool:
    """Check if token is an attention-sink article."""
    return _clean_token(token).lower() in SINK_WORDS


def plot_attention_web(
    conditions: List[Tuple[str, np.ndarray, Dict]],
    heads: List[Tuple[int, int]],
    output_path: str,
    source_region: str = "statement",
    target_region: str = "outcome",
    content_normalize: bool = False,
    skip_sink: bool = False,
    transpose: bool = False,
):
    """
    BertViz-style attention web: lines connect source→target tokens,
    thickness/opacity ∝ attention weight.

    Args:
        conditions: list of (label, attn_array, meta_dict) tuples.
                    First entry is used as reference for head labels/title.
        skip_sink: if True, remove article tokens (the/a/an) from target region
                   and renormalize attention over remaining tokens.
        transpose: if True, rows = conditions, cols = heads (default: rows = heads,
                   cols = conditions).

    If content_normalize, divides each condition's weights by its per-layer
    content fraction to compensate for attention "lost" to persona tokens.
    """
    n_heads = len(heads)
    n_conditions = len(conditions)

    if transpose:
        n_rows, n_cols = n_conditions, n_heads
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)
    else:
        n_rows, n_cols = n_heads, n_conditions
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

    line_color = '#4682B4'

    is_single_source = (source_region == 'last_token')

    ref_meta = conditions[0][2]  # first condition used as reference

    for head_i, (layer_idx, head_idx) in enumerate(heads):
        # First pass: find shared max weight for normalization across conditions
        weight_max = 0
        submatrices = []
        src_token_lists = []
        tgt_token_lists = []
        for label, attn, meta in conditions:
            tgt_start, tgt_end = meta['regions'][target_region]
            if is_single_source:
                last_tok = meta['regions']['last_token']
                sub = attn[layer_idx, head_idx, last_tok, tgt_start:tgt_end]
                sub = sub[np.newaxis, :]  # (1, n_tgt)
                src_toks = [meta['tokens'][last_tok]]
            else:
                src_start, src_end = meta['regions'][source_region]
                sub = attn[layer_idx, head_idx, src_start:src_end, tgt_start:tgt_end]
                src_toks, _ = get_region_tokens(meta, source_region)
            tgt_toks, _ = get_region_tokens(meta, target_region)
            if content_normalize:
                actual_layer = meta['layers'][layer_idx]
                frac = _content_fraction(meta, actual_layer)
                sub = sub / frac
            # Filter out sink tokens from target
            if skip_sink:
                keep = [j for j, t in enumerate(tgt_toks) if not _is_sink_token(t)]
                if keep:
                    sub = sub[:, keep]
                    tgt_toks = [tgt_toks[j] for j in keep]
            submatrices.append(sub)
            src_token_lists.append(src_toks)
            tgt_token_lists.append(tgt_toks)
            weight_max = max(weight_max, sub.max())

        if weight_max == 0:
            weight_max = 1.0  # avoid division by zero

        # Second pass: draw
        for cond_i, (label, attn, meta) in enumerate(conditions):
            if transpose:
                ax = axes[cond_i, head_i]
            else:
                ax = axes[head_i, cond_i]
            sub = submatrices[cond_i]

            src_tokens = src_token_lists[cond_i]
            tgt_tokens = tgt_token_lists[cond_i]
            n_src = len(src_tokens)
            n_tgt = len(tgt_tokens)

            # Vertical positions: evenly spaced, 0 at top, 1 at bottom
            src_y = np.linspace(0, 1, n_src) if n_src > 1 else [0.5]
            tgt_y = np.linspace(0, 1, n_tgt) if n_tgt > 1 else [0.5]

            x_left = 0.2
            x_right = 0.8

            # Target (earlier in sequence) on left, source (later) on right
            # Build line segments and properties
            segments = []
            alphas = []
            widths = []
            for i in range(n_src):
                for j in range(n_tgt):
                    w = sub[i, j]
                    norm_w = w / weight_max
                    segments.append([(x_left, tgt_y[j]), (x_right, src_y[i])])
                    alphas.append(float(np.clip(norm_w, 0.02, 1.0)))
                    widths.append(0.5 + 3.5 * float(norm_w))

            # Draw lines with LineCollection
            colors = [mcolors.to_rgba(line_color, alpha=a) for a in alphas]
            lc = LineCollection(segments, colors=colors, linewidths=widths)
            ax.add_collection(lc)

            # Token labels: target (attended-to) on left, source (attending-from) on right
            for j, tok in enumerate(tgt_tokens):
                color = _color_token_label(tok)
                ax.text(x_left - 0.02, tgt_y[j], _clean_token(tok),
                        ha='right', va='center', fontsize=9, color=color,
                        fontfamily='monospace')
            for i, tok in enumerate(src_tokens):
                color = _color_token_label(tok)
                if is_single_source:
                    ax.text(x_right + 0.02, src_y[i], '<last>',
                            ha='left', va='center', fontsize=9, color='#666666',
                            fontstyle='italic')
                else:
                    ax.text(x_right + 0.02, src_y[i], _clean_token(tok),
                            ha='left', va='center', fontsize=9, color=color,
                            fontfamily='monospace')

            # Axis setup
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.08, 1.08)
            ax.invert_yaxis()
            ax.set_axis_off()

            # Title logic depends on layout orientation
            layer_label = ref_meta['layers'][layer_idx]
            if transpose:
                # Column headers: head labels (top row only)
                if cond_i == 0:
                    ax.set_title(f"L{layer_label}H{head_idx}", fontsize=11)
                # Row labels: condition names (left column only)
                if head_i == 0:
                    ax.text(-0.12, 0.5, label, transform=ax.transAxes,
                            ha='right', va='center', fontsize=11, fontweight='bold',
                            rotation=0)
            else:
                if head_i == 0:
                    ax.set_title(f"{label}\nL{layer_label}H{head_idx}", fontsize=11)
                else:
                    ax.set_title(f"L{layer_label}H{head_idx}", fontsize=11)

    src_display = "Last Token" if is_single_source else source_region.capitalize()
    tgt_display = target_region.capitalize()
    suffixes = []
    if content_normalize:
        suffixes.append("content-normalized")
    if skip_sink:
        suffixes.append("no sink")
    norm_suffix = f" [{', '.join(suffixes)}]" if suffixes else ""
    fig.suptitle(
        f"{tgt_display} ← {src_display} Attention Web (Test {ref_meta['test_id']}){norm_suffix}",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


PERSONA_DISPLAY_NAMES = {
    'anti_gricean': 'Anti-Gricean',
    'literal_thinker': 'Literal Thinker',
    'helpful_teacher': 'Helpful Teacher',
    'pragmaticist': 'Pragmaticist',
}


def _label_from_npz_path(npz_path: str) -> str:
    """Derive display label from npz filename (e.g. test39_anti_gricean.npz -> Anti-Gricean)."""
    stem = Path(npz_path).stem  # e.g. "test39_anti_gricean"
    # Strip test ID prefix
    parts = stem.split('_', 1)
    if len(parts) > 1:
        key = parts[1]
    else:
        key = parts[0]
    return PERSONA_DISPLAY_NAMES.get(key, key.replace('_', ' ').title())


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize attention for a single test case across conditions'
    )
    parser.add_argument('--baseline', required=True, help='Baseline .npz file')
    parser.add_argument('--persona', nargs='+', required=True,
                        help='One or more persona .npz files')
    parser.add_argument('--persona-name', default=None,
                        help='Display name for persona (only used with single persona)')
    parser.add_argument('--heads', nargs='+',
                        help='Heads as L#H# (layer_array_idx, head). Auto-selects top 3 if omitted.')
    parser.add_argument('--n-heads', type=int, default=3, help='Number of auto-selected heads')
    parser.add_argument('--plot', nargs='+', default=['all'],
                        choices=['all', 'heatmap', 'last_outcome', 'last_statement',
                                 'web', 'web_last_outcome', 'web_last_statement'],
                        help='Which plots to generate')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: same as baseline)')
    parser.add_argument('--content-normalize', action='store_true',
                        help='Apply per-layer content-conditional normalization to web plots')
    parser.add_argument('--skip-sink', action='store_true',
                        help='Remove article tokens (the/a/an) from target region in web plots')
    parser.add_argument('--transpose', action='store_true',
                        help='Transpose web grid: rows=conditions, cols=heads')
    parser.add_argument('--suffix', default='',
                        help='Extra suffix for output filenames (e.g. "_all_personas")')

    args = parser.parse_args()

    # Load data
    baseline_attn, baseline_meta = load_npz_with_meta(args.baseline)
    print(f"Baseline: {baseline_attn.shape} ({baseline_meta['category']}, "
          f"test {baseline_meta['test_id']})")

    # Build conditions list: baseline first, then personas in given order
    all_conditions = [('Baseline', baseline_attn, baseline_meta)]
    for persona_path in args.persona:
        p_attn, p_meta = load_npz_with_meta(persona_path)
        if args.persona_name and len(args.persona) == 1:
            label = args.persona_name
        else:
            label = _label_from_npz_path(persona_path)
        all_conditions.append((label, p_attn, p_meta))
        print(f"{label}: {p_attn.shape} ({p_meta['category']}, "
              f"test {p_meta['test_id']})")

    # For backward compat: extract first persona for heatmap/bar plots
    persona_name = all_conditions[1][0]
    persona_attn = all_conditions[1][1]
    persona_meta = all_conditions[1][2]

    # Determine heads (from baseline)
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
    file_suffix = args.suffix
    plots = args.plot
    if 'all' in plots:
        plots = ['heatmap', 'last_outcome', 'last_statement',
                 'web', 'web_last_outcome', 'web_last_statement']

    if 'heatmap' in plots:
        plot_statement_outcome_heatmaps(
            baseline_attn, persona_attn,
            baseline_meta, persona_meta,
            heads,
            str(out_dir / f"test{test_id}_stmt_outcome_heatmap{file_suffix}.png"),
            persona_name=persona_name,
        )

    if 'last_outcome' in plots:
        plot_last_to_region_bars(
            baseline_attn, persona_attn,
            baseline_meta, persona_meta,
            heads, 'outcome',
            str(out_dir / f"test{test_id}_last_outcome_bars{file_suffix}.png"),
            persona_name=persona_name,
        )

    if 'last_statement' in plots:
        plot_last_to_region_bars(
            baseline_attn, persona_attn,
            baseline_meta, persona_meta,
            heads, 'statement',
            str(out_dir / f"test{test_id}_last_statement_bars{file_suffix}.png"),
            persona_name=persona_name,
        )

    sink_suffix = "_nosink" if args.skip_sink else ""

    if 'web' in plots:
        plot_attention_web(
            all_conditions,
            heads,
            str(out_dir / f"test{test_id}_stmt_outcome_web{file_suffix}{sink_suffix}.png"),
            source_region='statement',
            target_region='outcome',
            content_normalize=args.content_normalize,
            skip_sink=args.skip_sink,
            transpose=args.transpose,
        )

    if 'web_last_outcome' in plots or 'web_last_statement' in plots:
        # Auto-select heads by last→region mass in early layers
        for region, plot_key in [('outcome', 'web_last_outcome'),
                                 ('statement', 'web_last_statement')]:
            if plot_key not in plots:
                continue
            if args.heads:
                last_heads = heads
            else:
                last_heads = select_top_heads_last_token(
                    baseline_attn, baseline_meta, region,
                    n_heads=args.n_heads, max_layer=10,
                )
                layer_labels = [f"L{baseline_meta['layers'][li]}H{hi}"
                                for li, hi in last_heads]
                print(f"Auto-selected top {args.n_heads} heads "
                      f"(last→{region}, layers ≤10): {layer_labels}")
            plot_attention_web(
                all_conditions,
                last_heads,
                str(out_dir / f"test{test_id}_last_{region}_web{file_suffix}{sink_suffix}.png"),
                source_region='last_token',
                target_region=region,
                content_normalize=args.content_normalize,
                skip_sink=args.skip_sink,
                transpose=args.transpose,
            )


if __name__ == "__main__":
    main()
