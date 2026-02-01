#!/usr/bin/env python3
"""
Visualize attention matrices as heatmaps with region annotations.

Creates publication-quality figures showing attention patterns for specific heads.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional


def plot_attention_heatmap(
    attention: np.ndarray,
    tokens: List[str],
    regions: Dict,
    head_name: str,
    title: str,
    ax=None,
    highlight_rows: Optional[List[int]] = None,
    highlight_cols: Optional[List[int]] = None,
):
    """
    Plot attention matrix as heatmap with region annotations.

    Args:
        attention: (seq_len, seq_len) attention matrix
        tokens: list of token strings
        regions: dict with 'outcome', 'statement', 'scalar_terms', 'last_token'
        head_name: e.g., "L18H12"
        title: plot title
        highlight_rows: row indices to highlight (source positions)
        highlight_cols: column indices to highlight (target positions)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    seq_len = len(tokens)

    # Truncate tokens for display
    display_tokens = [t[:8] if len(t) > 8 else t for t in tokens]

    # Plot heatmap
    im = ax.imshow(attention, cmap='Blues', aspect='auto')

    # Add region boxes
    outcome_start, outcome_end = regions['outcome']
    stmt_start, stmt_end = regions['statement']
    last_tok = regions['last_token']

    # Highlight outcome region (columns)
    rect = mpatches.Rectangle(
        (outcome_start - 0.5, -0.5), outcome_end - outcome_start, seq_len,
        linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)

    # Highlight statement region (rows and columns)
    rect = mpatches.Rectangle(
        (stmt_start - 0.5, -0.5), stmt_end - stmt_start, seq_len,
        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)

    # Highlight last token row
    ax.axhline(y=last_tok, color='red', linewidth=1.5, linestyle='-', alpha=0.7)

    # Highlight scalar term if present
    if 'some' in regions.get('scalar_terms', {}):
        some_pos = regions['scalar_terms']['some']
        ax.axvline(x=some_pos, color='purple', linewidth=2, linestyle='-', alpha=0.7)

    # Axis labels (show every nth token for readability)
    n_show = max(1, seq_len // 20)
    ax.set_xticks(range(0, seq_len, n_show))
    ax.set_xticklabels([display_tokens[i] for i in range(0, seq_len, n_show)],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(0, seq_len, n_show))
    ax.set_yticklabels([display_tokens[i] for i in range(0, seq_len, n_show)], fontsize=7)

    ax.set_xlabel('Target token (attending to)', fontsize=10)
    ax.set_ylabel('Source token (attending from)', fontsize=10)
    ax.set_title(f'{title}\n{head_name}', fontsize=11, fontweight='bold')

    # Colorbar
    plt.colorbar(im, ax=ax, shrink=0.8, label='Attention weight')

    return ax


def plot_attention_row(
    attention: np.ndarray,
    tokens: List[str],
    regions: Dict,
    source_pos: int,
    head_name: str,
    title: str,
    ax=None,
):
    """
    Plot attention from a single source position as a bar chart.

    Useful for showing what the last token attends to.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    seq_len = len(tokens)
    attn_row = attention[source_pos, :source_pos + 1]  # Causal: only attend to prev tokens

    # Color bars by region
    colors = []
    outcome_start, outcome_end = regions['outcome']
    stmt_start, stmt_end = regions['statement']
    some_pos = regions.get('scalar_terms', {}).get('some')

    for i in range(len(attn_row)):
        if some_pos is not None and i == some_pos:
            colors.append('purple')
        elif outcome_start <= i < outcome_end:
            colors.append('green')
        elif stmt_start <= i < stmt_end:
            colors.append('blue')
        else:
            colors.append('gray')

    ax.bar(range(len(attn_row)), attn_row, color=colors, alpha=0.7)

    # Truncate tokens for display
    display_tokens = [t[:6] if len(t) > 6 else t for t in tokens[:len(attn_row)]]

    # Show every nth label
    n_show = max(1, len(attn_row) // 25)
    ax.set_xticks(range(0, len(attn_row), n_show))
    ax.set_xticklabels([display_tokens[i] for i in range(0, len(attn_row), n_show)],
                       rotation=45, ha='right', fontsize=7)

    ax.set_ylabel('Attention weight', fontsize=10)
    ax.set_xlabel('Target token', fontsize=10)
    ax.set_title(f'{title}\n{head_name} - Attention from position {source_pos}',
                 fontsize=11, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.7, label='Outcome'),
        mpatches.Patch(facecolor='blue', alpha=0.7, label='Statement'),
        mpatches.Patch(facecolor='purple', alpha=0.7, label='"some"'),
        mpatches.Patch(facecolor='gray', alpha=0.7, label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    return ax


def create_comparison_figure(
    data: Dict,
    head_name: str,
    output_path: str = None,
):
    """
    Create a figure comparing attention patterns across examples for one head.

    Shows true-quant vs underinf-quant side by side.
    """
    examples = data['examples']

    # Separate by category
    true_examples = [e for e in examples if e['category'] == 'true-quant']
    underinf_examples = [e for e in examples if e['category'] == 'underinf-quant']

    n_rows = max(len(true_examples), len(underinf_examples))
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows):
        # True example
        if i < len(true_examples):
            ex = true_examples[i]
            attn = np.array(ex['heads'][head_name])
            title = f"TRUE: {ex['statement'][:40]}..."
            plot_attention_row(
                attn, ex['tokens'], ex['regions'],
                ex['regions']['last_token'],
                head_name, title, ax=axes[i, 0]
            )
        else:
            axes[i, 0].axis('off')

        # Underinf example
        if i < len(underinf_examples):
            ex = underinf_examples[i]
            attn = np.array(ex['heads'][head_name])
            title = f"UNDERINF: {ex['statement'][:40]}..."
            plot_attention_row(
                attn, ex['tokens'], ex['regions'],
                ex['regions']['last_token'],
                head_name, title, ax=axes[i, 1]
            )
        else:
            axes[i, 1].axis('off')

    plt.suptitle(f'Attention Pattern Comparison: {head_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize attention matrices')
    parser.add_argument('input', help='JSON file with attention matrices')
    parser.add_argument('--output-dir', '-o', help='Output directory for figures')
    parser.add_argument('--head', help='Specific head to visualize (e.g., L18H12)')

    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)

    heads = [f"L{h['layer']}H{h['head']}" for h in data['heads']]

    if args.head:
        heads = [args.head]

    for head_name in heads:
        output_path = output_dir / f'attention_comparison_{head_name}.png'
        create_comparison_figure(data, head_name, str(output_path))


if __name__ == "__main__":
    main()
