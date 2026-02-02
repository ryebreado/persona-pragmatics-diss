#!/usr/bin/env python3
"""
Visualize attention patterns to persona text.

Creates plots comparing:
1. Overall persona attention by persona type
2. Attention to semantic regions within each persona
3. Attention patterns by stimulus category
4. Layer-wise attention profiles
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_persona_results(analysis_dir: str) -> dict:
    """Load all persona attention results."""
    analysis_path = Path(analysis_dir)

    results = {}
    personas = ['pragmaticist', 'anti_gricean', 'literal_thinker', 'helpful_teacher']

    for persona in personas:
        filepath = analysis_path / f"persona_attention_{persona}.json"
        if filepath.exists():
            with open(filepath) as f:
                results[persona] = json.load(f)

    # Load comparison if available
    comparison_path = analysis_path / "persona_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            results['comparison'] = json.load(f)

    return results


def plot_overall_persona_attention(results: dict, output_path: str):
    """Plot overall attention to persona by persona type."""
    if 'comparison' not in results:
        print("No comparison data found")
        return

    comparison = results['comparison']
    personas = list(comparison.keys())
    attention_values = [comparison[p]['persona_attention_overall'] for p in personas]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    bars = ax.bar(personas, attention_values, color=colors)

    ax.set_ylabel('Mean Attention to Persona', fontsize=12)
    ax.set_xlabel('Persona Type', fontsize=12)
    ax.set_title('Overall Attention to Persona by Type', fontsize=14)

    # Add value labels on bars
    for bar, val in zip(bars, attention_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.5f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_attention_profiles(results: dict, output_path: str):
    """Plot attention to persona across layers for each persona type."""
    if 'comparison' not in results:
        return

    comparison = results['comparison']
    personas = list(comparison.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    for persona, color in zip(personas, colors):
        layer_attn = comparison[persona]['persona_attention_by_layer']
        ax.plot(range(len(layer_attn)), layer_attn, label=persona, color=color, linewidth=2)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Attention to Persona', fontsize=12)
    ax.set_title('Attention to Persona Across Layers', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_semantic_region_attention(results: dict, output_path: str):
    """Plot attention to semantic regions within each persona."""
    if 'comparison' not in results:
        return

    comparison = results['comparison']
    personas = list(comparison.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for ax, persona in zip(axes, personas):
        region_data = comparison[persona].get('region_attention_means', {})
        if not region_data:
            ax.set_title(f'{persona}\n(no region data)')
            continue

        regions = list(region_data.keys())
        values = [region_data[r] for r in regions]

        bars = ax.barh(regions, values, color=colors[:len(regions)])
        ax.set_xlabel('Mean Attention', fontsize=10)
        ax.set_title(f'{persona}', fontsize=12, fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val, bar.get_y() + bar.get_height()/2,
                    f'{val:.5f}', va='center', fontsize=9)

    plt.suptitle('Attention to Semantic Regions Within Personas', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_keyword_attention(results: dict, output_path: str):
    """Plot attention to top keywords within each persona."""
    if 'comparison' not in results:
        return

    comparison = results['comparison']
    personas = list(comparison.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, persona in zip(axes, personas):
        keyword_data = comparison[persona].get('keyword_attention_means', {})
        if not keyword_data:
            ax.set_title(f'{persona}\n(no keyword data)')
            continue

        # Sort by attention value and take top 10
        sorted_kw = sorted(keyword_data.items(), key=lambda x: -x[1])[:10]
        keywords = [k for k, v in sorted_kw]
        values = [v for k, v in sorted_kw]

        bars = ax.barh(keywords, values, color='steelblue')
        ax.set_xlabel('Mean Attention', fontsize=10)
        ax.set_title(f'{persona}', fontsize=12, fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val, bar.get_y() + bar.get_height()/2,
                    f'{val:.5f}', va='center', fontsize=8)

    plt.suptitle('Top Keywords by Attention Within Each Persona', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_attention_by_category(results: dict, persona: str, output_path: str):
    """Plot persona attention broken down by stimulus category."""
    if persona not in results:
        return

    persona_results = results[persona]

    # Group by category
    by_category = defaultdict(list)
    for r in persona_results:
        # Get mean attention across layers and heads
        attn_mean = np.mean(r['last_to_persona_mean'])
        by_category[r['category']].append(attn_mean)

    categories = sorted(by_category.keys())
    means = [np.mean(by_category[c]) for c in categories]
    stds = [np.std(by_category[c]) for c in categories]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color by category type
    colors = []
    for cat in categories:
        if 'underinf' in cat:
            colors.append('#e74c3c')  # Red for underinformative
        elif 'true' in cat:
            colors.append('#2ecc71')  # Green for true
        else:
            colors.append('#3498db')  # Blue for false

    bars = ax.bar(categories, means, yerr=stds, color=colors, capsize=3)
    ax.set_ylabel('Mean Attention to Persona', fontsize=12)
    ax.set_xlabel('Stimulus Category', fontsize=12)
    ax.set_title(f'Attention to Persona by Category ({persona})', fontsize=14)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_underinf_vs_true_comparison(results: dict, output_path: str):
    """Compare persona attention between underinformative and true categories."""
    personas = ['pragmaticist', 'anti_gricean', 'literal_thinker', 'helpful_teacher']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(personas))
    width = 0.25

    underinf_means = []
    true_means = []
    false_means = []

    for persona in personas:
        if persona not in results:
            underinf_means.append(0)
            true_means.append(0)
            false_means.append(0)
            continue

        persona_results = results[persona]

        # Group by category type
        underinf = [np.mean(r['last_to_persona_mean']) for r in persona_results
                    if 'underinf' in r['category']]
        true_cat = [np.mean(r['last_to_persona_mean']) for r in persona_results
                    if r['category'].startswith('true')]
        false_cat = [np.mean(r['last_to_persona_mean']) for r in persona_results
                     if r['category'].startswith('false')]

        underinf_means.append(np.mean(underinf) if underinf else 0)
        true_means.append(np.mean(true_cat) if true_cat else 0)
        false_means.append(np.mean(false_cat) if false_cat else 0)

    bars1 = ax.bar(x - width, true_means, width, label='True', color='#2ecc71')
    bars2 = ax.bar(x, false_means, width, label='False', color='#3498db')
    bars3 = ax.bar(x + width, underinf_means, width, label='Underinformative', color='#e74c3c')

    ax.set_ylabel('Mean Attention to Persona', fontsize=12)
    ax.set_xlabel('Persona Type', fontsize=12)
    ax.set_title('Attention to Persona: True vs False vs Underinformative', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=15, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_by_category(results: dict, persona: str, output_path: str):
    """Plot layer-wise attention profiles broken down by category type."""
    if persona not in results:
        return

    persona_results = results[persona]

    # Group by category type
    underinf = [r['last_to_persona_mean'] for r in persona_results if 'underinf' in r['category']]
    true_cat = [r['last_to_persona_mean'] for r in persona_results if r['category'].startswith('true')]
    false_cat = [r['last_to_persona_mean'] for r in persona_results if r['category'].startswith('false')]

    fig, ax = plt.subplots(figsize=(12, 6))

    if underinf:
        underinf_arr = np.array(underinf)
        layer_mean = underinf_arr.mean(axis=0).mean(axis=1)  # Mean across examples and heads
        ax.plot(range(len(layer_mean)), layer_mean, label='Underinformative',
                color='#e74c3c', linewidth=2)

    if true_cat:
        true_arr = np.array(true_cat)
        layer_mean = true_arr.mean(axis=0).mean(axis=1)
        ax.plot(range(len(layer_mean)), layer_mean, label='True',
                color='#2ecc71', linewidth=2)

    if false_cat:
        false_arr = np.array(false_cat)
        layer_mean = false_arr.mean(axis=0).mean(axis=1)
        ax.plot(range(len(layer_mean)), layer_mean, label='False',
                color='#3498db', linewidth=2)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Attention to Persona', fontsize=12)
    ax.set_title(f'Persona Attention by Layer and Category ({persona})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_report(results: dict, output_path: str):
    """Create a text summary report of findings."""
    if 'comparison' not in results:
        return

    comparison = results['comparison']
    personas = list(comparison.keys())

    lines = []
    lines.append("=" * 70)
    lines.append("PERSONA ATTENTION ANALYSIS SUMMARY")
    lines.append("=" * 70)

    # Overall comparison
    lines.append("\n1. OVERALL ATTENTION TO PERSONA")
    lines.append("-" * 40)
    sorted_personas = sorted(personas, key=lambda p: -comparison[p]['persona_attention_overall'])
    for i, p in enumerate(sorted_personas, 1):
        val = comparison[p]['persona_attention_overall']
        lines.append(f"   {i}. {p}: {val:.6f}")

    # Layer analysis
    lines.append("\n2. ATTENTION BY LAYER DEPTH")
    lines.append("-" * 40)
    for persona in personas:
        layer_attn = comparison[persona]['persona_attention_by_layer']
        n = len(layer_attn)
        early = np.mean(layer_attn[:n//4])
        mid = np.mean(layer_attn[n//4:3*n//4])
        late = np.mean(layer_attn[3*n//4:])
        lines.append(f"\n   {persona}:")
        lines.append(f"      Early layers (0-{n//4-1}): {early:.6f}")
        lines.append(f"      Mid layers ({n//4}-{3*n//4-1}):  {mid:.6f}")
        lines.append(f"      Late layers ({3*n//4}-{n-1}): {late:.6f}")

    # Semantic region comparison
    lines.append("\n3. SEMANTIC REGION ATTENTION")
    lines.append("-" * 40)
    for persona in personas:
        region_data = comparison[persona].get('region_attention_means', {})
        if region_data:
            lines.append(f"\n   {persona}:")
            for region, val in sorted(region_data.items(), key=lambda x: -x[1]):
                lines.append(f"      {region}: {val:.6f}")

    # Key findings
    lines.append("\n4. KEY FINDINGS")
    lines.append("-" * 40)

    # Which persona gets most attention?
    max_persona = max(personas, key=lambda p: comparison[p]['persona_attention_overall'])
    min_persona = min(personas, key=lambda p: comparison[p]['persona_attention_overall'])
    lines.append(f"\n   - Most attended persona: {max_persona}")
    lines.append(f"   - Least attended persona: {min_persona}")

    # Category differences
    lines.append("\n5. CATEGORY DIFFERENCES")
    lines.append("-" * 40)
    for persona in personas:
        if persona not in results:
            continue
        persona_results = results[persona]

        underinf = [np.mean(r['last_to_persona_mean']) for r in persona_results
                    if 'underinf' in r['category']]
        true_cat = [np.mean(r['last_to_persona_mean']) for r in persona_results
                    if r['category'].startswith('true')]

        if underinf and true_cat:
            diff = np.mean(underinf) - np.mean(true_cat)
            lines.append(f"\n   {persona}:")
            lines.append(f"      Underinf mean: {np.mean(underinf):.6f}")
            lines.append(f"      True mean: {np.mean(true_cat):.6f}")
            lines.append(f"      Difference: {diff:+.6f}")

    report = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Saved: {output_path}")

    # Also print to console
    print(report)


def main(analysis_dir: str, output_dir: str):
    """Generate all visualizations."""
    results = load_persona_results(analysis_dir)

    if not results:
        print(f"No results found in {analysis_dir}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_overall_persona_attention(results, str(output_path / "overall_persona_attention.png"))
    plot_layer_attention_profiles(results, str(output_path / "layer_attention_profiles.png"))
    plot_semantic_region_attention(results, str(output_path / "semantic_region_attention.png"))
    plot_keyword_attention(results, str(output_path / "keyword_attention.png"))
    plot_underinf_vs_true_comparison(results, str(output_path / "category_comparison.png"))

    # Per-persona plots
    for persona in ['pragmaticist', 'anti_gricean', 'literal_thinker', 'helpful_teacher']:
        if persona in results:
            plot_attention_by_category(results, persona,
                                       str(output_path / f"attention_by_category_{persona}.png"))
            plot_layer_by_category(results, persona,
                                   str(output_path / f"layer_by_category_{persona}.png"))

    # Summary report
    create_summary_report(results, str(output_path / "summary_report.txt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize persona attention analysis')
    parser.add_argument('--analysis-dir', default='probing/analysis/persona_attention',
                        help='Directory with analysis results')
    parser.add_argument('--output-dir', default='probing/analysis/persona_attention/plots',
                        help='Output directory for plots')

    args = parser.parse_args()
    main(args.analysis_dir, args.output_dir)
