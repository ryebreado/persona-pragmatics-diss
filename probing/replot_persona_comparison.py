#!/usr/bin/env python3
"""
Replot persona comparison layer curves from saved JSON results.
Only shows error band (std fill) for baseline to reduce visual clutter.
"""

import json
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

_TAB10 = matplotlib.colormaps["tab10"]
PERSONA_COLORS = {
    "baseline": _TAB10(0),
    "literal_thinker": _TAB10(1),
    "helpful_teacher": _TAB10(2),
    "anti_gricean": _TAB10(3),
    "pragmaticist": _TAB10(4),
}


def main():
    parser = argparse.ArgumentParser(description='Replot persona comparison with baseline-only error band')
    parser.add_argument('json_path', help='Path to persona_comparison JSON results')
    parser.add_argument('--output', '-o', required=True, help='Output PNG path')
    parser.add_argument('--title', default=None, help='Plot title')
    parser.add_argument('--error-band', nargs='*', default=['baseline'],
                        help='Which conditions get error bands (default: baseline only)')
    parser.add_argument('--figsize', nargs=2, type=float, default=None, metavar=('W', 'H'),
                        help='Figure width and height in inches (default: 10 6)')
    parser.add_argument('--ymin', type=float, default=0.0, help='Y-axis minimum (default: 0.0)')
    parser.add_argument('--legend-loc', default='best', help='Legend location (e.g. "lower right")')
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    figsize = tuple(args.figsize) if args.figsize else (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    for entry in data:
        name = entry['name']
        layers = [l['layer'] for l in entry['layers']]
        accs = [l['accuracy'] for l in entry['layers']]
        stds = [l['std'] for l in entry['layers']]

        color = PERSONA_COLORS.get(name)
        ax.plot(layers, accs, marker='o', label=name, linewidth=2, markersize=4, color=color)

        if name in args.error_band and any(s > 0 for s in stds):
            ax.fill_between(
                layers,
                [a - s for a, s in zip(accs, stds)],
                [a + s for a, s in zip(accs, stds)],
                alpha=0.2,
                color=color
            )

    # Chance level
    n_classes = len(data[0]['layers'][0]['labels'])
    chance = 1.0 / n_classes
    ax.axhline(y=chance, color='gray', linestyle='--', label=f'Chance ({chance:.2f})')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    title = args.title if args.title is not None else Path(args.json_path).stem.replace('_', ' ').title()
    if title:
        ax.set_title(title)
    ax.legend(loc=args.legend_loc)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(args.ymin, 1)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved to {args.output}')
    plt.close()


if __name__ == '__main__':
    main()
