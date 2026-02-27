#!/usr/bin/env python3
"""
Batch extract content-word attention ratios across all examples and conditions.

For each example × condition × layer, computes what fraction of region-to-region
attention mass lands on content-word tokens (vs function words / punctuation).
This validates whether cherry-picked BertViz examples are representative.

Usage:
    # Extract only (fast dry-run on 5 examples)
    uv run python probing/extract_content_word_attention.py --num-examples 5

    # Full run: all 250 × 5 conditions
    uv run python probing/extract_content_word_attention.py

    # Plot from saved results
    uv run python probing/extract_content_word_attention.py --plot-only results.json
"""

import json
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

from attention_analysis import create_prompt_with_persona, find_token_regions
from content_words import extract_content_words, find_content_token_positions


DEFAULT_LAYERS = [0, 3, 5, 7, 10, 15, 17, 19, 25]

PERSONA_FILES = {
    "baseline": None,
    "anti_gricean": "personas/anti_gricean",
    "literal_thinker": "personas/literal_thinker",
    "helpful_teacher": "personas/helpful_teacher",
    "pragmaticist": "personas/pragmaticist",
}

PERSONA_DISPLAY = {
    "baseline": "Baseline",
    "anti_gricean": "Anti-Gricean",
    "literal_thinker": "Literal Thinker",
    "helpful_teacher": "Helpful Teacher",
    "pragmaticist": "Pragmaticist",
}

CATEGORY_ORDER = [
    "true-conj", "true-quant", "false-conj", "false-quant",
    "underinf-conj", "underinf-quant",
]


def compute_content_ratios(
    attn: torch.Tensor,
    tokens: List[str],
    regions: "TokenRegions",
    content_words: Dict[str, List[str]],
    layers: List[int],
    selected_layer_indices: List[int],
) -> Dict:
    """
    Compute content-word attention ratios for all three directions.

    Args:
        attn: (n_selected_layers, n_heads, seq, seq) attention tensor
        tokens: token string list
        regions: TokenRegions with outcome_span, statement_span
        content_words: {"object_words": [...], "scalar_terms": [...], "number_words": [...]}
        layers: actual layer numbers (e.g. [0,3,5,...])
        selected_layer_indices: indices into attn's first dim

    Returns dict with content ratios per layer per direction.
    """
    all_content = (
        content_words["object_words"]
        + content_words["scalar_terms"]
        + content_words["number_words"]
    )

    # Find content token positions within each region
    outcome_content_pos = find_content_token_positions(
        tokens, all_content, regions.outcome_span
    )
    statement_content_pos = find_content_token_positions(
        tokens, all_content, regions.statement_span
    )

    out_start, out_end = regions.outcome_span
    stmt_start, stmt_end = regions.statement_span
    n_outcome = out_end - out_start
    n_statement = stmt_end - stmt_start

    result = {
        "content_words_outcome": [tokens[i] for i in outcome_content_pos],
        "content_words_statement": [tokens[i] for i in statement_content_pos],
        "n_content_outcome": len(outcome_content_pos),
        "n_content_statement": len(statement_content_pos),
        "n_total_outcome": n_outcome,
        "n_total_statement": n_statement,
        "chance_outcome": len(outcome_content_pos) / max(n_outcome, 1),
        "chance_statement": len(statement_content_pos) / max(n_statement, 1),
        "s2o": {},  # statement → outcome
        "l2o": {},  # last → outcome
        "l2s": {},  # last → statement
    }

    for li, actual_layer in zip(selected_layer_indices, layers):
        layer_key = f"L{actual_layer}"
        # Average across all heads at this layer
        # attn shape: (n_layers, n_heads, seq, seq)
        attn_layer = attn[li]  # (n_heads, seq, seq)

        # S→O: statement tokens attending to outcome tokens
        if n_outcome > 0 and n_statement > 0:
            s2o_all = attn_layer[:, stmt_start:stmt_end, out_start:out_end]
            # Total attention from statement to outcome
            s2o_total = s2o_all.sum(dim=-1).mean().item()  # mean over heads and src tokens
            if s2o_total > 0 and outcome_content_pos:
                # Attention to content tokens only (relative indices within outcome)
                content_rel = [p - out_start for p in outcome_content_pos]
                s2o_content = attn_layer[:, stmt_start:stmt_end, :][:, :, [p for p in outcome_content_pos]]
                s2o_content_sum = s2o_content.sum(dim=-1).mean().item()
                result["s2o"][layer_key] = s2o_content_sum / s2o_total if s2o_total > 0 else 0.0
            else:
                result["s2o"][layer_key] = 0.0
        else:
            result["s2o"][layer_key] = 0.0

        # L→O: last token attending to outcome tokens
        if n_outcome > 0:
            l2o_all = attn_layer[:, regions.last_token, out_start:out_end]  # (n_heads, n_out)
            l2o_total = l2o_all.sum(dim=-1).mean().item()
            if l2o_total > 0 and outcome_content_pos:
                content_rel = [p - out_start for p in outcome_content_pos]
                l2o_content = l2o_all[:, content_rel].sum(dim=-1).mean().item()
                result["l2o"][layer_key] = l2o_content / l2o_total if l2o_total > 0 else 0.0
            else:
                result["l2o"][layer_key] = 0.0
        else:
            result["l2o"][layer_key] = 0.0

        # L→S: last token attending to statement tokens
        if n_statement > 0:
            l2s_all = attn_layer[:, regions.last_token, stmt_start:stmt_end]  # (n_heads, n_stmt)
            l2s_total = l2s_all.sum(dim=-1).mean().item()
            if l2s_total > 0 and statement_content_pos:
                content_rel = [p - stmt_start for p in statement_content_pos]
                l2s_content = l2s_all[:, content_rel].sum(dim=-1).mean().item()
                result["l2s"][layer_key] = l2s_content / l2s_total if l2s_total > 0 else 0.0
            else:
                result["l2s"][layer_key] = 0.0
        else:
            result["l2s"][layer_key] = 0.0

    return result


def extract_all(
    model_name: str,
    data_path: str,
    layers: List[int],
    conditions: Dict[str, Optional[str]],
    device: str = "mps",
    num_examples: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """Run extraction across all examples and conditions."""
    with open(data_path) as f:
        test_cases = json.load(f)

    if num_examples:
        test_cases = test_cases[:num_examples]

    # Load personas
    persona_prompts = {}
    for name, path in conditions.items():
        if path:
            with open(path) as f:
                persona_prompts[name] = f.read().strip()
        else:
            persona_prompts[name] = None

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    n_layers_model = model.config.num_hidden_layers
    valid_layers = [l for l in layers if l < n_layers_model]
    print(f"Model has {n_layers_model} layers, extracting at: {valid_layers}")

    results = []
    total = len(test_cases) * len(conditions)
    done = 0
    t0 = time.time()

    for tc in test_cases:
        content_words = extract_content_words(tc)
        example_result = {
            "test_id": tc.get("test_id"),
            "category": tc["category"],
            "content_words": content_words,
            "conditions": {},
        }

        for cond_name, persona_prompt in persona_prompts.items():
            prompt = create_prompt_with_persona(
                tc["scenario"], tc["items"], tc["outcome"],
                tc["question"], tc["statement"], persona_prompt,
            )
            regions = find_token_regions(tokenizer, tc, persona_prompt)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # Stack only requested layers
            attn_list = []
            for layer_idx in valid_layers:
                attn_list.append(outputs.attentions[layer_idx].squeeze(0).cpu())
            attn = torch.stack(attn_list, dim=0)  # (n_selected, n_heads, seq, seq)

            # Free GPU memory
            del outputs, inputs
            if device == "mps":
                torch.mps.empty_cache()

            ratios = compute_content_ratios(
                attn, tokens, regions, content_words,
                valid_layers, list(range(len(valid_layers))),
            )
            example_result["conditions"][cond_name] = {
                "s2o": ratios["s2o"],
                "l2o": ratios["l2o"],
                "l2s": ratios["l2s"],
            }

            # Store per-condition token counts (may differ due to persona length)
            if "chance_outcome" not in example_result:
                example_result["chance_outcome"] = ratios["chance_outcome"]
                example_result["chance_statement"] = ratios["chance_statement"]
                example_result["n_content_outcome"] = ratios["n_content_outcome"]
                example_result["n_content_statement"] = ratios["n_content_statement"]
                example_result["n_total_outcome"] = ratios["n_total_outcome"]
                example_result["n_total_statement"] = ratios["n_total_statement"]

            del attn
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] test {tc.get('test_id')} × {cond_name} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        results.append(example_result)

    output = {
        "meta": {
            "model": model_name,
            "layers": valid_layers,
            "n_examples": len(test_cases),
            "conditions": list(conditions.keys()),
        },
        "results": results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {output_path}")

    return output


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIRECTION_LABELS = {
    "s2o": "Statement → Outcome",
    "l2o": "Last Token → Outcome",
    "l2s": "Last Token → Statement",
}

CONDITION_COLORS = {
    "baseline": "#2196F3",
    "anti_gricean": "#F44336",
    "literal_thinker": "#FF9800",
    "helpful_teacher": "#4CAF50",
    "pragmaticist": "#9C27B0",
}


def _gather_by_category(data: Dict, direction: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Gather content ratios grouped by category → condition → layer → list of values.
    """
    layers = data["meta"]["layers"]
    by_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for ex in data["results"]:
        cat = ex["category"]
        for cond, cond_data in ex["conditions"].items():
            for layer_key, val in cond_data[direction].items():
                by_cat[cat][cond][layer_key].append(val)

    return by_cat


def plot_content_ratio_by_layer(data: Dict, output_dir: str):
    """
    Plot 1: Content attention ratio by layer, one panel per category × direction.
    Lines for each condition, dashed horizontal for chance.
    """
    layers = data["meta"]["layers"]
    conditions = data["meta"]["conditions"]

    for direction in ["s2o", "l2o", "l2s"]:
        by_cat = _gather_by_category(data, direction)
        chance_key = "chance_outcome" if direction != "l2s" else "chance_statement"

        cats = [c for c in CATEGORY_ORDER if c in by_cat]
        n_cats = len(cats)
        fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 4), squeeze=False)

        for col, cat in enumerate(cats):
            ax = axes[0, col]

            # Compute mean chance for this category
            chances = [ex[chance_key] for ex in data["results"] if ex["category"] == cat]
            mean_chance = np.mean(chances) if chances else 0

            for cond in conditions:
                layer_means = []
                layer_sems = []
                for layer in layers:
                    lk = f"L{layer}"
                    vals = by_cat[cat][cond].get(lk, [])
                    if vals:
                        layer_means.append(np.mean(vals))
                        layer_sems.append(np.std(vals) / np.sqrt(len(vals)))
                    else:
                        layer_means.append(0)
                        layer_sems.append(0)

                color = CONDITION_COLORS.get(cond, "gray")
                label = PERSONA_DISPLAY.get(cond, cond)
                ax.plot(layers, layer_means, "o-", color=color, label=label,
                        markersize=4, linewidth=1.5)
                ax.fill_between(
                    layers,
                    np.array(layer_means) - np.array(layer_sems),
                    np.array(layer_means) + np.array(layer_sems),
                    alpha=0.15, color=color,
                )

            ax.axhline(mean_chance, color="gray", ls="--", lw=1, label="Chance")
            ax.set_title(cat, fontsize=11)
            ax.set_xlabel("Layer")
            if col == 0:
                ax.set_ylabel("Content word attention ratio")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        axes[0, -1].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Content-Word Attention Ratio: {DIRECTION_LABELS[direction]}",
                     fontsize=13)
        plt.tight_layout()
        path = str(Path(output_dir) / f"content_ratio_{direction}_by_layer.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {path}")


def plot_distribution_at_peak(data: Dict, output_dir: str):
    """
    Plot 2: Box plots of content ratio distribution at the peak layer,
    per category, baseline only. Shows how tight the distribution is.
    """
    layers = data["meta"]["layers"]

    for direction in ["s2o", "l2o", "l2s"]:
        by_cat = _gather_by_category(data, direction)
        chance_key = "chance_outcome" if direction != "l2s" else "chance_statement"

        cats = [c for c in CATEGORY_ORDER if c in by_cat]

        # Find peak layer per category (baseline, highest mean)
        fig, axes = plt.subplots(1, len(cats), figsize=(3.5 * len(cats), 4), squeeze=False)

        for col, cat in enumerate(cats):
            ax = axes[0, col]
            baseline_data = by_cat[cat].get("baseline", {})

            # Find peak layer
            best_layer = None
            best_mean = -1
            for lk, vals in baseline_data.items():
                m = np.mean(vals) if vals else 0
                if m > best_mean:
                    best_mean = m
                    best_layer = lk

            if best_layer is None:
                continue

            # Collect values at peak layer for all conditions
            box_data = []
            box_labels = []
            box_colors = []
            for cond in data["meta"]["conditions"]:
                vals = by_cat[cat][cond].get(best_layer, [])
                if vals:
                    box_data.append(vals)
                    box_labels.append(PERSONA_DISPLAY.get(cond, cond))
                    box_colors.append(CONDITION_COLORS.get(cond, "gray"))

            if not box_data:
                continue

            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                            widths=0.6, showfliers=True)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            # Chance line
            chances = [ex[chance_key] for ex in data["results"] if ex["category"] == cat]
            ax.axhline(np.mean(chances), color="gray", ls="--", lw=1, label="Chance")

            ax.set_title(f"{cat}\n(peak: {best_layer})", fontsize=10)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")
            if col == 0:
                ax.set_ylabel("Content word attention ratio")

        fig.suptitle(f"Distribution at Peak Layer: {DIRECTION_LABELS[direction]}",
                     fontsize=13)
        plt.tight_layout()
        path = str(Path(output_dir) / f"content_ratio_{direction}_boxplot.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {path}")


def plot_category_comparison(data: Dict, output_dir: str):
    """
    Plot 4: Bar chart comparing content ratios across categories at peak layer.
    Baseline only. Shows whether model attends to content words more in
    underinf cases than true/false.
    """
    layers = data["meta"]["layers"]

    for direction in ["s2o", "l2o", "l2s"]:
        by_cat = _gather_by_category(data, direction)
        chance_key = "chance_outcome" if direction != "l2s" else "chance_statement"
        cats = [c for c in CATEGORY_ORDER if c in by_cat]

        # For each category, find peak layer and get mean ± SEM (baseline)
        means = []
        sems = []
        chances = []
        for cat in cats:
            baseline_data = by_cat[cat].get("baseline", {})
            best_layer = max(baseline_data.keys(),
                             key=lambda lk: np.mean(baseline_data[lk]) if baseline_data[lk] else 0,
                             default=None)
            if best_layer and baseline_data[best_layer]:
                vals = baseline_data[best_layer]
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                means.append(0)
                sems.append(0)
            cat_chances = [ex[chance_key] for ex in data["results"] if ex["category"] == cat]
            chances.append(np.mean(cat_chances) if cat_chances else 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(cats))
        width = 0.35

        ax.bar(x - width/2, means, width, yerr=sems, capsize=4,
               color="#2196F3", alpha=0.7, label="Content ratio (baseline)")
        ax.bar(x + width/2, chances, width,
               color="gray", alpha=0.4, label="Chance (uniform)")

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_ylabel("Attention ratio")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(f"Content-Word Attention by Category: {DIRECTION_LABELS[direction]}\n"
                     f"(baseline, at peak layer per category)")

        plt.tight_layout()
        path = str(Path(output_dir) / f"content_ratio_{direction}_category_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {path}")


def print_summary_stats(data: Dict):
    """
    Print summary statistics with t-tests against chance.
    """
    layers = data["meta"]["layers"]

    print("\n" + "=" * 80)
    print("SUMMARY: Content-word attention vs chance (one-sample t-test)")
    print("=" * 80)

    for direction in ["s2o", "l2o", "l2s"]:
        print(f"\n--- {DIRECTION_LABELS[direction]} ---")
        by_cat = _gather_by_category(data, direction)
        chance_key = "chance_outcome" if direction != "l2s" else "chance_statement"

        for cat in CATEGORY_ORDER:
            if cat not in by_cat:
                continue
            baseline_data = by_cat[cat].get("baseline", {})
            if not baseline_data:
                continue

            # Find peak layer
            best_layer = max(baseline_data.keys(),
                             key=lambda lk: np.mean(baseline_data[lk]) if baseline_data[lk] else 0,
                             default=None)
            if not best_layer or not baseline_data[best_layer]:
                continue

            vals = np.array(baseline_data[best_layer])
            cat_chances = [ex[chance_key] for ex in data["results"] if ex["category"] == cat]
            chance = np.mean(cat_chances) if cat_chances else 0

            t_stat, p_val = stats.ttest_1samp(vals, chance)
            n = len(vals)
            print(f"  {cat:20s} @ {best_layer}: "
                  f"{np.mean(vals):.3f} ± {np.std(vals):.3f} "
                  f"(chance={chance:.3f}, t={t_stat:.2f}, p={p_val:.2e}, n={n})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract content-word attention ratios across all examples/conditions"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--device", default="mps", help="Device")
    parser.add_argument("--data", default="data/scalar_implicature_250.json", help="Test data")
    parser.add_argument("--layers", default=",".join(str(l) for l in DEFAULT_LAYERS),
                        help="Comma-separated layer indices")
    parser.add_argument("--conditions", nargs="+", default=list(PERSONA_FILES.keys()),
                        help="Conditions to run")
    parser.add_argument("--num-examples", type=int, help="Limit number of examples")
    parser.add_argument("--output", "-o", default="probing/results/content_word_attention.json",
                        help="Output JSON path")
    parser.add_argument("--output-dir", default="probing/results",
                        help="Directory for plots")
    parser.add_argument("--plot-only", metavar="JSON",
                        help="Skip extraction, just plot from saved JSON")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    if args.plot_only:
        with open(args.plot_only) as f:
            data = json.load(f)
        print(f"Loaded {len(data['results'])} examples from {args.plot_only}")
    else:
        conditions = {k: PERSONA_FILES[k] for k in args.conditions if k in PERSONA_FILES}
        data = extract_all(
            model_name=args.model,
            data_path=args.data,
            layers=layers,
            conditions=conditions,
            device=args.device,
            num_examples=args.num_examples,
            output_path=args.output,
        )

    if not args.no_plot:
        out_dir = args.output_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plot_content_ratio_by_layer(data, out_dir)
        plot_distribution_at_peak(data, out_dir)
        plot_category_comparison(data, out_dir)

    print_summary_stats(data)


if __name__ == "__main__":
    main()
