#!/usr/bin/env python3
"""McNemar's test for comparing baseline vs persona conditions."""

import argparse
import json
from pathlib import Path

from statsmodels.stats.contingency_tables import mcnemar


def load_results(filepath: Path) -> dict[int, bool]:
    """Load results and return dict mapping test_id -> correct."""
    with open(filepath) as f:
        data = json.load(f)
    return {r["test_id"]: r["correct"] for r in data["results"]}


def build_contingency_table(baseline: dict[int, bool], persona: dict[int, bool]) -> tuple[int, int, int, int]:
    """Build 2x2 contingency table for McNemar's test.

    Returns (a, b, c, d) where:
        a: both correct
        b: baseline correct, persona incorrect
        c: baseline incorrect, persona correct
        d: both incorrect
    """
    a = b = c = d = 0
    for test_id in baseline:
        base_correct = baseline[test_id]
        pers_correct = persona[test_id]
        if base_correct and pers_correct:
            a += 1
        elif base_correct and not pers_correct:
            b += 1
        elif not base_correct and pers_correct:
            c += 1
        else:
            d += 1
    return a, b, c, d


def run_mcnemar_test(run_dir: Path) -> None:
    """Run McNemar's test comparing baseline to each persona."""
    json_files = list(run_dir.glob("*.json"))

    # Find baseline file
    baseline_file = None
    persona_files = []
    for f in json_files:
        if "_baseline_" in f.name:
            baseline_file = f
        else:
            persona_files.append(f)

    if baseline_file is None:
        raise FileNotFoundError(f"No baseline file found in {run_dir}")

    baseline_results = load_results(baseline_file)

    print(f"McNemar's Test Results: {run_dir.name}")
    print("=" * 50)

    for persona_file in sorted(persona_files):
        # Extract persona name from filename
        name = persona_file.stem
        for part in name.split("_"):
            if part in ["anti", "helpful", "literal", "pragmaticist"]:
                idx = name.split("_").index(part)
                if part == "anti":
                    persona_name = "anti_gricean"
                elif part == "helpful":
                    persona_name = "helpful_teacher"
                elif part == "literal":
                    persona_name = "literal_thinker"
                else:
                    persona_name = "pragmaticist"
                break
        else:
            persona_name = "unknown"

        persona_results = load_results(persona_file)
        a, b, c, d = build_contingency_table(baseline_results, persona_results)

        # Build table for scipy (needs [[a, b], [c, d]] format)
        table = [[a, b], [c, d]]

        # Run McNemar's test (exact=False uses chi-square approximation)
        # Use exact=True for small samples (discordant pairs < 25)
        discordant = b + c
        use_exact = discordant < 25
        result = mcnemar(table, exact=use_exact)

        # Calculate odds ratio (b/c), handle division by zero
        odds_ratio = b / c if c > 0 else float('inf')

        print(f"\nBaseline vs {persona_name}:")
        print(f"  Contingency table:")
        print(f"                      Persona Correct  Persona Incorrect")
        print(f"  Baseline Correct         {a:4d}             {b:4d}")
        print(f"  Baseline Incorrect       {c:4d}             {d:4d}")
        print(f"")
        print(f"  Discordant pairs: {b} + {c} = {discordant}")
        if result.pvalue < 0.0001:
            p_str = "p < 0.0001"
        else:
            p_str = f"p = {result.pvalue:.4f}"
        if use_exact:
            print(f"  McNemar exact test {p_str}")
        else:
            print(f"  McNemar χ² = {result.statistic:.2f}, {p_str}")
        print(f"  Odds ratio (b/c): {odds_ratio:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run McNemar's test on persona experiment results")
    parser.add_argument("run_dir", type=Path, help="Path to run directory containing result JSON files")
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.run_dir}")

    run_mcnemar_test(args.run_dir)


if __name__ == "__main__":
    main()
