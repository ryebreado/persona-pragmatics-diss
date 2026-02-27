#!/usr/bin/env python3
"""
Extract content words from scalar implicature test cases.

Content words = object nouns/adjectives from `items` field + scalar terms (some, all, and).
Used to measure whether attention patterns in BertViz examples are representative.
"""

import re
from typing import Dict, List, Tuple


def extract_object_words(items: str) -> List[str]:
    """
    Parse object words from the items field.

    Handles two patterns:
      "There's a coin, a button, a ring and a marble." → [coin, button, ring, marble]
      "There are hills and slopes." → [hills, slopes]
      "There's a purple potion, a green potion, ..." → [purple, potion, green, potion, ...]
    """
    # Strip "There's a/an " or "There are "
    text = re.sub(r"^There(?:'s|'s| is) (?:a |an )?", "", items.strip().rstrip("."))
    text = re.sub(r"^There are ", "", text)

    # Strip trailing location phrases ("in the river", "in the picture book")
    text = re.sub(r"\s+(?:in|on|at) the .+$", "", text)

    # Split on ", a/an ", ", " and " and a/an " / " and "
    parts = re.split(r",\s*(?:a |an )?| and (?:a |an )?", text)
    parts = [p.strip() for p in parts if p.strip()]

    # Each part may be multi-word ("purple potion") — keep all words
    words = []
    for part in parts:
        for w in part.split():
            words.append(w.lower())
    return words


SCALAR_TERMS = ["some", "all", "and"]

NUMBER_WORDS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
]


def extract_content_words(test_case: Dict) -> Dict[str, List[str]]:
    """
    Extract content words for a single test case.

    Returns:
        {"object_words": [...], "scalar_terms": [...], "number_words": [...]}
    """
    object_words = extract_object_words(test_case["items"])

    # Filter scalar terms to those actually present in outcome or statement
    text = (test_case["outcome"] + " " + test_case["statement"]).lower()
    present_terms = [t for t in SCALAR_TERMS if f" {t} " in f" {text} "]

    # Extract number words from "N out of M" patterns in outcome
    present_numbers = [n for n in NUMBER_WORDS if f" {n} " in f" {text} "]

    return {
        "object_words": object_words,
        "scalar_terms": present_terms,
        "number_words": present_numbers,
    }


def _clean_bpe(token: str) -> str:
    """Strip BPE markers and lowercase."""
    return token.replace("▁", "").replace("\u0120", "").strip().lower()


def find_content_token_positions(
    tokens: List[str],
    content_words: List[str],
    region_span: Tuple[int, int],
) -> List[int]:
    """
    Find token indices within a region that match content words.

    Uses greedy forward concatenation to handle BPE splits:
    e.g. "eucalyptus" → ['e', 'uc', 'aly', 'pt', 'us'] → all 5 tokens matched.

    Args:
        tokens: Full token list from tokenizer
        content_words: Words to search for (lowercase)
        region_span: (start, end) token indices defining the region

    Returns:
        Sorted list of token indices (absolute, not relative to region)
    """
    start, end = region_span
    matched = set()

    # Build cleaned token list for the region
    cleaned = []
    for idx in range(start, end):
        cleaned.append(_clean_bpe(tokens[idx]))

    # For each position, try to greedily match a content word by
    # concatenating consecutive cleaned tokens
    idx = 0
    while idx < len(cleaned):
        found = False
        concat = ""
        for j in range(idx, len(cleaned)):
            concat += cleaned[j]
            if not concat:
                break
            # Check if any content word matches exactly or starts with concat
            for word in content_words:
                if concat == word:
                    # Full match — mark all tokens from idx..j
                    for k in range(idx, j + 1):
                        matched.add(start + k)
                    found = True
                    break
                elif word.startswith(concat):
                    # Still building — keep going
                    continue
                # concat overshoots all words: no point extending further
            if found:
                break
            # If no word starts with concat, stop extending
            if not any(w.startswith(concat) for w in content_words):
                break

        # Also do single-token prefix match for common words that are
        # superstrings of short tokens: e.g. cleaned token "blue" matches
        # content word "blueberries" (token is prefix of word — already
        # handled above), or "butterflies" matches "butter" (word prefix
        # of token — need this fallback)
        if not found:
            tok_c = cleaned[idx]
            if tok_c:
                for word in content_words:
                    if tok_c.startswith(word) or word.startswith(tok_c):
                        matched.add(start + idx)
                        break

        idx += 1

    return sorted(matched)


if __name__ == "__main__":
    import json

    with open("data/scalar_implicature_250.json") as f:
        data = json.load(f)

    # Spot-check a few examples
    for tc in data[:6]:
        cw = extract_content_words(tc)
        print(f"Test {tc['test_id']} ({tc['category']}): "
              f"objects={cw['object_words']}, scalars={cw['scalar_terms']}, "
              f"numbers={cw['number_words']}")
        print(f"  items: {tc['items']}")
        print(f"  outcome: {tc['outcome']}")
        print()

    # Also check a quant case with "out of" pattern
    for tc in data:
        if tc['test_id'] == 62:
            cw = extract_content_words(tc)
            print(f"Test {tc['test_id']} ({tc['category']}): "
                  f"objects={cw['object_words']}, scalars={cw['scalar_terms']}, "
                  f"numbers={cw['number_words']}")
            print(f"  items: {tc['items']}")
            print(f"  outcome: {tc['outcome']}")
            break
