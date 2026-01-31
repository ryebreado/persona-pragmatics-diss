# Probing & Attention Analysis

## Overview

This directory contains interpretability experiments for analyzing how LLMs process scalar implicature under different persona conditions.

## Experiments

### 1. Linear Probing (`train_probes.py`)

Trains logistic regression probes on model activations to test what information is linearly decodable at each layer.

**Tasks:**
- `underinf_binary`: Classify underinformative vs everything else
- `underinf_conj` / `underinf_quant`: Subcategory-specific probes
- `three_way`: True vs false vs underinformative

**Token positions:**
- `last_token`: Activations at the final prompt token (decision point)
- `mean_pooled`: Mean across all prompt tokens
- `keyword_some`, `keyword_all`, `keyword_and`: Activations at scalar term positions

### 2. Behavioral Outcome Probing

Predicts behavioral outcomes from pre-generation activations.

**Example:** For Anti-Gricean persona on underinformative items, probe whether the model will accept (incorrectly) or reject (correctly) based on activations *before* generation starts.

High probe accuracy → decision is "baked in" to representation
Low probe accuracy → decision emerges during decoding

### 3. Attention Pattern Analysis (`attention_analysis.py`)

Extracts attention weights between key token regions to understand information flow.

## Attention Analysis: Methodology

### Token Regions

Each test case has these regions identified:

| Region | Description | Example |
|--------|-------------|---------|
| **Outcome** | The factual statement (line 3 of prompt) | "The rabbit collected the coin and the ring." |
| **Statement** | The answer being evaluated (in quotes) | "The rabbit collected the coin" |
| **Scalar term** | Key word within statement | "some", "all", "and" |
| **Last token** | Final token before generation | End of "Answer (yes/no):" |

### Attention Metrics

#### `last_to_outcome` / `last_to_statement`
**What it measures:** How much the decision token attends to a region.

**Calculation:** For multi-token regions, we **sum** attention weights to all tokens in the region.

```
last_to_statement[layer, head] = Σ attention[last_token → token_i]
                                  for token_i in statement_span
```

**Interpretation:** Higher values mean the decision point is gathering more information from that region. Summing (rather than averaging) preserves the total attention budget allocated to the region.

#### `statement_to_outcome`
**What it measures:** How much the statement representation attends to the factual outcome.

**Calculation:** For region-to-region attention, we take the **mean** over source positions and **sum** over target positions.

```
statement_to_outcome[layer, head] = mean over src_i in statement:
                                      Σ attention[src_i → tgt_j]
                                      for tgt_j in outcome_span
```

**Interpretation:** Shows whether tokens in the statement are "looking at" the outcome to verify correctness.

#### `last_to_some` (scalar term attention)
**What it measures:** Attention from decision point to a single token (the scalar term).

**Calculation:** Direct attention weight (no aggregation needed).

```
last_to_some[layer, head] = attention[last_token → position_of_"some"]
```

**Interpretation:** Higher attention to the scalar term may indicate pragmatic processing—the model focusing on "some" to evaluate whether it's appropriate given the outcome.

### Reading the Visualizations

#### Line plots (attention by layer)
- **X-axis:** Layer number (0 = embedding, higher = later processing)
- **Y-axis:** Attention weight (summed over target region, averaged over heads)
- **Lines:** Different categories (true/false/underinf)

Peaks indicate layers where that attention pattern is most active.

#### Heatmaps (layer × head)
- **X-axis:** Layer
- **Y-axis:** Attention head
- **Color:** Attention intensity

Bright spots reveal specific "attention heads" that specialize in particular patterns.

## Key Findings

### Behavioral Outcome Probe (Anti-Gricean)
- ~93% accuracy at predicting accept/reject from layer 0 onwards
- Flat curve across layers (no emergence)
- **Conclusion:** Persona effect is encoded in representation, not decoding

### Attention to Scalar Terms
- Layer 24 spike for underinf-quant vs true-quant
- May indicate where pragmatic implications of "some" are processed

## File Structure

```
probing/
├── train_probes.py          # Linear probe training
├── run_probe_experiments.py # Batch experiment runner
├── attention_analysis.py    # Attention extraction & region identification
├── README.md                # This file
└── results/
    └── {model}_run_XX/
        ├── *_probe.json     # Probe accuracy by layer
        ├── *_probe.png      # Probe accuracy plots
        ├── attention_*.json # Raw attention data
        └── attention_*.png  # Attention visualizations
```
