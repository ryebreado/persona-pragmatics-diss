# Persona-mediated scalar implicature in LLMs

Looking at how persona prompts affect comprehension of scalar implicature through behavioral and mechanistic approaches.

Created for my undergraduate dissertation at Cambridge.

## Dataset

The scalar implicature dataset (`data/scalar_implicature_250.json`) is adapted from Katsos & Bishop (2011), who developed a paradigm for testing pragmatic tolerance in children and adults. Our stimuli follow their design: participants (LLMs) see a scenario and must judge whether a given statement is a correct answer to a question about what happened.

**Categories (250 total examples):**

- `true-conj` (41): True conjunctive statements (e.g., "ate the apple and the banana" when both were eaten)
- `true-quant` (41): True quantifier statements (e.g., "ate some of the apples" when some but not all were eaten)
- `false-conj` (40): False conjunctive statements (factually incorrect)
- `false-quant` (40): False quantifier statements (factually incorrect)
- `underinf-conj` (44): Underinformative conjunctive (e.g., "ate the apple" when both apple and banana were eaten)
- `underinf-quant` (44): Underinformative quantifier (e.g., "ate some" when all were eaten—the classic scalar implicature case)

**Reference:** Katsos, N., & Bishop, D. V. M. (2011). Pragmatic tolerance: Implications for the acquisition of informativeness and implicature. *Cognition*, 120(1), 67–81. https://doi.org/10.1016/j.cognition.2011.02.015

## Running Experiments

### Quick Start

**Local Models:**
```bash
# Full experiment with all personas
python behavioral/run_persona_experiment.py meta-llama/Llama-3.2-1B-Instruct --device mps

# With activation tracking
python behavioral/run_persona_experiment.py meta-llama/Llama-3.2-1B-Instruct \
    --device mps --track-activations --layers "0,10,20"
```

**API Models:**
```bash
python behavioral/run_persona_experiment.py gpt-4o-mini
```

### What It Does

The script automatically:
1. Detects model type (local vs API based on name)
2. Runs baseline evaluation without persona
3. Runs each persona evaluation from `personas/` directory
4. Compares results and generates analysis

### Common Options

- `--device {mps,cuda,cpu}` - Device for local models (default: mps)
- `--use-logprobs` - Use logprobs for evaluation (recommended for small models)
- `--track-activations` - Track model activations (local models only)
- `--layers "0,10,20"` - Comma-separated layer indices to track
- `--skip-baseline` - Skip baseline run (use existing results)
- `--skip-personas` - Skip persona runs (use existing results)
- `--comparison-only` - Only run comparisons (skip evaluation runs)
- `--verbose` - Show detailed output

### Personas

Current personas in `personas/`:
- `anti_gricean` - Explicitly anti-pragmatic
- `literal_thinker` - Takes statements literally; intended to be implicitly anti-pragmatic
- `pragmaticist` - Explicitly pro-pragmatic
- `helpful_teacher` - Maximally helpful; intended to be implicitly pro-pragmatic

### Output

Results saved to `results/` with automatic naming:
- Baseline: `scalar_implicature_{model}_baseline_{timestamp}.json`
- Personas: `scalar_implicature_{model}_{persona}_{timestamp}.json`