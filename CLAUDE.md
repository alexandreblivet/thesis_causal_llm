# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

MSc thesis project evaluating whether LLMs can correctly identify when correlations do (or don't) imply causation in marketing contexts. Uses DAG-generated synthetic data with known ground truth to test causal reasoning capabilities. The experiment varies prompt framing and variable naming to diagnose why models fail.

## Commands

All commands use `uv run` from the project root:

```bash
# Generate the 30 synthetic scenarios (writes to data/scenarios.json)
uv run python -m thesis_causal_llm.generate_scenarios

# Run the full experiment across all scenarios, models, and prompt conditions
uv run python -m thesis_causal_llm.run_experiment

# Test a single scenario with a specific prompt condition
uv run python -m thesis_causal_llm.test_single --scenario direct_1 --model claude-haiku-4-5-20251001 --prompt-condition neutral
uv run python -m thesis_causal_llm.test_single -s abs_direct_1 -m claude-sonnet-4-6 -pc structure_given

# Filter by variable type
uv run python -m thesis_causal_llm.test_single -s direct_1 -m claude-opus-4-6 -vt marketing

# List available scenarios and models
uv run python -m thesis_causal_llm.test_single --list-scenarios
uv run python -m thesis_causal_llm.test_single --list-models

# Run analysis and generate plots to images/
uv run python -m thesis_causal_llm.analyze_results
uv run python -m thesis_causal_llm.analyze_results --results-file data/results/results_TIMESTAMP.csv
```

## Architecture

**Data flow:** `generate_scenarios.py` → `data/scenarios.json` → `run_experiment.py` → `data/results/results_TIMESTAMP.csv` → `analyze_results.py` → `images/*.png`

Key modules in `src/thesis_causal_llm/`:

- **`generate_scenarios.py`** — Generates 30 synthetic scenarios (15 marketing + 15 abstract) using numpy with DAG-based causal structures. Each scenario has 100 observations, effect strength 0.7, noise scale 0.3.
- **`models.py`** — LLM abstraction layer using the Anthropic API exclusively. Loads API key from `.env` via `python-dotenv`.
- **`run_experiment.py`** — Main experiment runner. Iterates all scenario × model × prompt_condition combinations (skipping experiment_stated × confounding), creates prompts from templates, parses yes/no responses, writes incremental CSV results.
- **`test_single.py`** — CLI tool for debugging individual scenario/model pairs. Supports `--scenario/-s`, `--model/-m`, `--prompt-condition/-pc`, `--variable-type/-vt`, `--list-scenarios/-ls`, `--list-models/-lm` flags.
- **`analyze_results.py`** — Generates seaborn/matplotlib plots to `images/` and prints statistical summaries (binomial tests, precision/recall/F1, error analysis). Optional `--results-file` flag; defaults to latest CSV.

## Experimental Design

### Factorial Design: 3 structures × 3 prompt conditions × 2 variable types

- **3 causal structures** (5 scenarios each = 15 per variable type):
  - **Direct causation** (X → M → Y) — ground truth: yes, correlation implies causation
  - **Confounding** (Z → X, Z → Y) — ground truth: no, spurious correlation
  - **Reverse causation** (Y → X) — ground truth: no, direction is reversed

- **3 prompt conditions**:
  - **Neutral** — presents correlation data only, asks if X causes Y
  - **Structure-given** — provides the true DAG alongside correlation data
  - **Experiment-stated** — frames data as from a randomized controlled experiment

- **2 variable types**:
  - **Marketing** — domain-specific names (ad_spend, sales, etc.)
  - **Abstract** — generic names (A, B, C / X1, Y1, etc.)

- **Exclusion**: Experiment-stated × confounding combinations are excluded (RCTs eliminate confounding by design), reducing total from 90 to 80 test cases per model.

- **4 models**: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`), Claude Sonnet 4.6 (`claude-sonnet-4-6`), Claude Opus 4.6 (`claude-opus-4-6`)
- **Zero-shot prompting** with correlation statistics (no few-shot examples)
- **Evaluation**: Binary correct/incorrect against known ground truth

### Hypotheses

| Condition | Direct (Yes) | Confounded (No) | Reverse (No) | Prediction |
|---|---|---|---|---|
| Neutral | Mostly No | Mostly No | Mostly No | Skepticism bias persists |
| Structure-given | More Yes | Still No | Still No | Models can reason causally when given structure |
| Experiment-stated | Mostly Yes | N/A* | Mostly Yes | Models understand RCT implies causation |

*Experiment-stated × confounding cells are excluded (RCT eliminates confounding by design).

## Key Technical Details

- **Random seed**: 42 (in `generate_scenarios.py`)
- **Temperature**: 0.0 (deterministic inference)
- **Max tokens**: 1024
- **Python**: >=3.12, managed with `uv`
- **API keys**: `ANTHROPIC_API_KEY` in `.env` (gitignored)
- **Results format**: CSV with columns: timestamp, scenario_id, structure, dag, variable_type, prompt_condition, model_name, prompt, response, predicted_answer, ground_truth, correct
