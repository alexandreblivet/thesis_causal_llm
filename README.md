# Causal Reasoning in LLMs: Marketing Context

MSc Thesis Project - Evaluating whether LLMs can correctly identify when correlations do (or don't) imply causation in marketing contexts. Uses DAG-generated synthetic data with known ground truth to test causal reasoning capabilities. The experiment varies prompt framing and variable naming to diagnose why models fail.

## Research Question

Can LLMs correctly identify when correlations do (or don't) imply causation in marketing contexts?

## Key Findings

**All models exhibit a strong "correlation ≠ causation" skepticism bias.** Every model tested — from Haiku 4.5 to Opus 4.6 — correctly rejects spurious correlations (confounding: 100%, reverse causation: 85.8%) but struggles to recognise genuine direct causation (44.2%), especially under neutral prompting where **no model scores above 0%** on direct causation.

**Providing causal structure is the most effective intervention.** The structure-given condition achieves 86.7% overall accuracy, the highest of all prompt conditions. Opus 4.6 reaches 100% on direct causation when given the DAG, and Sonnet 4.5 reaches 70%.

**The RCT framing creates a new failure mode for newer models.** While experiment-stated prompting helps with direct causation (Opus 4.6 and Sonnet 4.6 reach 100%), it causes Opus 4.6 (20%) and Sonnet 4.6 (10%) to incorrectly affirm causation on reverse causation scenarios — they over-trust the experimental framing and stop checking directionality.

**Marketing variable names slightly improve accuracy.** Unlike the prior open-source model finding, all Claude models perform comparably or slightly better on marketing-named variables (77.5%) vs abstract (70.0%).

### Results Summary

**Overall Accuracy: 73.8% (236/320)**

| Model | Confounding | Direct Causation | Reverse Causation | Overall |
|---|---|---|---|---|
| Sonnet 4.5 | 100% | 43.3% | 100% | 78.8% |
| Opus 4.6 | 100% | 66.7% | 73.3% | 77.5% |
| Sonnet 4.6 | 100% | 50.0% | 70.0% | 70.0% |
| Haiku 4.5 | 100% | 16.7% | 100% | 68.8% |

### Direct Causation Breakdown (where models fail most)

| Model | Neutral | Structure-Given | Experiment-Stated |
|---|---|---|---|
| Opus 4.6 | 0% | 100% | 100% |
| Sonnet 4.6 | 0% | 50% | 100% |
| Sonnet 4.5 | 0% | 70% | 60% |
| Haiku 4.5 | 0% | 20% | 30% |

### Reverse Causation × Experiment-Stated (new failure mode)

| Model | Experiment-Stated | Neutral | Structure-Given |
|---|---|---|---|
| Haiku 4.5 | 100% | 100% | 100% |
| Sonnet 4.5 | 100% | 100% | 100% |
| Opus 4.6 | 20% | 100% | 100% |
| Sonnet 4.6 | 10% | 100% | 100% |

**Key Insight:** Models have over-learned the "correlation ≠ causation" heuristic. They default to "no" when presented with correlational data, even when the correlation reflects genuine causation. Providing the true causal structure is the most reliable fix. However, the RCT framing introduces a trade-off: newer, more capable models (Opus 4.6, Sonnet 4.6) gain perfect direct causation recognition but lose the ability to reject reverse causation under experimental framing.

## Experimental Design

### Factorial Design: 3 Structures × 3 Prompt Conditions × 2 Variable Types

**3 causal structures** (5 scenarios each = 15 per variable type, 30 total):
1. **Direct Causation** (X → M → Y) — ground truth: yes, correlation implies causation
2. **Confounding** (Z → X, Z → Y) — ground truth: no, spurious correlation
3. **Reverse Causation** (Y → X) — ground truth: no, direction is reversed

**3 prompt conditions:**
- **Neutral** — presents correlation data only, asks if X causes Y
- **Structure-given** — provides the true DAG alongside correlation data
- **Experiment-stated** — frames data as from a randomized controlled experiment

**2 variable types:**
- **Marketing** — domain-specific names (ad_spend, sales, etc.)
- **Abstract** — generic names (A, B, C / X1, Y1, etc.)

**Exclusion:** Experiment-stated × confounding combinations are excluded (RCTs eliminate confounding by design), yielding 80 test cases per model.

### Models Tested

| Model | Model ID |
|---|---|
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| Claude Opus 4.6 | `claude-opus-4-6` |

### Methodology
- Zero-shot prompting with correlation statistics (no few-shot examples)
- Temperature: 0.0 (deterministic inference)
- Max tokens: 1024
- Binary evaluation: correct/incorrect against known ground truth

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- API key: `ANTHROPIC_API_KEY` in a `.env` file

### Installation

```bash
# Clone repository
git clone https://github.com/alexandreblivet/thesis_causal_llm.git
cd thesis_causal_llm

# Install dependencies with uv
uv sync

# Create .env file with API key
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

## Usage

```bash
# Generate the 30 synthetic scenarios (writes to data/scenarios.json)
uv run python -m thesis_causal_llm.generate_scenarios

# Run the full experiment across all scenarios, models, and prompt conditions
uv run python -m thesis_causal_llm.run_experiment

# Test a single scenario with a specific prompt condition
uv run python -m thesis_causal_llm.test_single --scenario direct_1 --model claude-opus-4-6 --prompt-condition neutral

# Filter by variable type
uv run python -m thesis_causal_llm.test_single -s direct_1 -m claude-opus-4-6 -vt marketing

# List available scenarios and models
uv run python -m thesis_causal_llm.test_single --list-scenarios
uv run python -m thesis_causal_llm.test_single --list-models

# Analyze results and generate plots to images/
uv run python -m thesis_causal_llm.analyze_results
```

## Project Structure

```
thesis-causal-llm/
├── data/
│   ├── scenarios.json          # 30 synthetic scenarios (15 marketing + 15 abstract)
│   └── results/                # Experiment output CSVs
├── src/thesis_causal_llm/
│   ├── generate_scenarios.py   # DAG-based synthetic data generation
│   ├── models.py               # LLM abstraction (Anthropic API)
│   ├── run_experiment.py       # Main experiment runner
│   ├── test_single.py          # CLI tool for debugging individual scenarios
│   └── analyze_results.py      # Results analysis and plot generation
├── images/                     # Generated plots (seaborn/matplotlib)
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## License

MIT
