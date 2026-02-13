# Causal Reasoning in LLMs: Marketing Context

MSc Thesis Project - Evaluating whether small LLMs can correctly identify when correlations do (or don't) imply causation in marketing contexts. Uses DAG-generated synthetic data with known ground truth to test causal reasoning capabilities. The experiment varies prompt framing and variable naming to diagnose why models fail.

## Research Question

Can small LLMs correctly identify when correlations do (or don't) imply causation in marketing contexts?

## Key Findings

**All models exhibit a strong "correlation ≠ causation" skepticism bias.** Every model tested — from Llama 3.1 8B to Claude Opus — correctly rejects spurious correlations (confounding, reverse causation) but struggles to recognise genuine direct causation, especially under neutral prompting where **no model scores above 0%** on direct causation.

**Prompt framing matters.** Providing the true causal structure or framing data as from an RCT substantially improves performance on direct causation for Claude models (Opus reaches 100%), but has no effect on Llama 3.1 8B (stays at 0%).

**Variable naming affects open-source models disproportionately.** Llama 3.1 8B drops from 66.7% accuracy on abstract variables to 10.5% on marketing-named variables, suggesting domain-specific terms trigger different heuristics. Claude models show minimal sensitivity.

### Results Summary

**Overall Accuracy: 71.1% (197/277)**

| Model | Confounding | Direct Causation | Reverse Causation | Overall |
|---|---|---|---|---|
| Claude Opus | 100% | 66.7% | 83.3% | 81.2% |
| Claude Sonnet | 100% | 43.3% | 100% | 78.8% |
| Claude Haiku | 100% | 16.7% | 100% | 68.8% |
| Llama 3.1 8B | 78.6% | 0% | 100% | 37.8% |

### Direct Causation Breakdown (where models fail)

| Model | Neutral | Structure-Given | Experiment-Stated |
|---|---|---|---|
| Claude Opus | 0% | 100% | 100% |
| Claude Sonnet | 0% | 70% | 60% |
| Claude Haiku | 0% | 20% | 30% |
| Llama 3.1 8B | 0% | 0% | 0% |

**Key Insight:** Models have over-learned the "correlation ≠ causation" heuristic. They default to "no" when presented with correlational data, even when the correlation reflects genuine causation. Larger Claude models can overcome this bias when given structural information or experimental framing, but smaller models cannot.

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

| Model | Type | Parameters |
|---|---|---|
| Llama 3.1 8B Instruct | Open-source (HuggingFace API) | 8B |
| Claude Haiku 4.5 | Anthropic API | — |
| Claude Sonnet 4.5 | Anthropic API | — |
| Claude Opus 4.5 | Anthropic API | — |

Additional models defined but not yet run: Gemma 2 9B, Qwen 2.5 7B.

### Methodology
- Zero-shot prompting with correlation statistics (no few-shot examples)
- Temperature: 0.0 (deterministic inference)
- Max tokens: 1024
- Binary evaluation: correct/incorrect against known ground truth

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- API keys: `HF_TOKEN` (HuggingFace) and `ANTHROPIC_API_KEY` in a `.env` file

### Installation

```bash
# Clone repository
git clone https://github.com/alexandreblivet/thesis_causal_llm.git
cd thesis_causal_llm

# Install dependencies with uv
uv sync

# Create .env file with API keys
echo "HF_TOKEN=your_token_here" >> .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

## Usage

```bash
# Generate the 30 synthetic scenarios (writes to data/scenarios.json)
uv run python -m thesis_causal_llm.generate_scenarios

# Run the full experiment across all scenarios, models, and prompt conditions
uv run python -m thesis_causal_llm.run_experiment

# Test a single scenario with a specific prompt condition
uv run python -m thesis_causal_llm.test_single --scenario direct_1 --model claude-haiku --prompt-condition neutral

# Filter by variable type
uv run python -m thesis_causal_llm.test_single -s direct_1 -m claude-haiku -vt marketing

# List available scenarios and models
uv run python -m thesis_causal_llm.test_single --list-scenarios
uv run python -m thesis_causal_llm.test_single --list-models

# Run the analysis notebook
uv run jupyter notebook notebooks/analysis.ipynb
```

## Project Structure

```
thesis-causal-llm/
├── data/
│   ├── scenarios.json          # 30 synthetic scenarios (15 marketing + 15 abstract)
│   └── results/                # Experiment output CSVs
├── src/thesis_causal_llm/
│   ├── generate_scenarios.py   # DAG-based synthetic data generation
│   ├── models.py               # LLM abstraction (HuggingFace + Anthropic APIs)
│   ├── run_experiment.py       # Main experiment runner
│   └── test_single.py          # CLI tool for debugging individual scenarios
├── notebooks/
│   └── analysis.ipynb          # Results analysis and visualisation
├── tests/
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## License

MIT
