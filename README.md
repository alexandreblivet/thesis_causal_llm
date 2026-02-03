# Causal Reasoning in LLMs: Marketing Context

MSc Thesis Project - Evaluating small language models' ability to distinguish correlation from causation in marketing scenarios.

## Research Question

Can small LLMs correctly identify when correlations do (or don't) imply causation in marketing contexts?

## Key Finding

**Phi-3 Mini shows a systematic "no" bias:** The model correctly rejects spurious correlations (confounding: 100%, reverse causation: 100%) but incorrectly rejects ALL genuine causal relationships (direct causation: 0%).

**Implication:** Small LLMs trained on "correlation ≠ causation" heuristics overgeneralize and miss real causal effects, leading to false negatives in marketing analytics.

## Experimental Design

### DAG-Based Synthetic Data
- Generate numerical datasets from known causal structures
- Present correlation coefficients (not text descriptions)
- Ground truth derived from underlying DAG

### Causal Structures (3 types, 5 scenarios each)
1. **Direct Causation** (X → M → Y): Mediated causation through intermediate variable
2. **Confounding** (Z → X, Z → Y): Spurious correlation
3. **Reverse Causation** (Y → X): Correlation exists but direction is wrong

### Models Tested
- **Phi-3 Mini** (3.8B params, local via Ollama)
- Future: Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B

### Methodology
- Zero-shot prompting only
- Binary evaluation (correct/incorrect causal identification)
- 15 scenarios per model

## Setup

### Prerequisites
- WSL2 with Ubuntu
- Python 3.12+
- uv (Python package manager)
- Ollama (for local models)

### Installation

```bash
# Clone repository
git clone https://github.com/alexandreblivet/thesis_causal_llm.git
cd thesis_causal_llm

# Install dependencies with uv
uv sync

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Pull models (start with smallest)
ollama pull phi3:mini          # 3.8B params, 2.3GB RAM
# ollama pull llama3.1:8b      # Requires 8GB WSL memory
# ollama pull mistral:7b
# ollama pull qwen2.5:7b
```

**Note:** If using WSL with limited RAM, start with `phi3:mini`. To use larger models, increase WSL memory in `.wslconfig`:
```ini
[wsl2]
memory=8GB
```

## Usage

```bash
# Generate DAG-based scenarios (already done, but reproducible)
uv run python -m thesis_causal_llm.generate_scenarios

# Test single scenario
uv run python -m thesis_causal_llm.test_single

# Run full experiment (15 scenarios, ~15 minutes)
uv run python -m thesis_causal_llm.run_experiment

# Results saved to data/results/results_TIMESTAMP.csv
```

## Project Structure

```
thesis-causal-llm/
├── data/
│   ├── scenarios.json          # 15 marketing scenarios
│   └── results/                # Experiment outputs
├── src/
│   ├── models.py               # LLM interface
│   └── run_experiment.py       # Main runner
├── tests/
├── notebooks/
│   └── analysis.ipynb          # Results analysis
├── pyproject.toml
└── README.md
```

## Results (Phi-3 Mini)

**Overall Accuracy: 66.67% (10/15)**

| Causal Structure    | Accuracy | Pattern |
|---------------------|----------|---------|
| Confounding         | 100% (5/5) | ✅ Correctly rejects spurious correlation |
| Reverse Causation   | 100% (5/5) | ✅ Correctly rejects wrong causal direction |
| Direct Causation    | 0% (0/5)   | ❌ Incorrectly rejects genuine causation |

**Key Insight:** Model has learned "correlation ≠ causation, always doubt" heuristic, leading to:
- High precision (no false positives)
- Low recall (misses all true causal relationships)

See `data/results/results_20260203_115706.csv` for detailed responses.

## Next Steps

1. **Add more models** (Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B) - requires 8GB WSL memory
2. **Test prompt variations:**
   - Add "consider mediated causation" hint
   - Few-shot examples with correct direct causation cases
3. **Human baseline:** Test if humans also struggle with mediated causation
4. **Analysis notebook:** Visualize confusion matrices and response patterns
5. **CI/CD:** Add GitHub Actions for automated testing

## License

MIT