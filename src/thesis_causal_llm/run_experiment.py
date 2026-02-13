"""
Main experiment runner for causal reasoning evaluation.
Tests whether LLMs can distinguish correlation from causation
using DAG-generated synthetic data.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .models import LLMInterface, get_available_models


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_scenarios(scenarios_path: str | None = None) -> list[dict]:
    """Load marketing scenarios from JSON file."""
    if scenarios_path is None:
        scenarios_path = get_project_root() / "data" / "scenarios.json"
    with open(scenarios_path, "r") as f:
        return json.load(f)


def format_correlations(data_summary: dict) -> str:
    """
    Format correlation data for the prompt.

    Args:
        data_summary: Dictionary with n_samples and correlation values

    Returns:
        Formatted string showing correlations
    """
    lines = [f"n = {data_summary['n_samples']} observations"]

    for key, value in data_summary.items():
        if key.startswith("corr_"):
            # Convert corr_var1__var2 to corr(var1, var2) format
            # Uses __ as separator between variable names
            raw = key[len("corr_"):]
            parts = raw.split("__")
            if len(parts) == 2:
                lines.append(f"corr({parts[0]}, {parts[1]}) = {value}")

    return "\n".join(lines)


PROMPT_CONDITIONS = ["neutral", "structure_given", "experiment_stated"]


def create_prompt(scenario: dict, prompt_condition: str = "neutral") -> str:
    """
    Create prompt from scenario with correlation data under a given condition.

    Args:
        scenario: Dictionary containing scenario details
        prompt_condition: One of "neutral", "structure_given", "experiment_stated"

    Returns:
        Formatted prompt string
    """
    correlations = format_correlations(scenario["data_summary"])

    variables = scenario["variables"]
    x_var = variables["x"]
    y_var = variables["y"]

    if prompt_condition == "neutral":
        prompt = f"""You are given data about relationships between variables.

{correlations}

Question: Does {x_var} cause {y_var}?

Answer Yes or No, then explain briefly."""

    elif prompt_condition == "structure_given":
        dag = scenario["dag"]
        prompt = f"""You are given data about relationships between variables.

The true causal structure is: {dag}

{correlations}

Question: Does {x_var} cause {y_var}?

Answer Yes or No, then explain briefly."""

    elif prompt_condition == "experiment_stated":
        prompt = f"""You are given data from a randomized controlled experiment where {x_var} was randomly assigned.

{correlations}

Question: Does {x_var} cause {y_var}?

Answer Yes or No, then explain briefly."""

    else:
        raise ValueError(f"Unknown prompt condition: {prompt_condition}")

    return prompt


def parse_response(response: str) -> dict:
    """
    Extract answer and reasoning from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        Dictionary with 'answer' (yes/no/unclear) and 'raw_response' fields
    """
    response_lower = response.lower().strip()

    # Try to extract yes/no answer
    if response_lower.startswith("yes"):
        answer = "yes"
    elif response_lower.startswith("no"):
        answer = "no"
    elif "**yes**" in response_lower[:100]:
        answer = "yes"
    elif "**no**" in response_lower[:100]:
        answer = "no"
    elif "yes" in response_lower[:100] and "no" not in response_lower[:50]:
        answer = "yes"
    elif "no" in response_lower[:100] and "yes" not in response_lower[:50]:
        answer = "no"
    else:
        answer = "unclear"

    return {
        "answer": answer,
        "raw_response": response
    }


def evaluate_answer(predicted: str, ground_truth: bool) -> bool:
    """
    Check if prediction matches ground truth.

    Args:
        predicted: "yes", "no", or "unclear"
        ground_truth: True or False

    Returns:
        True if correct, False otherwise
    """
    if predicted == "unclear":
        return False

    predicted_bool = predicted == "yes"
    return predicted_bool == ground_truth


def run_experiment(output_dir: str | None = None):
    """
    Run the full experiment across all scenarios and models.

    Args:
        output_dir: Directory to save results (default: data/results)
    """
    # Setup
    if output_dir is None:
        output_dir = get_project_root() / "data" / "results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()
    models = get_available_models()

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.csv"

    fieldnames = [
        "timestamp", "scenario_id", "structure", "dag", "variable_type",
        "prompt_condition", "model_name", "prompt", "response",
        "predicted_answer", "ground_truth", "correct"
    ]

    # Build list of (scenario, prompt_condition) pairs, skipping invalid combos
    test_cases = []
    for scenario in scenarios:
        for condition in PROMPT_CONDITIONS:
            # RCTs eliminate confounding by design — skip this combination
            if condition == "experiment_stated" and scenario["structure"] == "confounding":
                continue
            test_cases.append((scenario, condition))

    # Run experiment
    results = []
    total_queries = len(test_cases) * len(models)

    print(f"Running experiment: {len(test_cases)} test cases x {len(models)} models = {total_queries} queries")
    print(f"  Scenarios: {len(scenarios)} ({len([s for s in scenarios if s['variable_type'] == 'marketing'])} marketing + {len([s for s in scenarios if s['variable_type'] == 'abstract'])} abstract)")
    print(f"  Prompt conditions: {', '.join(PROMPT_CONDITIONS)}")
    print(f"  Models: {', '.join(models)}\n")

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(total=total_queries, desc="Running experiment") as pbar:
            for model_name in models:
                # Initialize model
                try:
                    llm = LLMInterface(model_name)
                except Exception as e:
                    print(f"\nError initializing {model_name}: {e}")
                    pbar.update(len(test_cases))
                    continue

                for scenario, condition in test_cases:
                    pbar.set_postfix(model=model_name, scenario=scenario["id"], cond=condition)

                    # Create prompt and query model
                    prompt = create_prompt(scenario, condition)

                    try:
                        response = llm.query(prompt)
                        parsed = parse_response(response)
                        correct = evaluate_answer(
                            parsed["answer"],
                            scenario["ground_truth"]
                        )

                        # Record result
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "scenario_id": scenario["id"],
                            "structure": scenario["structure"],
                            "dag": scenario["dag"],
                            "variable_type": scenario.get("variable_type", "marketing"),
                            "prompt_condition": condition,
                            "model_name": model_name,
                            "prompt": prompt,
                            "response": parsed["raw_response"],
                            "predicted_answer": parsed["answer"],
                            "ground_truth": scenario["ground_truth"],
                            "correct": correct
                        }

                        results.append(result)
                        writer.writerow(result)
                        f.flush()  # Save incrementally

                    except Exception as e:
                        print(f"\nError querying {model_name} on {scenario['id']} ({condition}): {e}")

                    pbar.update(1)

    print(f"\nExperiment complete! Results saved to: {results_file}")
    print_summary(results)

    return results_file


def print_summary(results: list):
    """Print accuracy summary by model, structure, prompt condition, and variable type."""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Overall accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Helper to compute accuracy for a filtered subset
    def _acc(subset):
        if not subset:
            return "N/A"
        c = sum(1 for r in subset if r["correct"])
        return f"{c / len(subset):5.1%} ({c}/{len(subset)})"

    # By model
    print("\nAccuracy by Model:")
    print("-" * 40)
    model_names = sorted({r["model_name"] for r in results})
    for model in model_names:
        subset = [r for r in results if r["model_name"] == model]
        print(f"  {model:20} {_acc(subset)}")

    # By causal structure
    print("\nAccuracy by Causal Structure:")
    print("-" * 40)
    struct_names = sorted({r["structure"] for r in results})
    for struct in struct_names:
        subset = [r for r in results if r["structure"] == struct]
        print(f"  {struct:20} {_acc(subset)}")

    # By prompt condition
    print("\nAccuracy by Prompt Condition:")
    print("-" * 40)
    cond_names = sorted({r["prompt_condition"] for r in results})
    for cond in cond_names:
        subset = [r for r in results if r["prompt_condition"] == cond]
        print(f"  {cond:20} {_acc(subset)}")

    # By variable type
    print("\nAccuracy by Variable Type:")
    print("-" * 40)
    var_types = sorted({r["variable_type"] for r in results})
    for vt in var_types:
        subset = [r for r in results if r["variable_type"] == vt]
        print(f"  {vt:20} {_acc(subset)}")

    # Cross-tabulation: Model x Structure x Prompt Condition
    print("\nModel x Structure x Prompt Condition Breakdown:")
    print("-" * 70)

    for model in model_names:
        print(f"\n  {model}:")
        header = f"    {'':20}" + "".join(f"{s[:12]:>14}" for s in struct_names)
        print(header)
        for cond in cond_names:
            row = f"    {cond:20}"
            for struct in struct_names:
                subset = [r for r in results
                          if r["model_name"] == model
                          and r["structure"] == struct
                          and r["prompt_condition"] == cond]
                if subset:
                    acc = sum(1 for r in subset if r["correct"]) / len(subset)
                    row += f"{acc:13.1%} "
                else:
                    row += f"{'—':>14}"
            print(row)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_experiment()
