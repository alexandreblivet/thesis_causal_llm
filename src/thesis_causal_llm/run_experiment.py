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
            # Convert corr_x_y to corr(X, Y) format
            parts = key.replace("corr_", "").split("_")
            if len(parts) >= 2:
                var1 = parts[0]
                var2 = "_".join(parts[1:])  # Handle multi-word variable names
                lines.append(f"corr({var1}, {var2}) = {value}")

    return "\n".join(lines)


def create_prompt(scenario: dict) -> str:
    """
    Create zero-shot prompt from scenario with correlation data.

    Args:
        scenario: Dictionary containing scenario details

    Returns:
        Formatted prompt string
    """
    correlations = format_correlations(scenario["data_summary"])

    # Get variable names for context
    variables = scenario["variables"]
    x_var = variables["x"]
    y_var = variables["y"]

    prompt = f"""A marketing analyst collected the following data:

{correlations}

The analyst wants to know: {scenario['question']}

Based solely on the correlation data provided, answer Yes or No: Does {x_var} cause {y_var}?

Important: Correlation does not imply causation. Consider whether the data supports a causal claim or if alternative explanations (confounding, reverse causation) are possible.

Answer with "Yes" or "No" first, then explain your reasoning."""

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
        "timestamp", "scenario_id", "structure", "dag", "model_name",
        "prompt", "response", "predicted_answer", "ground_truth", "correct"
    ]

    # Run experiment
    results = []
    total_queries = len(scenarios) * len(models)

    print(f"Running experiment: {len(scenarios)} scenarios x {len(models)} models = {total_queries} queries")
    print(f"Models: {', '.join(models)}\n")

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
                    pbar.update(len(scenarios))
                    continue

                for scenario in scenarios:
                    pbar.set_postfix(model=model_name, scenario=scenario["id"])

                    # Create prompt and query model
                    prompt = create_prompt(scenario)

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
                        print(f"\nError querying {model_name} on {scenario['id']}: {e}")

                    pbar.update(1)

    print(f"\nExperiment complete! Results saved to: {results_file}")
    print_summary(results)

    return results_file


def print_summary(results: list):
    """Print accuracy summary by model and structure."""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Overall accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")

    # By model
    print("\nAccuracy by Model:")
    print("-" * 40)
    models = {}
    for r in results:
        model = r["model_name"]
        if model not in models:
            models[model] = {"correct": 0, "total": 0}
        models[model]["total"] += 1
        if r["correct"]:
            models[model]["correct"] += 1

    for model, stats in sorted(models.items()):
        acc = stats["correct"] / stats["total"]
        print(f"  {model:20} {acc:6.1%} ({stats['correct']}/{stats['total']})")

    # By causal structure
    print("\nAccuracy by Causal Structure:")
    print("-" * 40)
    structures = {}
    for r in results:
        struct = r["structure"]
        if struct not in structures:
            structures[struct] = {"correct": 0, "total": 0}
        structures[struct]["total"] += 1
        if r["correct"]:
            structures[struct]["correct"] += 1

    for struct, stats in sorted(structures.items()):
        acc = stats["correct"] / stats["total"]
        print(f"  {struct:20} {acc:6.1%} ({stats['correct']}/{stats['total']})")

    # Cross-tabulation: Model x Structure
    print("\nModel x Structure Breakdown:")
    print("-" * 60)

    # Header
    struct_names = sorted(structures.keys())
    header = f"{'Model':20}" + "".join(f"{s[:12]:>14}" for s in struct_names)
    print(header)

    for model in sorted(models.keys()):
        row = f"{model:20}"
        for struct in struct_names:
            model_struct = [r for r in results if r["model_name"] == model and r["structure"] == struct]
            if model_struct:
                acc = sum(1 for r in model_struct if r["correct"]) / len(model_struct)
                row += f"{acc:13.1%} "
            else:
                row += f"{'N/A':>14}"
        print(row)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_experiment()
