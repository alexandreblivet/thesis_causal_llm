"""
Main experiment runner for causal reasoning evaluation.
"""

import json
import csv
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


def create_prompt(scenario: dict) -> str:
    """
    Create zero-shot prompt from scenario.

    Args:
        scenario: Dictionary containing scenario details

    Returns:
        Formatted prompt string
    """
    prompt = f"""{scenario['scenario']}

Question: {scenario['question']}

Please answer with either "Yes" or "No" and provide your reasoning."""

    return prompt


def parse_response(response: str) -> dict:
    """
    Extract answer and reasoning from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        Dictionary with 'answer' (yes/no) and 'reasoning' fields
    """
    response_lower = response.lower().strip()

    # Try to extract yes/no answer
    if response_lower.startswith("yes"):
        answer = "yes"
    elif response_lower.startswith("no"):
        answer = "no"
    elif "yes" in response_lower[:100] and "no" not in response_lower[:50]:
        answer = "yes"
    elif "no" in response_lower[:100] and "yes" not in response_lower[:50]:
        answer = "no"
    else:
        answer = "unclear"

    return {
        "answer": answer,
        "reasoning": response,
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

    predicted_bool = (predicted.lower() == "yes")
    return predicted_bool == ground_truth


def run_experiment(output_dir: str = "data/results"):
    """
    Run the full experiment across all scenarios and models.

    Args:
        output_dir: Directory to save results
    """
    # Setup
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()
    models = get_available_models()

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"results_{timestamp}.csv"

    fieldnames = [
        "timestamp", "scenario_id", "structure", "model_type", "model_name",
        "prompt", "response", "predicted_answer", "ground_truth", "correct"
    ]

    # Run experiment
    results = []
    total_queries = len(scenarios) * len(models)

    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(total=total_queries, desc="Running experiment") as pbar:
            for model_type, model_name in models:
                # Initialize model
                try:
                    llm = LLMInterface(model_type, model_name)
                except Exception as e:
                    print(f"\nError initializing {model_name}: {e}")
                    pbar.update(len(scenarios))
                    continue

                for scenario in scenarios:
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
                            "model_type": model_type,
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

    print(f"\n✓ Experiment complete! Results saved to: {results_file}")
    print_summary(results)

    return results_file


def print_summary(results: list):
    """Print basic accuracy summary."""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)

    # Overall accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")

    # By model
    print("\nAccuracy by Model:")
    models = {}
    for r in results:
        model_key = f"{r['model_type']}:{r['model_name']}"
        if model_key not in models:
            models[model_key] = {"correct": 0, "total": 0}
        models[model_key]["total"] += 1
        if r["correct"]:
            models[model_key]["correct"] += 1

    for model_key, stats in models.items():
        acc = stats["correct"] / stats["total"]
        print(f"  {model_key}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    # By causal structure
    print("\nAccuracy by Causal Structure:")
    structures = {}
    for r in results:
        struct = r["structure"]
        if struct not in structures:
            structures[struct] = {"correct": 0, "total": 0}
        structures[struct]["total"] += 1
        if r["correct"]:
            structures[struct]["correct"] += 1

    for struct, stats in structures.items():
        acc = stats["correct"] / stats["total"]
        print(f"  {struct}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    print("="*50 + "\n")


if __name__ == "__main__":
    run_experiment()