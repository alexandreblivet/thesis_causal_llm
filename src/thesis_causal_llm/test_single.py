"""
Test script for running a single scenario against Ollama models.
Useful for debugging and verifying setup before running the full experiment.
"""

import argparse
import sys

from .models import LLMInterface, get_available_models
from .run_experiment import (
    load_scenarios,
    create_prompt,
    parse_response,
    evaluate_answer,
    PROMPT_CONDITIONS,
)


def test_single_scenario(
    scenario_id: str | None = None,
    model_filter: str | None = None,
    prompt_condition: str = "neutral",
    variable_type: str | None = None,
):
    """
    Test a single scenario against available models.

    Args:
        scenario_id: Specific scenario ID to test (default: first scenario)
        model_filter: Test only this specific model
        prompt_condition: Prompt condition to use (neutral, structure_given, experiment_stated)
        variable_type: Filter scenarios by variable type (marketing or abstract)
    """
    # Load scenarios
    scenarios = load_scenarios()

    # Filter by variable type if specified
    if variable_type:
        scenarios = [s for s in scenarios if s.get("variable_type") == variable_type]
        if not scenarios:
            print(f"Error: No scenarios with variable_type '{variable_type}'.")
            sys.exit(1)

    # Select scenario
    if scenario_id:
        scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
        if not scenario:
            all_scenarios = load_scenarios()
            print(f"Error: Scenario '{scenario_id}' not found.")
            print(f"Available scenarios: {[s['id'] for s in all_scenarios]}")
            sys.exit(1)
    else:
        scenario = scenarios[0]

    # Validate prompt condition vs structure
    if prompt_condition == "experiment_stated" and scenario["structure"] == "confounding":
        print(f"Warning: experiment_stated × confounding is excluded by design (RCTs eliminate confounding).")
        print("Running anyway for inspection purposes.\n")

    print("=" * 70)
    print(f"SCENARIO: {scenario['id']}")
    print(f"Structure: {scenario['structure']}")
    print(f"DAG: {scenario['dag']}")
    print(f"Variable type: {scenario.get('variable_type', 'marketing')}")
    print(f"Prompt condition: {prompt_condition}")
    print("=" * 70)

    # Create prompt
    prompt = create_prompt(scenario, prompt_condition)
    print("\n--- PROMPT ---")
    print(prompt)
    print("-" * 70)

    # Get models to test
    models = get_available_models()
    if model_filter:
        if model_filter in models:
            models = [model_filter]
        else:
            print(f"Error: Model '{model_filter}' not in available models: {models}")
            sys.exit(1)

    # Test each model
    for model_name in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print("=" * 70)

        try:
            llm = LLMInterface(model_name)
            response = llm.query(prompt)
            parsed = parse_response(response)
            correct = evaluate_answer(parsed["answer"], scenario["ground_truth"])

            print("\n--- RESPONSE ---")
            print(response)
            print("-" * 70)
            print(f"\nParsed answer: {parsed['answer']}")
            print(f"Ground truth:  {'yes' if scenario['ground_truth'] else 'no'}")
            print(f"Correct:       {'YES' if correct else 'NO'}")

            if not correct:
                print(f"\nExpected reasoning: {scenario['reasoning']}")

        except RuntimeError as e:
            print(f"\nError: {e}")
        except Exception as e:
            print(f"\nUnexpected error: {type(e).__name__}: {e}")


def list_scenarios():
    """Print all available scenarios."""
    scenarios = load_scenarios()
    print("\nAvailable scenarios:")
    print("-" * 75)
    print(f"{'ID':<25} {'Structure':<20} {'Var Type':<12} {'Ground Truth'}")
    print("-" * 75)
    for s in scenarios:
        gt = "Yes (causal)" if s["ground_truth"] else "No (not causal)"
        vt = s.get("variable_type", "marketing")
        print(f"{s['id']:<25} {s['structure']:<20} {vt:<12} {gt}")


def list_models():
    """Print all available models."""
    models = get_available_models()
    print("\nAvailable models:")
    print("-" * 40)
    for m in models:
        print(f"  {m}")
    print("\nTo pull a model: ollama pull <model_name>")


def main():
    parser = argparse.ArgumentParser(
        description="Test a single causal reasoning scenario"
    )
    parser.add_argument(
        "--scenario", "-s",
        help="Scenario ID to test (default: first scenario)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Test only this specific model"
    )
    parser.add_argument(
        "--list-scenarios", "-ls",
        action="store_true",
        help="List all available scenarios"
    )
    parser.add_argument(
        "--list-models", "-lm",
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--prompt-condition", "-pc",
        choices=PROMPT_CONDITIONS,
        default="neutral",
        help="Prompt condition (default: neutral)"
    )
    parser.add_argument(
        "--variable-type", "-vt",
        choices=["marketing", "abstract"],
        default=None,
        help="Filter scenarios by variable type"
    )

    args = parser.parse_args()

    if args.list_scenarios:
        list_scenarios()
        return

    if args.list_models:
        list_models()
        return

    test_single_scenario(args.scenario, args.model, args.prompt_condition, args.variable_type)


if __name__ == "__main__":
    main()
