"""
Test script for running a single scenario against both models.
Useful for debugging and verifying setup before running the full experiment.
"""

import argparse
import sys
from pathlib import Path

from .models import LLMInterface, get_available_models
from .run_experiment import load_scenarios, create_prompt, parse_response, evaluate_answer


def test_single_scenario(scenario_id: str | None = None, model_filter: str | None = None):
    """
    Test a single scenario against available models.

    Args:
        scenario_id: Specific scenario ID to test (default: first scenario)
        model_filter: Filter to specific model type ("ollama" or "openai")
    """
    # Load scenarios
    scenarios = load_scenarios()

    # Select scenario
    if scenario_id:
        scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
        if not scenario:
            print(f"Error: Scenario '{scenario_id}' not found.")
            print(f"Available scenarios: {[s['id'] for s in scenarios]}")
            sys.exit(1)
    else:
        scenario = scenarios[0]

    print("=" * 60)
    print(f"TESTING SCENARIO: {scenario['id']}")
    print(f"Structure: {scenario['structure']}")
    print("=" * 60)

    # Create prompt
    prompt = create_prompt(scenario)
    print("\n--- PROMPT ---")
    print(prompt)
    print("-" * 40)

    # Get models to test
    models = get_available_models()
    if model_filter:
        models = [(t, n) for t, n in models if t == model_filter]
        if not models:
            print(f"Error: No models found for filter '{model_filter}'")
            sys.exit(1)

    # Test each model
    for model_type, model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_type}:{model_name}")
        print("=" * 60)

        try:
            llm = LLMInterface(model_type, model_name)
            response = llm.query(prompt)
            parsed = parse_response(response)
            correct = evaluate_answer(parsed["answer"], scenario["ground_truth"])

            print("\n--- RESPONSE ---")
            print(response)
            print("-" * 40)
            print(f"\nParsed answer: {parsed['answer']}")
            print(f"Ground truth: {'yes' if scenario['ground_truth'] else 'no'}")
            print(f"Correct: {'YES' if correct else 'NO'}")

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
    print("-" * 40)
    for s in scenarios:
        print(f"  {s['id']:20} ({s['structure']})")


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
        choices=["ollama", "openai"],
        help="Test only this model type"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available scenarios"
    )

    args = parser.parse_args()

    if args.list:
        list_scenarios()
        return

    test_single_scenario(args.scenario, args.model)


if __name__ == "__main__":
    main()
