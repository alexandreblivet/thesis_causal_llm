"""
DAG-based synthetic scenario generator for causal reasoning experiments.

Generates datasets from three causal structures:
1. Direct causation: X → Y (or X → M → Y)
2. Confounding: Z → X, Z → Y (spurious correlation)
3. Reverse causation: Y → X (direction is opposite to question)

Each scenario includes:
- Synthetic data generated from the true causal DAG
- Correlation matrix (what the LLM sees)
- Ground truth causal structure (for evaluation)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Seed for reproducibility
RANDOM_SEED = 42

# Marketing domain variable sets for realistic scenarios
VARIABLE_SETS = {
    "direct": [
        {"x": "ad_spend", "y": "sales", "m": "website_traffic"},
        {"x": "email_opens", "y": "purchases", "m": "click_throughs"},
        {"x": "discount_rate", "y": "conversion", "m": "cart_additions"},
        {"x": "content_quality", "y": "engagement", "m": "time_on_page"},
        {"x": "personalization", "y": "retention", "m": "user_satisfaction"},
    ],
    "confounding": [
        {"x": "ad_spend", "y": "sales", "z": "seasonality"},
        {"x": "social_posts", "y": "revenue", "z": "product_launches"},
        {"x": "support_calls", "y": "churn", "z": "product_bugs"},
        {"x": "training_hours", "y": "sales_performance", "z": "employee_experience"},
        {"x": "influencer_posts", "y": "brand_awareness", "z": "market_trend"},
    ],
    "reverse": [
        {"x": "customer_reviews", "y": "product_quality"},
        {"x": "support_tickets", "y": "user_frustration"},
        {"x": "referrals", "y": "customer_satisfaction"},
        {"x": "repeat_purchases", "y": "product_value"},
        {"x": "social_shares", "y": "content_virality"},
    ],
}


@dataclass
class Scenario:
    """A causal reasoning scenario."""
    id: str
    structure: str
    dag: str
    variables: dict[str, str]
    data_summary: dict
    question: str
    ground_truth: bool
    reasoning: str


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_direct_causation(
    n_samples: int = 100,
    effect_strength: float = 0.7,
    noise_scale: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from direct causation DAG: X → M → Y

    X causes M, M causes Y. The correlation between X and Y
    reflects genuine (mediated) causation.

    Args:
        n_samples: Number of data points
        effect_strength: Strength of causal effects (0-1)
        noise_scale: Scale of noise terms

    Returns:
        Tuple of (X, M, Y) arrays
    """
    # X is exogenous
    x = np.random.randn(n_samples)

    # M is caused by X
    m = effect_strength * x + noise_scale * np.random.randn(n_samples)

    # Y is caused by M (and thus indirectly by X)
    y = effect_strength * m + noise_scale * np.random.randn(n_samples)

    return x, m, y


def generate_confounding(
    n_samples: int = 100,
    effect_strength: float = 0.7,
    noise_scale: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from confounding DAG: Z → X, Z → Y

    Z (confounder) causes both X and Y. The correlation between
    X and Y is spurious - not causal.

    Args:
        n_samples: Number of data points
        effect_strength: Strength of Z's effects on X and Y
        noise_scale: Scale of noise terms

    Returns:
        Tuple of (X, Y, Z) arrays
    """
    # Z is the confounder (exogenous)
    z = np.random.randn(n_samples)

    # X is caused by Z (not by Y)
    x = effect_strength * z + noise_scale * np.random.randn(n_samples)

    # Y is caused by Z (not by X)
    y = effect_strength * z + noise_scale * np.random.randn(n_samples)

    return x, y, z


def generate_reverse_causation(
    n_samples: int = 100,
    effect_strength: float = 0.7,
    noise_scale: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data from reverse causation DAG: Y → X

    Y causes X, but the question asks if X causes Y.
    The correlation exists but the causal direction is reversed.

    Args:
        n_samples: Number of data points
        effect_strength: Strength of Y's effect on X
        noise_scale: Scale of noise terms

    Returns:
        Tuple of (X, Y) arrays
    """
    # Y is exogenous (the true cause)
    y = np.random.randn(n_samples)

    # X is caused by Y (reverse of what question implies)
    x = effect_strength * y + noise_scale * np.random.randn(n_samples)

    return x, y


def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation, rounded to 2 decimal places."""
    corr = np.corrcoef(a, b)[0, 1]
    return round(corr, 2)


def create_direct_scenario(
    scenario_num: int,
    variables: dict[str, str],
    n_samples: int = 100
) -> Scenario:
    """Create a direct causation scenario."""
    x, m, y = generate_direct_causation(n_samples)

    x_name = variables["x"]
    m_name = variables["m"]
    y_name = variables["y"]

    return Scenario(
        id=f"direct_{scenario_num}",
        structure="direct_causation",
        dag=f"{x_name} → {m_name} → {y_name}",
        variables=variables,
        data_summary={
            "n_samples": n_samples,
            f"corr_{x_name}_{y_name}": compute_correlation(x, y),
            f"corr_{x_name}_{m_name}": compute_correlation(x, m),
            f"corr_{m_name}_{y_name}": compute_correlation(m, y),
        },
        question=f"Given these correlations, does {x_name} cause {y_name}?",
        ground_truth=True,
        reasoning=f"{x_name} causes {m_name}, which causes {y_name}. "
                  f"The correlation reflects genuine (mediated) causation."
    )


def create_confounding_scenario(
    scenario_num: int,
    variables: dict[str, str],
    n_samples: int = 100
) -> Scenario:
    """Create a confounding scenario."""
    x, y, z = generate_confounding(n_samples)

    x_name = variables["x"]
    y_name = variables["y"]
    z_name = variables["z"]

    return Scenario(
        id=f"confounding_{scenario_num}",
        structure="confounding",
        dag=f"{z_name} → {x_name}, {z_name} → {y_name}",
        variables=variables,
        data_summary={
            "n_samples": n_samples,
            f"corr_{x_name}_{y_name}": compute_correlation(x, y),
            f"corr_{z_name}_{x_name}": compute_correlation(z, x),
            f"corr_{z_name}_{y_name}": compute_correlation(z, y),
        },
        question=f"Given these correlations, does {x_name} cause {y_name}?",
        ground_truth=False,
        reasoning=f"{z_name} is a confounder that causes both {x_name} and {y_name}. "
                  f"The correlation between {x_name} and {y_name} is spurious."
    )


def create_reverse_scenario(
    scenario_num: int,
    variables: dict[str, str],
    n_samples: int = 100
) -> Scenario:
    """Create a reverse causation scenario."""
    x, y = generate_reverse_causation(n_samples)

    x_name = variables["x"]
    y_name = variables["y"]

    return Scenario(
        id=f"reverse_{scenario_num}",
        structure="reverse_causation",
        dag=f"{y_name} → {x_name}",
        variables=variables,
        data_summary={
            "n_samples": n_samples,
            f"corr_{x_name}_{y_name}": compute_correlation(x, y),
        },
        question=f"Given this correlation, does {x_name} cause {y_name}?",
        ground_truth=False,
        reasoning=f"Reverse causation: {y_name} actually causes {x_name}, "
                  f"not the other way around."
    )


def generate_all_scenarios(n_samples: int = 100) -> list[dict]:
    """
    Generate all 15 scenarios (5 per structure type).

    Args:
        n_samples: Number of data points per scenario

    Returns:
        List of scenario dictionaries
    """
    set_seed(RANDOM_SEED)
    scenarios = []

    # Direct causation scenarios
    for i, variables in enumerate(VARIABLE_SETS["direct"], 1):
        scenario = create_direct_scenario(i, variables, n_samples)
        scenarios.append(scenario)

    # Confounding scenarios
    for i, variables in enumerate(VARIABLE_SETS["confounding"], 1):
        scenario = create_confounding_scenario(i, variables, n_samples)
        scenarios.append(scenario)

    # Reverse causation scenarios
    for i, variables in enumerate(VARIABLE_SETS["reverse"], 1):
        scenario = create_reverse_scenario(i, variables, n_samples)
        scenarios.append(scenario)

    # Convert to dictionaries
    return [
        {
            "id": s.id,
            "structure": s.structure,
            "dag": s.dag,
            "variables": s.variables,
            "data_summary": s.data_summary,
            "question": s.question,
            "ground_truth": s.ground_truth,
            "reasoning": s.reasoning,
        }
        for s in scenarios
    ]


def save_scenarios(scenarios: list[dict], output_path: Path | str):
    """Save scenarios to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"Saved {len(scenarios)} scenarios to {output_path}")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def main():
    """Generate and save scenarios."""
    scenarios = generate_all_scenarios(n_samples=100)

    output_path = get_project_root() / "data" / "scenarios.json"
    save_scenarios(scenarios, output_path)

    # Print summary
    print("\nScenario Summary:")
    print("-" * 40)
    for structure in ["direct_causation", "confounding", "reverse_causation"]:
        count = sum(1 for s in scenarios if s["structure"] == structure)
        gt_true = sum(1 for s in scenarios if s["structure"] == structure and s["ground_truth"])
        print(f"  {structure}: {count} scenarios (ground_truth=True: {gt_true})")

    print("\nSample scenario:")
    print(json.dumps(scenarios[0], indent=2))


if __name__ == "__main__":
    main()
