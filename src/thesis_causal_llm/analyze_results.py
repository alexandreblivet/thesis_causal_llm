"""
Analyze experiment results and generate publication-ready plots.
Saves plots to images/ and prints statistical summaries to console.

Usage:
    uv run python -m thesis_causal_llm.analyze_results
    uv run python -m thesis_causal_llm.analyze_results --results-file data/results/results_TIMESTAMP.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def find_latest_results(results_dir: Path) -> Path:
    """Find the most recent results CSV file."""
    csvs = sorted(results_dir.glob("results_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    return csvs[-1]


def load_results(results_file: Path | None = None) -> pd.DataFrame:
    """Load results CSV into DataFrame."""
    if results_file is None:
        results_dir = get_project_root() / "data" / "results"
        results_file = find_latest_results(results_dir)

    print(f"Loading results from: {results_file}")
    df = pd.read_csv(results_file)

    # Ensure correct types
    df["correct"] = df["correct"].astype(bool)
    df["ground_truth"] = df["ground_truth"].astype(bool)

    return df


def setup_style():
    """Set up matplotlib/seaborn styling."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def short_model_name(name: str) -> str:
    """Shorten model IDs for display."""
    mapping = {
        "claude-haiku-4-5-20251001": "Haiku 4.5",
        "claude-sonnet-4-5-20250929": "Sonnet 4.5",
        "claude-sonnet-4-6": "Sonnet 4.6",
        "claude-opus-4-6": "Opus 4.6",
        # Legacy names from older runs
        "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
        "google/gemma-2-9b-it": "Gemma 2 9B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
        "claude-haiku": "Haiku",
        "claude-sonnet": "Sonnet",
        "claude-opus": "Opus",
        "claude-opus-4-5-20251101": "Opus 4.5",
    }
    return mapping.get(name, name)


def plot_accuracy_by_model(df: pd.DataFrame, output_dir: Path):
    """Bar chart of accuracy per model with 50% chance line."""
    acc = df.groupby("model_name")["correct"].mean().reset_index()
    acc["model_short"] = acc["model_name"].map(short_model_name)
    acc = acc.sort_values("correct", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(acc["model_short"], acc["correct"], color=sns.color_palette("viridis", len(acc)))
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="50% chance")
    ax.set_xlabel("Accuracy")
    ax.set_title("Overall Accuracy by Model")
    ax.set_xlim(0, 1)
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, acc["correct"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)

    fig.savefig(output_dir / "accuracy_by_model.png")
    plt.close(fig)
    print("  Saved accuracy_by_model.png")


def plot_accuracy_by_structure(df: pd.DataFrame, output_dir: Path):
    """Bar chart by causal structure."""
    structure_colors = {
        "direct_causation": "#2ecc71",
        "confounding": "#e74c3c",
        "reverse_causation": "#9b59b6",
    }
    acc = df.groupby("structure")["correct"].mean().reset_index()
    acc = acc.sort_values("structure")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [structure_colors.get(s, "#95a5a6") for s in acc["structure"]]
    bars = ax.bar(acc["structure"], acc["correct"], color=colors)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="50% chance")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Causal Structure")
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bar, val in zip(bars, acc["correct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.1%}", ha="center", fontsize=10)

    fig.savefig(output_dir / "accuracy_by_structure.png")
    plt.close(fig)
    print("  Saved accuracy_by_structure.png")


def plot_accuracy_by_condition(df: pd.DataFrame, output_dir: Path):
    """Bar chart by prompt condition."""
    acc = df.groupby("prompt_condition")["correct"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(acc["prompt_condition"], acc["correct"],
                  color=sns.color_palette("Set2", len(acc)))
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="50% chance")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Prompt Condition")
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bar, val in zip(bars, acc["correct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.1%}", ha="center", fontsize=10)

    fig.savefig(output_dir / "accuracy_by_condition.png")
    plt.close(fig)
    print("  Saved accuracy_by_condition.png")


def plot_accuracy_by_variable_type(df: pd.DataFrame, output_dir: Path):
    """Marketing vs abstract comparison."""
    acc = df.groupby(["model_name", "variable_type"])["correct"].mean().reset_index()
    acc["model_short"] = acc["model_name"].map(short_model_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = acc.pivot(index="model_short", columns="variable_type", values="correct")
    pivot.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22"])
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="50% chance")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_title("Accuracy by Variable Type")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Variable Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.savefig(output_dir / "accuracy_by_variable_type.png")
    plt.close(fig)
    print("  Saved accuracy_by_variable_type.png")


def plot_factorial_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap grid: one subplot per model, structure x condition."""
    models = sorted(df["model_name"].unique())
    n_models = len(models)
    cols = min(n_models, 2)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // cols][idx % cols]
        sub = df[df["model_name"] == model]
        pivot = sub.pivot_table(
            index="structure", columns="prompt_condition",
            values="correct", aggfunc="mean"
        )
        sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax, cbar=idx == 0)
        ax.set_title(short_model_name(model))
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Accuracy: Structure x Prompt Condition (per model)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "factorial_heatmap.png")
    plt.close(fig)
    print("  Saved factorial_heatmap.png")


def plot_confusion_matrices(df: pd.DataFrame, output_dir: Path):
    """Overall + per-model confusion matrices."""
    models = sorted(df["model_name"].unique())
    n_plots = 1 + len(models)
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    def _cm(sub, ax, title):
        """Plot a single confusion matrix."""
        pred_yes = sub["predicted_answer"] == "yes"
        truth_yes = sub["ground_truth"]

        cm = np.array([
            [(~pred_yes & ~truth_yes).sum(), (pred_yes & ~truth_yes).sum()],
            [(~pred_yes & truth_yes).sum(), (pred_yes & truth_yes).sum()],
        ])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred No", "Pred Yes"],
                    yticklabels=["True No", "True Yes"],
                    cbar=False)
        ax.set_title(title)

    # Overall
    _cm(df, axes[0][0], "Overall")

    # Per model
    for i, model in enumerate(models, start=1):
        row, col = i // cols, i % cols
        _cm(df[df["model_name"] == model], axes[row][col], short_model_name(model))

    # Hide empty
    for idx in range(n_plots, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png")
    plt.close(fig)
    print("  Saved confusion_matrices.png")


def print_statistics(df: pd.DataFrame):
    """Print comprehensive statistics to console."""
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    # Overall accuracy
    overall_acc = df["correct"].mean()
    n_correct = df["correct"].sum()
    n_total = len(df)
    print(f"\nOverall Accuracy: {overall_acc:.1%} ({n_correct}/{n_total})")

    # Binomial test vs 50%
    binom_result = stats.binomtest(n_correct, n_total, 0.5)
    print(f"Binomial test vs 50%: p = {binom_result.pvalue:.4f} {'***' if binom_result.pvalue < 0.001 else '**' if binom_result.pvalue < 0.01 else '*' if binom_result.pvalue < 0.05 else 'ns'}")

    # Per-model breakdown
    print("\n--- Accuracy by Model ---")
    print(f"{'Model':<30} {'Accuracy':>10} {'n':>6} {'p vs 50%':>10}")
    print("-" * 60)
    for model in sorted(df["model_name"].unique()):
        sub = df[df["model_name"] == model]
        acc = sub["correct"].mean()
        n = len(sub)
        nc = sub["correct"].sum()
        p = stats.binomtest(nc, n, 0.5).pvalue
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {short_model_name(model):<28} {acc:>9.1%} {n:>6} {p:>8.4f} {sig}")

    # Per-structure
    print("\n--- Accuracy by Causal Structure ---")
    print(f"{'Structure':<25} {'Accuracy':>10} {'n':>6}")
    print("-" * 45)
    for struct in sorted(df["structure"].unique()):
        sub = df[df["structure"] == struct]
        print(f"  {struct:<23} {sub['correct'].mean():>9.1%} {len(sub):>6}")

    # Per-condition
    print("\n--- Accuracy by Prompt Condition ---")
    print(f"{'Condition':<25} {'Accuracy':>10} {'n':>6}")
    print("-" * 45)
    for cond in sorted(df["prompt_condition"].unique()):
        sub = df[df["prompt_condition"] == cond]
        print(f"  {cond:<23} {sub['correct'].mean():>9.1%} {len(sub):>6}")

    # Per-variable type
    print("\n--- Accuracy by Variable Type ---")
    print(f"{'Type':<25} {'Accuracy':>10} {'n':>6}")
    print("-" * 45)
    for vt in sorted(df["variable_type"].unique()):
        sub = df[df["variable_type"] == vt]
        print(f"  {vt:<23} {sub['correct'].mean():>9.1%} {len(sub):>6}")

    # Precision / Recall / F1 per model
    print("\n--- Precision / Recall / F1 per Model ---")
    print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    for model in sorted(df["model_name"].unique()):
        sub = df[df["model_name"] == model]
        pred_yes = sub["predicted_answer"] == "yes"
        truth_yes = sub["ground_truth"]

        tp = (pred_yes & truth_yes).sum()
        fp = (pred_yes & ~truth_yes).sum()
        fn = (~pred_yes & truth_yes).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  {short_model_name(model):<28} {precision:>9.1%} {recall:>9.1%} {f1:>9.3f}")

    # Error analysis
    print("\n--- Error Analysis ---")
    errors = df[~df["correct"]]
    if len(errors) > 0:
        print(f"Total errors: {len(errors)}/{n_total} ({len(errors)/n_total:.1%})")
        print("\nErrors by structure x condition:")
        error_ct = errors.groupby(["structure", "prompt_condition"]).size().reset_index(name="count")
        total_ct = df.groupby(["structure", "prompt_condition"]).size().reset_index(name="total")
        merged = error_ct.merge(total_ct, on=["structure", "prompt_condition"])
        merged["error_rate"] = merged["count"] / merged["total"]
        merged = merged.sort_values("error_rate", ascending=False)
        print(f"  {'Structure':<22} {'Condition':<20} {'Errors':>8} {'Rate':>8}")
        print("  " + "-" * 60)
        for _, row in merged.iterrows():
            print(f"  {row['structure']:<22} {row['prompt_condition']:<20} {row['count']:>5}/{row['total']:<3} {row['error_rate']:>7.1%}")

    # Key findings
    print("\n--- Key Findings ---")
    # Best/worst model
    model_acc = df.groupby("model_name")["correct"].mean()
    best = model_acc.idxmax()
    worst = model_acc.idxmin()
    print(f"  Best model:  {short_model_name(best)} ({model_acc[best]:.1%})")
    print(f"  Worst model: {short_model_name(worst)} ({model_acc[worst]:.1%})")

    # Direct causation accuracy (the hard case)
    direct = df[df["structure"] == "direct_causation"]
    if len(direct) > 0:
        direct_acc = direct.groupby("model_name")["correct"].mean()
        print(f"\n  Direct causation accuracy (the key challenge):")
        for model in sorted(direct_acc.index):
            print(f"    {short_model_name(model):<28} {direct_acc[model]:.1%}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and generate plots"
    )
    parser.add_argument(
        "--results-file", "-r",
        type=Path,
        default=None,
        help="Path to results CSV (default: latest in data/results/)"
    )
    args = parser.parse_args()

    # Load data
    df = load_results(args.results_file)
    print(f"Loaded {len(df)} results across {df['model_name'].nunique()} models")

    # Setup
    setup_style()
    output_dir = get_project_root() / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_by_model(df, output_dir)
    plot_accuracy_by_structure(df, output_dir)
    plot_accuracy_by_condition(df, output_dir)
    plot_accuracy_by_variable_type(df, output_dir)
    plot_factorial_heatmap(df, output_dir)
    plot_confusion_matrices(df, output_dir)

    # Print statistics
    print_statistics(df)


if __name__ == "__main__":
    main()
