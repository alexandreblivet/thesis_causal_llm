"""
Streamlit dashboard for thesis results: Can Small LLMs Distinguish Correlation from Causation?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binomtest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

STRUCTURE_COLORS = {
    "direct_causation": "#2ecc71",
    "confounding": "#e74c3c",
    "reverse_causation": "#9b59b6",
}
CONDITION_COLORS = {
    "neutral": "#3498db",
    "structure_given": "#f39c12",
    "experiment_stated": "#1abc9c",
}
EVAL_COLORS = {"correct": "#27ae60", "incorrect": "#c0392b"}

STRUCTURE_LABELS = {
    "direct_causation": "Direct Causation",
    "confounding": "Confounding",
    "reverse_causation": "Reverse Causation",
}
CONDITION_LABELS = {
    "neutral": "Neutral",
    "structure_given": "Structure Given",
    "experiment_stated": "Experiment Stated",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_results() -> pd.DataFrame:
    results_dir = PROJECT_ROOT / "data" / "results"
    files = sorted(results_dir.glob("results_*.csv"))
    if not files:
        st.error("No results CSV found in data/results/")
        st.stop()
    df = pd.read_csv(files[-1])
    # Normalise boolean-ish columns
    df["correct"] = df["correct"].astype(bool)
    df["ground_truth"] = df["ground_truth"].astype(bool)
    return df


@st.cache_data
def load_scenarios() -> list[dict]:
    with open(PROJECT_ROOT / "data" / "scenarios.json") as f:
        return json.load(f)


def format_correlations(data_summary: dict) -> str:
    lines = [f"n = {data_summary['n_samples']} observations"]
    for key, value in data_summary.items():
        if key.startswith("corr_"):
            raw = key[len("corr_"):]
            parts = raw.split("__")
            if len(parts) == 2:
                lines.append(f"corr({parts[0]}, {parts[1]}) = {value}")
    return "\n".join(lines)


def create_prompt(scenario: dict, prompt_condition: str) -> str:
    correlations = format_correlations(scenario["data_summary"])
    x_var = scenario["variables"]["x"]
    y_var = scenario["variables"]["y"]

    if prompt_condition == "neutral":
        return f"""You are given data about relationships between variables.

{correlations}

Question: Does {x_var} cause {y_var}?

Answer Yes or No, then explain briefly."""

    elif prompt_condition == "structure_given":
        dag = scenario["dag"]
        return f"""You are given data about relationships between variables.

The true causal structure is: {dag}

{correlations}

Question: Does {x_var} cause {y_var}?

Answer Yes or No, then explain briefly."""

    elif prompt_condition == "experiment_stated":
        return f"""You are given data from a randomized controlled experiment where {x_var} was randomly assigned.

{correlations}

Question: Does {x_var} cause {y_var}?

Answer Yes or No, then explain briefly."""

    return ""


# ---------------------------------------------------------------------------
# Page functions
# ---------------------------------------------------------------------------


def page_overview():
    st.title("Can Small LLMs Distinguish Correlation from Causation?")
    st.markdown("**MSc Thesis — Causal Reasoning Evaluation in Marketing Contexts**")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenarios", "30")
    c2.metric("Models", "6")
    c3.metric("Test Cases / Model", "80")
    c4.metric("Prompt Conditions", "3")

    st.header("Experimental Design")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Causal Structures")
        st.markdown(
            """
| Structure | DAG | Ground Truth |
|---|---|---|
| **Direct** | X → M → Y | Yes |
| **Confounding** | Z → X, Z → Y | No |
| **Reverse** | Y → X | No |
"""
        )

    with col2:
        st.subheader("Prompt Conditions")
        st.markdown(
            """
- **Neutral** — presents correlation data only
- **Structure Given** — provides the true DAG
- **Experiment Stated** — frames data as from an RCT
"""
        )

    with col3:
        st.subheader("Variable Types")
        st.markdown(
            """
- **Marketing** — domain names (ad_spend, sales, …)
- **Abstract** — generic names (A, B, C / X1, Y1, …)
"""
        )

    st.header("Prompt Examples")
    scenarios = load_scenarios()
    sample = scenarios[0]  # direct_1

    for cond in ["neutral", "structure_given", "experiment_stated"]:
        with st.expander(f"{CONDITION_LABELS[cond]} prompt"):
            st.code(create_prompt(sample, cond), language=None)

    st.header("Methodology")
    st.markdown(
        """
- **Temperature**: 0.0 (deterministic)
- **Zero-shot** prompting with correlation statistics
- **Binary evaluation**: correct / incorrect against known ground truth
- **Exclusion**: experiment_stated × confounding combinations excluded (RCTs eliminate confounding by design)
"""
    )


# ---------------------------------------------------------------------------


def page_results(df: pd.DataFrame):
    st.title("Results Overview")

    # --- Headline metrics ---
    overall_acc = df["correct"].mean()
    model_acc = df.groupby("model_name")["correct"].mean()
    best_model = model_acc.idxmax()
    struct_acc = df.groupby("structure")["correct"].mean()
    hardest = struct_acc.idxmin()

    yes_pct = (df["predicted_answer"] == "yes").mean()
    answer_bias = f"{yes_pct:.0%} Yes"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{overall_acc:.1%}")
    c2.metric("Best Model", best_model, f"{model_acc[best_model]:.1%}")
    c3.metric("Hardest Structure", STRUCTURE_LABELS.get(hardest, hardest))
    c4.metric("Answer Bias", answer_bias)

    st.divider()

    # --- Model accuracy bar chart ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Accuracy by Model")
        ma = model_acc.reset_index()
        ma.columns = ["Model", "Accuracy"]
        ma = ma.sort_values("Accuracy")
        fig = px.bar(
            ma,
            y="Model",
            x="Accuracy",
            orientation="h",
            text=ma["Accuracy"].apply(lambda v: f"{v:.1%}"),
            range_x=[0, 1],
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50% chance")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Accuracy by Structure")
        sa = struct_acc.reset_index()
        sa.columns = ["Structure", "Accuracy"]
        sa["Label"] = sa["Structure"].map(STRUCTURE_LABELS)
        sa["Color"] = sa["Structure"].map(STRUCTURE_COLORS)
        fig = px.bar(
            sa,
            y="Label",
            x="Accuracy",
            orientation="h",
            text=sa["Accuracy"].apply(lambda v: f"{v:.1%}"),
            color="Label",
            color_discrete_map={v: STRUCTURE_COLORS[k] for k, v in STRUCTURE_LABELS.items()},
            range_x=[0, 1],
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Factorial heatmap per model ---
    st.subheader("Factorial Heatmap: Structure × Prompt Condition")
    models = sorted(df["model_name"].unique())
    tabs = st.tabs(models)

    for tab, model in zip(tabs, models):
        with tab:
            subset = df[df["model_name"] == model]
            pivot = subset.pivot_table(
                index="structure",
                columns="prompt_condition",
                values="correct",
                aggfunc="mean",
            )
            counts = subset.pivot_table(
                index="structure",
                columns="prompt_condition",
                values="correct",
                aggfunc="count",
            )

            # Reorder
            struct_order = ["direct_causation", "confounding", "reverse_causation"]
            cond_order = ["neutral", "structure_given", "experiment_stated"]
            pivot = pivot.reindex(index=struct_order, columns=cond_order)
            counts = counts.reindex(index=struct_order, columns=cond_order)

            annotations = []
            for i, s in enumerate(struct_order):
                for j, c in enumerate(cond_order):
                    val = pivot.loc[s, c] if pd.notna(pivot.loc[s, c]) else None
                    cnt = counts.loc[s, c] if pd.notna(counts.loc[s, c]) else None
                    if val is not None:
                        annotations.append(f"{val:.0%}<br>n={int(cnt)}")
                    else:
                        annotations.append("N/A")

            text_matrix = np.array(annotations).reshape(len(struct_order), len(cond_order))

            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=[CONDITION_LABELS.get(c, c) for c in cond_order],
                    y=[STRUCTURE_LABELS.get(s, s) for s in struct_order],
                    text=text_matrix,
                    texttemplate="%{text}",
                    colorscale="RdYlGn",
                    zmin=0,
                    zmax=1,
                    colorbar_title="Accuracy",
                )
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # --- Answer distribution ---
    st.subheader("Answer Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Predicted answers**")
        pred_counts = df["predicted_answer"].value_counts().reset_index()
        pred_counts.columns = ["Answer", "Count"]
        fig = px.bar(pred_counts, x="Answer", y="Count", color="Answer")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Expected (ground truth)**")
        gt_counts = df["ground_truth"].value_counts().reset_index()
        gt_counts.columns = ["Ground Truth", "Count"]
        gt_counts["Ground Truth"] = gt_counts["Ground Truth"].map({True: "yes", False: "no"})
        fig = px.bar(gt_counts, x="Ground Truth", y="Count", color="Ground Truth")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Accuracy by variable type ---
    st.subheader("Accuracy by Variable Type")
    vt_acc = df.groupby("variable_type")["correct"].mean().reset_index()
    vt_acc.columns = ["Variable Type", "Accuracy"]
    fig = px.bar(
        vt_acc,
        x="Variable Type",
        y="Accuracy",
        text=vt_acc["Accuracy"].apply(lambda v: f"{v:.1%}"),
        color="Variable Type",
        range_y=[0, 1],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------


def page_hypotheses(df: pd.DataFrame):
    st.title("Hypothesis Testing")

    # Build hypothesis comparison table
    conditions = ["neutral", "structure_given", "experiment_stated"]
    structures = ["direct_causation", "confounding", "reverse_causation"]

    predictions = {
        ("neutral", "direct_causation"): "Mostly No",
        ("neutral", "confounding"): "Mostly No",
        ("neutral", "reverse_causation"): "Mostly No",
        ("structure_given", "direct_causation"): "More Yes",
        ("structure_given", "confounding"): "Still No",
        ("structure_given", "reverse_causation"): "Still No",
        ("experiment_stated", "direct_causation"): "Mostly Yes",
        ("experiment_stated", "confounding"): "N/A",
        ("experiment_stated", "reverse_causation"): "Mostly Yes",
    }

    rows = []
    for cond in conditions:
        row = {"Condition": CONDITION_LABELS[cond]}
        for struct in structures:
            subset = df[(df["prompt_condition"] == cond) & (df["structure"] == struct)]
            pred = predictions[(cond, struct)]
            if len(subset) == 0:
                row[STRUCTURE_LABELS[struct]] = "N/A"
                continue
            yes_rate = (subset["predicted_answer"] == "yes").mean()
            actual = f"{yes_rate:.0%} Yes"
            row[STRUCTURE_LABELS[struct]] = f"{actual} (pred: {pred})"
        rows.append(row)

    st.subheader("Hypothesis Table")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Per-condition breakdown ---
    st.subheader("Neutral Condition — Skepticism Bias")
    st.markdown("**Prediction**: Models default to 'No' across all structures (skepticism bias).")
    neutral = df[df["prompt_condition"] == "neutral"]
    for struct in structures:
        sub = neutral[neutral["structure"] == struct]
        if len(sub) == 0:
            continue
        yes_rate = (sub["predicted_answer"] == "yes").mean()
        acc = sub["correct"].mean()
        label = STRUCTURE_LABELS[struct]
        st.markdown(f"- **{label}**: {yes_rate:.0%} answered Yes, accuracy {acc:.0%}")

    st.subheader("Structure Given — Causal Reasoning with DAG")
    st.markdown("**Prediction**: Providing the DAG increases 'Yes' for direct causation.")
    sg = df[df["prompt_condition"] == "structure_given"]
    neutral_direct_yes = (
        neutral[neutral["structure"] == "direct_causation"]["predicted_answer"] == "yes"
    ).mean()
    sg_direct_yes = (
        sg[sg["structure"] == "direct_causation"]["predicted_answer"] == "yes"
    ).mean()
    delta = sg_direct_yes - neutral_direct_yes
    if delta > 0:
        st.success(f"Yes-rate for direct causation: {neutral_direct_yes:.0%} (neutral) → {sg_direct_yes:.0%} (structure given). +{delta:.0%}")
    else:
        st.error(f"Yes-rate for direct causation: {neutral_direct_yes:.0%} (neutral) → {sg_direct_yes:.0%} (structure given). {delta:+.0%}")

    st.subheader("Experiment Stated — RCT Framing")
    st.markdown("**Prediction**: RCT framing increases 'Yes' for direct and reverse causation.")
    es = df[df["prompt_condition"] == "experiment_stated"]
    for struct in ["direct_causation", "reverse_causation"]:
        sub = es[es["structure"] == struct]
        if len(sub) == 0:
            continue
        yes_rate = (sub["predicted_answer"] == "yes").mean()
        label = STRUCTURE_LABELS[struct]
        st.markdown(f"- **{label}**: {yes_rate:.0%} answered Yes")

    # --- Binomial tests ---
    st.subheader("Statistical Tests: Binomial Test vs 50% Chance")
    st.markdown("Per-model accuracy tested against chance (H0: accuracy = 50%).")

    models = sorted(df["model_name"].unique())
    test_rows = []
    for model in models:
        sub = df[df["model_name"] == model]
        n = len(sub)
        k = sub["correct"].sum()
        result = binomtest(int(k), n, 0.5)
        test_rows.append(
            {
                "Model": model,
                "n": n,
                "Correct": int(k),
                "Accuracy": f"{k / n:.1%}",
                "p-value": f"{result.pvalue:.4f}",
                "Significant (α=0.05)": "Yes" if result.pvalue < 0.05 else "No",
            }
        )

    st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)

    # --- Key findings ---
    st.subheader("Key Findings")
    overall = df["correct"].mean()
    if overall > 0.5:
        st.success(f"Overall accuracy ({overall:.1%}) is above chance (50%).")
    else:
        st.error(f"Overall accuracy ({overall:.1%}) is at or below chance (50%).")

    # Check skepticism bias
    neutral_yes = (neutral["predicted_answer"] == "yes").mean()
    if neutral_yes < 0.3:
        st.success(f"Skepticism bias confirmed: only {neutral_yes:.0%} 'Yes' answers in neutral condition.")
    else:
        st.error(f"Skepticism bias NOT confirmed: {neutral_yes:.0%} 'Yes' answers in neutral condition.")


# ---------------------------------------------------------------------------


def page_drilldown(df: pd.DataFrame, scenarios: list[dict]):
    st.title("Drill-Down Explorer")

    scenarios_map = {s["id"]: s for s in scenarios}

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        sel_model = st.selectbox("Model", ["All"] + sorted(df["model_name"].unique()))
        sel_struct = st.selectbox("Structure", ["All"] + sorted(df["structure"].unique()))
        sel_cond = st.selectbox("Prompt Condition", ["All"] + sorted(df["prompt_condition"].unique()))
    with col2:
        sel_vt = st.selectbox("Variable Type", ["All"] + sorted(df["variable_type"].unique()))
        sel_scenario = st.selectbox("Scenario ID", ["All"] + sorted(df["scenario_id"].unique()))
        show_errors = st.checkbox("Show only errors")

    filtered = df.copy()
    if sel_model != "All":
        filtered = filtered[filtered["model_name"] == sel_model]
    if sel_struct != "All":
        filtered = filtered[filtered["structure"] == sel_struct]
    if sel_cond != "All":
        filtered = filtered[filtered["prompt_condition"] == sel_cond]
    if sel_vt != "All":
        filtered = filtered[filtered["variable_type"] == sel_vt]
    if sel_scenario != "All":
        filtered = filtered[filtered["scenario_id"] == sel_scenario]
    if show_errors:
        filtered = filtered[~filtered["correct"]]

    st.markdown(f"**{len(filtered)} test cases**")

    # Comparison mode
    compare = st.toggle("Comparison mode (same scenario across models)")

    if compare and sel_scenario != "All":
        comp = df[df["scenario_id"] == sel_scenario]
        comp_display = comp[["model_name", "prompt_condition", "predicted_answer", "ground_truth", "correct"]].copy()
        comp_display["correct"] = comp_display["correct"].map({True: "✓", False: "✗"})
        st.dataframe(comp_display, use_container_width=True, hide_index=True)

    # Test case cards
    for _, row in filtered.head(20).iterrows():
        scenario_meta = scenarios_map.get(row["scenario_id"], {})
        correct = row["correct"]
        banner_color = EVAL_COLORS["correct"] if correct else EVAL_COLORS["incorrect"]
        banner_text = "CORRECT" if correct else "INCORRECT"

        with st.expander(
            f"{'✓' if correct else '✗'} {row['scenario_id']} | {row['model_name']} | {row['prompt_condition']}"
        ):
            st.markdown(
                f"<div style='background-color:{banner_color};color:white;padding:8px;border-radius:4px;'>"
                f"<b>{banner_text}</b> — Predicted: {row['predicted_answer']} | Ground truth: {row['ground_truth']}</div>",
                unsafe_allow_html=True,
            )

            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"**ID**: {row['scenario_id']}")
                st.markdown(f"**Structure**: {row['structure']}")
                st.markdown(f"**DAG**: {row['dag']}")
            with mc2:
                st.markdown(f"**Variable Type**: {row['variable_type']}")
                st.markdown(f"**Ground Truth**: {row['ground_truth']}")

            if scenario_meta and "data_summary" in scenario_meta:
                st.markdown("**Correlation Data**")
                corr_data = {
                    k: v
                    for k, v in scenario_meta["data_summary"].items()
                    if k.startswith("corr_")
                }
                if corr_data:
                    st.dataframe(
                        pd.DataFrame(
                            [{"Pair": k.replace("corr_", "").replace("__", " ↔ "), "r": v} for k, v in corr_data.items()]
                        ),
                        hide_index=True,
                    )

            st.markdown("**Prompt**")
            st.code(row["prompt"], language=None)

            st.markdown("**Model Response**")
            st.text(row["response"][:2000] if isinstance(row["response"], str) else "")

    if len(filtered) > 20:
        st.info(f"Showing first 20 of {len(filtered)} results. Use filters to narrow down.")


# ---------------------------------------------------------------------------


def page_confusion(df: pd.DataFrame):
    st.title("Confusion Matrix & Error Analysis")

    # Map to binary labels
    df_cm = df.copy()
    df_cm["pred_binary"] = df_cm["predicted_answer"].apply(
        lambda x: "Yes" if x == "yes" else "No"
    )
    df_cm["truth_binary"] = df_cm["ground_truth"].map({True: "Yes", False: "No"})

    dim = st.selectbox(
        "Break down by",
        ["model_name", "structure", "prompt_condition", "variable_type"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # --- Overall confusion matrix ---
    st.subheader("Overall Confusion Matrix")
    labels = ["Yes", "No"]
    cm = pd.crosstab(df_cm["pred_binary"], df_cm["truth_binary"])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm.values,
            x=[f"True {l}" for l in labels],
            y=[f"Pred {l}" for l in labels],
            text=cm.values,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(height=300, width=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)

    # --- Breakdown matrices ---
    st.subheader(f"Confusion Matrices by {dim.replace('_', ' ').title()}")
    values = sorted(df_cm[dim].unique())
    tabs = st.tabs([str(v) for v in values])

    for tab, val in zip(tabs, values):
        with tab:
            sub = df_cm[df_cm[dim] == val]
            cm_sub = pd.crosstab(sub["pred_binary"], sub["truth_binary"])
            cm_sub = cm_sub.reindex(index=labels, columns=labels, fill_value=0)

            fig = go.Figure(
                data=go.Heatmap(
                    z=cm_sub.values,
                    x=[f"True {l}" for l in labels],
                    y=[f"Pred {l}" for l in labels],
                    text=cm_sub.values,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig.update_layout(height=300, width=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig)

    # --- Error table ---
    st.subheader("Error Table")
    errors = df_cm[~df_cm["correct"]].copy()
    errors["error_type"] = errors.apply(
        lambda r: "False Positive" if r["predicted_answer"] == "yes" else "False Negative",
        axis=1,
    )
    error_display = errors[
        ["scenario_id", "model_name", "structure", "prompt_condition", "variable_type",
         "predicted_answer", "ground_truth", "error_type"]
    ]
    st.dataframe(error_display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download error table as CSV",
        error_display.to_csv(index=False),
        "errors.csv",
        "text/csv",
    )

    # --- Precision / Recall / F1 ---
    st.subheader("Precision / Recall / F1 per Model (Yes = positive class)")
    models = sorted(df_cm["model_name"].unique())
    prf_rows = []
    for model in models:
        sub = df_cm[df_cm["model_name"] == model]
        tp = ((sub["pred_binary"] == "Yes") & (sub["truth_binary"] == "Yes")).sum()
        fp = ((sub["pred_binary"] == "Yes") & (sub["truth_binary"] == "No")).sum()
        fn = ((sub["pred_binary"] == "No") & (sub["truth_binary"] == "Yes")).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        prf_rows.append(
            {
                "Model": model,
                "Precision": f"{precision:.2f}",
                "Recall": f"{recall:.2f}",
                "F1": f"{f1:.2f}",
            }
        )
    st.dataframe(pd.DataFrame(prf_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Causal LLM Thesis Dashboard",
        page_icon="📊",
        layout="wide",
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Thesis Overview",
            "Results Overview",
            "Hypothesis Testing",
            "Drill-Down Explorer",
            "Confusion Matrix & Errors",
        ],
    )

    if page == "Thesis Overview":
        page_overview()
    else:
        df = load_results()
        scenarios = load_scenarios()

        if page == "Results Overview":
            page_results(df)
        elif page == "Hypothesis Testing":
            page_hypotheses(df)
        elif page == "Drill-Down Explorer":
            page_drilldown(df, scenarios)
        elif page == "Confusion Matrix & Errors":
            page_confusion(df)


if __name__ == "__main__":
    main()
