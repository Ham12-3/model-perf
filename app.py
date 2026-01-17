"""
Scaling Impact Dashboard

A Streamlit app that compares model performance with different scaling approaches.
Demonstrates the impact of feature scaling on various ML algorithms using sklearn pipelines
to prevent data leakage during cross-validation.
"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    fetch_california_housing,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    cross_validate,
    KFold,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# -----------------------------------------------------------------------------
# Constants and Configuration
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
CV_FOLDS = 5

# Dataset definitions
DATASETS = {
    "Wine": {"loader": load_wine, "task": "classification"},
    "Breast Cancer": {"loader": load_breast_cancer, "task": "classification"},
    "California Housing": {"loader": fetch_california_housing, "task": "regression"},
    "Diabetes": {"loader": load_diabetes, "task": "regression"},
}

# Scaling options
SCALING_OPTIONS = {
    "Raw": None,
    "Normalised": MinMaxScaler(),
    "Standardised": StandardScaler(),
}

# Classification models with sensible hyperparameters
CLASSIFICATION_MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
    ),
    "Neural Net": MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

# Regression models with sensible hyperparameters
REGRESSION_MODELS = {
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "SVR RBF": SVR(kernel="rbf", C=1.0, gamma="scale"),
    "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Neural Net": MLPRegressor(
        hidden_layer_sizes=(50,),
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
    ),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


# -----------------------------------------------------------------------------
# Data Loading Functions (cached)
# -----------------------------------------------------------------------------


@st.cache_data
def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray, str, int, int]:
    """
    Load a dataset by name and return features, target, task type, and shape info.

    Returns:
        X: Feature matrix
        y: Target vector
        task: 'classification' or 'regression'
        n_samples: Number of samples
        n_features: Number of features
    """
    dataset_info = DATASETS[dataset_name]
    loader = dataset_info["loader"]
    task = dataset_info["task"]

    data = loader()
    X, y = data.data, data.target

    return X, y, task, X.shape[0], X.shape[1]


# -----------------------------------------------------------------------------
# Model Training and Evaluation Functions
# -----------------------------------------------------------------------------


def get_models_for_task(task: str) -> dict[str, Any]:
    """Return appropriate models based on task type."""
    if task == "classification":
        return CLASSIFICATION_MODELS
    return REGRESSION_MODELS


def create_pipeline(model: Any, scaler_name: str) -> Pipeline:
    """
    Create a sklearn Pipeline with optional scaling step.

    Using Pipeline ensures the scaler is fit only on training data within each
    CV fold, preventing data leakage.
    """
    # Clone the model to avoid state issues across runs
    from sklearn.base import clone
    model_clone = clone(model)

    scaler = SCALING_OPTIONS[scaler_name]

    if scaler is None:
        # No scaling - just the model
        return Pipeline([("model", model_clone)])
    else:
        # Clone scaler too to avoid state issues
        scaler_clone = clone(scaler)
        return Pipeline([("scaler", scaler_clone), ("model", model_clone)])


def get_cv_splitter(task: str):
    """Return appropriate cross-validation splitter for the task."""
    if task == "classification":
        return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def get_scoring_metrics(task: str) -> dict[str, str]:
    """Return scoring metrics for cross_validate based on task type."""
    if task == "classification":
        return {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
        }
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }


def run_single_experiment(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    dataset_name: str,
    model_name: str,
    model: Any,
    scaler_name: str,
) -> dict[str, Any]:
    """
    Run a single experiment with cross-validation and return results.

    Returns a dictionary with all metrics and timing information.
    """
    # Create pipeline
    pipeline = create_pipeline(model, scaler_name)

    # Get CV splitter and scoring metrics
    cv = get_cv_splitter(task)
    scoring = get_scoring_metrics(task)

    # Measure total wall clock time
    start_time = time.perf_counter()

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    end_time = time.perf_counter()
    total_run_time = end_time - start_time

    # Build result dictionary
    result = {
        "Task": task.capitalize(),
        "Dataset": dataset_name,
        "Model": model_name,
        "Scaling": scaler_name,
        "CV fit time (mean, s)": np.mean(cv_results["fit_time"]),
        "CV score time (mean, s)": np.mean(cv_results["score_time"]),
        "Total run time (s)": total_run_time,
    }

    # Add task-specific metrics
    if task == "classification":
        result["Accuracy (mean)"] = np.mean(cv_results["test_accuracy"])
        result["F1 macro (mean)"] = np.mean(cv_results["test_f1_macro"])
    else:
        # Convert negative scores to positive for RMSE and MAE
        result["RMSE (mean)"] = -np.mean(cv_results["test_rmse"])
        result["MAE (mean)"] = -np.mean(cv_results["test_mae"])
        result["R2 (mean)"] = np.mean(cv_results["test_r2"])

    return result


def run_all_experiments(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    dataset_name: str,
    selected_models: list[str],
    selected_scalers: list[str],
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run all selected model/scaler combinations and return results DataFrame.
    """
    all_models = get_models_for_task(task)
    results = []

    total_runs = len(selected_models) * len(selected_scalers)
    current_run = 0

    for model_name in selected_models:
        if model_name not in all_models:
            continue
        model = all_models[model_name]

        for scaler_name in selected_scalers:
            current_run += 1

            if progress_callback:
                progress_callback(
                    current_run / total_runs,
                    f"Running {model_name} with {scaler_name} scaling... ({current_run}/{total_runs})"
                )

            result = run_single_experiment(
                X, y, task, dataset_name, model_name, model, scaler_name
            )
            results.append(result)

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------


def create_comparison_chart(
    df: pd.DataFrame,
    metric: str,
    task: str,
) -> plt.Figure:
    """
    Create a bar chart comparing models across scaling options.

    Args:
        df: Results DataFrame
        metric: The metric column to plot
        task: Task type for labeling

    Returns:
        matplotlib Figure object
    """
    # Pivot data for grouped bar chart
    pivot_df = df.pivot(index="Model", columns="Scaling", values=metric)

    # Ensure consistent column order
    scaling_order = ["Raw", "Normalised", "Standardised"]
    available_scalings = [s for s in scaling_order if s in pivot_df.columns]
    pivot_df = pivot_df[available_scalings]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot grouped bars
    x = np.arange(len(pivot_df.index))
    width = 0.25
    multiplier = 0

    colors = {"Raw": "#FF6B6B", "Normalised": "#4ECDC4", "Standardised": "#45B7D1"}

    for scaling in available_scalings:
        offset = width * multiplier
        bars = ax.bar(
            x + offset,
            pivot_df[scaling],
            width,
            label=scaling,
            color=colors.get(scaling, "#888888"),
            edgecolor="white",
            linewidth=0.5,
        )
        multiplier += 1

    # Customize chart
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{metric} by Model and Scaling Method",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    ax.legend(title="Scaling", loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add some padding
    ax.margins(y=0.1)

    plt.tight_layout()

    return fig


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


def main():
    """Main Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="Scaling Impact Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("Scaling Impact Dashboard")
    st.markdown(
        """
        Compare how different feature scaling approaches affect model performance.
        Uses sklearn Pipelines to ensure scaling is fit only on training folds (no data leakage).
        """
    )

    # -------------------------------------------------------------------------
    # Sidebar Controls
    # -------------------------------------------------------------------------

    with st.sidebar:
        st.header("Configuration")

        # Dataset selection
        dataset_name = st.selectbox(
            "Select Dataset",
            options=list(DATASETS.keys()),
            help="Choose a dataset for the comparison experiment",
        )

        # Load dataset to determine task type
        X, y, task, n_samples, n_features = load_dataset(dataset_name)

        # Get available models for this task
        available_models = list(get_models_for_task(task).keys())

        # Model selection
        selected_models = st.multiselect(
            "Select Models",
            options=available_models,
            default=available_models,
            help="Choose which models to compare",
        )

        # Scaling selection
        selected_scalers = st.multiselect(
            "Select Scaling Options",
            options=list(SCALING_OPTIONS.keys()),
            default=list(SCALING_OPTIONS.keys()),
            help="Choose which scaling methods to compare",
        )

        st.divider()

        # Run button
        run_button = st.button(
            "Run Comparison",
            type="primary",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # Main Area - Dataset Info
    # -------------------------------------------------------------------------

    st.header("Dataset Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Task Type", task.capitalize())
    with col2:
        st.metric("Samples", f"{n_samples:,}")
    with col3:
        st.metric("Features", n_features)

    st.divider()

    # -------------------------------------------------------------------------
    # Validation and Experiment Execution
    # -------------------------------------------------------------------------

    if run_button:
        # Validate selections
        if not selected_models:
            st.warning("Please select at least one model to run the comparison.")
            st.stop()

        if not selected_scalers:
            st.warning("Please select at least one scaling option to run the comparison.")
            st.stop()

        # Run experiments with progress indicator
        st.header("Running Experiments...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress: float, message: str):
            progress_bar.progress(progress)
            status_text.text(message)

        # Run all experiments
        results_df = run_all_experiments(
            X, y, task, dataset_name,
            selected_models, selected_scalers,
            progress_callback=update_progress,
        )

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Store results in session state
        st.session_state["results"] = results_df
        st.session_state["task"] = task

        st.success(f"Completed {len(results_df)} experiments!")

    # -------------------------------------------------------------------------
    # Results Display
    # -------------------------------------------------------------------------

    if "results" in st.session_state and st.session_state["results"] is not None:
        results_df = st.session_state["results"]
        current_task = st.session_state["task"]

        st.header("Results")

        # Format numeric columns for display
        display_df = results_df.copy()
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if "time" in col.lower():
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

        # Show results table
        st.subheader("Comparison Table")
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # -------------------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------------------

        st.subheader("Performance Visualization")

        # Metric selection for chart
        if current_task == "classification":
            metric_options = ["Accuracy (mean)", "F1 macro (mean)"]
        else:
            metric_options = ["RMSE (mean)", "MAE (mean)", "R2 (mean)"]

        # Add timing metrics
        metric_options.extend([
            "CV fit time (mean, s)",
            "CV score time (mean, s)",
            "Total run time (s)",
        ])

        selected_metric = st.selectbox(
            "Select Metric for Chart",
            options=metric_options,
            help="Choose which metric to visualize in the bar chart",
        )

        # Create and display chart
        fig = create_comparison_chart(results_df, selected_metric, current_task)
        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------------------------------------------------
        # Insights
        # -------------------------------------------------------------------------

        st.divider()
        st.subheader("Key Insights")

        # Find best performing combinations
        if current_task == "classification":
            primary_metric = "Accuracy (mean)"
        else:
            primary_metric = "R2 (mean)"

        best_idx = results_df[primary_metric].idxmax()
        best_row = results_df.loc[best_idx]

        st.info(
            f"**Best {primary_metric}:** {best_row['Model']} with {best_row['Scaling']} scaling "
            f"({best_row[primary_metric]:.4f})"
        )

        # Compare scaling impact for scale-sensitive models
        scale_sensitive = ["KNN", "SVM RBF", "Logistic Regression", "Neural Net",
                          "KNN Regressor", "SVR RBF", "Ridge Regression"]

        sensitive_in_results = [m for m in scale_sensitive if m in results_df["Model"].values]

        if sensitive_in_results and len(results_df["Scaling"].unique()) > 1:
            st.markdown(
                """
                **Note:** Scale-sensitive models (KNN, SVM, Logistic Regression, Neural Networks, Ridge)
                typically show improved performance with scaling, while tree-based models
                (Decision Tree, Random Forest) are generally unaffected by feature scaling.
                """
            )

    else:
        # No results yet
        st.info(
            "Configure your experiment in the sidebar and click **Run Comparison** to see results."
        )


if __name__ == "__main__":
    main()
