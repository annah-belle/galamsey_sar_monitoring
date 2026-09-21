"""
Train and save Sentinel-1 SAR classification models.

This script is based on the supplied thesis training workflow.

Expected input CSV columns
--------------------------
- plot_id
- Class
- VV
- VH

The script derives:
- VV-VH difference
- VV/VH ratio
- Normalized Polarisation Index (NPI)

Models:
- Logistic Regression
- Random Forest
- Support Vector Classifier
- ExtraTrees

Validation:
- Plot-based held-out split
- GroupKFold for grouped hyperparameter tuning

Important implementation note
-----------------------------
The supplied thesis script creates a held-out test split, but the
original code does not calculate held-out test metrics. This cleaned
version preserves that behavior and does not claim held-out test
performance that the code does not produce.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LOGGER = logging.getLogger(__name__)

FEATURES = [
    "VV",
    "VH",
    "VV_VH_diff",
    "VV_VH_ratio",
    "NPI",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Sentinel-1 galamsey classification models."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV containing plot_id, Class, VV and VH columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for trained models and training summary.",
    )
    parser.add_argument(
        "--test-plots",
        nargs="+",
        default=["1", "5"],
        help="Plot IDs reserved as the held-out spatial test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used where supported by the original workflow.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of RandomizedSearchCV parameter settings per model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_dataset(input_csv: Path) -> pd.DataFrame:
    """Load and validate the training CSV."""
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    data = pd.read_csv(input_csv)

    # Preserve the supplied script's support for alternative band names.
    rename_map = {}
    if "BAND_1" in data.columns and "VV" not in data.columns:
        rename_map["BAND_1"] = "VV"
    if "BAND_2" in data.columns and "VH" not in data.columns:
        rename_map["BAND_2"] = "VH"

    if rename_map:
        data = data.rename(columns=rename_map)

    required = {"plot_id", "Class", "VV", "VH"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if data.empty:
        raise ValueError("The input CSV contains no rows.")

    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the Sentinel-1-derived features used by the classifiers."""
    result = data.copy()

    vv = pd.to_numeric(result["VV"], errors="coerce")
    vh = pd.to_numeric(result["VH"], errors="coerce")

    result["VV_VH_diff"] = vv - vh

    # The supplied workflow treats VV/VH as dB values and converts them
    # to linear scale for ratio and NPI calculations.
    vv_linear = 10 ** (vv / 10.0)
    vh_linear = 10 ** (vh / 10.0)

    epsilon = 1e-12
    result["VV_VH_ratio"] = vv_linear / (vh_linear + epsilon)
    result["NPI"] = (
        (vv_linear - vh_linear)
        / (vv_linear + vh_linear + epsilon)
    )

    return result.replace([np.inf, -np.inf], np.nan)


def split_by_plot(
    data: pd.DataFrame,
    test_plots: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate training and held-out observations by plot ID."""
    plot_ids = data["plot_id"].astype(str)
    test_set = set(str(plot) for plot in test_plots)

    test_mask = plot_ids.isin(test_set)

    if not test_mask.any():
        raise ValueError(
            f"None of the requested test plots were found: {sorted(test_set)}"
        )

    train_data = data.loc[~test_mask].copy()
    test_data = data.loc[test_mask].copy()

    if train_data.empty:
        raise ValueError("No training observations remain after plot split.")

    return train_data, test_data


def make_models(random_state: int) -> dict[str, Any]:
    """Create the four classifier pipelines."""
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVC": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def parameter_distributions(random_state: int) -> dict[str, dict[str, list[Any]]]:
    """
    Return the supplied workflow's hyperparameter search spaces.

    These are kept in one place so the search configuration is visible
    and reproducible.
    """
    return {
        "LogisticRegression": {
            "model__C": np.logspace(-3, 3, 20).tolist(),
            "model__solver": ["liblinear", "lbfgs"],
        },
        "RandomForest": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        "SVC": {
            "model__C": np.logspace(-2, 2, 20).tolist(),
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "linear"],
        },
        "ExtraTrees": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
    }


def fit_models(
    train_data: pd.DataFrame,
    models: dict[str, Any],
    search_spaces: dict[str, dict[str, list[Any]]],
    n_iter: int,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune and fit the four classifiers using grouped cross-validation."""
    X = train_data[FEATURES]
    y = train_data["Class"]
    groups = train_data["plot_id"].astype(str)

    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(
            "At least two unique training plots are required for GroupKFold."
        )

    n_splits = min(3, n_groups)
    cv = GroupKFold(n_splits=n_splits)

    fitted_models: dict[str, Any] = {}
    summaries: dict[str, Any] = {}

    for name, model in models.items():
        LOGGER.info("Tuning %s using %d grouped folds.", name, n_splits)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=search_spaces[name],
            n_iter=n_iter,
            scoring="recall",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )

        search.fit(X, y, groups=groups)

        fitted_models[name] = search.best_estimator_
        summaries[name] = {
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "cv_splits": n_splits,
        }

        LOGGER.info(
            "%s: best grouped-CV recall = %.4f",
            name,
            search.best_score_,
        )

    return fitted_models, summaries


def save_outputs(
    models: dict[str, Any],
    summaries: dict[str, Any],
    output_dir: Path,
    test_data: pd.DataFrame,
) -> None:
    """Save trained models and a JSON training summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        path = output_dir / f"{name}.joblib"
        joblib.dump(model, path)
        LOGGER.info("Saved %s", path)

    summary = {
        "models": summaries,
        "held_out_test_plots": sorted(
            test_data["plot_id"].astype(str).unique().tolist()
        ),
        "held_out_test_evaluation": (
            "Not calculated by the supplied training workflow."
        ),
    }

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    LOGGER.info("Saved %s", summary_path)


def main() -> None:
    """Run the model-training workflow."""
    args = parse_args()
    configure_logging(args.verbose)

    if args.n_iter < 1:
        raise ValueError("--n-iter must be at least 1.")

    data = load_dataset(args.input_csv)
    data = engineer_features(data)

    train_data, test_data = split_by_plot(
        data,
        test_plots=args.test_plots,
    )

    LOGGER.info(
        "Training observations: %d; held-out observations: %d",
        len(train_data),
        len(test_data),
    )
    LOGGER.info(
        "Training plots: %s",
        sorted(train_data["plot_id"].astype(str).unique()),
    )
    LOGGER.info(
        "Held-out plots: %s",
        sorted(test_data["plot_id"].astype(str).unique()),
    )

    models = make_models(args.random_state)
    search_spaces = parameter_distributions(args.random_state)

    fitted_models, summaries = fit_models(
        train_data=train_data,
        models=models,
        search_spaces=search_spaces,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )

    save_outputs(
        models=fitted_models,
        summaries=summaries,
        output_dir=args.output_dir,
        test_data=test_data,
    )

    LOGGER.info("Model training completed.")


if __name__ == "__main__":
    main()
