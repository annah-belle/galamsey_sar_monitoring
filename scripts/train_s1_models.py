#!/usr/bin/env python3
"""Train and save Sentinel-1 classifiers using plot-based cross-validation.

Inputs require plot_id, Class, VV and VH columns. BAND_1/BAND_2 are accepted
as aliases for VV/VH. VV and VH are assumed to be in dB.

The script preserves the supplied thesis training design:
- held-out plots are excluded from model training;
- GroupKFold is used on training plots for hyperparameter tuning;
- recall for Class=1 (galamsey) is the tuning objective;
- Logistic Regression, Random Forest, SVC and ExtraTrees are tuned;
- best estimators and a JSON training summary are saved.

Important: the original script creates X_test/y_test but does not calculate
held-out test metrics. This version records the held-out split metadata but does
not claim test performance that was not computed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ["VV", "VH", "VV_VH_diff", "VV_VH_ratio", "NPI"]
MODEL_FILENAMES = {
    "LR": "best_lr.joblib",
    "RF": "best_rf.joblib",
    "SVC": "best_svc.joblib",
    "ExtraTrees": "best_extratrees.joblib",
}


@dataclass
class TrainConfig:
    csv_path: Path
    out_dir: Path
    test_plots: list[str]
    seed: int
    n_iter_lr: int = 20
    n_iter_rf: int = 30
    n_iter_svc: int = 30
    n_iter_et: int = 30


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename thesis sample aliases BAND_1/BAND_2 to VV/VH when needed."""
    rename_map: dict[str, str] = {}
    if "BAND_1" in df.columns and "VV" not in df.columns:
        rename_map["BAND_1"] = "VV"
    if "BAND_2" in df.columns and "VH" not in df.columns:
        rename_map["BAND_2"] = "VH"
    return df.rename(columns=rename_map) if rename_map else df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create the five Sentinel-1 features used by the supplied thesis script."""
    required = {"plot_id", "Class", "VV", "VH"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    vv_db = pd.to_numeric(result["VV"], errors="coerce")
    vh_db = pd.to_numeric(result["VH"], errors="coerce")
    result["VV_VH_diff"] = vv_db - vh_db

    vv_linear = 10 ** (vv_db / 10.0)
    vh_linear = 10 ** (vh_db / 10.0)
    eps = 1e-6
    result["VV_VH_ratio"] = vv_linear / (vh_linear + eps)
    result["NPI"] = (vv_linear - vh_linear) / (vv_linear + vh_linear + eps)
    result[FEATURES] = result[FEATURES].replace([np.inf, -np.inf], np.nan)
    result["Class"] = pd.to_numeric(result["Class"], errors="raise").astype(int)

    labels = set(result["Class"].dropna().unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError(f"Class must contain both binary labels 0 and 1; found {sorted(labels)}")
    return result, FEATURES.copy()


def split_by_plots(
    df: pd.DataFrame, features: list[str], test_plots: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by plot ID so test plots are excluded from training."""
    test_mask = df["plot_id"].astype(str).isin({str(p) for p in test_plots})
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()
    if train_df.empty:
        raise ValueError("Training set is empty. Check --test-plots.")
    if test_df.empty:
        raise ValueError("Test set is empty. Check --test-plots.")
    if train_df["Class"].nunique() < 2:
        raise ValueError("Training data must contain both Class=0 and Class=1.")
    return train_df, test_df


def make_group_kfold(groups: np.ndarray) -> GroupKFold:
    """Create GroupKFold with up to three training plots."""
    n_splits = min(3, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two unique training plots for GroupKFold.")
    return GroupKFold(n_splits=n_splits)


def build_searches(cfg: TrainConfig, cv: GroupKFold, scorer) -> dict[str, RandomizedSearchCV]:
    """Build the four RandomizedSearchCV objects from the thesis script."""
    searches: dict[str, RandomizedSearchCV] = {}

    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed)),
    ])
    searches["LR"] = RandomizedSearchCV(
        lr,
        {"clf__C": np.logspace(-3, 2, 20), "clf__solver": ["lbfgs"]},
        n_iter=cfg.n_iter_lr, scoring=scorer, cv=cv, n_jobs=-1, verbose=1, random_state=cfg.seed,
    )

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=cfg.seed)),
    ])
    searches["RF"] = RandomizedSearchCV(
        rf,
        {
            "rf__n_estimators": [200, 400, 600, 800],
            "rf__max_depth": [None, 10, 20, 30, 40],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__max_features": ["sqrt", "log2", 0.5],
        },
        n_iter=cfg.n_iter_rf, scoring=scorer, cv=cv, n_jobs=-1, verbose=1, random_state=cfg.seed,
    )

    svc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=cfg.seed)),
    ])
    searches["SVC"] = RandomizedSearchCV(
        svc,
        {"svc__C": np.logspace(-2, 2, 20), "svc__gamma": np.logspace(-4, 0, 20)},
        n_iter=cfg.n_iter_svc, scoring=scorer, cv=cv, n_jobs=-1, verbose=1, random_state=cfg.seed,
    )

    et = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("et", ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=cfg.seed)),
    ])
    searches["ExtraTrees"] = RandomizedSearchCV(
        et,
        {
            "et__n_estimators": [300, 600, 900],
            "et__max_depth": [None, 10, 20, 30, 40],
            "et__min_samples_split": [2, 5, 10],
            "et__min_samples_leaf": [1, 2, 4],
            "et__max_features": ["sqrt", "log2", 0.5],
        },
        n_iter=cfg.n_iter_et, scoring=scorer, cv=cv, n_jobs=-1, verbose=1, random_state=cfg.seed,
    )
    return searches


def fit_and_save(
    searches: dict[str, RandomizedSearchCV],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    """Fit searches, save best estimators, and return CV summaries."""
    summary: dict[str, Any] = {}
    for name, search in searches.items():
        print(f"\n=== Training {name} ===")
        search.fit(x_train, y_train, groups=groups_train)
        model_path = out_dir / MODEL_FILENAMES[name]
        joblib.dump(search.best_estimator_, model_path)
        summary[name] = {
            "best_cv_recall_galamsey": float(search.best_score_),
            "best_params": search.best_params_,
            "saved_model": model_path.name,
        }
        print(f"{name} best CV recall (galamsey): {search.best_score_:.4f}")
    return summary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Sentinel-1 classifiers with plot-based GroupKFold CV.")
    parser.add_argument("--csv-path", required=True, type=Path, help="CSV containing plot_id, Class, VV, VH.")
    parser.add_argument("--out-dir", default=Path("./model_outputs_s1"), type=Path, help="Output directory for models and summary.")
    parser.add_argument("--test-plots", nargs="+", default=["1", "5"], help="Plot IDs reserved for held-out testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    return TrainConfig(csv_path=args.csv_path, out_dir=args.out_dir, test_plots=[str(p) for p in args.test_plots], seed=args.seed)


def main() -> None:
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    df = standardize_columns(pd.read_csv(cfg.csv_path))
    df, features = engineer_features(df)
    train_df, test_df = split_by_plots(df, features, cfg.test_plots)

    x_train = train_df[features]
    y_train = train_df["Class"].to_numpy()
    groups_train = train_df["plot_id"].astype(str).to_numpy()

    print("\nData split summary")
    print("------------------")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)} | Train plots: {sorted(train_df['plot_id'].astype(str).unique())}")
    print(f"Test rows: {len(test_df)} | Test plots: {sorted(test_df['plot_id'].astype(str).unique())}")
    print(f"Features: {features}")

    scorer = make_scorer(recall_score, pos_label=1)
    cv = make_group_kfold(groups_train)
    searches = build_searches(cfg, cv, scorer)
    summary = fit_and_save(searches, x_train, y_train, groups_train, cfg.out_dir)

    summary_path = cfg.out_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "csv_path": str(cfg.csv_path),
                "test_plots": cfg.test_plots,
                "seed": cfg.seed,
                "features": features,
                "n_rows_total": len(df),
                "n_rows_train": len(train_df),
                "n_rows_held_out": len(test_df),
                "held_out_metrics_computed": False,
                "summary": summary,
            },
            handle,
            indent=2,
        )
    print(f"\nTraining complete. Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
