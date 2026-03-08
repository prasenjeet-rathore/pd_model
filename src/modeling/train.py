"""
Training script for the PD model with MLflow tracking.

This file answers: what is being trained and in what order?
All model-specific logic lives in src/modeling/trainers/.

Artifact strategy:
  - models/production/ → written by this script; read by inference.py
  - mlruns/            → MLflow experiment tracking (metrics + params committed;
                          mlruns/**/artifacts/ gitignored per .gitignore)

Usage:
    uv run python -m src.modeling.train
    uv run python -m src.modeling.train --model lr      # train only logistic regression
    uv run python -m src.modeling.train --model xgb     # train only xgboost baseline
    uv run python -m src.modeling.train --tune          # run Optuna HPO for XGBoost
    uv run python -m src.modeling.train --n-trials 150  # set number of Optuna trials
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.trainers import (
    LRTrainer,
    OptunaParams,
    BaselineParams,
    TrainingData,
    XGBTrainer,
)
from src.utils.config import PATHS, XGB_CATEGORICAL_COLS

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────

def load_data() -> tuple:
    """Load pre-engineered WoE and raw feature sets from final_dir."""
    final_dir = Path(PATHS["final_dir"])

    X_train_woe = pd.read_parquet(final_dir / "X_train_woe_tree.parquet")
    X_val_woe   = pd.read_parquet(final_dir / "X_val_woe_tree.parquet")
    X_oot_woe   = pd.read_parquet(final_dir / "X_oot_woe_tree.parquet")

    X_train = pd.read_parquet(final_dir / "X_train.parquet")
    X_val   = pd.read_parquet(final_dir / "X_val.parquet")
    X_oot   = pd.read_parquet(final_dir / "X_oot.parquet")

    y_train = pd.read_parquet(final_dir / "y_train.parquet").squeeze()
    y_val   = pd.read_parquet(final_dir / "y_val.parquet").squeeze()
    y_oot   = pd.read_parquet(final_dir / "y_oot.parquet").squeeze()

    print(f"LR features (WoE): {X_train_woe.shape[1]}")
    print(f"XGBoost features (raw): {X_train.shape[1]}")
    print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | OOT: {len(y_oot):,}")
    print(
        f"Default rate — Train: {y_train.mean():.2%} | "
        f"Val: {y_val.mean():.2%} | OOT: {y_oot.mean():.2%}"
    )

    return (
        X_train_woe, X_val_woe, X_oot_woe,
        X_train, X_val, X_oot,
        y_train, y_val, y_oot,
    )


def prepare_xgb_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_oot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert XGB_CATEGORICAL_COLS to integer codes for XGBoost."""
    cat_cols = [c for c in XGB_CATEGORICAL_COLS if c in X_train.columns]
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_oot   = X_oot.copy()
    for col in cat_cols:
        for df in [X_train, X_val, X_oot]:
            df[col] = df[col].astype(str).astype("category").cat.codes.replace(-1, np.nan)
    return X_train, X_val, X_oot


def _build_training_data() -> TrainingData:
    """Load all splits and return a TrainingData bundle for the trainers."""
    (
        X_train_woe, X_val_woe, X_oot_woe,
        X_train, X_val, X_oot,
        y_train, y_val, y_oot,
    ) = load_data()

    X_train_xgb, X_val_xgb, X_oot_xgb = prepare_xgb_categoricals(X_train, X_val, X_oot)

    return TrainingData(
        X_train_woe=X_train_woe, X_val_woe=X_val_woe, X_oot_woe=X_oot_woe,
        X_train=X_train_xgb,    X_val=X_val_xgb,     X_oot=X_oot_xgb,
        y_train=y_train,         y_val=y_val,          y_oot=y_oot,
        imbalance_ratio=(y_train == 0).sum() / (y_train == 1).sum(),
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main(model_choice: str = "all", tune: bool = False, n_trials: int = 75) -> None:
    data = _build_training_data()

    if model_choice in ["all", "lr"]:
        LRTrainer(data).run()

    if model_choice in ["all", "xgb"]:
        XGBTrainer(data, strategy=BaselineParams()).run()

        if tune:
            XGBTrainer(
                data,
                strategy=OptunaParams(n_trials=n_trials),
                run_name="XGBoost_Tuned_Model",
                artifact_prefix="xgb_tuned",
                registered_model_name="PD_XGB_Tuned",
            ).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PD model artifacts")
    parser.add_argument(
        "--model",
        choices=["lr", "xgb", "all"],
        default="all",
        help="Which model type to train (default: all)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter search for XGBoost",
    )
    # For quick pipeline validation: --n-trials 2
    parser.add_argument(
        "--n-trials", type=int, default=75,
        help="Number of Optuna trials (default: 75)",
    )
    args = parser.parse_args()
    main(model_choice=args.model, tune=args.tune, n_trials=args.n_trials)
