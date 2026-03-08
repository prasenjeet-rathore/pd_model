"""
lr.py — Logistic Regression trainer.

All LR-specific logic lives here: which data splits to use (WoE-transformed),
hyperparameters, cross-validation, MLflow run name, registered model name,
and artifact filenames. Debug LR issues by reading this file only.
"""

import joblib
import mlflow.sklearn
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.modeling.trainers.base import BaseModelTrainer, TrainingData
from src.utils.config import PATHS, RANDOM_STATE
from src.utils.evaluation import fit_platt_scaling, report_auc_all

PRODUCTION_DIR = Path(PATHS["production_models_dir"])


class LRTrainer(BaseModelTrainer):
    """Trains a Logistic Regression model on WoE-transformed features."""

    def __init__(self, data: TrainingData) -> None:
        super().__init__(data)
        self._cv_scores: np.ndarray | None = None

    def train(self) -> tuple:
        print("\n--- Training Logistic Regression ---")
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model.fit(self.data.X_train_woe, self.data.y_train)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        self._cv_scores = cross_val_score(
            model, self.data.X_train_woe, self.data.y_train,
            cv=cv, scoring="roc_auc",
        )
        print(f"LR 5-fold CV AUC: {self._cv_scores.mean():.4f} +/- {self._cv_scores.std():.4f}")

        metadata = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_features": self.data.X_train_woe.shape[1],
        }
        return model, metadata

    def evaluate(self) -> dict:
        return report_auc_all(
            self.data.y_train, self.model.predict_proba(self.data.X_train_woe)[:, 1],
            self.data.y_val,   self.model.predict_proba(self.data.X_val_woe)[:, 1],
            self.data.y_oot,   self.model.predict_proba(self.data.X_oot_woe)[:, 1],
            model_name="Logistic Regression",
        )

    def calibrate(self):
        return fit_platt_scaling(self.model, self.data.X_val_woe, self.data.y_val)

    def get_run_name(self) -> str:
        return "Logistic_Regression_Baseline"

    def get_mlflow_params(self) -> dict:
        return self._train_metadata

    def get_extra_metrics(self) -> dict:
        return {
            "cv_auc_mean": float(self._cv_scores.mean()),
            "cv_auc_std":  float(self._cv_scores.std()),
        }

    def export(self) -> None:
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

        # Save to production dir so inference.py can load by path
        joblib.dump(self.model,      PRODUCTION_DIR / "lr_model.joblib")
        joblib.dump(self.calibrator, PRODUCTION_DIR / "lr_platt_calibrator.joblib")

        # Register model in MLflow and log platt calibrator as artifact
        mlflow.sklearn.log_model(
            self.model,
            artifact_path="lr",
            registered_model_name="PD_LR_Baseline",
            input_example=None,
        )
        mlflow.log_artifact(
            str(PRODUCTION_DIR / "lr_platt_calibrator.joblib"),
            artifact_path="lr",
        )
        print(f"Exported LR artifacts to {PRODUCTION_DIR}/")
