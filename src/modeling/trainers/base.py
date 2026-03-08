"""
base.py — Abstract base class for all PD model trainers.

Implements the Template Method pattern: run() defines the invariant training
pipeline (train → evaluate → calibrate → log → export). Concrete subclasses
implement only the steps that differ per model type.
"""

import mlflow
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainingData:
    """All data splits passed to every trainer as a single object.

    WoE splits are used by LR. Raw splits are used by XGBoost.
    Both are carried here so BaseModelTrainer needs only one argument.
    """
    X_train_woe: pd.DataFrame
    X_val_woe: pd.DataFrame
    X_oot_woe: pd.DataFrame
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_oot: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_oot: pd.Series
    imbalance_ratio: float


class BaseModelTrainer(ABC):
    """Template Method base for PD model trainers.

    run() defines the invariant sequence and must not be overridden.
    Subclasses implement the abstract steps that differ per model type.
    """

    def __init__(self, data: TrainingData) -> None:
        self.data = data
        self.model = None
        self.calibrator = None
        self.aucs: dict = {}
        self._train_metadata: dict = {}

    def run(self) -> None:
        """Execute the full training pipeline. Do not override in subclasses."""
        self.model, self._train_metadata = self.train()
        self.aucs = self.evaluate()
        self.calibrator = self.calibrate()

        mlflow.set_experiment("PD_Model_Training")
        with mlflow.start_run(run_name=self.get_run_name()):
            mlflow.log_params(self.get_mlflow_params())
            mlflow.log_metrics({
                "train_auc": self.aucs["train"],
                "val_auc":   self.aucs["val"],
                "oot_auc":   self.aucs["oot"],
                **self.get_extra_metrics(),
            })
            self.export()

    @abstractmethod
    def train(self) -> tuple:
        """Fit the model. Return (fitted_model, metadata_dict)."""

    @abstractmethod
    def evaluate(self) -> dict:
        """Compute AUC on all splits. Return {'train': float, 'val': float, 'oot': float}."""

    @abstractmethod
    def calibrate(self):
        """Fit and return a Platt calibrator on the validation set."""

    @abstractmethod
    def get_run_name(self) -> str:
        """Return the MLflow run name string."""

    @abstractmethod
    def get_mlflow_params(self) -> dict:
        """Return the params dict for mlflow.log_params()."""

    @abstractmethod
    def export(self) -> None:
        """Write artifacts to production dir and log to the active MLflow run."""

    def get_extra_metrics(self) -> dict:
        """Optional extra metrics beyond train/val/oot AUC. Override if needed."""
        return {}
