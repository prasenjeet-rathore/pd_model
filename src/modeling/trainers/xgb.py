"""
xgb.py — XGBoost trainer with pluggable hyperparameter strategy.

Implements two patterns:
  - Strategy: TuningStrategy protocol lets BaselineParams and OptunaParams
    be swapped without changing XGBTrainer.
  - Template Method: XGBTrainer(BaseModelTrainer) fills in the XGBoost-specific
    steps of the base training pipeline.

Usage:
    XGBTrainer(data, strategy=BaselineParams()).run()
    XGBTrainer(data, strategy=OptunaParams(n_trials=75),
               run_name="XGBoost_Tuned_Model",
               artifact_prefix="xgb_tuned",
               registered_model_name="PD_XGB_Tuned").run()
"""

import joblib
import mlflow.xgboost
import xgboost as xgb
from pathlib import Path
from typing import Protocol

from src.modeling.trainers.base import BaseModelTrainer, TrainingData
from src.utils.config import PATHS, RANDOM_STATE
from src.utils.evaluation import fit_platt_scaling, report_auc_all

PRODUCTION_DIR = Path(PATHS["production_models_dir"])


# ── Strategy Protocol ──────────────────────────────────────────────────────────

class TuningStrategy(Protocol):
    """Protocol for hyperparameter acquisition strategies.

    get_params()     — returns constructor kwargs for XGBClassifier
    get_fit_kwargs() — returns kwargs for model.fit()
    """

    def get_params(
        self,
        X_train, X_val, y_train, y_val,
        imbalance_ratio: float,
    ) -> dict: ...

    def get_fit_kwargs(self, X_val, y_val) -> dict: ...


class BaselineParams:
    """Fixed hyperparameter set for the XGBoost baseline.

    Uses early stopping during training to find the best iteration.
    """

    def get_params(self, X_train, X_val, y_train, y_val, imbalance_ratio: float) -> dict:
        return {
            "n_estimators":        500,
            "learning_rate":       0.05,
            "max_depth":           5,
            "scale_pos_weight":    imbalance_ratio,
            "early_stopping_rounds": 50,
            "eval_metric":         "auc",
            "random_state":        RANDOM_STATE,
        }

    def get_fit_kwargs(self, X_val, y_val) -> dict:
        return {"eval_set": [(X_val, y_val)], "verbose": False}


class OptunaParams:
    """Runs an Optuna study and returns the best hyperparameters found.

    The final model is trained on the full training set without early stopping —
    the optimal n_estimators was already determined by the search.
    """

    def __init__(self, n_trials: int = 75) -> None:
        self.n_trials = n_trials
        self.best_value: float | None = None

    def get_params(self, X_train, X_val, y_train, y_val, imbalance_ratio: float) -> dict:
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError("optuna is required for tuning: uv add optuna")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "max_depth":         trial.suggest_int("max_depth", 3, 7),
                "min_child_weight":  trial.suggest_int("min_child_weight", 5, 100),
                "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "scale_pos_weight":  imbalance_ratio,
                "eval_metric":       "auc",
                "tree_method":       "hist",
                "random_state":      RANDOM_STATE,
                "verbosity":         0,
            }
            from sklearn.metrics import roc_auc_score
            model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        print(f"\n--- Optuna XGBoost HPO ({self.n_trials} trials) ---")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE),
            study_name="xgb_pd_model",
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_value = study.best_value
        best_params = study.best_params.copy()
        print(f"Best Val AUC: {study.best_value:.4f}")
        print(f"Best params: {best_params}")

        return {
            **best_params,
            "scale_pos_weight": imbalance_ratio,
            "eval_metric":      "auc",
            "tree_method":      "hist",
            "random_state":     RANDOM_STATE,
        }

    def get_fit_kwargs(self, X_val, y_val) -> dict:
        # Train on all training data with fixed n_estimators — no early stopping
        return {"verbose": False}


# ── Trainer ────────────────────────────────────────────────────────────────────

class XGBTrainer(BaseModelTrainer):
    """XGBoost trainer. The strategy determines how hyperparameters are chosen."""

    def __init__(
        self,
        data: TrainingData,
        strategy: TuningStrategy,
        run_name: str = "XGBoost_Baseline",
        artifact_prefix: str = "xgb",
        registered_model_name: str = "PD_XGB_Baseline",
    ) -> None:
        super().__init__(data)
        self.strategy = strategy
        self._run_name = run_name
        self.artifact_prefix = artifact_prefix
        self.registered_model_name = registered_model_name

    def train(self) -> tuple:
        print(f"\n--- Training XGBoost ({self._run_name}) ---")
        params = self.strategy.get_params(
            self.data.X_train, self.data.X_val,
            self.data.y_train, self.data.y_val,
            self.data.imbalance_ratio,
        )
        fit_kwargs = self.strategy.get_fit_kwargs(self.data.X_val, self.data.y_val)

        model = xgb.XGBClassifier(**params, verbosity=0)
        model.fit(self.data.X_train, self.data.y_train, **fit_kwargs)

        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            print(f"  Best iteration: {model.best_iteration}")

        return model, params

    def evaluate(self) -> dict:
        return report_auc_all(
            self.data.y_train, self.model.predict_proba(self.data.X_train)[:, 1],
            self.data.y_val,   self.model.predict_proba(self.data.X_val)[:, 1],
            self.data.y_oot,   self.model.predict_proba(self.data.X_oot)[:, 1],
            model_name=f"XGBoost ({self._run_name})",
        )

    def calibrate(self):
        return fit_platt_scaling(self.model, self.data.X_val, self.data.y_val)

    def get_run_name(self) -> str:
        return self._run_name

    def get_mlflow_params(self) -> dict:
        params = {f"best_{k}": v for k, v in self._train_metadata.items()}
        if isinstance(self.strategy, OptunaParams):
            params["n_optuna_trials"] = self.strategy.n_trials
        return params

    def get_extra_metrics(self) -> dict:
        extras = {}
        if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
            extras["best_iteration"] = self.model.best_iteration
        if isinstance(self.strategy, OptunaParams) and self.strategy.best_value is not None:
            extras["optuna_best_val_auc"] = self.strategy.best_value
        return extras

    def export(self) -> None:
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

        # Save to production dir so PipelineFactory can load by path
        model_path = PRODUCTION_DIR / f"{self.artifact_prefix}_model.json"
        platt_path = PRODUCTION_DIR / f"{self.artifact_prefix}_platt_calibrator.joblib"
        self.model.save_model(str(model_path))
        joblib.dump(self.calibrator, platt_path)

        # Register model in MLflow and log platt calibrator as artifact
        mlflow.xgboost.log_model(
            self.model,
            artifact_path=self.artifact_prefix,
            registered_model_name=self.registered_model_name,
        )
        mlflow.log_artifact(str(platt_path), artifact_path=self.artifact_prefix)
        print(f"Exported {self.artifact_prefix} artifacts to {PRODUCTION_DIR}/")
