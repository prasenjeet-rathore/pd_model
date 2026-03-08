"""
Deployable production pipeline wiring

    from src.modeling.inference import pipeline
    probs = pipeline.predict_proba(df)

Set PD_MODEL_TYPE env var to switch the serving model without code changes:
    PD_MODEL_TYPE=lr          (default) — Logistic Regression + Platt
    PD_MODEL_TYPE=xgb         — XGBoost baseline + Platt
    PD_MODEL_TYPE=xgb_tuned   — Optuna-tuned XGBoost + Platt
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.config import PATHS, XGB_CATEGORICAL_COLS
from src.utils.woe import transform_woe


class ProductionPipeline:
    """Production artifact for PD model """

    def __init__(
        self,
        woe_rules: dict,
        selected_vars: list[str],
        model,
        calibrator,
        version: str | None = None,
        preprocessor=None,
    ) -> None:
        self.woe_rules = woe_rules
        self.model = model
        self.calibrator = calibrator
        self.selected_vars = selected_vars
        self.version = version or f"PD_EE_v6_{datetime.now().strftime('%Y%m%d')}"
        # None → default WoE transform (LR); pass a callable for other model types
        self._preprocessor = preprocessor

    def transform_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the WoE transformation used in training."""
        return transform_woe(X, self.woe_rules, self.selected_vars)

    def predict_proba(self, X_new: pd.DataFrame) -> np.ndarray:
        """Return calibrated PDs in [0, 1] for each row in X_new."""
        if self._preprocessor is not None:
            X_ready = self._preprocessor(X_new)
        else:
            X_ready = self.transform_woe(X_new[self.selected_vars])
        raw_pd = self.model.predict_proba(X_ready)[:, 1]
        return self.calibrator.predict_proba(raw_pd.reshape(-1, 1))[:, 1]


class PipelineFactory:
    """Creates the correct ProductionPipeline for a given model type.

    Centralises artifact path resolution so app.py and test_prediction.py
    need no changes when switching the serving model — set PD_MODEL_TYPE instead.
    """

    @classmethod
    def create(cls, model_type: str) -> ProductionPipeline:
        final_dir  = Path(PATHS["final_dir"])
        models_dir = Path(PATHS["production_models_dir"])

        woe_rules     = joblib.load(final_dir / "woe_rules_tree.joblib")
        selected_vars = joblib.load(final_dir / "selected_vars_tree.joblib")

        if model_type == "lr":
            model      = joblib.load(models_dir / "lr_model.joblib")
            calibrator = joblib.load(models_dir / "lr_platt_calibrator.joblib")
            version    = "PD_EE_v1_logistic"
            preprocessor = None  # default WoE transform

        elif model_type == "xgb":
            model = xgb.XGBClassifier()
            model.load_model(str(models_dir / "xgb_model.json"))
            calibrator   = joblib.load(models_dir / "xgb_platt_calibrator.joblib")
            version      = "PD_EE_v2_xgboost_baseline"
            preprocessor = cls._xgb_preprocessor

        elif model_type == "xgb_tuned":
            model = xgb.XGBClassifier()
            model.load_model(str(models_dir / "xgb_tuned_model.json"))
            calibrator   = joblib.load(models_dir / "xgb_tuned_platt_calibrator.joblib")
            version      = "PD_EE_v3_xgboost_tuned"
            preprocessor = cls._xgb_preprocessor

        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose: lr, xgb, xgb_tuned"
            )

        return ProductionPipeline(
            woe_rules=woe_rules,
            selected_vars=selected_vars,
            model=model,
            calibrator=calibrator,
            version=version,
            preprocessor=preprocessor,
        )

    @staticmethod
    def _xgb_preprocessor(X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same categorical encoding used during XGBoost training."""
        cat_cols = [c for c in XGB_CATEGORICAL_COLS if c in X.columns]
        X = X.copy()
        for col in cat_cols:
            X[col] = X[col].astype(str).astype("category").cat.codes.replace(-1, np.nan)
        return X


def top_xgb_shap_contributions(
    pipeline,
    X_row: pd.DataFrame,
    n_top: int = 4,
) -> List[Dict]:
    """Return top-N per-prediction SHAP contributions for an XGBoost pipeline.

    Positive shap_value → feature pushed this loan's PD higher.
    Negative shap_value → feature pushed this loan's PD lower.
    """
    import json
    import shap

    if X_row.shape[0] != 1:
        raise ValueError("X_row must contain exactly one row")

    # Apply the same categorical encoding used during training
    X_encoded = pipeline._preprocessor(X_row)

    # Fix base_score string formatting — required for SHAP TreeExplainer compatibility
    booster = pipeline.model.get_booster()
    config = json.loads(booster.save_config())
    config["learner"]["learner_model_param"]["base_score"] = str(
        float(config["learner"]["learner_model_param"]["base_score"].strip("[]"))
    )
    booster.load_config(json.dumps(config))

    explainer = shap.TreeExplainer(pipeline.model)
    shap_values = explainer.shap_values(X_encoded)  # shape (1, n_features)

    items = [
        {"feature": feat, "shap_value": round(float(val), 6)}
        for feat, val in zip(X_encoded.columns.tolist(), shap_values[0])
    ]
    return sorted(items, key=lambda x: abs(x["shap_value"]), reverse=True)[:n_top]


def top_lr_feature_contributions(
    pipeline,
    X_row: pd.DataFrame,
    n_top: int = 4,
) -> List[Dict]:
    """
    Compute top-N feature contributions for a single observation
    for the existing LR WoE model.

    Returns a list of dicts like:
    [
      {
        "feature": "...",
        "woe_value": ...,
        "coefficient": ...,
        "contribution": ...,
      },
      ...
    ]
    """
    if X_row.shape[0] != 1:
        raise ValueError("X_row must contain exactly one row")

    selected_vars = pipeline.selected_vars

    # Apply WoE transformation the model expects
    X_woe = pipeline.transform_woe(X_row[selected_vars])

    # Coefficients and values for this row
    coefs  = np.asarray(pipeline.model.coef_[0], dtype=float)
    values = X_woe.iloc[0].values.astype(float)

    # Linear contributions before the logistic link function
    contributions = coefs * values

    items: List[Dict] = []
    for feat, val, coef, contr in zip(selected_vars, values, coefs, contributions):
        items.append(
            {
                "feature":     feat,
                "woe_value":   float(val),
                "coefficient": float(coef),
                "contribution": float(contr),
            }
        )

    # Sort and return top N
    items_sorted = sorted(items, key=lambda d: abs(d["contribution"]), reverse=True)
    return items_sorted[:n_top]


# Used by both FastAPI app and local scripts
_MODEL_TYPE = os.environ.get("PD_MODEL_TYPE", "lr")
print(f"Loading production PD model artifacts (type={_MODEL_TYPE})...")
pipeline = PipelineFactory.create(_MODEL_TYPE)
print(f"Model loaded successfully: {pipeline.version}")
