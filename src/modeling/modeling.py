"""
Deployable production pipeline wiring

    from src.modeling.modeling import pipeline
    probs = pipeline.predict_proba(df)
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict
import joblib
import numpy as np
import pandas as pd

from src.utils.config import PATHS
from src.utils.woe import transform_woe


class ProductionPipeline:
    """Production artifact for EE 12-month PD model (Tree-WoE + LR + Platt calibration)."""

    def __init__(
        self,
        woe_rules: dict,
        selected_vars: list[str],
        model,
        calibrator,
        version: str | None = None,
    ) -> None:
        self.woe_rules = woe_rules
        self.model = model
        self.calibrator = calibrator
        self.selected_vars = selected_vars
        self.version = version or f"PD_EE_v6_{datetime.now().strftime('%Y%m%d')}"

    def transform_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the WoE transformation used in training."""
        return transform_woe(X, self.woe_rules, self.selected_vars)

    def predict_proba(self, X_new: pd.DataFrame) -> np.ndarray:
        """Return calibrated PDs in [0, 1] for each row in X_new."""
        X_woe = self.transform_woe(X_new[self.selected_vars])
        raw_pd = self.model.predict_proba(X_woe)[:, 1]
        return self.calibrator.predict_proba(raw_pd.reshape(-1, 1))[:, 1]


def _load_production_pipeline() -> ProductionPipeline:
    """Load artifacts saved by 3_feature_engineering and 04_modeling notebooks."""
    final_dir = Path(PATHS["final_dir"])
    models_dir = Path(PATHS["models_dir"])

    woe_rules_path = final_dir / "woe_rules_tree.joblib"
    selected_vars_path = final_dir / "selected_vars_tree.joblib"
    lr_model_path = models_dir / "lr_model.joblib"
    calibrator_path = models_dir / "lr_platt_calibrator.joblib"

    woe_rules = joblib.load(woe_rules_path)
    selected_vars = joblib.load(selected_vars_path)
    lr_model = joblib.load(lr_model_path)
    lr_platt = joblib.load(calibrator_path)

    version = "PD_EE_v1_logistic"
    return ProductionPipeline(
        woe_rules=woe_rules,
        selected_vars=selected_vars,
        model=lr_model,
        calibrator=lr_platt,
        version=version,
    )

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
        "direction": "risk_up" | "risk_down",
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
    coefs = np.asarray(pipeline.model.coef_[0], dtype=float)
    values = X_woe.iloc[0].values.astype(float)

    # Linear contributions before the logistic link function
    contributions = coefs * values

    items: List[Dict] = []
    for feat, val, coef, contr in zip(selected_vars, values, coefs, contributions):
        items.append(
            {
                "feature": feat,
                "woe_value": float(val),
                "coefficient": float(coef),
                "contribution": float(contr),
            }
        )

    # Sort and return top N
    items_sorted = sorted(items, key=lambda d: abs(d["contribution"]), reverse=True)
    return items_sorted[:n_top]

#  used by both FastAPI app and local scripts
print("Loading production PD model artifacts...")
pipeline = _load_production_pipeline()
print(f"✅ Model loaded successfully: {pipeline.version}")


