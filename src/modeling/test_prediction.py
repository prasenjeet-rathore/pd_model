"""
Simple local prediction tester.

When copied into src/modeling/test_prediction.py, run from project root with:
    uv run python -m src.modeling.test_prediction
or
    uv run python src/modeling/test_prediction.py
depending on how you prefer to execute modules.
"""

from pprint import pprint

import pandas as pd

from src.modeling.inference import pipeline, top_lr_feature_contributions, top_xgb_shap_contributions
from src.utils.config import PATHS


def main() -> None:

    X_oot = pd.read_parquet(PATHS["final_dir"] / "X_oot.parquet")
    example_loan = X_oot.iloc[0].to_dict()  # picking one row from oot sample for prediction to see api working

    if not example_loan:
        print("Please populate `example_loan` with the full feature set before running.")
        return

    df = pd.DataFrame([example_loan])
    proba = float(pipeline.predict_proba(df)[0])

    # Top 4 features — LR uses per-prediction WoE contributions, XGBoost uses per-prediction SHAP values
    if hasattr(pipeline.model, "coef_"):
        top_features = top_lr_feature_contributions(pipeline=pipeline, X_row=df, n_top=4)
    else:
        top_features = top_xgb_shap_contributions(pipeline=pipeline, X_row=df, n_top=4)

    result = {
            "input": example_loan,
            "probability_of_default": proba,
            "model_version": pipeline.version,
            "top_features": top_features,
            }
    pprint(result)


if __name__ == "__main__":
    main()

