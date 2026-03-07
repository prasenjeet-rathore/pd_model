"""
Deployable FastAPI app using the global `pipeline` from inference.py.

run with:
    uv run uvicorn app.app:app --reload
"""

from fastapi import FastAPI
import pandas as pd

from src.modeling.inference import pipeline
from src.modeling.inference import top_lr_feature_contributions

app = FastAPI(title="PD Model API", version=pipeline.version)


@app.post("/predict")
async def predict_pd(loan: dict):
    """
    Expect a JSON body with raw feature values matching training columns.
    Returns probability of default for that single loan.
    Also returns features ordered by importance which led to the current prediction. 
    """
    try:
        df_input = pd.DataFrame([loan])
        pd_score = float(pipeline.predict_proba(df_input)[0])
        top_items = top_lr_feature_contributions(pipeline, df_input, n_top=2)
        return {
            "probability_of_default": round(pd_score, 6),
            "top_feature_contributions": top_items,
            "model_version": pipeline.version,
            "status": "success",
        }
    except Exception as e:  #  simple safety net
        return {"status": "error", "error": str(e), "model_version": pipeline.version}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": pipeline.version}

