# PD MODEL

<p align="center">
  <a href="https://cookiecutter-data-science.drivendata.org/"><img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a>
  <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" />
  <img src="https://img.shields.io/badge/mlflow-%230194E2.svg?style=flat&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white" />
</p>

The following is a project for experimentation on PD model for credit risk modeling
By Prasenjeet Rathore

## Project Organization

```
├── LICENSE                 <- Open-source license MIT
├── README.md               <- The top-level README for using this project.
├── app
│   └── app.py              <- FastAPI app exposing the production model as an endpoint.
├── data
│   ├── 01_raw              <- The original, immutable data dump.
│   ├── 02_processed        <- Intermediate data that has been transformed.
│   └── 03_final            <- The final data sets for modeling.
│
├── models
│   └── production/         <- Serialized production artifacts read by inference.py
│
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering).
│
├── pyproject.toml          <- Project configuration and pinned dependencies (managed with uv).
│
├── references              <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports
│   └── figures             <- Generated graphics and figures from notebooks.
│
├── requirements.txt        <- Dev environment requirements (uv pip freeze > requirements.txt).
│
└── src
    ├── __init__.py
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── train.py            <- Orchestrator: loads data, calls models for training (LR / XGBoost).
    │   ├── inference.py        <- ProductionPipeline + PipelineFactory; loaded by app.py.
    │   ├── test_prediction.py  <- Local one-row prediction tester using the OOT sample.
    │   └── trainers/
    │       ├── __init__.py
    │       ├── base.py         <- BaseModelTrainer ABC + TrainingData dataclass.
    │       ├── lr.py           <- LRTrainer (Template Method, WoE features).
    │       └── xgb.py          <- XGBTrainer + TuningStrategy (Strategy pattern).
    │
    └── utils
        ├── __init__.py
        ├── config.py           <- Central config: paths, random state, categorical columns.
        ├── data_cleaning.py    <- Util functions for data processing.
        ├── evaluation.py       <- Util functions for evaluating models.
        ├── features.py         <- Util functions to create features for modeling.
        ├── target.py           <- Util functions to create target variable.
        └── woe.py              <- Util functions for WoE transformation and binning.
```


## 1. How to test the model prediction?

This repository includes a two-stage `Dockerfile` at the project root. It uses:

- **Builder stage** – installs dependencies
- **Runtime stage** – copies the virtual environment and project code into a slim Python image.

The serving model is controlled by the `PD_MODEL_TYPE` environment variable (default: `lr`).

| Value | Model |
|---|---|
| `lr` | Logistic Regression + Platt calibration (default) |
| `xgb` | XGBoost baseline + Platt calibration |
| `xgb_tuned` | Optuna-tuned XGBoost + Platt calibration |

### Build the image
*(use either docker or podman, I used podman for my Fedora workstation)*

*Build may take 2-3 minutes because it is a 2-stage build to reduce attack surface and image size.*

### 1.1 From the project root run:

```bash
docker build -t model-pd .
```

### 1.2 Run the API server in detached mode

Default model (Logistic Regression):

```bash
docker run -d -p 8000:8000 --name pd-service model-pd
```

To serve XGBoost instead, pass the env var:

```bash
docker run -d -e PD_MODEL_TYPE=xgb -p 8000:8000 --name pd-service model-pd
```

### 1.3 Check if container is created and running

```bash
docker ps
```

### 1.4 Run a one-off prediction from the OOT sample

Default model (Logistic Regression) — top features show per-prediction WoE contributions:

```bash
docker run --rm model-pd python -m src.modeling.test_prediction
```

XGBoost — top features show per-prediction SHAP values:

```bash
docker run --rm -e PD_MODEL_TYPE=xgb model-pd python -m src.modeling.test_prediction
```

### 1.5 How to stop the container

```bash
docker ps                        # find the container id
docker stop <container-id>       # first few characters + Tab autocompletes
```

---

## 2. MLflow experiment tracking

Training runs are tracked with MLflow. After running `train.py`, launch the UI from the project root:

```bash
uv run mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser.

- **Experiments** tab — all training runs with metrics (train/val/OOT AUC, CV scores).
- **Models** tab — registered model versions for `PD_LR_Baseline`, `PD_XGB_Baseline`, and `PD_XGB_Tuned`.

### Retrain models

```bash
uv run python -m src.modeling.train               # train LR + XGBoost baseline
uv run python -m src.modeling.train --model lr    # LR only
uv run python -m src.modeling.train --model xgb   # XGBoost baseline only
uv run python -m src.modeling.train --tune        # also run Optuna HPO for XGBoost
uv run python -m src.modeling.train --tune --n-trials 150
```

---

## 3. How to run notebooks?

Install dev dependencies with `requirements.txt` (not `requirements-prod.txt`, which is for the production container only). Make sure to install into a `.venv` folder at the project root.

Set the Python interpreter path in your IDE to point to `.venv` so Jupyter notebooks pick it up. In VSCode: `Ctrl+Shift+P` → *Python: Select Interpreter*.

--------