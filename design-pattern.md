# Design Patterns — PD Credit Risk Model

**Project:** Probability of Default (PD) — Estonian Loan Data
**Date:** 2026-03-08
**Scope:** Full repository analysis — `src/`, `app/`, training pipeline, inference pipeline

---

## Executive Summary

This document analyses the codebase for design pattern opportunities and prescribes the
minimal, high-impact set to adopt. The goal is not architectural novelty — it is
legibility, debuggability, and controlled extensibility. The current codebase is already
well-structured at the utility layer. The problems are localised to `train.py` (repetition
and mixed concerns in `main()`) and `inference.py` (hard-coded to a single model type).
Three patterns address these directly. A fourth is noted as a deferred investment.

---

## Patterns Already Implemented Well

### Facade — `src/utils/`

The entire `utils/` directory is a deliberate Facade over complex domain operations.
`woe.py`, `evaluation.py`, `features.py`, `target.py`, and `data_cleaning.py` each expose
a clean, flat API of pure functions. Callers in notebooks and `train.py` import a named
function and do not need to know about the internal binning strategies, Laplace smoothing,
or VIF calculation internals. This is working correctly — no changes warranted.

### Singleton — `inference.py` Global Pipeline

`inference.py` loads the production pipeline once at module import time:

```python
pipeline = _load_production_pipeline()
```

Both `app.py` and `test_prediction.py` import this symbol. For a FastAPI process this is
appropriate — artifact deserialization is expensive and should not happen per-request.
The only weakness is the hard-coding of LR-only artifacts, addressed by the Factory Method below.

### Single Source of Truth — `src/utils/config.py`

`config.py` acts as a module-level configuration object. All paths, random seeds, feature
lists, split dates, and thresholds are defined once and imported everywhere. This eliminates
the class of bug where a notebook and a production script disagree on column definitions.
Do not fragment this file.

---

## Patterns to Implement

### Pattern 1 — Template Method (Critical)

**New file:** `src/modeling/trainers/base.py`

#### Problem

`main()` in `train.py` contains three near-identical blocks — one per model type (LR,
XGBoost baseline, XGBoost tuned). Each block follows the same invariant sequence:

1. Train the model
2. Evaluate on train / val / OOT
3. Calibrate with Platt scaling
4. Open an MLflow run
5. Log parameters and metrics
6. Export artifacts

The sequence is fixed. Only the details differ per model. This is the canonical Template
Method problem: a fixed algorithm skeleton with variable steps.

#### Solution

A `BaseModelTrainer` abstract class defines the algorithm in `run()`. Each step —
`train()`, `evaluate()`, `calibrate()`, `get_run_name()`, `get_mlflow_params()`,
`export()` — is abstract and must be overridden by each concrete trainer. `run()` itself
is never overridden; it owns the MLflow context and calls each step in sequence.

```
BaseModelTrainer (base.py)
├── run()                  ← template method, never overridden
├── train()                ← abstract
├── evaluate()             ← abstract
├── calibrate()            ← abstract
├── get_run_name()         ← abstract
├── get_mlflow_params()    ← abstract
├── export()               ← abstract
└── get_extra_metrics()    ← optional hook, default returns {}

LRTrainer(BaseModelTrainer)    ← lr.py
XGBTrainer(BaseModelTrainer)   ← xgb.py
```

#### Benefit

- `train.py`'s `main()` shrinks from ~90 lines to ~25 lines
- Debugging LR means opening `lr.py` only — XGBoost logic is not present
- Adding LightGBM means creating `lgbm.py` — `main()` and `base.py` are unchanged
- The MLflow run lifecycle lives in one place (`base.py`)

#### Files

| File | Action |
|---|---|
| `src/modeling/trainers/base.py` | New — `TrainingData` dataclass + `BaseModelTrainer` ABC |
| `src/modeling/trainers/lr.py` | New — `LRTrainer(BaseModelTrainer)` |
| `src/modeling/trainers/xgb.py` | New — `XGBTrainer(BaseModelTrainer)` |
| `src/modeling/trainers/__init__.py` | New — re-exports |
| `src/modeling/train.py` | Refactored — `main()` becomes the clean orchestrator |

---

### Pattern 2 — Strategy (Critical, companion to Template Method)

**File:** `src/modeling/trainers/xgb.py`

#### Problem

XGBoost has two training strategies: a fixed-hyperparameter baseline and an Optuna-driven
search. Currently `tune_xgboost()` is a standalone function and a conditional `if tune:`
in `main()` creates a third training block. The tuned and baseline XGBoost models share
everything except *how their hyperparameters are chosen*.

#### Solution

The hyperparameter acquisition step is extracted into a pluggable `TuningStrategy`.
`XGBTrainer` accepts a strategy in its constructor. Two concrete strategies implement the
`TuningStrategy` protocol:

- `BaselineParams` — returns the hardcoded parameter dict; instructs the trainer to use
  early stopping with an eval set
- `OptunaParams(n_trials)` — runs the Optuna study and returns `study.best_params`;
  instructs the trainer to fit on all training data without early stopping (the optimal
  `n_estimators` was already found by the search)

The `--tune` CLI flag in `train.py` maps cleanly to strategy selection:

```python
strategy = OptunaParams(n_trials) if tune else BaselineParams()
XGBTrainer(data, strategy=strategy).run()
```

#### Benefit

- The Optuna objective function is isolated in `OptunaParams` — it has no awareness of
  MLflow, export paths, or calibration
- Debugging a failed Optuna run means reading `xgb.py` only
- Future strategies (grid search, Hyperopt) slot in without modifying `XGBTrainer`
- `main()` has a single XGBoost instantiation call regardless of whether tuning is used

#### Files

| File | Action |
|---|---|
| `src/modeling/trainers/xgb.py` | `TuningStrategy` protocol + `BaselineParams` + `OptunaParams` + `XGBTrainer` |

---

### Pattern 3 — Factory Method (High Priority)

**File:** `src/modeling/inference.py`

#### Problem

`_load_production_pipeline()` is hard-coded to load LR artifacts by name
(`lr_model.joblib`, `lr_platt_calibrator.joblib`). Serving an XGBoost model from the same
FastAPI endpoint requires manually editing the function. `app.py` and
`test_prediction.py` are both coupled to the LR pipeline because they import the global
`pipeline` symbol which is created by that one hard-coded loader.

#### Solution

A `PipelineFactory` class with a `create(model_type: str) -> ProductionPipeline` class
method centralises all artifact path resolution. It knows the artifact filenames for each
model type. The global singleton becomes:

```python
_MODEL_TYPE = os.environ.get("PD_MODEL_TYPE", "lr")
pipeline = PipelineFactory.create(_MODEL_TYPE)
```

Switching the serving model from LR to tuned XGBoost requires only an environment
variable change in Docker — no code change.

#### Benefit

- `app.py` and `test_prediction.py` require zero changes
- The Docker container can be configured at runtime: `-e PD_MODEL_TYPE=xgb_tuned`
- Adding a new serving model type requires one new branch in `PipelineFactory.create()`
- `ProductionPipeline` itself is unchanged — it remains model-agnostic

#### Files

| File | Action |
|---|---|
| `src/modeling/inference.py` | Add `PipelineFactory`; replace hard-coded loader with factory + env var |

---

### Pattern 4 — Registry (Deferred)

Not implemented now. With three model types, a registry adds indirection without removing
meaningful repetition. The Template Method already solves duplication. Revisit when adding
a fourth model type — at that point, a `TRAINER_REGISTRY` dict maps string keys to trainer
classes and `main()` resolves trainers dynamically rather than via `if/elif` branching.

---

## Summary

| Pattern | Location | Priority | Benefit |
|---|---|---|---|
| Template Method | `src/modeling/trainers/base.py` | Critical | Eliminates 3 near-identical training blocks; `main()` becomes ~25 lines |
| Strategy | `src/modeling/trainers/xgb.py` | Critical | Optuna search isolated from training loop; single XGBTrainer for all XGB variants |
| Factory Method | `src/modeling/inference.py` | High | Any model type serveable via env var; no code change to swap serving model |
| Registry | `src/modeling/train.py` | Deferred | Not worth it at 3 models; revisit at 4+ |

## Unchanged

The following are well-designed and require no structural changes:

- `src/utils/` — all 6 modules (pure functions, Facade pattern)
- `app/app.py` — imports `pipeline` from inference; unchanged after factory refactor
- `src/modeling/test_prediction.py` — same import; unchanged
- `src/utils/config.py` — single source of truth; do not fragment
