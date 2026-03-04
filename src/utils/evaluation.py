"""
evaluation.py — Model evaluation, calibration, and diagnostic utilities.


Covers:
  - AUC/Gini reporting
  - ROC curves
  - Brier score & reliability plots
  - Platt scaling & isotonic recalibration
  - Hosmer-Lemeshow test
  - Feature importance (XGBoost gain, LR coefficients, SHAP)
  - PSI (Population Stability Index) for drift monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.stats import chi2


# ─────────────────────────────────────────────────────────────────────
# AUC / GINI REPORTING
# ─────────────────────────────────────────────────────────────────────

def report_auc(y_true, y_probs, label=""):
    """Compute and print AUC + Gini for a single dataset."""
    auc = roc_auc_score(y_true, y_probs)
    gini = 2 * auc - 1
    if label:
        print(f"  {label:5s} AUC: {auc:.4f} | Gini: {gini:.4f}")
    return auc, gini


def report_auc_all(y_train, probs_train, y_val, probs_val, y_oot, probs_oot, model_name="Model"):
    """Print AUC/Gini for train, val, and OOT in one table."""
    print(f"\n--- {model_name} Performance ---")
    auc_train, _ = report_auc(y_train, probs_train, "Train")
    auc_val, _ = report_auc(y_val, probs_val, "Val")
    auc_oot, _ = report_auc(y_oot, probs_oot, "OOT")
    return {'train': auc_train, 'val': auc_val, 'oot': auc_oot}


def model_comparison_table(lr_aucs, xgb_aucs, lr_cv_scores, xgb_cv_scores):
    """Build a side-by-side comparison DataFrame for two models."""
    return pd.DataFrame({
        'Model': ['Logistic Regression (WoE)', 'XGBoost (Tuned)'],
        'Train AUC': [lr_aucs['train'], xgb_aucs['train']],
        'Val AUC':   [lr_aucs['val'], xgb_aucs['val']],
        'OOT AUC':   [lr_aucs['oot'], xgb_aucs['oot']],
        'Train Gini': [2*lr_aucs['train']-1, 2*xgb_aucs['train']-1],
        'Val Gini':   [2*lr_aucs['val']-1, 2*xgb_aucs['val']-1],
        'OOT Gini':   [2*lr_aucs['oot']-1, 2*xgb_aucs['oot']-1],
        'CV AUC (mean)': [lr_cv_scores.mean(), xgb_cv_scores.mean()],
        'CV AUC (std)':  [lr_cv_scores.std(), xgb_cv_scores.std()],
    })


# ─────────────────────────────────────────────────────────────────────
# ROC CURVES
# ─────────────────────────────────────────────────────────────────────

def plot_roc_comparison(y_oot, lr_probs, xgb_probs, lr_auc, xgb_auc):
    """Plot side-by-side ROC curves for LR and XGBoost on OOT data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fpr_lr, tpr_lr, _ = roc_curve(y_oot, lr_probs)
    axes[0].plot(fpr_lr, tpr_lr, 'b-', label=f'LR (AUC={lr_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title('Logistic Regression — OOT ROC')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

    fpr_xgb, tpr_xgb, _ = roc_curve(y_oot, xgb_probs)
    axes[1].plot(fpr_xgb, tpr_xgb, 'r-', label=f'XGB (AUC={xgb_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_title('XGBoost (Baseline) — OOT ROC')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────

def fit_platt_scaling(model, X_val, y_val):
    """Fit Platt scaling (logistic recalibration) on validation set.

    Platt scaling fits a sigmoid on top of raw model scores to correct
    systematic over/under-confidence from class_weight or scale_pos_weight.

    Returns:
        calibrator: fitted LogisticRegression (maps score → calibrated PD)
    """
    raw_probs = model.predict_proba(X_val)[:, 1]
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(raw_probs.reshape(-1, 1), y_val)
    return calibrator


def fit_isotonic(model, X_val, y_val):
    """Fit isotonic regression calibration on validation set.

    Non-parametric alternative to Platt — more flexible but needs more data.

    Returns:
        calibrator: fitted IsotonicRegression
    """
    raw_probs = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, y_val)
    return calibrator


def apply_platt(calibrator, raw_probs):
    """Apply a fitted Platt calibrator to raw probabilities."""
    return calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]


def apply_isotonic(calibrator, raw_probs):
    """Apply a fitted isotonic calibrator to raw probabilities."""
    return calibrator.predict(raw_probs)


# ─────────────────────────────────────────────────────────────────────
# HOSMER-LEMESHOW TEST
# ─────────────────────────────────────────────────────────────────────

def hosmer_lemeshow_test(y_true, y_pred, g=10):
    """Hosmer-Lemeshow goodness-of-fit test for calibration.

    Groups predicted PDs into g deciles, computes observed vs expected
    defaults, and tests whether differences are statistically significant.

    H0: Model is well-calibrated (observed ≈ expected)
    p < 0.05 → reject → model is miscalibrated
    p >= 0.05 → fail to reject → calibration acceptable

    Returns:
        hl_stat: chi-squared test statistic
        p_value: p-value
        summary: decile-level DataFrame
    """
    data = pd.DataFrame({'y_true': y_true.values, 'y_pred': y_pred})
    data['decile'] = pd.qcut(data['y_pred'], q=g, duplicates='drop')

    summary = data.groupby('decile', observed=True).agg(
        n=('y_true', 'count'),
        obs_default=('y_true', 'sum'),
        mean_pred_pd=('y_pred', 'mean')
    )
    summary['obs_non_default'] = summary['n'] - summary['obs_default']
    summary['exp_default'] = summary['n'] * summary['mean_pred_pd']
    summary['exp_non_default'] = summary['n'] * (1 - summary['mean_pred_pd'])
    summary['obs_default_rate'] = summary['obs_default'] / summary['n']

    hl_stat = (
        ((summary['obs_default'] - summary['exp_default'])**2 / summary['exp_default']) +
        ((summary['obs_non_default'] - summary['exp_non_default'])**2 / summary['exp_non_default'])
    ).sum()

    p_value = 1 - chi2.cdf(hl_stat, df=len(summary) - 2)
    return hl_stat, p_value, summary


# ─────────────────────────────────────────────────────────────────────
# CALIBRATION PLOTS
# ─────────────────────────────────────────────────────────────────────

def plot_calibration_raw(y_oot, lr_probs, xgb_probs):
    """Pre-calibration reliability plots for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, probs, name, color in [
        (axes[0], lr_probs, 'Logistic Regression', 'blue'),
        (axes[1], xgb_probs, 'XGBoost', 'red')
    ]:
        frac, mean_pred = calibration_curve(y_oot, probs, n_bins=10, strategy='quantile')
        ax.plot(mean_pred, frac, 's-', color=color, label=f'{name} (raw)')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
        ax.set_xlabel('Mean Predicted PD')
        ax.set_ylabel('Observed Default Rate')
        ax.set_title(f'{name} — Calibration (Pre-calibration)')
        ax.legend(loc='upper left')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig


def plot_calibration_comparison(y_oot, lr_probs_dict, xgb_probs_dict):
    """Post-calibration comparison: raw vs Platt vs isotonic for both models.

    Args:
        lr_probs_dict: {'raw': array, 'platt': array, 'isotonic': array}
        xgb_probs_dict: same structure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # LR calibration variants
    ax = axes[0, 0]
    for key, style in [('raw', 'b--'), ('platt', 'b-'), ('isotonic', 'b:')]:
        frac, mean_pred = calibration_curve(y_oot, lr_probs_dict[key], n_bins=10, strategy='quantile')
        ax.plot(mean_pred, frac, style, label=key.capitalize(), linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_title('LR — Calibration Comparison')
    ax.set_xlabel('Mean Predicted PD')
    ax.set_ylabel('Observed Default Rate')
    ax.legend()

    # XGB calibration variants
    ax = axes[0, 1]
    for key, style in [('raw', 'r--'), ('platt', 'r-'), ('isotonic', 'r:')]:
        frac, mean_pred = calibration_curve(y_oot, xgb_probs_dict[key], n_bins=10, strategy='quantile')
        ax.plot(mean_pred, frac, style, label=key.capitalize(), linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_title('XGB — Calibration Comparison')
    ax.set_xlabel('Mean Predicted PD')
    ax.set_ylabel('Observed Default Rate')
    ax.legend()

    # PD distributions
    for ax, probs, title, colors in [
        (axes[1, 0], lr_probs_dict.get('platt', lr_probs_dict['raw']),
         'LR (Platt) — PD Distribution', ('green', 'red')),
        (axes[1, 1], xgb_probs_dict.get('platt', xgb_probs_dict['raw']),
         'XGB (Platt) — PD Distribution', ('green', 'red')),
    ]:
        ax.hist(probs[y_oot == 0], bins=50, alpha=0.5, label='Non-default', density=True, color=colors[0])
        ax.hist(probs[y_oot == 1], bins=50, alpha=0.5, label='Default', density=True, color=colors[1])
        ax.set_title(title)
        ax.set_xlabel('Calibrated PD')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────

def lr_coefficient_table(model, feature_names):
    """Extract LR coefficients sorted by absolute magnitude."""
    return pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)


def xgb_importance_table(model, feature_names):
    """Extract XGBoost gain-based feature importance."""
    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)


def plot_feature_importance(importance_df, title='Feature Importance', top_n=20, color='steelblue'):
    """Horizontal bar chart of top-N features."""
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top.iloc[:, 1].values, color=color)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.iloc[:, 0].values)
    ax.invert_yaxis()
    ax.set_xlabel(top.columns[1])
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
# MODEL SELECTION LOGIC
# ─────────────────────────────────────────────────────────────────────

def select_model(lr_auc_oot, xgb_auc_oot, threshold=0.02):
    """Select between LR and XGBoost based on OOT AUC gap.

    If XGBoost OOT advantage > threshold: choose XGBoost.
    Otherwise: choose LR (interpretability, regulatory compliance).
    """
    gap = xgb_auc_oot - lr_auc_oot
    if gap > threshold:
        chosen = 'XGBoost (Tuned)'
        print(f"Selected: XGBoost (OOT AUC advantage of {gap:.4f})")
    else:
        chosen = 'Logistic Regression (WoE)'
        print(f"Selected: Logistic Regression (OOT AUC gap only {gap:.4f})")
        print("  → LR preferred: interpretable, regulatory-friendly, simpler deployment")
    return chosen
