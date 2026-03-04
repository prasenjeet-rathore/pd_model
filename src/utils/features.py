"""
features.py — Feature engineering, leakage screening, and temporal splitting.


Design:
  - engineer_features() creates derived features (loan_to_income, approval_ratio)
  - drop_leakage() removes all known post-origination columns
  - correlation_analysis() identifies highly correlated pairs
  - temporal_split() splits data by LoanDate into train/val/OOT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────

def engineer_features(df):
    """Create derived features with business meaning.

    Features:
      - loan_to_income: How large is the loan vs monthly income?
        High values → borrower is stretching beyond their means.
      - approval_ratio: Did lender cut the requested amount?
        Ratio < 1 means lender reduced the loan → risk signal.

    Args:
        df: DataFrame (modified in-place)
    Returns:
        df with new columns
    """
    df['loan_to_income'] = df['Amount'] / df['IncomeTotal'].replace(0, np.nan)
    df['approval_ratio'] = df['Amount'] / df['AppliedAmount'].replace(0, np.nan)

    print("Engineered features: loan_to_income, approval_ratio")
    return df


# ─────────────────────────────────────────────────────────────────────
# LEAKAGE REMOVAL
# ─────────────────────────────────────────────────────────────────────

def drop_leakage(df, leakage_columns):
    """Remove all known leakage/meta columns. Safe to call even if some don't exist."""
    to_drop = [c for c in leakage_columns if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    print(f"Dropped {len(to_drop)} leakage/meta columns")
    return df


# ─────────────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def find_correlated_pairs(df, feature_cols, threshold=0.7):
    """Find pairs of numeric features with |correlation| > threshold.

    Returns:
        DataFrame with columns: Feature 1, Feature 2, Correlation
    """
    cont = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[cont].corr()

    pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                pairs.append({
                    'Feature 1': corr_matrix.index[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': round(r, 4)
                })

    pair_df = pd.DataFrame(pairs).sort_values('Correlation', key=abs, ascending=False)
    print(f"\nHighly correlated pairs (|r| > {threshold}): {len(pair_df)}")
    if len(pair_df) > 0:
        print(pair_df.to_string(index=False))
    return pair_df


def plot_correlation_heatmap(df, feature_cols, top_n=20, figsize=(14, 11)):
    """Plot correlation heatmap for the top-N features by variance."""
    cont = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    top_var = df[cont].var().nlargest(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df[top_var].corr(), annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax, annot_kws={'size': 7})
    ax.set_title(f'Pearson Correlation — Top {top_n} Features (by Variance)')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
# TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────

def temporal_split(df, loan_dates, target_col, feature_cols, train_end, val_end):
    """Split data into Train / Validation / OOT by LoanDate.

    Business rationale:
      - Train on older loans, validate on recent ones, test on newest.
      - This mimics production: the model only sees past data at scoring time.
      - Temporal split avoids random leakage where a borrower's later loan
        appears in train while their earlier loan is in test.

    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_oot, y_oot
    """
    train_mask = loan_dates <= train_end
    val_mask = (loan_dates > train_end) & (loan_dates <= val_end)
    oot_mask = loan_dates > val_end

    # Ensure feature_cols only includes columns actually present
    feature_cols = [c for c in feature_cols if c in df.columns]

    splits = {
        'X_train': df.loc[train_mask, feature_cols].copy(),
        'y_train': df.loc[train_mask, target_col].copy(),
        'X_val':   df.loc[val_mask, feature_cols].copy(),
        'y_val':   df.loc[val_mask, target_col].copy(),
        'X_oot':   df.loc[oot_mask, feature_cols].copy(),
        'y_oot':   df.loc[oot_mask, target_col].copy(),
    }

    for name in ['train', 'val', 'oot']:
        y = splits[f'y_{name}']
        print(f"  {name.upper():5s}: {len(y):>8,}  |  Default rate: {y.mean():.2%}")

    return splits




from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """Cramér's V between two categorical Series (NaNs dropped)."""
    mask = x.notna() & y.notna()
    ct = pd.crosstab(x[mask], y[mask])
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    # Bias-corrected formula (Bergsma & Wicher, 2013)
    phi2 = max(0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    return np.sqrt(phi2 / denom) if denom > 0 else 0.0