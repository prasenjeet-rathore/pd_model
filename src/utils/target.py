"""
target.py — Target variable construction and validation.

Business logic:
  A loan is considered "defaulted within 12 months" if lender marked it
  as defaulted (DefaultDate is not null) AND the time between loan issuance
  and default is within the observation horizon (365 days).

  The modelling universe is restricted to loans issued before the cutoff
  date (snapshot_date - horizon_days) so every loan has had a full 12-month
  window to potentially default.
"""

import pandas as pd
import numpy as np


def build_modelling_universe(df, snapshot_date, horizon_days=365):
    """Filter to loans that have a full observation window.

    Only loans issued at least `horizon_days` before the snapshot can be
    labelled — newer loans haven't had time to default yet.

    Returns:
        df_model: filtered DataFrame (copy)
        cutoff_date: the computed cutoff
    """
    cutoff_date = snapshot_date - pd.DateOffset(days=horizon_days)
    df_model = df[df['LoanDate'] <= cutoff_date].copy()
    print(f"Modelling universe: {len(df_model):,} loans")
    print(f"  LoanDate range: {df_model['LoanDate'].min().date()} → {df_model['LoanDate'].max().date()}")
    print(f"  Cutoff date: {cutoff_date.date()} (snapshot {snapshot_date.date()} - {horizon_days}d)")
    return df_model, cutoff_date


def create_default_target(df, horizon_days=365, target_col='into_12m_default_ind'):
    """Construct binary target: 1 if loan defaulted within `horizon_days` of origination.

    Creates two columns:
      - days_to_default: days between LoanDate and DefaultDate
      - target_col: binary indicator (1 = default within horizon)

    Returns:
        df with new columns added in-place
    """
    df['days_to_default'] = (df['DefaultDate'] - df['LoanDate']).dt.days
    df[target_col] = (
        (df['DefaultDate'].notna()) &
        (df['days_to_default'] >= 0) &
        (df['days_to_default'] <= horizon_days)
    ).astype(int)

    n_pos = df[target_col].sum()
    n_total = len(df)
    print(f"Target '{target_col}' created:")
    print(f"  Default (1): {n_pos:,} ({n_pos/n_total:.2%})")
    print(f"  Non-default (0): {n_total - n_pos:,} ({(n_total - n_pos)/n_total:.2%})")
    return df


def validate_target(df, horizon_days=365, target_col='into_12m_default_ind'):
    """Run sanity checks on the constructed target variable.

    Checks:
      1. No impossible combination (DefaultDate is null but target = 1)
      2. All target=1 loans have days_to_default in [0, horizon_days]
      3. No DefaultDate before LoanDate (negative days_to_default)

    Returns:
        True if all checks pass, False otherwise
    """
    errors = []

    # Check 1: No impossible default_ind=1 when DefaultDate is null
    impossible = ((df['DefaultDate'].isna()) & (df[target_col] == 1)).sum()
    status = 'PASS' if impossible == 0 else 'FAIL'
    print(f"[1] No null DefaultDate with target=1 : {status}")
    if impossible > 0:
        errors.append("Impossible: null DefaultDate but target=1")

    # Check 2: All target=1 within [0, horizon_days]
    pos = df[df[target_col] == 1]['days_to_default']
    in_range = pos.between(0, horizon_days).all() if len(pos) > 0 else True
    status = 'PASS' if in_range else 'FAIL'
    print(f"[2] All target=1 within [0, {horizon_days}] days : {status}")
    if not in_range:
        errors.append("Out of range days_to_default for target=1")

    # Check 3: No DefaultDate before LoanDate
    neg_days = (df['days_to_default'].dropna() < 0).sum()
    status = 'PASS' if neg_days == 0 else 'FAIL'
    print(f"[3] No DefaultDate before LoanDate    : {status}")
    if neg_days > 0:
        errors.append(f"{neg_days} loans with DefaultDate before LoanDate")

    if errors:
        print(f"\n⚠ {len(errors)} validation error(s): {errors}")
        return False
    print("\n✓ All target validation checks passed")
    return True
