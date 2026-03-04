"""
data_cleaning.py — Reusable data cleaning functions for the PD pipeline.

Used in: Notebook 1 (Data Preparation), will be also useful in production pipeline

Functions handle: loading, type detection, 
sentinel replacement which are values that are -1 categorical columns which should Nan, 
missingness analysis, leakage detection, and zero-variance checks.
"""

import re
import numpy as np
import pandas as pd
from tabulate import tabulate


# ─────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────

def load_raw_data(filepath, sep=';'):
    """Load the loan data CSV and print basic shape info."""
    df = pd.read_csv(filepath, sep=sep, low_memory=False)
    print(f"Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────
# COLUMN-LEVEL CLEANING
# ─────────────────────────────────────────────────────────────────────

def drop_empty_columns(df):
    """Drop columns where every value is null."""
    empty_cols = df.columns[df.isnull().all()].tolist()
    df.drop(columns=empty_cols, inplace=True)
    print(f"Dropped {len(empty_cols)} fully empty columns: {empty_cols}")
    return df, empty_cols


def drop_constant_columns(df, protect=None):
    """Drop columns with only 1 unique value (zero variance).

    Args:
        df: DataFrame
        protect: list of column names to keep even if constant (e.g. 'ReportAsOfEOD')
    """
    protect = protect or []
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    to_drop = [c for c in constant_cols if c not in protect]
    df.drop(columns=to_drop, inplace=True)
    print(f"Zero-variance columns ({len(constant_cols)} found, {len(to_drop)} dropped): {to_drop}")
    if protect:
        kept = [c for c in constant_cols if c in protect]
        if kept:
            print(f"  Protected (kept): {kept}")
    return df, constant_cols


def detect_date_columns(df):
    """Auto-detect object columns containing date-like strings (YYYY-MM-DD).
    Returns list of column names that look like dates.
    """
    date_regex = r'\d{4}[-/]\d{2}[-/]\d{2}'
    date_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
        if re.search(date_regex, str(first_val)):
            date_cols.append(col)
    print(f"Detected {len(date_cols)} date columns: {date_cols}")
    return date_cols


def convert_date_columns(df, date_cols):
    """Convert detected date columns to datetime dtype."""
    # Show dtypes BEFORE conversion
    print('--- Before conversion ---')
    print(df[date_cols].dtypes)

    # Convert
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Show dtypes AFTER conversion
    print('\n--- After conversion ---')
    print(df[date_cols].dtypes)
    return df


# ─────────────────────────────────────────────────────────────────────
# SENTINEL VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────

def find_sentinel_columns(df, sentinel=-1):
    """Find columns containing a sentinel value (e.g. -1 used for missing)."""
    cols = df.columns[(df == sentinel).any()].tolist()
    if cols:
        counts = (df[cols] == sentinel).sum().sort_values(ascending=False)
        print(f"Found {len(cols)} columns with sentinel value {sentinel}:")
        print(counts.to_string())
    return cols


def replace_sentinel_with_nan(df, columns, sentinel=-1, exclude=None):
    """Replace sentinel values with NaN.

    Args:
        exclude: list of columns to skip (e.g. 'FreeCash' where -1 is valid)
    """
    exclude = exclude or []
    cols_to_fix = [c for c in columns if c not in exclude]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(sentinel, np.nan)
    print(f"Replaced {sentinel} → NaN in {len(cols_to_fix)} columns (excluded: {exclude})")
    return df


# ─────────────────────────────────────────────────────────────────────
# MISSINGNESS ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def missingness_summary(df, title="Missingness Analysis"):
    """Bucket features by their missing percentage.
    Returns: (summary_table, missing_pct_series)
    """
    missing_pct = df.isnull().mean() * 100
    bins = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0%', '0.1-10%', '10-20%', '20-30%', '30-40%',
              '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    buckets = pd.cut(missing_pct, bins=bins, labels=labels, right=True)
    summary = buckets.value_counts().sort_index().reset_index()
    summary.columns = ['Missingness Range', 'Number of Features']

    high_miss90 = missing_pct[(missing_pct > 90)].sort_values(ascending=True).index.tolist()


    print(f"\n### {title} ###")
    print(tabulate(summary, headers='keys', tablefmt='psql', showindex=False))
    print(f"\nFeatures with >90% missingness : {high_miss90}")
    return summary, missing_pct


def drop_high_missing(df, missing_pct, threshold_pct=90):
    """Drop columns exceeding the missingness threshold."""
    high_miss = missing_pct[missing_pct > threshold_pct].index.tolist()
    df.drop(columns=[c for c in high_miss if c in df.columns], inplace=True)
    print(f"Dropped {len(high_miss)} columns with >{threshold_pct}% missing: {high_miss}")
    return df, high_miss


# ─────────────────────────────────────────────────────────────────────
# LEAKAGE DETECTION
# ─────────────────────────────────────────────────────────────────────

def find_post_default_numeric(df, default_col='DefaultDate'):
    """Find numeric columns that are 100% empty for non-defaulted loans
    but populated for defaulted ones → leakage proxy.
    """
    is_healthy = df[default_col].isna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    proxy_cols = []
    for col in numeric_cols:
        empty_for_healthy = df.loc[is_healthy, col].isnull().all()
        has_data_for_defaulted = df.loc[~is_healthy, col].notnull().any()
        if empty_for_healthy and has_data_for_defaulted:
            proxy_cols.append(col)
    print(f"Post-default numeric proxies (empty when healthy): {proxy_cols}")
    return proxy_cols


def find_post_default_categorical(df, default_col='DefaultDate'):
    """Find categorical columns only populated for defaulted loans."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    null_at_t0 = [
        col for col in cat_cols
        if df.loc[df[default_col].isna(), col].isnull().all()
        and df.loc[df[default_col].notna(), col].notnull().any()
    ]
    print(f"Post-default categorical proxies: {null_at_t0}")
    return null_at_t0


def find_future_date_columns(df, reference_col='LoanDate'):
    """Find date columns where any value is strictly after the reference date."""
    dt_cols = df.select_dtypes(include=['datetime64']).columns
    future_cols = [
        col for col in dt_cols
        if col != reference_col and (df[col] > df[reference_col]).any()
    ]
    print(f"Future date columns (post-origination events): {future_cols}")
    return future_cols


# ─────────────────────────────────────────────────────────────────────
# DATA QUALITY TESTS  (from the test suite in data_preparation.py)
# ─────────────────────────────────────────────────────────────────────

def test_unique_ratio(df, threshold=0.90):
    """Identify potential identifiers (columns with very high cardinality)."""
    n_rows = len(df)
    results = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        ratio = n_unique / n_rows if n_rows > 0 else 0
        results.append({'column': col, 'n_unique': n_unique, 'unique_ratio': ratio})
    result_df = pd.DataFrame(results).sort_values('unique_ratio', ascending=False)
    high = result_df[result_df['unique_ratio'] > threshold]
    print(f"\nColumns with unique_ratio > {threshold} (potential identifiers):")
    print(high.to_string(index=False))
    return result_df


def test_near_zero_variance(df, dominant_pct_threshold=95):
    """Identify features where a single value dominates >threshold%."""
    results = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique == 0:
            dominant_pct = 100.0
        else:
            dominant_pct = df[col].value_counts(normalize=True, dropna=False).iloc[0] * 100
        results.append({'column': col, 'n_unique': n_unique, 'dominant_value_pct': round(dominant_pct, 1)})
    result_df = pd.DataFrame(results).sort_values('dominant_value_pct', ascending=False)
    near_zero = result_df[result_df['dominant_value_pct'] > dominant_pct_threshold]
    print(f"\nColumns with dominant value >{dominant_pct_threshold}% (near-zero variance):")
    print(near_zero.to_string(index=False))
    return result_df


def test_temporal_leak(df, loan_date_col='LoanDate'):
    """Check which date columns have values after LoanDate (post-origination leak)."""
    dt_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    loan_date = df[loan_date_col]
    results = []
    for col in dt_cols:
        if col == loan_date_col:
            continue
        mask = df[col].notna() & loan_date.notna()
        if mask.sum() == 0:
            pct_after = np.nan
        else:
            pct_after = ((df.loc[mask, col] > loan_date[mask]).sum() / mask.sum()) * 100
        diffs = (df[col] - loan_date).dt.days
        median_diff = diffs.median()
        results.append({
            'column': col,
            'pct_after_LoanDate': round(pct_after, 1) if not np.isnan(pct_after) else 'N/A',
            'median_days_from_loan': round(median_diff, 0) if not np.isnan(median_diff) else 'N/A',
            'non_null_pct': round(df[col].notna().mean() * 100, 1)
        })
    result_df = pd.DataFrame(results).sort_values('column')
    print("\nTemporal leak check (date columns vs LoanDate):")
    print(result_df.to_string(index=False))
    return result_df


def test_structural_missingness(df, target_col, exclude_cols=None):
    """Check if fill rate differs dramatically between default/non-default.
    A >30pp difference means the column's missingness leaks the target.
    """
    exclude_cols = exclude_cols or ['DefaultDate', 'days_to_default', target_col]
    results = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        fill_default = df.loc[df[target_col] == 1, col].notna().mean()
        fill_no_default = df.loc[df[target_col] == 0, col].notna().mean()
        diff = abs(fill_default - fill_no_default)
        if diff > 0.30:
            results.append({
                'column': col,
                'fill_rate_default': f"{fill_default:.1%}",
                'fill_rate_no_default': f"{fill_no_default:.1%}",
                'absolute_diff': f"{diff:.1%}"
            })
    result_df = pd.DataFrame(results).sort_values('absolute_diff', ascending=False)
    print("\nStructural missingness leak (>30pp fill rate gap):")
    print(result_df.to_string(index=False) if len(result_df) > 0 else "  None found.")
    return result_df
