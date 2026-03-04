"""
woe.py — Weight of Evidence (WoE) and Information Value (IV) utilities.

WoE encoding is the industry standard for logistic regression credit scorecards.
It transforms each feature into a single numeric score that represents its
log-odds relationship with the target. Benefits:
  - Handles missing values and outliers naturally
  - Monotonic with log-odds → coefficients are directly interpretable
  - IV provides a built-in feature importance metric
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier


# ─────────────────────────────────────────────────────────────────────
# BINNING STRATEGIES
# ─────────────────────────────────────────────────────────────────────

def tree_bin_edges(series, target, max_bins=5, min_samples_leaf=0.05):
    """Find optimal bin edges using a decision tree.

    Fits a shallow tree on a single feature vs the target. The tree's
    split thresholds become bin boundaries — they're chosen to maximize
    information gain (i.e. separation between default / non-default).

    Why this is better than quantile binning:
      - Supervised: bins are optimized for the target, not just equal-frequency
      - Handles skewed distributions naturally
      - Produces bins where the WoE pattern is already close to monotonic

    Args:
        series: continuous feature values
        target: binary target (1 = default)
        max_bins: maximum number of bins (= max_leaf_nodes in the tree)
        min_samples_leaf: minimum fraction of samples per bin (default 5%)

    Returns:
        bin_edges: sorted array with -inf and +inf at boundaries
    """
    # Drop NaN for tree fitting
    mask = series.notna() & target.notna()
    X_tree = series[mask].values.reshape(-1, 1)
    y_tree = target[mask].values

    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    tree.fit(X_tree, y_tree)

    # Extract unique thresholds (internal split points)
    thresholds = tree.tree_.threshold[tree.tree_.feature != -2]  # -2 = leaf node
    thresholds = np.sort(np.unique(thresholds))

    # Build edges: -inf → thresholds → +inf
    bin_edges = np.concatenate([[-np.inf], thresholds, [np.inf]])
    return bin_edges


def compute_woe_with_edges(train_series, target_series, bin_edges):
    """Compute WoE using pre-defined bin edges (from tree or manual).

    Same WoE math as compute_woe_continuous, but uses your bin edges
    instead of quantile binning. Works with edges from tree_bin_edges()
    or any custom boundaries.

    Returns:
        woe_map: dict mapping bin intervals → WoE values
        iv: Information Value
        bin_edges: the edges passed in (returned for consistency with fit_woe)
    """
    data = pd.DataFrame({'feature': train_series, 'target': target_series}).dropna(subset=['feature'])
    data['bin'] = pd.cut(data['feature'], bins=bin_edges, include_lowest=True)

    grouped = data.groupby('bin', observed=True)['target'].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    grouped['good_pct'] = (grouped['good'] + 0.5) / (total_good + 0.5 * len(grouped))
    grouped['bad_pct'] = (grouped['bad'] + 0.5) / (total_bad + 0.5 * len(grouped))
    # Standard Industry Formula: ln(Bad % / Good %)
    grouped['woe'] = np.log(grouped['bad_pct'] / grouped['good_pct'])
    grouped['iv'] = (grouped['bad_pct'] - grouped['good_pct']) * grouped['woe']

    iv = grouped['iv'].sum()
    woe_map = grouped['woe'].to_dict()
    return woe_map, iv, bin_edges

# ─────────────────────────────────────────────────────────────────────
# WOE COMPUTATION (fit on train only)
# ─────────────────────────────────────────────────────────────────────


def compute_woe_continuous(train_series, target_series, n_bins=10):
    """Compute WoE for a continuous variable using quantile binning.

    Business intuition: We split the variable into equal-frequency bins
    and measure how much each bin shifts the odds of default relative to
    the population average. Bins with more defaults get negative WoE.

    Args:
        train_series: feature values (training set only)
        target_series: binary target (1=default)
        n_bins: number of quantile bins

    Returns:
        woe_map: dict mapping bin intervals → WoE values
        iv: Information Value (sum of bin-level IVs)
        bin_edges: array of bin edges for applying to new data
    """
    data = pd.DataFrame({'feature': train_series, 'target': target_series}).dropna(subset=['feature'])

    try:
        data['bin'], bin_edges = pd.qcut(data['feature'], q=n_bins,
                                          duplicates='drop', retbins=True)
    except ValueError:
        data['bin'], bin_edges = pd.qcut(data['feature'], q=5,
                                          duplicates='drop', retbins=True)

    grouped = data.groupby('bin', observed=True)['target'].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    # Laplace smoothing to avoid log(0)
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    grouped['good_pct'] = (grouped['good'] + 0.5) / (total_good + 0.5 * len(grouped))
    grouped['bad_pct'] = (grouped['bad'] + 0.5) / (total_bad + 0.5 * len(grouped))
    # Standard Industry Formula: ln(Bad % / Good %)
    grouped['woe'] = np.log(grouped['bad_pct'] / grouped['good_pct'])
    grouped['iv'] = (grouped['bad_pct'] - grouped['good_pct']) * grouped['woe']

    iv = grouped['iv'].sum()
    woe_map = grouped['woe'].to_dict()
    return woe_map, iv, bin_edges


def compute_woe_categorical(train_series, target_series):
    """Compute WoE for a categorical variable.

    Each category level gets its own WoE value based on its default rate
    relative to the population.

    Returns:
        woe_map: dict mapping category label → WoE
        iv: Information Value
    """
    data = pd.DataFrame({'feature': train_series.astype(str), 'target': target_series})
    grouped = data.groupby('feature')['target'].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    grouped['good_pct'] = (grouped['good'] + 0.5) / (total_good + 0.5 * len(grouped))
    grouped['bad_pct'] = (grouped['bad'] + 0.5) / (total_bad + 0.5 * len(grouped))
    grouped['woe'] = np.log(grouped['good_pct'] / grouped['bad_pct'])
    grouped['iv'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']

    iv = grouped['iv'].sum()
    woe_map = grouped['woe'].to_dict()
    return woe_map, iv


# ─────────────────────────────────────────────────────────────────────
# WOE APPLICATION (apply to any dataset using stored rules)
# ─────────────────────────────────────────────────────────────────────

def apply_woe_continuous(series, bin_edges, woe_map):
    """Apply pre-computed WoE bins to new data.

    Uses positional mapping: bin 0 → first WoE value, bin 1 → second, etc.
    Unseen values (outside bin range) get WoE = 0 (neutral).
    """
    edges = list(bin_edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    binned = pd.cut(series, bins=edges, labels=list(range(len(edges) - 1)),
                     include_lowest=True)
    positional_woe = {i: woe for i, (_, woe) in enumerate(woe_map.items())}
    return binned.map(positional_woe).astype(float).fillna(0.0)


def apply_woe_categorical(series, woe_map):
    """Apply pre-computed categorical WoE to new data. Unknown levels get WoE=0."""
    return series.astype(str).map(woe_map).fillna(0).astype(float)


# ─────────────────────────────────────────────────────────────────────
# FULL WOE PIPELINE
# ─────────────────────────────────────────────────────────────────────

def fit_woe(X_train, y_train, continuous_cols, categorical_cols, n_bins=10, binning='quantile'):
    """Fit WoE rules on the training set for all features.

    Args:
        binning: 'quantile' (default) or 'tree'

    Returns:
        woe_rules: dict[col_name → {type, woe_map, bin_edges (if continuous)}]
        iv_df: DataFrame with IV per feature, sorted descending
    """
    continuous_cols = [c for c in continuous_cols if c in X_train.columns]
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]

    woe_rules = {}
    iv_table = []

    print(f"\n--- Computing WoE / IV on Training Set (Binning: {binning}) ---")
    
    for col in continuous_cols:
        if binning == 'tree':
            # 1. Find optimal edges using the tree
            edges = tree_bin_edges(X_train[col], y_train, max_bins=n_bins)
            # 2. Calculate WoE using those specific edges
            woe_map, iv, bin_edges = compute_woe_with_edges(X_train[col], y_train, edges)
        else:
            # Fallback to standard quantile binning
            woe_map, iv, bin_edges = compute_woe_continuous(X_train[col], y_train, n_bins)
            
        woe_rules[col] = {'type': 'continuous', 'woe_map': woe_map, 'bin_edges': bin_edges}
        iv_table.append({'Variable': col, 'IV': iv, 'Type': 'Continuous'})
        print(f"  {col:40s} IV = {iv:.4f}")

    for col in categorical_cols:
        woe_map, iv = compute_woe_categorical(X_train[col], y_train)
        woe_rules[col] = {'type': 'categorical', 'woe_map': woe_map}
        iv_table.append({'Variable': col, 'IV': iv, 'Type': 'Categorical'})
        print(f"  {col:40s} IV = {iv:.4f}")

    iv_df = pd.DataFrame(iv_table).sort_values('IV', ascending=False)
    return woe_rules, iv_df


def select_by_iv(iv_df, threshold=0.02):
    """Select features whose IV exceeds the threshold.

    IV interpretation:
      < 0.02  → not useful (drop)
      0.02-0.1 → weak predictor
      0.1-0.3  → medium predictor
      > 0.3    → strong predictor (check for leakage if > 0.5)

    Returns:
        list of selected variable names
    """
    selected = iv_df[iv_df['IV'] >= threshold]['Variable'].tolist()
    print(f"\nSelected features (IV >= {threshold}): {len(selected)} of {len(iv_df)}")
    return selected


def transform_woe(df, woe_rules, selected_vars):
    """Apply WoE transformation to a DataFrame using pre-computed rules.

    This is the function used in both notebooks AND production scoring.
    It takes raw features and returns WoE-encoded features.

    Args:
        df: raw feature DataFrame
        woe_rules: fitted rules from fit_woe()
        selected_vars: list of variable names to transform

    Returns:
        DataFrame with columns named {original}_woe
    """
    df_woe = pd.DataFrame(index=df.index)
    for col in selected_vars:
        if col not in woe_rules:
            continue
        rule = woe_rules[col]
        if rule['type'] == 'continuous':
            df_woe[f'{col}_woe'] = apply_woe_continuous(
                df[col], rule['bin_edges'], rule['woe_map']
            )
        else:
            df_woe[f'{col}_woe'] = apply_woe_categorical(df[col], rule['woe_map'])
    return df_woe


# ─────────────────────────────────────────────────────────────────────
# VIF CHECK
# ─────────────────────────────────────────────────────────────────────

def check_vif(df_woe, threshold=5.0):
    """Compute Variance Inflation Factor for WoE features.

    VIF > 5-10 indicates multicollinearity that inflates LR coefficient
    standard errors and makes them unreliable.

    Returns:
        vif_df: DataFrame with Feature, VIF columns
        high_vif_cols: list of columns exceeding threshold
    """
    X = df_woe.replace([np.inf, -np.inf], np.nan).fillna(0)
    vif_data = pd.DataFrame({
        'Feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values('VIF', ascending=False)

    print("\n--- VIF Check on WoE Features ---")
    print(vif_data.to_string(index=False))

    high_vif = vif_data[vif_data['VIF'] > threshold]['Feature'].tolist()
    if high_vif:
        print(f"\n⚠ {len(high_vif)} features with VIF > {threshold}: {high_vif}")
    else:
        print(f"\nAll VIF values < {threshold} — no multicollinearity concerns.")
    return vif_data, high_vif


# ─────────────────────────────────────────────────────────────────────
# WOE DETAIL TABLES (for EDA and reporting)
# ─────────────────────────────────────────────────────────────────────

def woe_detail_table(series, target, variable_name=None):
    """Build a detailed WoE breakdown table for a single variable.

    Useful for EDA: shows per-level counts, proportions, WoE, IV,
    and the difference in WoE/default-rate between adjacent levels.

    Works for both categorical and binned continuous variables.
    For continuous variables, pass in a pre-binned Series (e.g. via pd.qcut).

    Args:
        series: feature values (can be categorical or pre-binned)
        target: binary target Series (1 = default / bad)
        variable_name: optional label for the first column

    Returns:
        DataFrame with columns: level, n_obs, prop_n_obs, n_good, n_bad,
        prop_n_good, prop_n_bad, WoE, IV_bin, IV_total, diff_prop_good, diff_WoE
    """
    name = variable_name or series.name or 'feature'
    df = pd.DataFrame({name: series.astype(str), 'target': target})

    grouped = df.groupby(name)['target'].agg(['count', 'mean'])
    grouped.columns = ['n_obs', 'prop_bad']
    grouped['prop_good'] = 1 - grouped['prop_bad']

    grouped['prop_n_obs'] = grouped['n_obs'] / grouped['n_obs'].sum()
    grouped['n_good'] = grouped['prop_good'] * grouped['n_obs']
    grouped['n_bad'] = grouped['prop_bad'] * grouped['n_obs']

    # Proportions of total goods/bads (with smoothing for zero bins)
    total_good = grouped['n_good'].sum()
    total_bad = grouped['n_bad'].sum()
    grouped['prop_n_good'] = (grouped['n_good'] + 0.5) / (total_good + 0.5 * len(grouped))
    grouped['prop_n_bad'] = (grouped['n_bad'] + 0.5) / (total_bad + 0.5 * len(grouped))

    grouped['WoE'] = np.log(grouped['prop_n_bad'] / grouped['prop_n_good'])
    grouped['IV_bin'] = (grouped['prop_n_bad'] - grouped['prop_n_good']) * grouped['WoE']

    # Sort by WoE for readable output
    grouped = grouped.sort_values('WoE').reset_index()

    # Adjacent-level diffs — useful for spotting where to merge bins
    grouped['diff_prop_good'] = grouped['prop_good'].diff().abs()
    grouped['diff_WoE'] = grouped['WoE'].diff().abs()

    # Total IV as a summary column
    grouped['IV_total'] = grouped['IV_bin'].sum()

    return grouped

def woe_detail_all(X, y, continuous_cols, categorical_cols, n_bins=10, binning='quantile'):
    """Build WoE detail tables for all features at once.

    Returns:
        dict[variable_name → detail DataFrame]
    """
    tables = {}

    for col in continuous_cols:
        if col not in X.columns:
            continue
            
        # 1. Handle the binning strategy
        if binning == 'tree':
            # Use tree to find optimal edges
            edges = tree_bin_edges(X[col], y, max_bins=n_bins)
            # Apply those edges to create bins
            binned = pd.cut(X[col], bins=edges, include_lowest=True)
        else:
            # Fallback to standard quantile binning
            try:
                binned = pd.qcut(X[col], q=n_bins, duplicates='drop')
            except ValueError:
                binned = pd.qcut(X[col], q=5, duplicates='drop')
                
        # 2. Generate the detail table using the binned series
        tables[col] = woe_detail_table(binned, y, variable_name=col)

    for col in categorical_cols:
        if col not in X.columns:
            continue
        tables[col] = woe_detail_table(X[col], y, variable_name=col)

    return tables


# ─────────────────────────────────────────────────────────────────────
# WOE PLOTTING
# ─────────────────────────────────────────────────────────────────────

def plot_woe(detail_df, rotation=0, figsize=(18, 6)):
    """Plot WoE values per level for a single variable.

    Shows the monotonicity (or lack thereof) of the WoE pattern.
    Non-monotonic patterns may indicate the variable needs re-binning
    or that the relationship with default is non-linear.

    Args:
        detail_df: output from woe_detail_table()
        rotation: x-axis label rotation in degrees
        figsize: figure size tuple

    Returns:
        matplotlib Figure
    """
    var_name = detail_df.columns[0]
    x = detail_df[var_name].astype(str).values
    y = detail_df['WoE'].values
    iv_total = detail_df['IV_total'].iloc[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, marker='o', linestyle='--', color='k')
    ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
    ax.set_xlabel(var_name)
    ax.set_ylabel('Weight of Evidence')
    ax.set_title(f'WoE by {var_name}  (IV = {iv_total:.4f})')
    ax.tick_params(axis='x', rotation=rotation)
    plt.tight_layout()
    return fig


def plot_woe_grid(detail_tables, cols_per_row=3, figsize_per_plot=(6, 4)):
    """Plot WoE charts for multiple variables in a grid.

    Args:
        detail_tables: dict from woe_detail_all()
        cols_per_row: number of charts per row
        figsize_per_plot: size of each subplot

    Returns:
        matplotlib Figure
    """
    n = len(detail_tables)
    n_rows = (n + cols_per_row - 1) // cols_per_row
    fig_w = figsize_per_plot[0] * cols_per_row
    fig_h = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(fig_w, fig_h))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (var_name, detail_df) in enumerate(detail_tables.items()):
        ax = axes[idx]
        x = detail_df[var_name].astype(str).values
        y = detail_df['WoE'].values
        iv_total = detail_df['IV_total'].iloc[0]

        ax.plot(x, y, marker='o', linestyle='--', color='k', markersize=4)
        ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
        ax.set_title(f'{var_name}\nIV={iv_total:.4f}', fontsize=9)
        ax.set_ylabel('WoE', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig