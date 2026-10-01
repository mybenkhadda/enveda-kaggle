"""Quantitative train/test distribution-shift metrics.

Deliberately not histograms-by-eye: every function here returns a number a notebook can sort,
threshold, or plot as a bar chart across many columns at once, so "does train look like test"
becomes a table instead of a slideshow.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def numeric_shift(train_values, test_values, name=None):
    """KS statistic + Wasserstein distance between two numeric samples (NaNs dropped from
    each side independently). Wasserstein is in the column's own units, so it's only
    comparable across columns that share units -- KS (bounded in [0, 1]) is the one to sort
    heterogeneous columns by.

    Returns a dict: {name, n_train, n_test, ks_stat, ks_pvalue, wasserstein}.
    """
    train_values = pd.Series(train_values).dropna().to_numpy(dtype=float)
    test_values = pd.Series(test_values).dropna().to_numpy(dtype=float)
    if len(train_values) == 0 or len(test_values) == 0:
        return {"name": name, "n_train": len(train_values), "n_test": len(test_values),
                "ks_stat": float("nan"), "ks_pvalue": float("nan"), "wasserstein": float("nan")}
    ks = ks_2samp(train_values, test_values)
    return {
        "name": name,
        "n_train": len(train_values),
        "n_test": len(test_values),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(wasserstein_distance(train_values, test_values)),
    }


def categorical_shift(train_values, test_values, name=None):
    """Jensen-Shannon distance (base 2, so it's in [0, 1]) between the category-frequency
    distributions of two samples, over the union of categories seen in either. A category
    present in only one split contributes its full mass to the divergence rather than being
    dropped.

    Returns a dict: {name, n_train, n_test, n_categories_train, n_categories_test,
    n_categories_test_only, jensen_shannon_distance}.
    """
    train_counts = pd.Series(train_values).dropna().value_counts()
    test_counts = pd.Series(test_values).dropna().value_counts()
    categories = sorted(set(train_counts.index) | set(test_counts.index))

    train_dist = np.array([train_counts.get(c, 0) for c in categories], dtype=float)
    test_dist = np.array([test_counts.get(c, 0) for c in categories], dtype=float)
    train_dist = train_dist / train_dist.sum() if train_dist.sum() else train_dist
    test_dist = test_dist / test_dist.sum() if test_dist.sum() else test_dist

    return {
        "name": name,
        "n_train": int(train_counts.sum()),
        "n_test": int(test_counts.sum()),
        "n_categories_train": int((train_counts > 0).sum()),
        "n_categories_test": int((test_counts > 0).sum()),
        "n_categories_test_only": len(set(test_counts.index) - set(train_counts.index)),
        "jensen_shannon_distance": float(jensenshannon(train_dist, test_dist, base=2)),
    }


def shift_report(train_df, test_df, numeric_cols=(), categorical_cols=()):
    """Run `numeric_shift`/`categorical_shift` over several columns at once. Returns a
    DataFrame, one row per column, numeric and categorical columns concatenated (differing
    fields are NaN across the two groups) and sorted by `ks_stat`/`jensen_shannon_distance`
    descending within each -- the biggest shifts first."""
    numeric_rows = [numeric_shift(train_df[c], test_df[c], name=c) for c in numeric_cols]
    categorical_rows = [categorical_shift(train_df[c], test_df[c], name=c) for c in categorical_cols]

    numeric_report = pd.DataFrame(numeric_rows).sort_values("ks_stat", ascending=False) if numeric_rows else pd.DataFrame()
    categorical_report = (
        pd.DataFrame(categorical_rows).sort_values("jensen_shannon_distance", ascending=False)
        if categorical_rows else pd.DataFrame()
    )
    return numeric_report.reset_index(drop=True), categorical_report.reset_index(drop=True)
