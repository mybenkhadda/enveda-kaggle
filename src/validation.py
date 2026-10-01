"""Leakage-safe, group-aware validation-fold construction.

CASMI scoring is at the molecule (connectivity) level, so every fold must guarantee that no
connectivity key appears in both the train and validation partitions of that fold -- this
module is the single source of truth for that assignment; later notebooks should read the
saved fold artifact rather than regenerating folds independently.
"""
import numpy as np
import pandas as pd


def assign_group_folds(groups, n_splits=5, seed=42):
    """Assign each row an integer fold in [0, n_splits) such that all rows sharing the same
    `groups` value get the same fold. Groups are shuffled before slicing so fold sizes are
    balanced regardless of the input's original ordering."""
    groups = pd.Series(groups).reset_index(drop=True)
    unique_groups = groups.dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    group_to_fold = {g: i % n_splits for i, g in enumerate(unique_groups)}
    return groups.map(group_to_fold)


def summarize_fold_balance(df, fold_col, group_col, numeric_cols=(), categorical_cols=()):
    """Per-fold row counts, unique-group counts, and distribution checks for a few key columns."""
    rows = []
    for fold, sub in df.groupby(fold_col):
        rec = {
            "fold": fold,
            "n_rows": len(sub),
            "n_unique_groups": sub[group_col].nunique(),
        }
        for c in numeric_cols:
            if c in sub.columns:
                rec[f"{c}_median"] = sub[c].median()
        for c in categorical_cols:
            if c in sub.columns and len(sub):
                top = sub[c].value_counts(normalize=True)
                rec[f"{c}_top_share"] = float(top.iloc[0]) if len(top) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def max_similarity_to_train(query_fps, train_fps, bulk_tanimoto_fn):
    """For each query fingerprint, the maximum Tanimoto similarity to anything in train_fps.
    O(n_query * n_train); intended for sampled novelty audits, not full-scale retrieval."""
    out = np.zeros(len(query_fps))
    for i, fp in enumerate(query_fps):
        if fp is None:
            out[i] = np.nan
            continue
        sims = bulk_tanimoto_fn(fp, train_fps)
        out[i] = sims.max() if len(sims) else np.nan
    return out
