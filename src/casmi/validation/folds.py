"""Group-aware CV fold assignment.

Both the primary (connectivity) and harder secondary (Murcko scaffold) splits share one core
algorithm: `sklearn.model_selection.GroupKFold` applied at the SPECTRUM-ROW level (not
pre-deduplicated), so folds stay balanced by spectrum count while guaranteeing every group --
every `connectivity_key`, or every scaffold -- lands entirely inside one fold. `GroupKFold`
itself has no `random_state` (its assignment is a deterministic function of the groups array),
so there is no seed to thread through here.

The output is always deduplicated back down to one row per group: that's the artifact that
gets saved (`connectivity_key | fold`), not a per-spectrum column -- matching
`casmi.data.aggregation.build_molecule_metadata`'s one-row-per-`connectivity_key` convention.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from casmi.validation.checks import CheckResult


def _group_kfold_table(group_key_col, groups, n_splits):
    groups = pd.Series(groups).reset_index(drop=True)
    if groups.isna().any():
        raise ValueError(f"{group_key_col} contains null values -- drop or fill before folding")

    dummy_X = np.zeros((len(groups), 1))
    fold_of_row = np.empty(len(groups), dtype=int)
    for fold_idx, (_, valid_idx) in enumerate(GroupKFold(n_splits=n_splits).split(dummy_X, groups=groups.values)):
        fold_of_row[valid_idx] = fold_idx

    return (
        pd.DataFrame({group_key_col: groups.values, "fold": fold_of_row})
        .drop_duplicates(group_key_col)
        .sort_values(group_key_col)
        .reset_index(drop=True)
    )


def build_connectivity_folds(train_metadata, group_col="connectivity_key", n_splits=5):
    """Primary CV: one fold per unique `group_col` value (typically `connectivity_key`).

    `train_metadata`: the full per-spectrum table (e.g. `train_spectrum_metadata`), NOT
    deduplicated -- GroupKFold balances fold size by row count, so passing pre-deduplicated
    keys would balance by molecule count instead of the spectrum count that actually determines
    training-set size. Returns a DataFrame with one row per unique key: [group_col, "fold"].
    """
    return _group_kfold_table(group_col, train_metadata[group_col], n_splits)


def build_scaffold_folds(train_metadata, connectivity_to_scaffold, connectivity_col="connectivity_key",
                          scaffold_col="scaffold_smiles", n_splits=5):
    """Harder secondary CV: same algorithm, grouped by Murcko scaffold instead of connectivity
    -- a much coarser split (many connectivities per scaffold), so it stress-tests whether a
    model generalizes to genuinely new chemistry rather than interpolating within a scaffold
    family it has already seen.

    `connectivity_to_scaffold`: Series/dict/mapping of connectivity_key -> scaffold_smiles
    (e.g. `structure_table.set_index("connectivity_key")["scaffold_smiles"]`, or joined via
    `molecule_metadata`). An acyclic molecule's empty scaffold (`""`) is a valid group like any
    other -- it is not treated as missing.
    """
    scaffolds = train_metadata[connectivity_col].map(connectivity_to_scaffold)
    return _group_kfold_table(scaffold_col, scaffolds, n_splits)


def check_group_leakage(train_metadata, fold_table, group_col="connectivity_key"):
    """CheckResult: True iff no `group_col` value spans more than one fold once
    `train_metadata` (per-spectrum) is joined to `fold_table` (per-group). `GroupKFold` already
    guarantees this by construction -- this check exists to catch a bug in the surrounding
    join/build code (e.g. joining on the wrong key), not to second-guess sklearn."""
    df = train_metadata.dropna(subset=[group_col])
    merged = df[[group_col]].merge(fold_table, on=group_col, how="left")
    n_folds_per_group = merged.groupby(group_col)["fold"].nunique()
    leaking = n_folds_per_group[n_folds_per_group > 1]
    return CheckResult(
        name=f"no {group_col} leakage across folds",
        passed=len(leaking) == 0,
        detail="no leakage" if leaking.empty else f"{len(leaking)} groups span >1 fold: {list(leaking.index[:5])}",
    )


def train_valid_connectivity_disjointness(query_table, fold_col="fold", connectivity_col="true_connectivity_key"):
    """Per-fold, COMPUTED (never hard-coded) train/valid connectivity-disjointness table for a
    leave-one-fold-out ranker (e.g. notebook 09's `train_table`/`dev_ranking_queries`): for
    every held-out fold, the set of TRUE connectivities used as training positives vs. the set
    used as validation positives must never overlap -- `GroupKFold` already guarantees this at
    the spectrum-metadata level (`check_group_leakage`), but this re-derives it at the level the
    ranker actually trains at (one row per ranking QUERY, not per raw spectrum), catching a bug
    in the join between spectrum-level folds and query-level tables that `check_group_leakage`
    alone wouldn't see. Returns one row per fold: `[fold, n_train_connectivities,
    n_valid_connectivities, n_overlap, passed]`."""
    rows = []
    for held_out_fold in sorted(query_table[fold_col].unique()):
        train_keys = set(query_table.loc[query_table[fold_col] != held_out_fold, connectivity_col])
        valid_keys = set(query_table.loc[query_table[fold_col] == held_out_fold, connectivity_col])
        overlap = train_keys & valid_keys
        rows.append({
            "fold": held_out_fold,
            "n_train_connectivities": len(train_keys),
            "n_valid_connectivities": len(valid_keys),
            "n_overlap": len(overlap),
            "passed": len(overlap) == 0,
        })
    return pd.DataFrame(rows)


def fold_summary(train_metadata, fold_table, group_col="connectivity_key"):
    """Per-fold spectrum count and group count, for eyeballing fold balance before trusting any
    CV number built on top of it."""
    df = train_metadata.dropna(subset=[group_col])
    merged = df[[group_col]].merge(fold_table, on=group_col, how="left")
    g = merged.groupby("fold")
    return pd.DataFrame({"n_spectra": g.size(), "n_groups": g[group_col].nunique()}).reset_index()
