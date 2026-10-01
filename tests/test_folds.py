import pandas as pd

from casmi.validation.folds import (
    build_connectivity_folds, build_scaffold_folds, check_group_leakage, fold_summary,
    train_valid_connectivity_disjointness,
)


def _toy_train_metadata(n_groups=20, spectra_per_group=5):
    rows = []
    for g in range(n_groups):
        for _ in range(spectra_per_group):
            rows.append({"connectivity_key": f"key_{g}"})
    return pd.DataFrame(rows)


def test_every_group_lands_in_exactly_one_fold():
    train_metadata = _toy_train_metadata()
    fold_table = build_connectivity_folds(train_metadata, n_splits=5)

    assert set(fold_table["connectivity_key"]) == set(train_metadata["connectivity_key"])
    assert fold_table["connectivity_key"].is_unique
    check = check_group_leakage(train_metadata, fold_table)
    assert check.passed, check.detail


def test_folds_are_reasonably_balanced():
    train_metadata = _toy_train_metadata(n_groups=25, spectra_per_group=4)
    fold_table = build_connectivity_folds(train_metadata, n_splits=5)
    summary = fold_summary(train_metadata, fold_table)

    assert len(summary) == 5
    assert summary["n_spectra"].sum() == len(train_metadata)
    assert summary["n_spectra"].min() >= summary["n_spectra"].max() - 4


def test_scaffold_folds_group_by_scaffold_not_connectivity():
    train_metadata = pd.DataFrame({
        "connectivity_key": ["a", "a", "b", "c", "c", "c"],
    })
    # a and b share a scaffold -- they must end up in the same fold even though they're
    # different connectivity keys.
    connectivity_to_scaffold = {"a": "scaffold_1", "b": "scaffold_1", "c": "scaffold_2"}

    fold_table = build_scaffold_folds(train_metadata, connectivity_to_scaffold, n_splits=2)
    assert set(fold_table["scaffold_smiles"]) == {"scaffold_1", "scaffold_2"}
    assert fold_table.set_index("scaffold_smiles")["fold"].nunique() == 2


def test_train_valid_connectivity_disjointness_passes_for_clean_folds():
    query_table = pd.DataFrame({
        "fold": [0, 0, 1, 1, 2, 2],
        "true_connectivity_key": ["a", "b", "c", "d", "e", "f"],
    })
    result = train_valid_connectivity_disjointness(query_table)
    assert len(result) == 3
    assert result["passed"].all()
    assert result["n_overlap"].eq(0).all()
    assert result.loc[result["fold"] == 0, "n_valid_connectivities"].iloc[0] == 2
    assert result.loc[result["fold"] == 0, "n_train_connectivities"].iloc[0] == 4


def test_train_valid_connectivity_disjointness_catches_a_leaking_connectivity():
    # "b" appears as a true positive in BOTH fold 0 (validation) and fold 1 (training) -- a bug
    # a spectrum-level-only check could miss if the join to the query table went wrong.
    query_table = pd.DataFrame({
        "fold": [0, 0, 1, 1],
        "true_connectivity_key": ["a", "b", "b", "c"],
    })
    result = train_valid_connectivity_disjointness(query_table)
    fold0 = result[result["fold"] == 0].iloc[0]
    assert fold0["n_overlap"] == 1
    assert not fold0["passed"]


def test_null_group_raises():
    train_metadata = pd.DataFrame({"connectivity_key": ["a", None, "b"]})
    try:
        build_connectivity_folds(train_metadata, n_splits=2)
        assert False, "should reject null group values"
    except ValueError:
        pass
