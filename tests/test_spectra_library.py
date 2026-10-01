import pandas as pd
import pytest

from casmi.spectra.library import (
    assert_no_connectivity_leakage,
    build_reference_index,
    known_spectrum_reference_ids,
    unseen_connectivity_reference_ids,
)


def _train_metadata():
    return pd.DataFrame({
        "connectivity_key": ["A", "A", "A", "B", "B", "C"],
        "train_spectrum_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
    })


def test_build_reference_index_groups_by_connectivity():
    idx = build_reference_index(_train_metadata())
    assert idx["A"] == ["s1", "s2", "s3"]
    assert idx["B"] == ["s4", "s5"]


def test_known_spectrum_excludes_query_itself():
    idx = build_reference_index(_train_metadata())
    refs = known_spectrum_reference_ids(idx, "A", exclude_spectrum_id="s2")
    assert refs == ["s1", "s3"]


def test_known_spectrum_query_is_the_only_spectrum_of_its_connectivity():
    idx = build_reference_index(_train_metadata())
    refs = known_spectrum_reference_ids(idx, "C", exclude_spectrum_id="s6")
    assert refs == []


def test_unseen_connectivity_empty_for_query_own_fold():
    idx = build_reference_index(_train_metadata())
    connectivity_to_fold = {"A": 0, "B": 1, "C": 2}
    refs = unseen_connectivity_reference_ids(idx, "A", query_fold=0, connectivity_to_fold=connectivity_to_fold)
    assert refs == []  # A's own fold -- intentionally zero reference spectra


def test_unseen_connectivity_returns_other_fold_spectra():
    idx = build_reference_index(_train_metadata())
    connectivity_to_fold = {"A": 0, "B": 1, "C": 2}
    refs = unseen_connectivity_reference_ids(idx, "B", query_fold=0, connectivity_to_fold=connectivity_to_fold)
    assert refs == ["s4", "s5"]  # B is in fold 1, query is fold 0 -- not excluded


def test_assert_no_connectivity_leakage_passes_when_disjoint():
    assert_no_connectivity_leakage(["A", "B"], ["C", "D"])  # should not raise


def test_assert_no_connectivity_leakage_raises_on_overlap():
    with pytest.raises(AssertionError):
        assert_no_connectivity_leakage(["A", "B"], ["B", "C"])
