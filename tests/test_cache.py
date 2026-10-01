import pandas as pd

from casmi.candidates.cache import cached_dataframe, cached_mass_index
from casmi.candidates.mass_index import MassIndex
from casmi.io.artifacts import save_artifact


def test_cached_dataframe_hits_when_inputs_unchanged(tmp_path):
    calls = []

    def _build():
        calls.append(1)
        return pd.DataFrame({"x": [1, 2, 3]})

    df1, _, cached1 = cached_dataframe("thing", tmp_path, _build, input_fingerprints={"upstream": "v1"})
    df2, _, cached2 = cached_dataframe("thing", tmp_path, _build, input_fingerprints={"upstream": "v1"})

    assert not cached1
    assert cached2
    assert len(calls) == 1  # build_fn only ran once


def test_cached_dataframe_invalidates_when_upstream_fingerprint_changes(tmp_path):
    calls = []

    def _build():
        calls.append(1)
        return pd.DataFrame({"x": [1, 2, 3]})

    cached_dataframe("thing", tmp_path, _build, input_fingerprints={"upstream": "v1"})
    _, _, cached2 = cached_dataframe("thing", tmp_path, _build, input_fingerprints={"upstream": "v2"})

    assert not cached2
    assert len(calls) == 2  # rebuilt because the fingerprint changed


def test_cached_mass_index_hits_on_matching_library_fingerprint(tmp_path):
    calls = []

    def _build():
        calls.append(1)
        return MassIndex(["a", "b"], [100.0, 200.0])

    index1, _, cached1 = cached_mass_index(tmp_path, _build, library_fingerprint="fp-a")
    index2, _, cached2 = cached_mass_index(tmp_path, _build, library_fingerprint="fp-a")

    assert not cached1
    assert cached2
    assert len(calls) == 1
    assert list(index2.query(200.0, tolerance_ppm=5)) == ["b"]


def test_cached_mass_index_misses_on_library_fingerprint_change(tmp_path):
    calls = []

    def _build():
        calls.append(1)
        return MassIndex(["a", "b"], [100.0, 200.0])

    cached_mass_index(tmp_path, _build, library_fingerprint="fp-a")
    _, _, cached2 = cached_mass_index(tmp_path, _build, library_fingerprint="fp-b")

    assert not cached2
    assert len(calls) == 2


def test_cached_mass_index_forced_rebuild_even_with_matching_fingerprint(tmp_path):
    calls = []

    def _build():
        calls.append(1)
        return MassIndex(["a"], [100.0])

    cached_mass_index(tmp_path, _build, library_fingerprint="fp-a")
    _, _, cached2 = cached_mass_index(tmp_path, _build, library_fingerprint="fp-a", force=True)

    assert not cached2
    assert len(calls) == 2
