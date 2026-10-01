import numpy as np
import pandas as pd

from casmi.candidates.mass_index import MassIndex


def test_basic_query_returns_the_match():
    masses = [100.0, 200.0, 300.0]
    keys = ["a", "b", "c"]
    index = MassIndex(keys, masses)

    result = index.query(200.0, tolerance_ppm=5)
    assert list(result) == ["b"]


def test_query_closest_first():
    keys = ["a", "b", "c"]
    masses = [300.0000, 300.0005, 300.0020]
    index = MassIndex(keys, masses)

    result = index.query(300.0000, tolerance_da=0.003)
    assert list(result) == ["a", "b", "c"]


def test_tolerance_excludes_out_of_window_candidates():
    index = MassIndex(["a", "b"], [100.0, 100.1])
    assert list(index.query(100.0, tolerance_da=0.05)) == ["a"]
    assert set(index.query(100.0, tolerance_da=0.2)) == {"a", "b"}


def test_null_masses_are_dropped():
    index = MassIndex(["a", "b", "c"], [100.0, np.nan, 300.0])
    assert len(index) == 2
    assert index.candidate_count(500.0, tolerance_ppm=1_000_000) == 2


def test_tolerance_argument_validation():
    index = MassIndex(["a"], [100.0])
    try:
        index.query(100.0)
        assert False, "should require exactly one tolerance kwarg"
    except ValueError:
        pass
    try:
        index.query(100.0, tolerance_ppm=5, tolerance_da=0.01)
        assert False, "should reject both tolerance kwargs at once"
    except ValueError:
        pass


def test_from_molecule_table():
    table = pd.DataFrame({"connectivity_key": ["a", "b"], "exact_mass": [180.063, 342.116]})
    index = MassIndex.from_molecule_table(table)
    assert list(index.query(180.063, tolerance_ppm=5)) == ["a"]


def test_query_many_matches_query():
    index = MassIndex(["a", "b", "c"], [100.0, 200.0, 300.0])
    results = index.query_many([100.0, 200.0], tolerance_ppm=5)
    assert [list(r) for r in results] == [["a"], ["b"]]


def test_from_dataframe_is_an_alias_of_from_molecule_table():
    table = pd.DataFrame({"connectivity_key": ["a", "b"], "exact_mass": [180.063, 342.116]})
    index = MassIndex.from_dataframe(table)
    assert list(index.query(342.116, tolerance_ppm=5)) == ["b"]


def test_query_with_masses_returns_parallel_arrays_closest_first():
    index = MassIndex(["a", "b", "c"], [300.0020, 300.0000, 300.0005])
    keys, masses = index.query_with_masses(300.0000, tolerance_da=0.003)
    assert list(keys) == ["b", "c", "a"]
    assert list(masses) == [300.0000, 300.0005, 300.0020]


def test_query_with_masses_empty_window():
    index = MassIndex(["a"], [100.0])
    keys, masses = index.query_with_masses(500.0, tolerance_ppm=1)
    assert len(keys) == 0 and len(masses) == 0


def test_save_and_load_round_trip(tmp_path):
    index = MassIndex(["a", "b", "c"], [100.0, 200.0, 300.0])
    saved_path = index.save(tmp_path / "mass_index")

    loaded = MassIndex.load(saved_path)
    assert len(loaded) == len(index)
    assert list(loaded.query(200.0, tolerance_ppm=5)) == ["b"]
    assert (tmp_path / "mass_index.meta.json").exists()


def test_save_records_library_fingerprint_and_version(tmp_path):
    from casmi.candidates.mass_index import MASS_INDEX_VERSION

    index = MassIndex(["a"], [100.0])
    index.save(tmp_path / "mass_index", library_fingerprint="fp-123")
    meta = MassIndex.read_meta(tmp_path / "mass_index")
    assert meta["library_fingerprint"] == "fp-123"
    assert meta["mass_index_version"] == MASS_INDEX_VERSION
    assert meta["n_structures"] == 1


def test_read_meta_missing_sidecar_returns_none(tmp_path):
    assert MassIndex.read_meta(tmp_path / "does_not_exist") is None


def test_validate_against_brute_force_passes_on_real_data():
    from casmi.candidates.mass_index import validate_against_brute_force

    rng = np.random.RandomState(0)
    masses = np.concatenate([rng.uniform(50, 500, 500), [200.0, 200.0]])  # includes a duplicate mass
    table = pd.DataFrame({"connectivity_key": [f"k{i}" for i in range(len(masses))], "exact_mass": masses})
    index = MassIndex.from_dataframe(table)

    result = validate_against_brute_force(index, table, tolerance_ppms=(1, 10, 50), n_samples=50, seed=1)
    assert result.passed, result.detail


def test_validate_against_brute_force_catches_a_broken_index():
    from casmi.candidates.mass_index import validate_against_brute_force

    table = pd.DataFrame({"connectivity_key": ["a", "b", "c"], "exact_mass": [100.0, 200.0, 300.0]})
    index = MassIndex.from_dataframe(table)
    index.masses[1] = 999.0  # deliberately corrupt the sorted array so it disagrees with the table

    result = validate_against_brute_force(index, table, tolerance_ppms=(5,), n_samples=3, seed=1)
    assert not result.passed
