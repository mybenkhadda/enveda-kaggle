"""v4b deterministic top-N truncation: intensity DESC, then m/z ASC -- tied intensities at the
rank-100/101 boundary resolve by m/z, independent of input peak order."""
import numpy as np
import pytest

from casmi.spectra.preprocessing import top_n_boundary_diagnostics, truncate_top_peaks, truncate_top_peaks_legacy


def _spectrum_with_boundary_tie(n_top=99, n_tied=5):
    """99 clearly-top peaks, then `n_tied` peaks sharing the rank-100 intensity, then filler."""
    top_mz = np.arange(n_top) + 100.0
    top_int = np.linspace(100.0, 10.0, n_top)
    tied_mz = np.array([900.5, 350.25, 700.0, 425.75, 610.1])[:n_tied]
    tied_int = np.full(n_tied, 5.0)
    filler_mz = np.arange(20) + 1000.0
    filler_int = np.full(20, 1.0)
    return np.concatenate([top_mz, tied_mz, filler_mz]), np.concatenate([top_int, tied_int, filler_int])


def test_tie_at_cutoff_resolved_by_lowest_mz():
    mzs, ints = _spectrum_with_boundary_tie()
    kept_mz, kept_int = truncate_top_peaks(mzs, ints, max_peaks=100)
    assert len(kept_mz) == 100
    tied_kept = kept_mz[kept_int == 5.0]
    assert tied_kept.tolist() == [350.25]  # lowest m/z among the five tied peaks wins the single slot


def test_result_invariant_to_input_order():
    mzs, ints = _spectrum_with_boundary_tie()
    ref = truncate_top_peaks(mzs, ints, max_peaks=100)
    rng = np.random.default_rng(0)
    for _ in range(25):
        perm = rng.permutation(len(mzs))
        out = truncate_top_peaks(mzs[perm], ints[perm], max_peaks=100)
        assert np.array_equal(out[0], ref[0]) and np.array_equal(out[1], ref[1])


def test_output_sorted_by_mz_and_keeps_most_intense():
    mzs, ints = _spectrum_with_boundary_tie()
    kept_mz, kept_int = truncate_top_peaks(mzs, ints, max_peaks=100)
    assert np.all(np.diff(kept_mz) > 0)
    assert kept_int.min() == 5.0 and 1.0 not in kept_int


def test_two_slots_for_three_tied_peaks():
    mzs = np.array([10.0, 30.0, 20.0, 40.0])
    ints = np.array([9.0, 1.0, 1.0, 1.0])
    kept_mz, _ = truncate_top_peaks(mzs, ints, max_peaks=3)
    assert kept_mz.tolist() == [10.0, 20.0, 30.0]


def test_noop_under_limit_unchanged():
    mzs, ints = np.array([5.0, 1.0]), np.array([1.0, 2.0])
    out = truncate_top_peaks(mzs, ints, max_peaks=100)
    assert np.array_equal(out[0], mzs) and np.array_equal(out[1], ints)


def test_no_tie_matches_legacy_rule():
    rng = np.random.default_rng(1)
    mzs = np.sort(rng.uniform(50, 900, 300))
    ints = rng.permutation(np.arange(300, dtype=float) + 1.0)  # all distinct -> no ambiguity
    a, b = truncate_top_peaks(mzs, ints, 100), truncate_top_peaks_legacy(mzs, ints, 100)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_boundary_diagnostics():
    mzs, ints = _spectrum_with_boundary_tie()
    d = top_n_boundary_diagnostics(ints, max_peaks=100)
    assert d["truncated"] and d["exact_tie"] and d["near_tie"]
    assert d["intensity_at_n"] == 5.0 and d["intensity_at_n_plus_1"] == 5.0 and d["n_tied_at_boundary"] == 5
    d2 = top_n_boundary_diagnostics(np.arange(150, dtype=float), max_peaks=100)
    assert not d2["exact_tie"] and not d2["near_tie"]
    d3 = top_n_boundary_diagnostics(np.ones(10), max_peaks=100)
    assert not d3["truncated"] and not d3["exact_tie"]


@pytest.mark.parametrize("seed", range(5))
def test_deterministic_across_repeated_calls(seed):
    rng = np.random.default_rng(seed)
    mzs = rng.uniform(50, 900, 400)
    ints = rng.integers(1, 20, 400).astype(float)  # heavy ties everywhere
    a = truncate_top_peaks(mzs, ints, 100)
    b = truncate_top_peaks(mzs.copy(), ints.copy(), 100)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
