import numpy as np

from casmi.spectra.features import compute_spectrum_features
from casmi.spectra.preprocessing import merge_duplicate_mz, normalize_max, normalize_sum, remove_invalid_peaks, sort_peaks
from casmi.spectra.validation import is_clean, validate_spectrum


def test_empty_spectrum():
    flags = validate_spectrum([], [])
    assert flags["empty"]
    assert not flags["length_mismatch"]
    feats = compute_spectrum_features([], [])
    assert feats["n_peaks"] == 0
    assert feats["spectral_entropy"] == 0.0


def test_length_mismatch():
    flags = validate_spectrum([1.0, 2.0], [1.0])
    assert flags["length_mismatch"]


def test_normal_spectrum_is_clean_and_featurized():
    mzs = [100.0, 200.0, 300.0]
    intensities = [10.0, 100.0, 50.0]
    flags = validate_spectrum(mzs, intensities, precursor_mz=310.0)
    assert is_clean(flags)

    feats = compute_spectrum_features(mzs, intensities, precursor_mz=310.0)
    assert feats["n_peaks"] == 3
    assert feats["base_peak_mz"] == 200.0
    assert feats["base_peak_intensity"] == 100.0
    assert feats["total_intensity"] == 160.0
    assert abs(feats["top1_intensity_fraction"] - 100.0 / 160.0) < 1e-9
    assert feats["n_fragments_above_precursor"] == 0


def test_negative_intensity_flagged_and_removable():
    mzs, intensities = np.array([100.0, 200.0]), np.array([-5.0, 10.0])
    flags = validate_spectrum(mzs, intensities)
    assert flags["negative_intensity"]
    clean_mzs, clean_intensities = remove_invalid_peaks(mzs, intensities)
    assert len(clean_mzs) == 1
    assert clean_mzs[0] == 200.0


def test_duplicate_mz_flagged_and_mergeable():
    mzs, intensities = [100.0, 100.0, 200.0], [1.0, 2.0, 3.0]
    flags = validate_spectrum(mzs, intensities)
    assert flags["duplicate_mz"]
    merged_mz, merged_intensity = merge_duplicate_mz(mzs, intensities)
    assert len(merged_mz) == 2
    assert merged_intensity[list(merged_mz).index(100.0)] == 3.0


def test_unsorted_mz_flagged_and_sortable():
    mzs, intensities = [200.0, 100.0, 300.0], [1.0, 2.0, 3.0]
    flags = validate_spectrum(mzs, intensities)
    assert flags["unsorted_mz"]
    sorted_mz, sorted_intensity = sort_peaks(mzs, intensities)
    assert list(sorted_mz) == [100.0, 200.0, 300.0]
    assert list(sorted_intensity) == [2.0, 1.0, 3.0]


def test_single_peak_entropy_is_zero():
    # a single peak (or an all-equal-intensity spectrum with one peak) carries no information
    feats = compute_spectrum_features([100.0], [10.0])
    assert feats["spectral_entropy"] == 0.0
    assert feats["top1_intensity_fraction"] == 1.0


def test_fragment_above_precursor_detection():
    flags = validate_spectrum([100.0, 500.0], [1.0, 1.0], precursor_mz=200.0, fragment_above_precursor_da=5.0)
    assert flags["fragment_above_precursor"]
    feats = compute_spectrum_features([100.0, 500.0], [1.0, 1.0], precursor_mz=200.0)
    assert feats["n_fragments_above_precursor"] == 1
    assert feats["fraction_fragments_above_precursor"] == 0.5


def test_normalize_max_and_sum():
    intensities = np.array([10.0, 20.0, 30.0])
    assert normalize_max(intensities).max() == 1.0
    assert abs(normalize_sum(intensities).sum() - 1.0) < 1e-9
    # all-zero input should not raise (division by zero guarded)
    zeros = np.zeros(3)
    assert normalize_max(zeros).sum() == 0.0
    assert normalize_sum(zeros).sum() == 0.0


def test_truncate_top_peaks_keeps_most_intense():
    from casmi.spectra.preprocessing import truncate_top_peaks

    mzs = np.array([100.0, 200.0, 300.0, 400.0])
    intensities = np.array([1.0, 5.0, 2.0, 4.0])
    kept_mzs, kept_intensities = truncate_top_peaks(mzs, intensities, max_peaks=2)
    assert list(kept_mzs) == [200.0, 400.0]  # the two most intense, re-sorted by ascending m/z
    assert list(kept_intensities) == [5.0, 4.0]


def test_truncate_top_peaks_noop_when_under_limit():
    from casmi.spectra.preprocessing import truncate_top_peaks

    mzs = np.array([100.0, 200.0])
    intensities = np.array([1.0, 2.0])
    kept_mzs, kept_intensities = truncate_top_peaks(mzs, intensities, max_peaks=100)
    assert list(kept_mzs) == [100.0, 200.0]
    assert list(kept_intensities) == [1.0, 2.0]


def test_truncate_top_peaks_bounds_large_spectra():
    from casmi.spectra.preprocessing import truncate_top_peaks

    rng = np.random.RandomState(0)
    mzs = rng.uniform(50, 500, 5000)
    intensities = rng.uniform(0, 1, 5000)
    kept_mzs, kept_intensities = truncate_top_peaks(mzs, intensities, max_peaks=100)
    assert len(kept_mzs) == 100
    assert list(kept_mzs) == sorted(kept_mzs)  # re-sorted ascending
