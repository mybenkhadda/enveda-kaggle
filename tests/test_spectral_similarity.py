import math

import numpy as np

from casmi.spectra.binning import bin_spectrum
from casmi.spectra.neutral_loss import neutral_loss_cosine_similarity, neutral_loss_overlap, neutral_losses, valid_neutral_loss_mask
from casmi.spectra.similarity import binned_cosine_similarity, modified_cosine_similarity, peak_overlap


def test_bin_spectrum_sums_peaks_in_same_bin():
    bins, vals = bin_spectrum([100.01, 100.05, 200.0], [1.0, 1.0, 1.0], bin_width=0.1, l2_normalize=False)
    assert list(bins) == [1000, 2000]
    assert list(vals) == [2.0, 1.0]


def test_bin_spectrum_l2_normalized_by_default():
    bins, vals = bin_spectrum([100.0, 200.0], [3.0, 4.0], bin_width=0.1)
    assert math.isclose(np.linalg.norm(vals), 1.0)


def test_bin_spectrum_empty():
    bins, vals = bin_spectrum([], [])
    assert len(bins) == 0 and len(vals) == 0


def test_binned_cosine_identical_spectra_is_one():
    bins, vals = bin_spectrum([100.0, 200.0, 300.0], [1.0, 2.0, 3.0], bin_width=0.1)
    assert math.isclose(binned_cosine_similarity(bins, vals, bins, vals), 1.0)


def test_binned_cosine_disjoint_spectra_is_zero():
    bins_a, vals_a = bin_spectrum([100.0], [1.0], bin_width=0.1)
    bins_b, vals_b = bin_spectrum([500.0], [1.0], bin_width=0.1)
    assert binned_cosine_similarity(bins_a, vals_a, bins_b, vals_b) == 0.0


def test_binned_cosine_empty_returns_zero_not_nan():
    bins, vals = bin_spectrum([], [])
    assert binned_cosine_similarity(bins, vals, bins, vals) == 0.0


def test_peak_overlap_exact_match():
    result = peak_overlap([100.0, 200.0, 300.0], [100.01, 200.01, 999.0], tol_da=0.02)
    assert result["n_matched"] == 2
    assert math.isclose(result["frac_a_matched"], 2 / 3)
    assert math.isclose(result["frac_b_matched"], 2 / 3)


def test_peak_overlap_no_matches_outside_tolerance():
    result = peak_overlap([100.0], [100.5], tol_da=0.02)
    assert result["n_matched"] == 0


def test_peak_overlap_empty_input():
    result = peak_overlap([], [100.0])
    assert result["n_matched"] == 0
    assert result["frac_a_matched"] == 0.0


def test_modified_cosine_identical_spectrum_scores_one():
    mzs = [100.0, 150.0, 200.0]
    ints = [1.0, 2.0, 3.0]
    result = modified_cosine_similarity(mzs, ints, 300.0, mzs, ints, 300.0)
    assert math.isclose(result["score"], 1.0, rel_tol=1e-6)
    assert result["n_matched"] == 3


def test_modified_cosine_shifted_precursor_still_matches_via_shift():
    # spectrum B is spectrum A with every fragment shifted up by 14 Da (e.g. a +CH2 analogue),
    # and its precursor is also 14 Da higher -- modified cosine should recover a match that a
    # DIRECT (unshifted) comparison would completely miss.
    mzs_a = np.array([100.0, 150.0, 200.0])
    ints_a = np.array([1.0, 2.0, 3.0])
    shift = 14.0
    mzs_b = mzs_a + shift
    result = modified_cosine_similarity(mzs_a, ints_a, 300.0, mzs_b, ints_a, 300.0 + shift)
    assert result["n_matched"] == 3
    assert result["score"] > 0.9

    direct_overlap = peak_overlap(mzs_a, mzs_b, tol_da=0.02)
    assert direct_overlap["n_matched"] == 0  # confirms the shift was actually necessary


def test_modified_cosine_empty_spectrum_returns_zero():
    result = modified_cosine_similarity([], [], 300.0, [100.0], [1.0], 300.0)
    assert result["score"] == 0.0
    assert result["n_matched"] == 0


def test_neutral_losses_and_valid_mask():
    mzs = np.array([50.0, 100.0, 250.0])  # 250 is >= precursor (200) -> invalid
    precursor = 200.0
    mask = valid_neutral_loss_mask(mzs, precursor)
    assert list(mask) == [True, True, False]
    losses = neutral_losses(mzs[mask], precursor)
    assert list(losses) == [150.0, 100.0]


def test_neutral_loss_cosine_identical_spectra():
    mzs = [50.0, 100.0]
    ints = [1.0, 2.0]
    score = neutral_loss_cosine_similarity(mzs, ints, 200.0, mzs, ints, 200.0)
    assert math.isclose(score, 1.0)


def test_neutral_loss_cosine_all_fragments_invalid_returns_zero():
    score = neutral_loss_cosine_similarity([250.0], [1.0], 200.0, [50.0], [1.0], 200.0)
    assert score == 0.0


def test_neutral_loss_overlap_matches_shared_losses():
    # different fragment masses, but the SAME neutral loss (water, -18) from each precursor
    result = neutral_loss_overlap([282.0], 300.0, [182.0], 200.0, tol_da=0.02)
    assert result["n_matched"] == 1
