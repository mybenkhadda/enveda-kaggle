"""Classical MS/MS spectral similarity: binned cosine, peak overlap, and modified (precursor-
mass-shift-aware) cosine. Pure functions over raw peak arrays or `casmi.spectra.binning`
output -- no file I/O, no aggregation across multiple reference spectra (that's
`casmi.ranking.aggregation`).
"""
import numpy as np


def binned_cosine_similarity(bins_a, vals_a, bins_b, vals_b):
    """Cosine similarity between two ALREADY-L2-NORMALIZED binned spectra (see
    `casmi.spectra.binning.bin_spectrum`) -- a plain dot product over the bins the two spectra
    share, since an empty intersection contributes zero either way. Returns 0.0 for two empty
    spectra (not NaN)."""
    if len(bins_a) == 0 or len(bins_b) == 0:
        return 0.0
    common, idx_a, idx_b = np.intersect1d(bins_a, bins_b, assume_unique=True, return_indices=True)
    if len(common) == 0:
        return 0.0
    return float(np.dot(vals_a[idx_a], vals_b[idx_b]))


def peak_overlap(mzs_a, mzs_b, tol_da=0.02):
    """Simple greedy peak matching (each peak used at most once) within `tol_da`. Returns
    `{n_matched, frac_a_matched, frac_b_matched}`. Assumes `mzs_a`/`mzs_b` are peak-picked
    (already deduplicated), not raw profile data."""
    mzs_a = np.sort(np.asarray(mzs_a, dtype=float))
    mzs_b = np.sort(np.asarray(mzs_b, dtype=float))
    if len(mzs_a) == 0 or len(mzs_b) == 0:
        return {"n_matched": 0, "frac_a_matched": 0.0, "frac_b_matched": 0.0}

    i = j = n_matched = 0
    while i < len(mzs_a) and j < len(mzs_b):
        diff = mzs_a[i] - mzs_b[j]
        if abs(diff) <= tol_da:
            n_matched += 1
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return {
        "n_matched": n_matched,
        "frac_a_matched": n_matched / len(mzs_a),
        "frac_b_matched": n_matched / len(mzs_b),
    }


def _greedy_matched_score(mzs_a, ints_a, mzs_b, ints_b, tol_da, mass_shift):
    """Shared core for `modified_cosine_similarity`: every (i, j) pair within `tol_da` of a
    direct match OR a `mass_shift`-shifted match is a match candidate, scored by
    `ints_a[i] * ints_b[j]`; candidates are assigned greedily by descending score, each peak
    used at most once (the standard modified-cosine matching rule).

    The (i, j) x (i, j) match search is a vectorized NumPy broadcast (an `n_a x n_b` difference
    matrix), not a nested Python loop -- at typical MS/MS peak counts this is instant, whereas
    a pure-Python double loop measurably bottlenecks a pair table with hundreds of thousands of
    (query, candidate) rows. Only the final greedy ASSIGNMENT step -- inherently sequential,
    bounded by the number of actual matches, which is usually small -- stays a Python loop."""
    diff = np.asarray(mzs_a)[:, None] - np.asarray(mzs_b)[None, :]
    is_match = (np.abs(diff) <= tol_da) | (np.abs(diff - mass_shift) <= tol_da)
    if not is_match.any():
        return 0.0, 0

    score_matrix = np.outer(ints_a, ints_b)
    i_idx, j_idx = np.nonzero(is_match)
    scores = score_matrix[i_idx, j_idx]
    order = np.argsort(scores)[::-1]

    used_a, used_b = set(), set()
    matched_score = 0.0
    n_matched = 0
    for k in order:
        i, j, score = int(i_idx[k]), int(j_idx[k]), float(scores[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched_score += score
        n_matched += 1
    return matched_score, n_matched


def modified_cosine_similarity(mzs_a, intensities_a, precursor_a, mzs_b, intensities_b, precursor_b, tol_da=0.02):
    """Precursor-mass-shift-aware cosine: a fragment pair counts as matched if it aligns
    directly OR after shifting by `precursor_a - precursor_b` (the standard "modified cosine"
    used for analogue/adduct-shifted spectral matching). Returns `{score, n_matched}`; `score`
    is 0.0 for an empty spectrum, never NaN. O(n_peaks_a * n_peaks_b) -- fine for typical
    MS/MS peak counts (tens to low hundreds), not meant for profile-mode data."""
    mzs_a = np.asarray(mzs_a, dtype=float)
    intensities_a = np.asarray(intensities_a, dtype=float)
    mzs_b = np.asarray(mzs_b, dtype=float)
    intensities_b = np.asarray(intensities_b, dtype=float)
    if len(mzs_a) == 0 or len(mzs_b) == 0:
        return {"score": 0.0, "n_matched": 0}

    norm_a = np.linalg.norm(intensities_a)
    norm_b = np.linalg.norm(intensities_b)
    if norm_a == 0 or norm_b == 0:
        return {"score": 0.0, "n_matched": 0}

    mass_shift = precursor_a - precursor_b
    matched_score, n_matched = _greedy_matched_score(mzs_a, intensities_a, mzs_b, intensities_b, tol_da, mass_shift)
    return {"score": float(matched_score / (norm_a * norm_b)), "n_matched": n_matched}
