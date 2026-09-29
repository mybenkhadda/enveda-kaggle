"""Near-duplicate / mirror reference-spectrum detection: a reference spectrum that is a
near-identical technical replicate of the QUERY spectrum itself (same connectivity, near-1.0
spectral similarity, near-zero precursor-mass difference) would let a "realistic" host-holdout
Mode-A benchmark trivially match its own replicate under a different `spectrum_id` rather than
genuinely different spectral evidence -- inflating the number relative to what an unpublished
real test spectrum could expect. Flags, never deletes: the raw dataset is untouched, callers
decide whether to exclude flagged references from a given evaluation protocol.

Two distinct relationships, not one:

    MIRROR      -- the EXACT same physical spectrum, ingested twice under different
                    `spectrum_id`s (a data-provenance/dedup issue -- e.g. a public database
                    mirrored into this project's own library). Detected via `peak_hash`: an
                    exact match after canonicalization (sort, normalize, round, drop zeros).
                    No SPLASH library is available locally, so `mirror` here is peak_hash
                    equality alone -- the documented fallback when SPLASH isn't available.
    NEAR_DUPLICATE (non-mirror) -- a genuinely independent remeasurement that happens to be
                    spectrally very similar (cosine>=0.95, precursor within 0.01 Da) but is NOT
                    byte-for-byte identical -- a legitimate second measurement, not a data-
                    provenance issue. Conflating the two would treat real independent evidence
                    as if it were a database artifact.
"""
import hashlib

import numpy as np

from casmi.spectra.binning import bin_spectrum
from casmi.spectra.similarity import binned_cosine_similarity

DEFAULT_COSINE_THRESHOLD = 0.95
DEFAULT_PRECURSOR_DIFF_DA = 0.01


def is_near_duplicate_reference(query_peaks, reference_peaks, cosine_threshold=DEFAULT_COSINE_THRESHOLD,
                                 precursor_diff_da=DEFAULT_PRECURSOR_DIFF_DA, bin_width=0.1):
    """True if `reference_peaks` is a near-duplicate of `query_peaks`: precursor m/z difference
    below `precursor_diff_da` AND binned spectral cosine above `cosine_threshold`. Both
    `*_peaks` are `{"mzs", "intensities", "precursor_mz"}` dicts (same shape
    `casmi.ranking.features.compute_pair_spectral_scores` takes). Connectivity match is the
    caller's responsibility -- this is only ever meaningful when called on same-connectivity
    candidates (see `find_near_duplicate_references`); checking precursor mass first is a cheap
    short-circuit before the more expensive binning/cosine step."""
    precursor_diff = abs(query_peaks["precursor_mz"] - reference_peaks["precursor_mz"])
    if precursor_diff >= precursor_diff_da:
        return False
    q_bins, q_vals = bin_spectrum(query_peaks["mzs"], query_peaks["intensities"], bin_width=bin_width)
    r_bins, r_vals = bin_spectrum(reference_peaks["mzs"], reference_peaks["intensities"], bin_width=bin_width)
    cosine = binned_cosine_similarity(q_bins, q_vals, r_bins, r_vals)
    return cosine > cosine_threshold


def find_near_duplicate_references(query_peaks, reference_id_to_peaks, cosine_threshold=DEFAULT_COSINE_THRESHOLD,
                                    precursor_diff_da=DEFAULT_PRECURSOR_DIFF_DA, bin_width=0.1):
    """`reference_id_to_peaks`: `{spectrum_id: peaks_dict}` for candidate reference spectra
    (typically same-connectivity as the query). Returns the SET of spectrum_ids flagged as
    near-duplicates of `query_peaks` -- flag, don't delete (spec 5.1); callers filter their own
    reference-id list against this set. Never raises on an empty `reference_id_to_peaks`."""
    return {
        rid for rid, ref_peaks in reference_id_to_peaks.items()
        if is_near_duplicate_reference(query_peaks, ref_peaks, cosine_threshold, precursor_diff_da, bin_width)
    }


def compute_peak_hash(mzs, intensities, precursor_mz, mz_decimals=4, intensity_decimals=3, precursor_decimals=3):
    """A stable hash identifying a spectrum's EXACT peak content, for detecting `mirror`
    (byte-for-byte-identical, modulo float noise) spectra ingested twice under different
    `spectrum_id`s. Canonicalization: sort by m/z, max-normalize intensity, round both, drop
    peaks whose rounded intensity is 0, round the precursor m/z, hash the resulting canonical
    string. Two spectra with the same peak_hash are the SAME physical measurement; two spectra
    that are merely spectrally similar (different peak_hash) are NOT -- that's what
    `is_near_duplicate_reference` is for instead."""
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if len(mzs) == 0:
        peak_repr = ""
    else:
        max_intensity = intensities.max()
        normalized = intensities / max_intensity if max_intensity > 0 else intensities
        order = np.argsort(mzs)
        sorted_mzs = np.round(mzs[order], mz_decimals)
        sorted_intensities = np.round(normalized[order], intensity_decimals)
        keep = sorted_intensities > 0
        peak_repr = ";".join(
            f"{mz:.{mz_decimals}f}:{inten:.{intensity_decimals}f}"
            for mz, inten in zip(sorted_mzs[keep], sorted_intensities[keep])
        )
    precursor_repr = f"{round(float(precursor_mz), precursor_decimals):.{precursor_decimals}f}"
    canonical = f"{precursor_repr}|{peak_repr}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_mirror_reference(query_peaks, reference_peaks, query_peak_hash=None, reference_peak_hash=None, **peak_hash_kwargs):
    """True if `reference_peaks` has the EXACT same `peak_hash` as `query_peaks` -- the same
    physical spectrum under a different `spectrum_id`. No SPLASH library is available locally,
    so this is peak_hash equality alone (the documented SPLASH-unavailable fallback).

    Pass `query_peak_hash`/`reference_peak_hash` (precomputed via `compute_peak_hash`) to skip
    re-hashing the same spectrum on every call -- e.g. when checking one query against many
    reference candidates, hash the query once and reuse it, rather than recomputing its hash
    from scratch for every comparison."""
    q_hash = query_peak_hash if query_peak_hash is not None else \
        compute_peak_hash(query_peaks["mzs"], query_peaks["intensities"], query_peaks["precursor_mz"], **peak_hash_kwargs)
    r_hash = reference_peak_hash if reference_peak_hash is not None else \
        compute_peak_hash(reference_peaks["mzs"], reference_peaks["intensities"], reference_peaks["precursor_mz"], **peak_hash_kwargs)
    return q_hash == r_hash


def classify_reference_relationship(query_peaks, reference_peaks, cosine_threshold=DEFAULT_COSINE_THRESHOLD,
                                     precursor_diff_da=DEFAULT_PRECURSOR_DIFF_DA, bin_width=0.1,
                                     query_peak_hash=None, reference_peak_hash=None):
    """One of `"mirror"` / `"near_duplicate_non_mirror"` / `"other"` -- the three-way
    classification a `mirror` check collapsing into `near_duplicate` would lose. `mirror` is
    checked FIRST (exact peak_hash match implies near-duplicate too, but the reverse doesn't
    hold, and mirror is the stronger, more specific claim worth keeping distinct: a genuinely
    independent remeasurement that merely happens to be spectrally very similar is legitimate
    evidence, not a database artifact, and should never be silently bucketed with mirrors).

    `query_peak_hash`/`reference_peak_hash`: optional precomputed hashes, see `is_mirror_reference`.

    Superseded by `classify_identity_tier` (T1-T4, adds tolerant-mirror/T2 detection) for new
    code -- kept for backward compatibility with `09b`/`10v2`/`10v3`, which only ever needed the
    three-way split."""
    if is_mirror_reference(query_peaks, reference_peaks, query_peak_hash=query_peak_hash, reference_peak_hash=reference_peak_hash):
        return "mirror"
    if is_near_duplicate_reference(query_peaks, reference_peaks, cosine_threshold, precursor_diff_da, bin_width):
        return "near_duplicate_non_mirror"
    return "other"


DEFAULT_TOLERANT_MIRROR_PRECURSOR_DIFF_DA = 0.005
DEFAULT_TOLERANT_MIRROR_MIN_RELATIVE_INTENSITY = 0.01
DEFAULT_TOLERANT_MIRROR_PPM_TOL = 5.0
DEFAULT_TOLERANT_MIRROR_ABS_TOL_DA = 0.002
DEFAULT_TOLERANT_MIRROR_MATCH_FRACTION = 0.95
DEFAULT_TOLERANT_MIRROR_MIN_CORRELATION = 0.99

TIERS = ("T1", "T2", "T3", "T4")


def _match_peaks_greedy(query_mzs, query_intensities, ref_mzs, ref_intensities,
                         ppm_tol=DEFAULT_TOLERANT_MIRROR_PPM_TOL, abs_tol_da=DEFAULT_TOLERANT_MIRROR_ABS_TOL_DA):
    """Deterministic descending-query-intensity greedy one-to-one peak matching (each reference
    peak used at most once). Tolerance per query peak: `max(ppm_tol ppm of that peak's m/z,
    abs_tol_da)`. Ties among multiple in-tolerance reference peaks break by nearest m/z, then by
    lowest reference index (never by "whichever numpy happened to return first"). Returns
    `(matched_query_idx, matched_ref_idx)`, both arrays, in query-intensity-descending order --
    `matched_query_idx[i]` is matched to `matched_ref_idx[i]`."""
    query_mzs = np.asarray(query_mzs, dtype=float)
    query_intensities = np.asarray(query_intensities, dtype=float)
    ref_mzs = np.asarray(ref_mzs, dtype=float)
    ref_intensities = np.asarray(ref_intensities, dtype=float)

    query_order = np.argsort(-query_intensities, kind="stable")
    ref_available = np.ones(len(ref_mzs), dtype=bool)

    matched_query, matched_ref = [], []
    for qi in query_order:
        q_mz = query_mzs[qi]
        tol = max(ppm_tol * 1e-6 * q_mz, abs_tol_da)
        candidate_ref_idx = np.where(ref_available)[0]
        if len(candidate_ref_idx) == 0:
            continue
        deltas = np.abs(ref_mzs[candidate_ref_idx] - q_mz)
        within = deltas <= tol
        if not within.any():
            continue
        in_tol_idx = candidate_ref_idx[within]
        in_tol_deltas = deltas[within]
        # nearest m/z first, lowest reference index breaks exact ties -- deterministic
        best = in_tol_idx[np.lexsort((in_tol_idx, in_tol_deltas))[0]]
        ref_available[best] = False
        matched_query.append(qi)
        matched_ref.append(best)
    return np.array(matched_query, dtype=int), np.array(matched_ref, dtype=int)


def is_tolerant_mirror_reference(query_peaks, reference_peaks,
                                  precursor_diff_da=DEFAULT_TOLERANT_MIRROR_PRECURSOR_DIFF_DA,
                                  min_relative_intensity=DEFAULT_TOLERANT_MIRROR_MIN_RELATIVE_INTENSITY,
                                  ppm_tol=DEFAULT_TOLERANT_MIRROR_PPM_TOL, abs_tol_da=DEFAULT_TOLERANT_MIRROR_ABS_TOL_DA,
                                  match_fraction=DEFAULT_TOLERANT_MIRROR_MATCH_FRACTION,
                                  min_correlation=DEFAULT_TOLERANT_MIRROR_MIN_CORRELATION):
    """T2 ("tolerant mirror"): catches the same physical spectrum re-serialized with different
    m/z/intensity rounding -- not byte-identical (T1/`peak_hash` would miss it) but still the
    same measurement, not an independent remeasurement. MUST be called on the IDENTITY
    representation (full cleaned spectrum, no top-K truncation) -- see the module's
    representation-separation note.

    Requires: precursor within `precursor_diff_da`; then greedy one-to-one matching (peaks below
    `min_relative_intensity` of their own spectrum's base peak dropped first) at
    `tol = max(ppm_tol ppm, abs_tol_da)`; both matched-fraction(query) and matched-fraction(ref)
    >= `match_fraction`; and Pearson correlation of sqrt-intensities over the matched pairs >=
    `min_correlation`. Fewer than 2 matched pairs (correlation undefined) or an empty spectrum
    after thresholding both count as NOT a tolerant mirror -- ambiguous evidence never gets
    classified as identity-equivalent."""
    precursor_diff = abs(query_peaks["precursor_mz"] - reference_peaks["precursor_mz"])
    if precursor_diff > precursor_diff_da:
        return False

    def _thresholded(peaks):
        mzs = np.asarray(peaks["mzs"], dtype=float)
        intensities = np.asarray(peaks["intensities"], dtype=float)
        if len(intensities) == 0:
            return mzs, intensities
        base = intensities.max()
        if base <= 0:
            return mzs[:0], intensities[:0]
        keep = (intensities / base) >= min_relative_intensity
        return mzs[keep], intensities[keep]

    q_mzs, q_int = _thresholded(query_peaks)
    r_mzs, r_int = _thresholded(reference_peaks)
    if len(q_mzs) == 0 or len(r_mzs) == 0:
        return False

    matched_q_idx, matched_r_idx = _match_peaks_greedy(q_mzs, q_int, r_mzs, r_int, ppm_tol=ppm_tol, abs_tol_da=abs_tol_da)
    n_matched = len(matched_q_idx)
    if n_matched < 2:
        return False
    if (n_matched / len(q_mzs)) < match_fraction or (n_matched / len(r_mzs)) < match_fraction:
        return False

    q_sqrt = np.sqrt(q_int[matched_q_idx])
    r_sqrt = np.sqrt(r_int[matched_r_idx])
    if np.std(q_sqrt) == 0 or np.std(r_sqrt) == 0:
        return False
    correlation = float(np.corrcoef(q_sqrt, r_sqrt)[0, 1])
    return np.isfinite(correlation) and correlation >= min_correlation


def classify_identity_tier(query_peaks, reference_peaks, query_peak_hash=None, reference_peak_hash=None,
                            near_dup_cosine_threshold=DEFAULT_COSINE_THRESHOLD,
                            near_dup_precursor_diff_da=DEFAULT_PRECURSOR_DIFF_DA,
                            tolerant_mirror_kwargs=None, bin_width=0.1):
    """One of `"T1"` (exact mirror -- same `peak_hash`) / `"T2"` (tolerant mirror -- same
    measurement, different serialization/rounding) / `"T3"` (near-duplicate, non-mirror --
    genuinely independent remeasurement that's spectrally very similar) / `"T4"` (other),
    checked in that priority order (T1 implies T2/T3 would also match; T2 implies T3 would also
    match -- always report the STRONGEST true claim). `query_peaks`/`reference_peaks` should be
    the IDENTITY representation (full cleaned spectrum) for T1/T2 to be meaningful; T3's cosine
    check works on either representation but is typically also run on identity peaks for
    consistency with T1/T2 in the same call.

    `mirror = tier in {"T1", "T2"}`, `near_dup_non_mirror = tier == "T3"` -- derive these from
    the returned tier string, never re-check thresholds separately (spec 6.2's "no ambiguous
    combinations")."""
    if is_mirror_reference(query_peaks, reference_peaks, query_peak_hash=query_peak_hash, reference_peak_hash=reference_peak_hash):
        return "T1"
    if is_tolerant_mirror_reference(query_peaks, reference_peaks, **(tolerant_mirror_kwargs or {})):
        return "T2"
    if is_near_duplicate_reference(query_peaks, reference_peaks, near_dup_cosine_threshold, near_dup_precursor_diff_da, bin_width):
        return "T3"
    return "T4"
