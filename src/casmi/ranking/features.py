"""Pair-level (query x candidate) spectral feature construction: mass evidence already lives in
notebook 03's candidate pool (`abs_mass_error_ppm`); this adds spectral evidence -- binned
cosine, modified cosine, peak overlap, neutral-loss cosine -- aggregated across a candidate's
available reference spectra.
"""
from casmi.ranking.aggregation import aggregate_scores
from casmi.spectra.binning import bin_spectrum
from casmi.spectra.neutral_loss import neutral_loss_cosine_similarity
from casmi.spectra.similarity import binned_cosine_similarity, modified_cosine_similarity, peak_overlap

SPECTRAL_SCORE_PREFIXES = ("cosine", "modified_cosine", "peak_overlap_frac", "neutral_loss_cosine")
AGGREGATION_METHODS = ("max", "top3_mean")


def compute_pair_spectral_scores(query_peaks, reference_peaks_list, bin_width=0.1, peak_tol_da=0.02):
    """`query_peaks`: `{"mzs", "intensities", "precursor_mz"}` for ONE query spectrum.
    `reference_peaks_list`: a list of such dicts, the candidate's available reference spectra
    (already regime-filtered and self-excluded by the caller -- see `casmi.spectra.library`).

    Returns aggregated `{"<metric>_<agg>": value, ...}` for every metric in
    `SPECTRAL_SCORE_PREFIXES` x `AGGREGATION_METHODS`, plus `n_reference_spectra` and
    `has_reference_spectrum`. An empty `reference_peaks_list` returns every spectral field as
    NaN with `has_reference_spectrum=False` -- never raises, so mass-only ranking always stays
    possible even for a candidate with no usable reference spectrum."""
    n_refs = len(reference_peaks_list)
    if n_refs == 0:
        out = {"n_reference_spectra": 0, "has_reference_spectrum": False}
        for prefix in SPECTRAL_SCORE_PREFIXES:
            for agg in AGGREGATION_METHODS:
                out[f"{prefix}_{agg}"] = float("nan")
        return out

    q_bins, q_vals = bin_spectrum(query_peaks["mzs"], query_peaks["intensities"], bin_width=bin_width)
    cosines, mod_cosines, overlaps, nl_cosines = [], [], [], []
    for ref in reference_peaks_list:
        r_bins, r_vals = bin_spectrum(ref["mzs"], ref["intensities"], bin_width=bin_width)
        cosines.append(binned_cosine_similarity(q_bins, q_vals, r_bins, r_vals))
        mod_cosines.append(modified_cosine_similarity(
            query_peaks["mzs"], query_peaks["intensities"], query_peaks["precursor_mz"],
            ref["mzs"], ref["intensities"], ref["precursor_mz"], tol_da=peak_tol_da,
        )["score"])
        overlaps.append(peak_overlap(query_peaks["mzs"], ref["mzs"], tol_da=peak_tol_da)["frac_a_matched"])
        nl_cosines.append(neutral_loss_cosine_similarity(
            query_peaks["mzs"], query_peaks["intensities"], query_peaks["precursor_mz"],
            ref["mzs"], ref["intensities"], ref["precursor_mz"], bin_width=bin_width,
        ))

    out = {"n_reference_spectra": n_refs, "has_reference_spectrum": True}
    for prefix, values in zip(SPECTRAL_SCORE_PREFIXES, (cosines, mod_cosines, overlaps, nl_cosines)):
        for agg_name, agg_value in aggregate_scores(values, methods=AGGREGATION_METHODS).items():
            out[f"{prefix}_{agg_name}"] = agg_value
    return out


def compute_single_pair_scores(query_peaks, reference_peaks, bin_width=0.1, peak_tol_da=0.02):
    """The same four metrics `compute_pair_spectral_scores` computes per-reference inside its
    own loop, exposed for ONE (query, reference) pair -- for callers that need to compute and
    CACHE per-reference scores once (e.g. a similarity cache keyed by (query, reference), reused
    across every candidate/protocol that reference could be walked under) rather than
    recomputing them inside an aggregation already scoped to one candidate's reference list.
    Returns `(cosine, modified_cosine, peak_overlap_frac, neutral_loss_cosine)`."""
    q_bins, q_vals = bin_spectrum(query_peaks["mzs"], query_peaks["intensities"], bin_width=bin_width)
    r_bins, r_vals = bin_spectrum(reference_peaks["mzs"], reference_peaks["intensities"], bin_width=bin_width)
    cosine = binned_cosine_similarity(q_bins, q_vals, r_bins, r_vals)
    modified_cosine = modified_cosine_similarity(
        query_peaks["mzs"], query_peaks["intensities"], query_peaks["precursor_mz"],
        reference_peaks["mzs"], reference_peaks["intensities"], reference_peaks["precursor_mz"], tol_da=peak_tol_da,
    )["score"]
    overlap = peak_overlap(query_peaks["mzs"], reference_peaks["mzs"], tol_da=peak_tol_da)["frac_a_matched"]
    nl_cosine = neutral_loss_cosine_similarity(
        query_peaks["mzs"], query_peaks["intensities"], query_peaks["precursor_mz"],
        reference_peaks["mzs"], reference_peaks["intensities"], reference_peaks["precursor_mz"], bin_width=bin_width,
    )
    return cosine, modified_cosine, overlap, nl_cosine


def aggregate_pair_scores_from_values(cosines, modified_cosines, peak_overlaps, neutral_loss_cosines):
    """Aggregate ALREADY-COMPUTED per-reference scores (e.g. from a similarity cache / QCR
    table's `accepted_<protocol>` rows) into the same `{"<metric>_<agg>": value, ...}` shape
    `compute_pair_spectral_scores` returns -- makes ZERO similarity calls, pure
    filter/aggregate. Empty input returns every field NaN with `has_reference_spectrum=False`,
    matching `compute_pair_spectral_scores`'s empty-list contract exactly."""
    n_refs = len(cosines)
    if n_refs == 0:
        out = {"n_reference_spectra": 0, "has_reference_spectrum": False}
        for prefix in SPECTRAL_SCORE_PREFIXES:
            for agg in AGGREGATION_METHODS:
                out[f"{prefix}_{agg}"] = float("nan")
        return out

    out = {"n_reference_spectra": n_refs, "has_reference_spectrum": True}
    for prefix, values in zip(SPECTRAL_SCORE_PREFIXES, (cosines, modified_cosines, peak_overlaps, neutral_loss_cosines)):
        for agg_name, agg_value in aggregate_scores(values, methods=AGGREGATION_METHODS).items():
            out[f"{prefix}_{agg_name}"] = agg_value
    return out
