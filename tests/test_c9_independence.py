"""C9 independent brute-force validation tests (spec sections 62-63): the brute-force path must
never depend on the production SimilarityCache (a reference the lazy walk never reached must
still be independently classifiable), and a genuine difference in independently-recomputed
evidence must be detected as a feature mismatch, not silently absorbed because both sides read
from the same source.

Uses casmi.qcr.builder.compute_reference_evidence -- the same low-level evidence function the
v4a.1 stage notebooks' production classify_fn (cache-wrapped) and C9 brute-force audit (called
directly on independently-loaded peaks) both call, per spec section 23's "brute force may share
ONLY low-level deterministic primitives" rule -- with real numeric peak arrays, not synthetic
tuples, so this exercises the actual chemistry/config wiring, not just the walk logic.
"""
import numpy as np
import pytest

from casmi.qcr.builder import compute_reference_evidence
from casmi.ranking.features import aggregate_pair_scores_from_values
from casmi.spectra.reference_selection import PROTOCOLS, compat_sort_key, walk_references
from casmi.spectra.similarity_cache import SimilarityCache
from casmi.validation.reference_audit import brute_force_reference_selection

SIMILARITY_CONFIG = {
    "bin_width_da": 0.1, "peak_tol_da": 0.02,
    "near_dup_cosine_threshold": 0.95, "near_dup_precursor_diff_da": 0.01,
    "tolerant_mirror_precursor_diff_da": 0.005, "tolerant_mirror_min_rel_intensity": 0.01,
    "tolerant_mirror_ppm_tol": 5.0, "tolerant_mirror_abs_tol_da": 0.002,
    "tolerant_mirror_match_fraction": 0.95, "tolerant_mirror_min_correlation": 0.99,
}


def _peaks(mzs, intensities, precursor_mz):
    return {"mzs": np.array(mzs, dtype=float), "intensities": np.array(intensities, dtype=float), "precursor_mz": precursor_mz}


def _query_meta(source="lib_a"):
    return {"adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV", "instrument": "Orbitrap", "source": source}


def _ref_meta(spectrum_id, source="lib_b"):
    return {"spectrum_id": spectrum_id, "adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV",
            "instrument": "Orbitrap", "source": source}


def _synthetic_dataset(n_refs=30):
    """A query plus n_refs references, each with a distinct, varied peak pattern (so their
    similarity scores are not all identical -- a real-shaped scenario). Offsets 0 = query,
    1..n_refs = references, used as keys into identity_peaks/similarity_peaks (mirroring how
    the real pipeline keys peaks by row offset)."""
    identity_peaks = {0: _peaks([100.0, 150.0, 200.0], [1.0, 0.6, 0.3], precursor_mz=300.15)}
    similarity_peaks = dict(identity_peaks)
    for i in range(1, n_refs + 1):
        drift = i * 0.05
        identity_peaks[i] = _peaks([100.0 + drift, 150.0, 200.0 - drift], [1.0, 0.55, 0.28], precursor_mz=300.15)
        similarity_peaks[i] = identity_peaks[i]
    ranked = [f"r{i:03d}" for i in range(1, n_refs + 1)]
    offset_of = {f"r{i:03d}": i for i in range(1, n_refs + 1)}
    offset_of["query"] = 0
    ref_meta = {rid: _ref_meta(rid) for rid in ranked}
    return identity_peaks, similarity_peaks, ranked, offset_of, ref_meta


def test_bruteforce_does_not_depend_on_production_similarity_cache():
    """The lazy walk stops once every protocol has 5 accepted references; with 30 available,
    non-repeating, all-eligible references, most of the tail is never passed to the production
    cache. A correct C9 audit must still classify those tail references successfully by calling
    compute_reference_evidence directly -- never through the (necessarily incomplete)
    production SimilarityCache."""
    identity_peaks, similarity_peaks, ranked, offset_of, ref_meta = _synthetic_dataset(n_refs=30)
    query_meta = _query_meta()

    cache = SimilarityCache(config_hash="test-config")

    def production_classify(rid):
        def compute():
            return compute_reference_evidence(offset_of["query"], offset_of[rid], identity_peaks, similarity_peaks, SIMILARITY_CONFIG)
        return cache.get_or_compute("dataset", "query", rid, compute)

    lazy = walk_references(ranked, query_meta, ref_meta, production_classify)
    assert lazy.n_references_walked < len(ranked)  # confirms early stop actually happened
    assert cache.n_computed < len(ranked)  # the cache was never asked about the tail references

    def brute_classify(rid):
        # independent of `cache` entirely -- must succeed for every reference, including ones
        # `cache` never saw.
        return compute_reference_evidence(offset_of["query"], offset_of[rid], identity_peaks, similarity_peaks, SIMILARITY_CONFIG)

    brute_accepted, brute_classified = brute_force_reference_selection(ranked, query_meta, ref_meta, brute_classify)
    assert len(brute_classified) == len(ranked)  # every reference WAS independently evaluated
    tail_ref = ranked[-1]
    assert not any(k[2] == tail_ref for k in cache._store.keys())  # production cache never touched the tail reference
    assert tail_ref in brute_classified  # yet the independent brute pass classified it successfully
    for p in PROTOCOLS:
        assert lazy.accepted[p] == brute_accepted[p]


def test_feature_mismatch_detected_when_brute_peaks_are_perturbed():
    """If the independently-loaded peaks used for the brute-force pass genuinely differ from
    what production computed (simulating a real bug), the aggregated feature comparison must
    catch it -- proving the comparison is not vacuous."""
    identity_peaks, similarity_peaks, ranked, offset_of, ref_meta = _synthetic_dataset(n_refs=8)
    query_meta = _query_meta()

    def lazy_classify(rid):
        return compute_reference_evidence(offset_of["query"], offset_of[rid], identity_peaks, similarity_peaks, SIMILARITY_CONFIG)

    lazy = walk_references(ranked, query_meta, ref_meta, lazy_classify)
    rows_lazy = {r["reference_spectrum_id"]: r for r in lazy.rows}
    accepted_ids = lazy.accepted["standard"]
    assert accepted_ids

    # independently-loaded "brute" peaks, deliberately perturbed for one accepted reference
    perturbed_offset = offset_of[accepted_ids[0]]
    brute_identity = dict(identity_peaks)
    brute_similarity = dict(similarity_peaks)
    brute_similarity[perturbed_offset] = _peaks([100.0, 150.0, 200.0], [0.1, 0.05, 0.02], precursor_mz=300.15)  # very different intensities

    def brute_classify(rid):
        return compute_reference_evidence(offset_of["query"], offset_of[rid], brute_identity, brute_similarity, SIMILARITY_CONFIG)

    _, brute_classified = brute_force_reference_selection(ranked, query_meta, ref_meta, brute_classify)

    lazy_feat = aggregate_pair_scores_from_values(
        [rows_lazy[r]["cosine"] for r in accepted_ids], [rows_lazy[r]["modified_cosine"] for r in accepted_ids],
        [rows_lazy[r]["peak_overlap_frac"] for r in accepted_ids], [rows_lazy[r]["neutral_loss_cosine"] for r in accepted_ids],
    )
    brute_feat = aggregate_pair_scores_from_values(
        [brute_classified[r][1] for r in accepted_ids], [brute_classified[r][2] for r in accepted_ids],
        [brute_classified[r][3] for r in accepted_ids], [brute_classified[r][4] for r in accepted_ids],
    )
    assert not np.isclose(lazy_feat["cosine_max"], brute_feat["cosine_max"], atol=1e-9)


def test_reference_ids_and_features_match_when_evidence_is_genuinely_consistent():
    """Positive control: when both sides use the exact same peaks (no perturbation), reference
    IDs AND aggregated features must match exactly -- the case that should PASS in a real run."""
    identity_peaks, similarity_peaks, ranked, offset_of, ref_meta = _synthetic_dataset(n_refs=20)
    query_meta = _query_meta()

    def classify(rid):
        return compute_reference_evidence(offset_of["query"], offset_of[rid], identity_peaks, similarity_peaks, SIMILARITY_CONFIG)

    lazy = walk_references(ranked, query_meta, ref_meta, classify)
    brute_accepted, brute_classified = brute_force_reference_selection(ranked, query_meta, ref_meta, classify)
    rows_lazy = {r["reference_spectrum_id"]: r for r in lazy.rows}

    for p in PROTOCOLS:
        assert lazy.accepted[p] == brute_accepted[p]
        accepted_ids = lazy.accepted[p]
        if not accepted_ids:
            continue
        lazy_feat = aggregate_pair_scores_from_values(
            [rows_lazy[r]["cosine"] for r in accepted_ids], [rows_lazy[r]["modified_cosine"] for r in accepted_ids],
            [rows_lazy[r]["peak_overlap_frac"] for r in accepted_ids], [rows_lazy[r]["neutral_loss_cosine"] for r in accepted_ids],
        )
        brute_feat = aggregate_pair_scores_from_values(
            [brute_classified[r][1] for r in accepted_ids], [brute_classified[r][2] for r in accepted_ids],
            [brute_classified[r][3] for r in accepted_ids], [brute_classified[r][4] for r in accepted_ids],
        )
        assert np.isclose(lazy_feat["cosine_max"], brute_feat["cosine_max"], atol=1e-9)
