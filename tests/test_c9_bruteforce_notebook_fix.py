"""Regression tests for the 10v4a.1 C9 notebook cell fix (independent brute-force equivalence
audit). These test the *pattern* the fixed notebook cell relies on -- brute-force classification
must not depend on what the lazy walk already computed, and lazy vs. brute features must be
genuinely independent comparisons -- using the same production primitives
(casmi.spectra.reference_selection.walk_references, casmi.validation.reference_audit.
brute_force_reference_selection, casmi.ranking.features.aggregate_pair_scores_from_values) that
the notebook's feature_check() calls. feature_check() itself lives only in the notebook build
script (not in the casmi package), so it is not imported directly here.

NOT EXECUTED as part of writing this fix -- see the notebook build script's "CODE-AUTHORING-ONLY
MODE" constraint. Run with: pytest tests/test_c9_bruteforce_notebook_fix.py -v
"""
import numpy as np
import pytest

from casmi.ranking.features import aggregate_pair_scores_from_values
from casmi.spectra.reference_selection import PROTOCOLS, walk_references
from casmi.validation.reference_audit import brute_force_reference_selection

FEATURE_COLS = ["cosine_max", "cosine_top3_mean", "modified_cosine_max", "modified_cosine_top3_mean",
                "peak_overlap_frac_max", "peak_overlap_frac_top3_mean", "neutral_loss_cosine_max", "neutral_loss_cosine_top3_mean"]


def _query(source="lib_a"):
    return {"adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV", "instrument": "Orbitrap", "source": source}


def _ref(spectrum_id, source="lib_b"):
    return {"spectrum_id": spectrum_id, "adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV",
            "instrument": "Orbitrap", "source": source}


def _long_scenario(n=30):
    """n references, all eligible under every protocol (cross-source, non-mirror tier) -- the
    lazy walk reaches its 5-per-protocol quota well before the end of a list this long, so
    references near the tail are never passed to any classify_fn during the lazy walk. This is
    the same shape as a real HOST/DEV query: most candidates have far more compatible
    references than the 5-per-protocol cap needs."""
    refs = [_ref(f"r{i:03d}") for i in range(n)]
    lookup = {r["spectrum_id"]: r for r in refs}
    ranked = [r["spectrum_id"] for r in refs]
    evidence = {f"r{i:03d}": ("T4", 0.5 + i * 0.001, 0.4 + i * 0.001, 0.3 + i * 0.001, 0.2 + i * 0.001) for i in range(n)}
    return ranked, lookup, evidence


def test_bruteforce_audit_does_not_require_every_reference_to_already_be_computed():
    """Regression test for the C9 bug where the brute-force classify_fn raised RuntimeError for
    any reference the lazy walk hadn't already computed (i.e. it required a SimilarityCache hit).
    The lazy walk stops as soon as every protocol has 5 accepted references -- here that happens
    well before reference r029 -- so a correct brute-force audit must independently compute
    r029's evidence itself and must never assume it exists anywhere already."""
    ranked, lookup, evidence = _long_scenario(n=30)
    query_meta = _query()

    lazy_computed = set()

    def lazy_classify(rid):
        lazy_computed.add(rid)
        return evidence[rid]

    lazy = walk_references(ranked, query_meta, lookup, lazy_classify)
    assert lazy.n_references_walked < len(ranked)  # confirms the walk really did stop early
    assert "r029" not in lazy_computed  # the tail reference was never touched by the lazy walk

    def brute_classify(rid):
        # independently recomputed -- does NOT consult lazy_computed, must not raise for r029
        return evidence[rid]

    brute_accepted, brute_classified = brute_force_reference_selection(ranked, query_meta, lookup, brute_classify)
    assert "r029" in brute_classified  # the tail reference WAS independently evaluated
    for p in PROTOCOLS:
        assert lazy.accepted[p] == brute_accepted[p]


def test_feature_mismatch_is_detected_when_brute_evidence_is_perturbed():
    """Regression test for the C9 bug where lazy and brute features were both read from
    rows_lazy, so the comparison could never fail even when the brute-force recomputation
    disagreed. Here brute_classified deliberately holds a different cosine for one accepted
    reference than what the lazy walk recorded -- the aggregated feature comparison must catch
    this, proving the two sides are genuinely independent rather than comparing a value to
    itself."""
    ranked, lookup, evidence = _long_scenario(n=10)
    query_meta = _query()

    def lazy_classify(rid):
        return evidence[rid]

    lazy = walk_references(ranked, query_meta, lookup, lazy_classify)
    rows_lazy = {r["reference_spectrum_id"]: r for r in lazy.rows}

    accepted_ids = lazy.accepted["standard"]
    assert accepted_ids, "scenario must actually select references under 'standard'"
    perturbed_id = accepted_ids[0]
    brute_classified = dict(evidence)
    tier, cosine, mcos, overlap, nl = evidence[perturbed_id]
    brute_classified[perturbed_id] = (tier, cosine + 0.3, mcos, overlap, nl)  # deliberate perturbation

    lazy_feat = aggregate_pair_scores_from_values(
        [rows_lazy[r]["cosine"] for r in accepted_ids], [rows_lazy[r]["modified_cosine"] for r in accepted_ids],
        [rows_lazy[r]["peak_overlap_frac"] for r in accepted_ids], [rows_lazy[r]["neutral_loss_cosine"] for r in accepted_ids],
    )
    brute_feat = aggregate_pair_scores_from_values(
        [brute_classified[r][1] for r in accepted_ids], [brute_classified[r][2] for r in accepted_ids],
        [brute_classified[r][3] for r in accepted_ids], [brute_classified[r][4] for r in accepted_ids],
    )
    assert not np.isclose(lazy_feat["cosine_max"], brute_feat["cosine_max"], atol=1e-9)


def test_reference_id_selection_matches_independent_brute_force_on_full_list():
    """Positive control: when the same evidence function backs both the lazy walk and an
    independent brute-force evaluation of the full reference list, every protocol's accepted
    reference-ID list must match exactly -- the case that should PASS in the real notebook
    run."""
    ranked, lookup, evidence = _long_scenario(n=25)
    query_meta = _query()

    def classify(rid):
        return evidence[rid]

    lazy = walk_references(ranked, query_meta, lookup, classify)
    brute_accepted, _ = brute_force_reference_selection(ranked, query_meta, lookup, classify)
    for p in PROTOCOLS:
        assert lazy.accepted[p] == brute_accepted[p]


def test_feature_values_match_independent_brute_force_when_evidence_is_consistent():
    """Positive control for the feature-level half of C9: aggregating features from the lazy
    walk's own recorded rows must equal aggregating features from an independently recomputed
    brute_classified dict, when both were built from the same underlying evidence function --
    the case that should PASS in the real notebook run."""
    ranked, lookup, evidence = _long_scenario(n=25)
    query_meta = _query()

    def classify(rid):
        return evidence[rid]

    lazy = walk_references(ranked, query_meta, lookup, classify)
    _, brute_classified = brute_force_reference_selection(ranked, query_meta, lookup, classify)
    rows_lazy = {r["reference_spectrum_id"]: r for r in lazy.rows}

    for p in PROTOCOLS:
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
        for feat_name in FEATURE_COLS:
            assert np.isclose(lazy_feat[feat_name], brute_feat[feat_name], atol=1e-9, rtol=0, equal_nan=True)
