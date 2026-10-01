import math

import pytest

from casmi.spectra.reference_selection import PROTOCOLS, walk_references
from casmi.validation.reference_audit import (
    assert_lazy_matches_brute_force, brute_force_reference_selection, reference_coverage_summary, walk_depth_summary,
)


def _query(source="lib_a"):
    return {"adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV", "instrument": "Orbitrap", "source": source}


def _ref(spectrum_id, source="lib_b"):
    return {"spectrum_id": spectrum_id, "adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV",
            "instrument": "Orbitrap", "source": source}


def _mixed_scenario(n=40):
    """A varied deterministic scenario: alternating source, and every 7th reference a mirror,
    every 5th (non-mirror) a near-duplicate -- enough variety for a real equivalence check."""
    refs = []
    for i in range(n):
        source = "lib_a" if i % 3 == 0 else "lib_b"
        refs.append(_ref(f"r{i:03d}", source=source))
    lookup = {r["spectrum_id"]: r for r in refs}
    ranked = [r["spectrum_id"] for r in refs]

    def classify(rid):
        i = int(rid[1:])
        if i % 7 == 0:
            tier = "T1"
        elif i % 5 == 0:
            tier = "T3"
        else:
            tier = "T4"
        return (tier, 0.5, 0.5, 0.5, 0.5)

    return ranked, lookup, classify


def test_brute_force_matches_lazy_walk_on_varied_scenario():
    ranked, lookup, classify = _mixed_scenario(n=40)
    query_meta = _query()

    lazy = walk_references(ranked, query_meta, lookup, classify)
    brute_accepted, _ = brute_force_reference_selection(ranked, query_meta, lookup, classify)

    assert_lazy_matches_brute_force(lazy, brute_accepted)  # must not raise
    for p in PROTOCOLS:
        assert lazy.accepted[p] == brute_accepted[p]


def test_brute_force_matches_lazy_walk_when_list_is_short():
    ranked, lookup, classify = _mixed_scenario(n=4)  # fewer than MAX_REFS -- exhausted case
    query_meta = _query()
    lazy = walk_references(ranked, query_meta, lookup, classify)
    brute_accepted, _ = brute_force_reference_selection(ranked, query_meta, lookup, classify)
    assert_lazy_matches_brute_force(lazy, brute_accepted)


def test_assert_lazy_matches_brute_force_raises_on_real_mismatch():
    ranked, lookup, classify = _mixed_scenario(n=10)
    lazy_result = walk_references(ranked, _query(), lookup, classify)
    # deliberately wrong brute-force result (empty selections) -- must raise, not silently pass
    wrong_brute = {p: [] for p in PROTOCOLS}
    with pytest.raises(AssertionError):
        assert_lazy_matches_brute_force(lazy_result, wrong_brute)


def test_reference_coverage_summary_exact_thresholds():
    ranked, lookup, classify = _mixed_scenario(n=40)
    query_meta = _query()
    wr_true = walk_references(ranked, query_meta, lookup, classify)
    wr_decoy_short = walk_references(ranked[:2], query_meta, {k: lookup[k] for k in ranked[:2]}, classify)

    walk_results = {("q1", "true_cand"): wr_true, ("q1", "decoy_cand"): wr_decoy_short}
    is_true_by_key = {("q1", "true_cand"): True, ("q1", "decoy_cand"): False}

    summary = reference_coverage_summary(walk_results, is_true_by_key)
    true_std = summary.query("protocol == 'standard' and group == 'true_candidates'").iloc[0]
    decoy_std = summary.query("protocol == 'standard' and group == 'decoys'").iloc[0]
    assert true_std["frac_ge5"] == 1.0  # 40 references, all eligible under standard
    assert decoy_std["frac_ge5"] == 0.0  # only 2 references available
    assert decoy_std["n_zero"] == 0


def test_walk_depth_summary_reports_exact_depth_and_far_fractions():
    ranked, lookup, classify = _mixed_scenario(n=40)
    query_meta = _query()
    wr = walk_references(ranked, query_meta, lookup, classify)
    walk_results = {("q1", "cand"): wr}

    summary = walk_depth_summary(walk_results)
    standard_row = summary.query("protocol == 'standard'").iloc[0]
    assert standard_row["max_depth"] == wr.walk_depth["standard"]
    assert not math.isnan(standard_row["median_depth"])
