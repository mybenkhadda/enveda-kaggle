"""v5.2 protocol filters over QCR: no blanket source exclusion for test_simulated, T1 always excluded,
strict excludes T2, relaxed permits T2, T3/T4 allowed, deterministic top-5, identical V1 aggregation,
parity with the existing eligibility flags, walk completeness, derived manifests, T2 rule."""
import inspect

import numpy as np
import pandas as pd
import pytest

from casmi.qcr.protocols import (PROTOCOL_DEFS, T2_RULE, aggregate_protocol, assert_walk_complete, check_existing_flag_parity,
                                 coverage_row, derive_protocol_manifest, eligible_mask, forbidden_tier_counts, protocol_eligible_counts,
                                 select_protocol_refs, t2_policy)
from casmi.ranking.features import aggregate_pair_scores_from_values
from casmi.spectra.reference_selection import is_eligible

METRICS = ("cosine", "modified_cosine", "peak_overlap_frac", "neutral_loss_cosine")


def _qcr():
    """One query, truth T with same-source refs of every tier + other-source refs; decoy D."""
    spec = [("T", "r1", 1, "T1", "libA"), ("T", "r2", 2, "T2", "libA"), ("T", "r3", 3, "T3", "libA"), ("T", "r4", 4, "T4", "libA"),
            ("T", "r5", 5, "T4", "libB"), ("T", "r6", 6, "T2", "libB"), ("T", "r7", 7, "T4", "libA"), ("T", "r8", 8, "T3", "libA"),
            ("D", "d1", 1, "T4", "libB"), ("D", "d2", 2, "T4", "libA")]
    rows = []
    for i, (cand, rid, rank, tier, src) in enumerate(spec):
        rows.append({"query_id": "q", "candidate_key": cand, "is_true": cand == "T", "ref_spectrum_id": rid, "compat_rank": rank, "tier": tier,
                     "ref_source": src, "query_source": "libA", "cosine": 0.1 * (i + 1), "modified_cosine": 0.05 * (i + 1),
                     "peak_overlap_frac": 0.02 * (i + 1), "neutral_loss_cosine": 0.03 * (i + 1)})
    q = pd.DataFrame(rows)
    q["eligible_standard"] = True
    q["eligible_mirror_aware"] = [is_eligible("mirror_aware", {"source": s}, {"source": "libA"}, t) for s, t in zip(q["ref_source"], q["tier"])]
    return q


POOL = pd.DataFrame({"query_id": ["q", "q"], "candidate_connectivity_key": ["T", "D"], "abs_mass_error_ppm": [1.0, 2.0], "is_true_candidate": [True, False]})


def test_test_simulated_has_no_blanket_source_exclusion_and_allows_t3_t4():
    q = _qcr()
    for p in ("test_simulated_strict", "test_simulated_relaxed"):
        assert not PROTOCOL_DEFS[p].source_exclusion and PROTOCOL_DEFS[p].allow_t3 and PROTOCOL_DEFS[p].allow_t4
        m = eligible_mask(q, PROTOCOL_DEFS[p])
        same_t34 = (q["ref_source"] == "libA") & q["tier"].isin(["T3", "T4"])
        assert m[same_t34.to_numpy()].all()


def test_t1_always_excluded_strict_excludes_t2_relaxed_permits():
    q = _qcr()
    for p in ("test_simulated_strict", "test_simulated_relaxed", "mirror_aware"):
        assert not eligible_mask(q, PROTOCOL_DEFS[p])[(q["tier"] == "T1").to_numpy()].any()
    assert not eligible_mask(q, PROTOCOL_DEFS["test_simulated_strict"])[(q["tier"] == "T2").to_numpy()].any()
    assert eligible_mask(q, PROTOCOL_DEFS["test_simulated_relaxed"])[(q["tier"] == "T2").to_numpy()].all()


def test_filters_reproduce_existing_flags():
    assert check_existing_flag_parity(_qcr()) == {"mirror_aware": 0, "standard": 0}


def test_deterministic_top5_in_compat_order():
    q = _qcr()
    sel = select_protocol_refs(q, PROTOCOL_DEFS["test_simulated_strict"])
    assert sel[sel["candidate_key"] == "T"]["ref_spectrum_id"].tolist() == ["r3", "r4", "r5", "r7", "r8"]
    sel2 = select_protocol_refs(q.sample(frac=1.0, random_state=0), PROTOCOL_DEFS["test_simulated_strict"])
    assert sorted(sel["ref_spectrum_id"]) == sorted(sel2["ref_spectrum_id"])
    rel = select_protocol_refs(q, PROTOCOL_DEFS["test_simulated_relaxed"])
    assert rel[rel["candidate_key"] == "T"]["ref_spectrum_id"].tolist() == ["r2", "r3", "r4", "r5", "r6"]


def test_same_aggregation_as_v1():
    q = _qcr()
    f = aggregate_protocol(q, POOL, PROTOCOL_DEFS["test_simulated_strict"]).set_index("candidate_connectivity_key")
    g = q[q["ref_spectrum_id"].isin(["r3", "r4", "r5", "r7", "r8"])]
    ref = aggregate_pair_scores_from_values(*(g[m].tolist() for m in METRICS))
    for m in METRICS:
        for a in ("max", "top3_mean"):
            assert f.loc["T", f"{m}_{a}"] == pytest.approx(ref[f"{m}_{a}"], abs=1e-12)
    assert f.loc["T", "n_reference_spectra"] == 5


def test_forbidden_tier_audit():
    q = _qcr()
    for p in ("test_simulated_strict", "test_simulated_relaxed", "mirror_aware"):
        c = forbidden_tier_counts(select_protocol_refs(q, PROTOCOL_DEFS[p]), PROTOCOL_DEFS[p])
        assert c["t1_violation_count"] == 0 and c["forbidden_tier_count"] == 0 and c["same_source_violation_count"] == 0
    bad = q.copy()
    assert forbidden_tier_counts(bad, PROTOCOL_DEFS["test_simulated_strict"])["t1_violation_count"] == 1


def test_counts_and_walk_completeness():
    q = _qcr()
    pairs = pd.DataFrame({"query_id": ["q", "q"], "candidate_key": ["T", "D"], "exhausted": [True, True]})
    c = protocol_eligible_counts(q, pairs, POOL, PROTOCOL_DEFS["test_simulated_strict"]).set_index("candidate_connectivity_key")
    assert c.loc["T", "n_eligible"] == 5 and c.loc["T", "count_is_exact"]
    early = pairs.assign(exhausted=[False, False])
    c2 = protocol_eligible_counts(q, early, POOL, PROTOCOL_DEFS["mirror_aware"])
    with pytest.raises(AssertionError):
        assert_walk_complete(c2)          # mirror has only 2 eligible for T but the walk "stopped early": QCR insufficient


def test_coverage_row_counts_all_queries():
    counts = pd.DataFrame({"query_id": ["a", "a", "b", "b"], "candidate_connectivity_key": ["T", "D", "T", "D"], "is_true_candidate": [True, False, True, False],
                           "n_eligible": [3, 0, 0, 1], "count_censored": [False, False, False, False]})
    r = coverage_row(counts, ["a", "b", "c"], "x")
    assert r["truth_ge1"] == pytest.approx(1 / 3) and r["truth_in_pool"] == pytest.approx(2 / 3)
    assert r["decoy_ge1_share_pooled"] == 0.5 and r["reference_availability_gap"] == pytest.approx(1 / 3 - 0.5)


def test_derived_manifest_is_subset_with_truth_evidence():
    man = pd.DataFrame({"query_id": ["a", "b", "c"], "connectivity_key": ["K1", "K2", "K3"], "selection_reason": "x"})
    before = man.copy()
    counts = pd.DataFrame({"query_id": ["a", "b", "c"], "candidate_connectivity_key": ["K1", "K2", "K3"], "is_true_candidate": [True, True, True],
                           "n_eligible": [2, 0, 1]})
    out, aud = derive_protocol_manifest(man, counts, "TESTSIM_STRICT")
    assert man.equals(before) and out["query_id"].tolist() == ["a", "c"] and aud["n_retained"] == 2
    assert set(out["connectivity_key"]) <= set(man["connectivity_key"])


def test_t2_rule():
    assert t2_policy(0.5, 0.5)["t2_policy"] == "STRICT_ONLY"
    assert t2_policy(0.8, 0.1)["protocols"] == ["test_simulated_strict"]
    r = t2_policy(0.49, 0.51)
    assert r["t2_policy"] == "STRICT_AND_RELAXED" and r["primary"] == "test_simulated_strict"
    assert set(r["protocols"]) == {"test_simulated_strict", "test_simulated_relaxed"}
    with pytest.raises(ValueError):
        t2_policy(1.2, 0.0)


def test_t2_rule_cannot_see_performance():
    params = set(inspect.signature(t2_policy).parameters)
    assert params == {"q_t34", "q_t12_only"} and T2_RULE["threshold"] == 0.50
