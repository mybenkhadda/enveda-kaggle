"""v5.2 provenance summaries (existing tiers reused, query-level not pair-weighted, NOT_AVAILABLE kept),
SCALE_GO (semantics before MRR, standard never selectable, HOST refused), and routing v2."""
import numpy as np
import pandas as pd
import pytest

from casmi.ranking.protocol_decision import NEXT_INVALID, NEXT_RUN, NEXT_T2_REVIEW, evaluate_protocol, scale_go_decision
from casmi.ranking.selection import HostLeakError
from casmi.validation.provenance import (CATEGORY, NOT_AVAILABLE, build_truth_reference_pairs, query_level_summary, session_fields_available,
                                         t2_rule_inputs, tier_shares)
from casmi.validation.regime_audit import ROUTE_V2_MODE_A, ROUTE_V2_REF_ABSENT, ROUTE_V2_RESOLVE, project_route_v2


def _truth_qcr():
    rows = []
    # q1: 10 references (T1 x8 selected-heavy); q2: 1 T3 reference
    for i in range(10):
        rows.append(("q1", f"r{i}", i + 1, "T1" if i < 8 else "T4", i < 5))
    rows.append(("q2", "s0", 1, "T3", True))
    df = pd.DataFrame(rows, columns=["query_id", "ref_spectrum_id", "compat_rank", "tier", "accepted_standard"])
    df["candidate_key"], df["is_true"] = "K", True
    for c in ("query_source", "ref_source"):
        df[c] = "libA"
    for c in ("query_instrument", "ref_instrument"):
        df[c] = "timsTOF"
    for c in ("query_adduct", "ref_adduct"):
        df[c] = "[M+H]+"
    df["query_ion_mode"], df["ref_ion_mode"], df["ref_ce"] = "positive", "positive", 20.0
    df["mirror"], df["near_dup_non_mirror"] = df["tier"].isin(["T1", "T2"]), df["tier"] == "T3"
    for p in ("standard", "cross_library", "mirror_aware", "near_dup_strict"):
        df[f"eligible_{p}"] = p == "standard"
    return df


META = pd.DataFrame({"precursor_mz": [300.0] * 13, "ce": [20.0] * 12 + [np.nan]}, index=["q1", "q2"] + [f"r{i}" for i in range(10)] + ["s0"])


def test_pairs_reuse_existing_tiers_and_keep_not_available():
    p = build_truth_reference_pairs(_truth_qcr(), ["q1", "q2"], META, peak_hash=None, identity_thresholds={"t2_precursor_diff_da": 0.005})
    assert (p["tier"].to_numpy() == _truth_qcr()["tier"].to_numpy()).all()          # tier column passed through, never re-classified
    assert set(p["semantic_category"]) <= set(CATEGORY.values())
    assert (p["differs_session"] == NOT_AVAILABLE).all() and (p["query_peak_hash"] == NOT_AVAILABLE).all()
    assert (p.loc[p["ref_spectrum_id"] == "s0", "differs_collision_energy"] == "UNKNOWN").all()   # missing CE never counts as different
    assert (p["precursor_beyond_t3_tolerance"] == NOT_AVAILABLE).all()                             # threshold not supplied


def test_session_fields():
    assert session_fields_available(["ingest_lib", "adduct"]) == NOT_AVAILABLE
    assert session_fields_available(["run_id", "adduct"]) == ["run_id"]


def test_query_level_not_pair_weighted():
    p = build_truth_reference_pairs(_truth_qcr(), ["q1", "q2"], META)
    q = query_level_summary(p, ["q1", "q2", "q3"])
    t = tier_shares(p, q)
    assert t["pair_level_tier_share"]["T1"] == pytest.approx(8 / 11)          # pair level is dominated by q1
    assert t["query_level"]["has_T1"] == pytest.approx(1 / 3)                 # query level: one of three queries
    assert q.set_index("query_id").loc["q1", "selected_all_T1_T2"] and q.set_index("query_id").loc["q2", "selected_has_T3_T4"]
    assert q.set_index("query_id").loc["q1", "highest_risk_tier"] == "T1" and q.set_index("query_id").loc["q1", "best_non_T1_T2_tier"] == "T4"
    assert q.set_index("query_id").loc["q3", "highest_risk_tier"] == "NONE"
    inp = t2_rule_inputs(q)
    assert inp["q_t34"] == 0.5 and inp["q_t12_only"] == 0.5 and inp["n_queries_with_selected_refs"] == 2


def _ev(protocol="test_simulated_strict", **kw):
    ev = {"protocol": protocol, "semantics_allowed": True, "n_eval_truth_evidence": 500, "tl_eval_truth_coverage": 0.9, "v1_mrr_all": 0.5,
          "b1_mrr_all": 0.2, "t1_violation_count": 0, "forbidden_tier_count": 0}
    ev.update(kw)
    return ev


def test_semantics_checked_before_mrr():
    r = evaluate_protocol(_ev(t1_violation_count=3, v1_mrr_all=0.99))
    assert not r["eligible_for_scaling"] and not r["semantic_ok"] and not r["performance_ok"]
    assert not evaluate_protocol(_ev(forbidden_tier_count=1))["eligible_for_scaling"]
    assert not evaluate_protocol(_ev(n_eval_truth_evidence=99))["eligible_for_scaling"]
    assert not evaluate_protocol(_ev(v1_mrr_all=0.1))["eligible_for_scaling"]
    assert evaluate_protocol(_ev())["eligible_for_scaling"]


def test_standard_never_selectable_even_with_best_score():
    r = evaluate_protocol(_ev(protocol="standard", v1_mrr_all=0.99))
    assert not r["eligible_for_scaling"]
    d = scale_go_decision({"standard": _ev(protocol="standard", v1_mrr_all=0.99), "test_simulated_strict": _ev(v1_mrr_all=0.1)}, {"t2_policy": "STRICT_ONLY"})
    assert d["selected_protocol"] is None and d["next_step"] == NEXT_INVALID


def test_scale_go_prefers_strict_and_never_auto_selects_relaxed():
    pol = {"t2_policy": "STRICT_AND_RELAXED"}
    d = scale_go_decision({"test_simulated_strict": _ev(), "test_simulated_relaxed": _ev(protocol="test_simulated_relaxed", v1_mrr_all=0.9)}, pol)
    assert d["selected_protocol"] == "test_simulated_strict" and d["next_step"] == NEXT_RUN
    d2 = scale_go_decision({"test_simulated_strict": _ev(v1_mrr_all=0.1), "test_simulated_relaxed": _ev(protocol="test_simulated_relaxed")}, pol)
    assert d2["selected_protocol"] is None and d2["next_step"] == NEXT_T2_REVIEW


def test_host_cannot_enter_decision():
    with pytest.raises(HostLeakError):
        evaluate_protocol({**_ev(), "host_mrr": 0.9})
    with pytest.raises(HostLeakError):
        evaluate_protocol({**_ev(), "anything": 1})


@pytest.mark.parametrize("shares, route", [((0.01, 0.98, 0.01), ROUTE_V2_RESOLVE), ((0.1, 0.2, 0.7), ROUTE_V2_REF_ABSENT), ((0.8, 0.1, 0.1), ROUTE_V2_MODE_A)])
def test_routing_v2(shares, route):
    r = project_route_v2(*shares)
    assert r["project_route"] == route and r["is_gate"] is False
