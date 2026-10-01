"""Leakage-audit helpers used by notebooks 13 / 14: failures raise, limitations are reported (never silently passed)."""
import pandas as pd
import pytest

from casmi.validation.c2_protocol import GATE_A_FAIL_PROTOCOL_INVALID
from casmi.validation.leakage_audit import LeakageAuditError, audit_analog, audit_gate_a
from casmi.workspace.cache_identity import fingerprint_values, open_namespace


def test_gate_a_audit():
    sweep = pd.DataFrame({"regime": ["C2"], "pool_share_with_reference": [1.0]})
    invalid = {"protocol_valid": False, "reasons": ["x"]}
    ok = audit_gate_a(sweep, {0: 0, 1: 0}, invalid, {"decision": GATE_A_FAIL_PROTOCOL_INVALID, "metrics": None},
                      truth_rows=pd.DataFrame({"reach_external": [False, True], "truth_sources": ["none", "COCONUT"]}))
    assert ok["passed"].all()
    with pytest.raises(LeakageAuditError, match="C3 truths"):
        audit_gate_a(sweep, {0: 2}, invalid, {"decision": GATE_A_FAIL_PROTOCOL_INVALID, "metrics": None})
    with pytest.raises(LeakageAuditError, match="carries no C2 metrics"):
        audit_gate_a(sweep, {0: 0}, invalid, {"decision": "RANKING", "metrics": {"recall_all@5ppm": 0.9}})
    with pytest.raises(LeakageAuditError, match="non-TRAIN provenance"):
        audit_gate_a(sweep, {0: 0}, invalid, {"decision": GATE_A_FAIL_PROTOCOL_INVALID, "metrics": None},
                     truth_rows=pd.DataFrame({"reach_external": [True], "truth_sources": ["none"]}))


def _analog_inputs(tmp_path, hidden_in_cache):
    regimes = pd.DataFrame({"query_id": ["q1", "q2", "q3"], "true_connectivity_key": ["K1", "K2", "K3"], "fold": [0, 0, 0],
                            "regime": ["C1", "C2", "C3"]})
    q = regimes.copy()
    ident = {"hidden_reference_keys": fingerprint_values(hidden_in_cache), "removed_structure_keys": fingerprint_values({"K3"}),
             "query_ids": fingerprint_values(["q1", "q2", "q3"])}
    ns = open_namespace(tmp_path / "fold=0", ident)
    runs = [{"fold": 0, "features_dir": str(ns)}]
    pos = pd.Series({"q1": 1, "q2": 1})
    disj = pd.DataFrame({"fold": [0], "n_overlap": [0]})
    feats = pd.DataFrame({"query_id": ["q1", "q2"], "score_x": [0.1, 0.2]})
    return q, pos, disj, feats, runs, regimes


def test_analog_audit_detects_stale_cache_and_reports_t2_limitation(tmp_path):
    q, pos, disj, feats, runs, regimes = _analog_inputs(tmp_path / "ok", {"K2", "K3"})
    t = audit_analog(q, pos, disj, feats, ["score_x"], ["abs_mass_error_ppm"], runs, regimes, "T2 not applied")
    assert t.loc[t["kind"] == "limitation", "detail"].str.contains("T2").any()
    q, pos, disj, feats, runs, regimes = _analog_inputs(tmp_path / "stale", {"K2"})      # cache built before K3 was hidden
    with pytest.raises(LeakageAuditError, match="no stale cache"):
        audit_analog(q, pos, disj, feats, ["score_x"], ["abs_mass_error_ppm"], runs, regimes, "T2 not applied")


def test_analog_audit_rejects_provenance_features_and_c3_positives(tmp_path):
    q, pos, disj, feats, runs, regimes = _analog_inputs(tmp_path, {"K2", "K3"})
    with pytest.raises(LeakageAuditError, match="TRAIN-provenance"):
        audit_analog(q, pos, disj, feats, ["score_x"], ["abs_mass_error_ppm", "train_present"], runs, regimes, "")
    with pytest.raises(LeakageAuditError, match="C3 truths are never candidates"):
        audit_analog(q, pd.Series({"q1": 1, "q3": 1}), disj, feats, ["score_x"], ["abs_mass_error_ppm"], runs, regimes, "")
