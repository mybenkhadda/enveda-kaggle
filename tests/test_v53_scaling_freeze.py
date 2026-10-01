"""v5.3: test_simulated_strict derived manifests (subset, nested, leakage-free, truth evidence), primary
protocol semantics + hash + persisted-definition verification, exact V1 feature contract, TL_EVAL-only
selection, paired bootstrap resample identity, fixed scale rule, freeze conditions, protocol-aware
molecule guard, artifact sidecars."""
import json

import numpy as np
import pandas as pd
import pytest

from casmi.qcr.protocol_features import sidecar_valid, write_sidecar
from casmi.qcr.protocols import (PRIMARY_PROTOCOL, PROTOCOL_DEFS, aggregate_protocol, derive_protocol_manifest, protocol_definition_record,
                                 protocol_semantic_hash, t2_policy, verify_protocol_definition_file)
from casmi.ranking.freeze import (FREEZE_CONDITIONS, FreezeIntegrityError, SpectrumModelNotFrozen, feature_order_hash, freeze_status,
                                  model_file_hashes, require_frozen_spectrum_model)
from casmi.ranking.lambdamart import BASE_FEATURES, FORBIDDEN_MODEL_FEATURES, assert_feature_list_allowed
from casmi.ranking.scaling import V53_MODELS, V53_NESTED_ORDER, V53_PAIRS, V53_PROTOCOL, V53_RULE, scale_decision, select_scale_headline
from casmi.ranking.selection import HostLeakError
from casmi.validation.cluster_bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap

V1 = ["abs_mass_error_ppm", "cosine_max", "cosine_top3_mean", "modified_cosine_max", "modified_cosine_top3_mean", "peak_overlap_frac_max",
      "peak_overlap_frac_top3_mean", "neutral_loss_cosine_max", "neutral_loss_cosine_top3_mean"]


# ---- manifests ------------------------------------------------------------------------------------

def _nested():
    ids = [f"q{i:03d}" for i in range(100)]
    base = pd.DataFrame({"query_id": ids, "connectivity_key": [f"K{i}" for i in range(100)], "selection_reason": "x"})
    counts = pd.DataFrame({"query_id": ids, "candidate_connectivity_key": [f"K{i}" for i in range(100)], "is_true_candidate": True,
                           "n_eligible": [(i * 7) % 4 for i in range(100)]})
    return {"TL_1K": base.iloc[:10], "TL_3K": base.iloc[:30], "TL_10K": base}, counts


def test_derived_manifests_subset_nested_and_truth_evidence():
    ms, counts = _nested()
    before = {k: v.copy() for k, v in ms.items()}
    der = {k: derive_protocol_manifest(v, counts, "TESTSIM_STRICT")[0] for k, v in ms.items()}
    for k in ms:
        assert ms[k].equals(before[k])                                          # originals immutable
        assert set(der[k]["query_id"]) <= set(ms[k]["query_id"])
        assert (der[k]["query_id"].map(counts.set_index("query_id")["n_eligible"]) >= 1).all()
    assert set(der["TL_1K"]["query_id"]) <= set(der["TL_3K"]["query_id"]) <= set(der["TL_10K"]["query_id"])


def test_derived_manifests_inherit_no_leakage():
    ms, counts = _nested()
    der = derive_protocol_manifest(ms["TL_10K"], counts, "TESTSIM_STRICT")[0]
    tl_eval_keys, host_keys = {"E1", "E2"}, {"H1"}
    assert not (set(der["connectivity_key"]) & tl_eval_keys) and not (set(der["connectivity_key"]) & host_keys)


# ---- protocol --------------------------------------------------------------------------------------

def test_primary_protocol_semantics():
    assert PRIMARY_PROTOCOL == V53_PROTOCOL == "test_simulated_strict"
    p = PROTOCOL_DEFS[PRIMARY_PROTOCOL]
    assert not p.source_exclusion and set(p.excluded_tiers) == {"T1", "T2"} and p.allow_t3 and p.allow_t4


def test_protocol_hash_stable_and_distinct():
    assert protocol_semantic_hash("test_simulated_strict") == protocol_semantic_hash("test_simulated_strict")
    assert len({protocol_semantic_hash(p) for p in PROTOCOL_DEFS}) == len(PROTOCOL_DEFS)


def test_persisted_definition_verification(tmp_path):
    t2 = t2_policy(0.9, 0.05)
    rec = protocol_definition_record(PROTOCOL_DEFS["test_simulated_strict"], t2, "code", "cfg")
    p = tmp_path / "def.json"
    p.write_text(json.dumps({"definitions": {"test_simulated_strict": rec}}))
    assert verify_protocol_definition_file(p, "test_simulated_strict")[0]
    tampered = dict(rec, exclude_T2=False)
    p.write_text(json.dumps({"definitions": {"test_simulated_strict": tampered}}))
    assert not verify_protocol_definition_file(p, "test_simulated_strict")[0]
    assert not verify_protocol_definition_file(tmp_path / "missing.json", "test_simulated_strict")[0]


# ---- features ---------------------------------------------------------------------------------------

def test_exact_v1_feature_contract_and_no_audit_columns():
    assert BASE_FEATURES == V1
    for col in ("n_reference_spectra", "has_reference_spectrum", "eligible_reference_count", "source", "popularity_level"):
        assert col in FORBIDDEN_MODEL_FEATURES
        with pytest.raises(AssertionError):
            assert_feature_list_allowed(V1 + [col])


def test_top3_mean_finite_with_one_or_two_refs():
    rows = [("q", "A", "a1", 1, "T4", 0.9), ("q", "B", "b1", 1, "T3", 0.2), ("q", "B", "b2", 2, "T4", 0.6), ("q", "B", "b3", 3, "T1", 0.99)]
    q = pd.DataFrame(rows, columns=["query_id", "candidate_key", "ref_spectrum_id", "compat_rank", "tier", "cosine"])
    for m in ("modified_cosine", "peak_overlap_frac", "neutral_loss_cosine"):
        q[m] = q["cosine"] / 2
    q["ref_source"], q["query_source"], q["is_true"] = "libA", "libA", False
    pool = pd.DataFrame({"query_id": ["q", "q"], "candidate_connectivity_key": ["A", "B"], "abs_mass_error_ppm": [1.0, 1.0], "is_true_candidate": [True, False]})
    f = aggregate_protocol(q, pool, PROTOCOL_DEFS["test_simulated_strict"]).set_index("candidate_connectivity_key")
    assert f.loc["A", "cosine_top3_mean"] == pytest.approx(0.9)
    assert f.loc["B", "cosine_top3_mean"] == pytest.approx(0.4)                     # T1 b3 excluded; mean of the two eligible
    assert np.isfinite(f[[c for c in V1 if c.endswith("top3_mean")]].to_numpy()).all()


# ---- scaling ------------------------------------------------------------------------------------------

def _tbl(mrrs):
    return pd.DataFrame({"model_id": list(V53_NESTED_ORDER), "tl_eval_mrr": mrrs, "n_train_queries": [990, 2950, 9800], "status": "OK"})


def test_selection_tl_eval_only_prefers_smaller_within_margin():
    assert select_scale_headline(_tbl([0.600, 0.603, 0.604]), margin=V53_RULE["headline_margin"], candidates=V53_NESTED_ORDER)[0] == "V1_TL_1K_TESTSIM_STRICT"
    assert select_scale_headline(_tbl([0.600, 0.620, 0.640]), margin=0.005, candidates=V53_NESTED_ORDER)[0] == "V1_TL_10K_TESTSIM_STRICT"
    with pytest.raises(HostLeakError):
        select_scale_headline(_tbl([0.6, 0.6, 0.6]).assign(host_mrr=0.9), candidates=V53_NESTED_ORDER)


def test_scale_rule_fixed_on_10k_vs_3k():
    assert "10K" in V53_RULE["scale_decision"] and "3K" in V53_RULE["scale_decision"] and "CI excludes 0" in V53_RULE["scale_decision"]
    assert ("V1_TL_10K_TESTSIM_STRICT", "V1_TL_3K_TESTSIM_STRICT") in V53_PAIRS
    ok = set(V53_NESTED_ORDER)
    up = {("V1_TL_10K_TESTSIM_STRICT", "V1_TL_3K_TESTSIM_STRICT"): {"delta": 0.02, "ci_low": 0.005, "ci_high": 0.03}}
    flat = {("V1_TL_10K_TESTSIM_STRICT", "V1_TL_3K_TESTSIM_STRICT"): {"delta": 0.02, "ci_low": -0.001, "ci_high": 0.03}}
    assert scale_decision(up, ok, nested_order=V53_NESTED_ORDER)["scaling_status"] == "CONTINUE_SCALING"
    assert scale_decision(flat, ok, nested_order=V53_NESTED_ORDER)["scaling_status"] == "PLATEAU"
    assert set(V53_MODELS.values()) == {"TL_1K_TESTSIM_STRICT", "TL_3K_TESTSIM_STRICT", "TL_10K_TESTSIM_STRICT"}


def test_paired_bootstrap_uses_identical_resamples():
    rng = np.random.default_rng(0)
    a, b = rng.random(60), rng.random(60)
    cl = np.repeat(np.arange(20), 3)
    paired = paired_cluster_bootstrap(a, b, cl, n_boot=300, seed=42)
    diff = cluster_bootstrap_mean(a - b, cl, n_boot=300, seed=42)
    assert paired["ci_low"] == pytest.approx(diff["ci_low"]) and paired["ci_high"] == pytest.approx(diff["ci_high"])


# ---- freeze -------------------------------------------------------------------------------------------

def test_freeze_requires_every_condition():
    all_ok = {c: True for c in FREEZE_CONDITIONS}
    assert freeze_status(all_ok) == ("FROZEN", [])
    for c in ("scale_decision_10k_vs_3k", "protocol_hash_persisted", "selection_on_tl_eval_only", "selection_locked_before_host"):
        st, missing = freeze_status({**all_ok, c: False})
        assert st == "PROVISIONAL" and missing == [c]
    assert freeze_status({})[0] == "PROVISIONAL"


def _frozen(tmp_path, status="FROZEN", protocol="test_simulated_strict", protocol_hash=None):
    mdir = tmp_path / "m"
    mdir.mkdir()
    for k in range(5):
        (mdir / f"fold_{k}.txt").write_text(f"b{k}")
    rec = {"freeze_status": status, "model_dir": str(mdir), "feature_names": V1, "feature_order_hash": feature_order_hash(V1),
           "model_hashes": model_file_hashes(mdir), "protocol": protocol,
           "protocol_hash": protocol_semantic_hash(protocol) if protocol_hash is None else protocol_hash}
    p = tmp_path / "spectrum_model.json"
    p.write_text(json.dumps(rec))
    return p


def test_guard_frozen_matching_hashes_allowed(tmp_path):
    assert require_frozen_spectrum_model(_frozen(tmp_path))["protocol"] == "test_simulated_strict"


def test_guard_refuses_provisional_and_wrong_protocol_hash(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with pytest.raises(SpectrumModelNotFrozen):
        require_frozen_spectrum_model(_frozen(a, status="PROVISIONAL"))
    with pytest.raises(FreezeIntegrityError):
        require_frozen_spectrum_model(_frozen(b, protocol_hash="deadbeef"))


# ---- sidecars ------------------------------------------------------------------------------------------

def test_sidecar_detects_content_and_protocol_change(tmp_path):
    p = tmp_path / "f.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(p)
    write_sidecar(p, protocol="test_simulated_strict", protocol_hash="H")
    assert sidecar_valid(p, protocol="test_simulated_strict", protocol_hash="H")
    assert not sidecar_valid(p, protocol_hash="OTHER")
    pd.DataFrame({"x": [1, 3]}).to_parquet(p)
    assert not sidecar_valid(p, protocol="test_simulated_strict")
