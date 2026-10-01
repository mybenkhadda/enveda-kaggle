"""C2 protocol validity: a TRAIN-only universe can never look like a valid Class-2 (external-universe) evaluation."""
import json
from pathlib import Path

import pandas as pd
import pytest

from casmi.validation.c2_protocol import (C2_MODE_EXTERNAL, C2_MODE_PRELIM_NO_UNIVERSE, C2_MODE_PRELIM_TRAIN_ONLY, GATE_A_FAIL_PROTOCOL_INVALID,
                                          assert_no_train_provenance_features, assess_c2_protocol, assess_universe, gate_a_decision,
                                          gate_a_protocol_status, universe_status_from_summary)
from casmi.workspace.artifact_registry import ArtifactRegistry
from casmi.workspace.config import load_v2_config

REPO = Path(__file__).resolve().parents[1]
GATE_CFG = {"recall_all_ppm": [2, 5, 10], "recall_at_k_ppm": 5, "pool_quantiles_ppm": 5, "decision_ppm": 10}


def _summary(n, ext):
    return {"n_candidates": n, "external_any": ext, "train_only": n - ext, "external_only": ext, "train_and_external": 0, "no_source": 0,
            "per_source": {"TRAIN": n - ext, **({"COCONUT": ext} if ext else {})}}


def test_universe_status_absent_train_only_external():
    a = universe_status_from_summary(None)
    assert (a.universe_present, a.universe_status, a.c2_mode, a.c2_protocol_valid) == (False, "absent", C2_MODE_PRELIM_NO_UNIVERSE, False)
    t = universe_status_from_summary(_summary(274195, 0), {"n_candidates": 274195, "finalized_at": "t"})
    assert (t.universe_status, t.c2_mode, t.c2_protocol_valid, t.external_source_present) == ("train_only", C2_MODE_PRELIM_TRAIN_ONLY, False, False)
    assert "No non-TRAIN candidate source" in t.reason
    e = universe_status_from_summary(_summary(300000, 25000), {"n_candidates": 300000, "finalized_at": "t"})
    assert (e.universe_status, e.c2_mode, e.c2_protocol_valid) == ("external", C2_MODE_EXTERNAL, True)
    tiny = universe_status_from_summary(_summary(10 ** 6, 3), {}, min_external_candidates=1, min_external_share=1e-4)
    assert tiny.c2_protocol_valid is False and tiny.external_source_present is True      # below the configured share


@pytest.fixture
def registry(tmp_path):
    cfg, P = load_v2_config(REPO / "configs" / "casmi_v2_colab.yaml",
                            environ={"ENVEDA_DRIVE_ROOT": str(tmp_path / "d"), "ENVEDA_REPO_ROOT": str(tmp_path / "r"),
                                     "ENVEDA_SCRATCH_DIR": str(tmp_path / "w")})
    return cfg, ArtifactRegistry(cfg, P)


def _write_universe(A, summary=None, buckets=None):
    A.universe_root.mkdir(parents=True, exist_ok=True)
    man = {"n_candidates": 6, "n_buckets": 2, "finalized_at": "2026-10-01", "universe_schema_version": "casmi-v2-universe-1"}
    if summary is not None:
        man["source_summary"] = summary
    A.universe_manifest.write_text(json.dumps(man))
    A.bucket_offsets.write_text(json.dumps({"offsets": {"A": 0, "B": 4}, "n_candidates": 6}))
    if buckets:
        A.universe_buckets.mkdir(parents=True, exist_ok=True)
        for b, sources in buckets.items():
            pd.DataFrame({"candidate_sources": sources}).to_parquet(A.universe_buckets / f"bucket={b}.parquet", index=False)


def test_manifest_existence_alone_is_not_an_external_universe(registry):
    cfg, A = registry
    _write_universe(A, buckets={"A": [["TRAIN"]] * 4, "B": [["TRAIN"]] * 2})       # manifest WITHOUT source summary
    u = assess_universe(A, cfg)
    assert u.universe_present and u.universe_status == "train_only" and not u.c2_protocol_valid
    _write_universe(A, buckets={"A": [["TRAIN"], ["TRAIN", "COCONUT"], ["COCONUT"], ["PUBCHEM"]], "B": [["TRAIN"]] * 2})
    u = assess_universe(A, cfg)
    assert u.universe_status == "external" and u.c2_protocol_valid and u.n_candidates == 6
    assert u.identity["external_any"] == 3


def test_assess_c2_protocol_requires_matching_external_regimes(registry):
    cfg, A = registry
    _write_universe(A, summary=_summary(6, 2))
    u = assess_universe(A, cfg)
    ok_meta = {"c2_mode": C2_MODE_EXTERNAL, "signature": {"universe_identity": u.identity}}
    assert assess_c2_protocol(u, ok_meta, 10)["protocol_valid"]
    p = assess_c2_protocol(u, {"c2_mode": C2_MODE_PRELIM_TRAIN_ONLY, "signature": {}}, 10)
    assert not p["protocol_valid"] and any("REBUILD" in r for r in p["reasons"])
    p = assess_c2_protocol(u, {"c2_mode": C2_MODE_EXTERNAL, "signature": {"universe_identity": {"n_candidates": 999}}}, 10)
    assert not p["protocol_valid"] and any("different universe" in r for r in p["reasons"])
    assert not assess_c2_protocol(u, ok_meta, 0)["protocol_valid"]
    t = universe_status_from_summary(_summary(6, 0), {"n_candidates": 6})
    assert not assess_c2_protocol(t, ok_meta, 10)["protocol_valid"]


def _recall_summary():
    rows = []
    for mode in ("external_only", "universe"):
        for ppm in (2, 5, 10):
            rows.append({"mode": mode, "regime": "C2", "ppm": ppm, "n_queries": 100, "recall_all": 0.5 + ppm / 100, "recall_at_25": 0.4,
                         "recall_at_100": 0.45, "pool_size_median": 10.0 * ppm, "pool_size_p90": 50.0, "pool_size_p99": 90.0})
    return pd.DataFrame(rows)


def test_gate_a_reports_metrics_only_under_a_valid_protocol():
    invalid = {"protocol_valid": False, "reasons": ["No non-TRAIN candidate source contributed"], "c2_mode": C2_MODE_PRELIM_TRAIN_ONLY}
    d = gate_a_decision(invalid, _recall_summary(), GATE_CFG)
    assert d["decision"] == GATE_A_FAIL_PROTOCOL_INVALID and d["metrics"] is None and "No non-TRAIN" in d["reason"]
    assert not gate_a_protocol_status(d)
    valid = {"protocol_valid": True, "reasons": [], "c2_mode": C2_MODE_EXTERNAL}
    d = gate_a_decision(valid, _recall_summary(), GATE_CFG)
    m = d["metrics"]
    assert d["protocol_valid"] and gate_a_protocol_status(d)
    assert m["recall_all@2ppm"] == pytest.approx(0.52) and m["recall_all@10ppm"] == pytest.approx(0.6)
    assert m["recall@100@5ppm"] == 0.45 and m["recall@25@5ppm"] == 0.4
    assert m["pool_median@5ppm"] == 50.0 and m["pool_p99@5ppm"] == 90.0


def test_train_provenance_features_are_rejected():
    assert assert_no_train_provenance_features(["abs_mass_error_ppm", "analog_rank", "source_count"]) == ["source_count"]
    for bad in ("train_present", "in_train", "has_reference_spectrum", "candidate_sources"):
        with pytest.raises(AssertionError):
            assert_no_train_provenance_features(["abs_mass_error_ppm", bad])
