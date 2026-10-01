"""ArtifactRegistry: the one place semantic artifact locations are derived from V2Paths roots."""
import json
from pathlib import Path

import pytest

from casmi.workspace.artifact_registry import (ARTIFACTS_SPECS, FROZEN_BUNDLE_IDENTITY, ArtifactRegistry, bundle_identity, check_bundle_identity,
                                               pipeline_status)
from casmi.workspace.config import INPUT_KEYS, V2Paths, load_v2_config
from casmi.workspace.versions import ARTIFACT_REGISTRY_API_VERSION

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "casmi_v2_colab.yaml"


@pytest.fixture
def ws(tmp_path):
    env = {"ENVEDA_DRIVE_ROOT": str(tmp_path / "drive"), "ENVEDA_REPO_ROOT": str(tmp_path / "repo"), "ENVEDA_SCRATCH_DIR": str(tmp_path / "scratch")}
    cfg, P = load_v2_config(CFG_PATH, environ=env)
    return cfg, P, ArtifactRegistry(cfg, P)


def test_every_spec_uses_a_v2paths_root_or_an_input(ws):
    cfg, P, A = ws
    assert A.API_VERSION == ARTIFACT_REGISTRY_API_VERSION
    for name, s in ARTIFACTS_SPECS.items():
        assert s.base == "inputs" or s.base in V2Paths.field_names(), name
        assert str(P.drive_root) in str(A.path(name)), name
    assert set(INPUT_KEYS) <= set(ARTIFACTS_SPECS)


def test_canonical_universe_layout(ws):
    cfg, P, A = ws
    assert A.universe_root == P.candidate_db_dir == A.candidate_staging_dir
    for name, rel in (("universe_manifest", "universe_manifest.json"), ("candidate_keys", "candidate_keys.npy"),
                      ("bucket_offsets", "bucket_offsets.json"), ("candidate_mass_index", "index"), ("candidate_formula_index", "formula_index"),
                      ("universe_buckets", "buckets"), ("universe_variants", "variants"), ("candidate_manifests_dir", "manifests")):
        assert A[name] == P.candidate_db_dir / rel, name


def test_semantic_locations(ws):
    cfg, P, A = ws
    assert A.regimes_dir == P.drive_root / "validation" / "regimes"
    assert A.validation_regimes == A.regimes_dir / "validation_regimes.parquet"
    assert A.dev_queries == P.drive_root / "data" / "processed" / "dev_queries.parquet"
    assert A.external_coconut_dir == P.external_dir / "coconut"
    assert A.fingerprint_cache_dir == P.cache_dir / "fingerprints"
    assert A.experiment_records_dir == P.experiments_dir / "records"
    assert A.gate_a_decision.parent == P.reports_dir / "candidate_recall"


def test_unknown_names_raise_with_suggestions(ws):
    cfg, P, A = ws
    with pytest.raises(KeyError, match="did you mean"):
        A.path("validation_regime")
    with pytest.raises(AttributeError):
        A.universe                               # retired alias -> not a registry name either
    assert {"name", "exists", "producer", "path"} <= set(A.table(["universe_manifest"]).columns)


def test_bundle_identity_uses_labels_only(ws):
    cfg, P, A = ws
    assert check_bundle_identity(A) and "missing" in check_bundle_identity(A)[0]
    P.bundle_dir.mkdir(parents=True)
    (P.bundle_dir / "config.json").write_text(json.dumps({**FROZEN_BUNDLE_IDENTITY, "other": 1}))
    assert bundle_identity(A) == FROZEN_BUNDLE_IDENTITY and check_bundle_identity(A) == []
    (P.bundle_dir / "config.json").write_text(json.dumps({**FROZEN_BUNDLE_IDENTITY, "CONFIG_HASH": "x"}))
    assert check_bundle_identity(A)


def _universe(A, external_any, n=10):
    A.universe_root.mkdir(parents=True, exist_ok=True)
    summary = {"n_candidates": n, "external_any": external_any, "train_only": n - external_any, "external_only": external_any,
               "train_and_external": 0, "no_source": 0, "per_source": {"TRAIN": n - external_any, "COCONUT": external_any}}
    A.universe_manifest.write_text(json.dumps({"n_candidates": n, "n_buckets": 1, "finalized_at": "t", "universe_schema_version": "v",
                                               "source_summary": summary}))
    A.bucket_offsets.write_text(json.dumps({"offsets": {"A": 0}, "n_candidates": n}))


def test_pipeline_status_blocks_train_only_universe_and_notebook_14(ws):
    cfg, P, A = ws
    P.ensure()
    t, nxt = pipeline_status(cfg, P)
    st = t.set_index("stage")["status"]
    assert st["candidate universe"] == "TODO" and st["analog baseline"] == "BLOCKED"
    _universe(A, external_any=0)
    t, _ = pipeline_status(cfg, P)
    row = t.set_index("stage").loc["candidate universe"]
    assert row["status"] == "BLOCKED" and "No non-TRAIN candidate source" in row["reason"]
    _universe(A, external_any=4)
    t, _ = pipeline_status(cfg, P)
    assert t.set_index("stage").loc["candidate universe", "status"] == "DONE"
    A.gate_a_decision.parent.mkdir(parents=True, exist_ok=True)
    A.gate_a_decision.write_text(json.dumps({"decision": "FAIL_PROTOCOL_INVALID", "protocol_valid": False, "reason": "x"}))
    t, _ = pipeline_status(cfg, P)
    st = t.set_index("stage")["status"]
    assert st["Gate A"] == "BLOCKED" and st["analog baseline"] == "BLOCKED"
