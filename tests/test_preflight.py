"""Preflight / bootstrap cell / versions: the layer that stops a notebook before it runs against the wrong source, a
missing artifact or an obsolete API."""
import ast
import os
import subprocess
from pathlib import Path

import pytest

from casmi.workspace.artifact_registry import ArtifactRegistry
from casmi.workspace.config import DEPRECATED_PATH_ALIASES, V2Paths, load_v2_config
from casmi.workspace.notebook_cells import BOOTSTRAP_CELL, BOOTSTRAP_MARKER_PREFIX, bootstrap_cell_version
from casmi.workspace.preflight import (CompatibilityError, MissingDependencyError, check_artifact_names, check_path_fields, check_repo,
                                       check_signatures, check_versions, preflight, repo_status, validate_inputs)
from casmi.workspace.versions import ARTIFACT_REGISTRY_API_VERSION, BOOTSTRAP_CELL_VERSION, NOTEBOOK_API_VERSION

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "casmi_v2_colab.yaml"


@pytest.fixture
def ws(tmp_path):
    env = {"ENVEDA_DRIVE_ROOT": str(tmp_path / "drive"), "ENVEDA_REPO_ROOT": str(tmp_path / "repo"), "ENVEDA_SCRATCH_DIR": str(tmp_path / "scratch")}
    cfg, P = load_v2_config(CFG_PATH, environ=env)
    return cfg, P, ArtifactRegistry(cfg, P)


def test_version_skew_is_reported_with_a_fix(ws):
    cfg, P, A = ws
    assert check_versions(NOTEBOOK_API_VERSION, BOOTSTRAP_CELL_VERSION, cfg, A) == []
    msg = check_versions("casmi-v2-notebooks-3")[0]
    assert msg.startswith("ERROR:") and "Fix:" in msg and NOTEBOOK_API_VERSION in msg and ARTIFACT_REGISTRY_API_VERSION in msg
    assert "bootstrap cell" in check_versions(NOTEBOOK_API_VERSION, "casmi-v2-bootstrap-1")[0]
    assert "config schema" in check_versions(NOTEBOOK_API_VERSION, cfg={**cfg, "schema_version": "casmi-v2-config-2"})[0]

    class OldRegistry:
        API_VERSION = "casmi-v2-artifacts-1"
    assert "artifact registry API" in check_versions(NOTEBOOK_API_VERSION, registry=OldRegistry())[0]


def test_deprecated_aliases_name_their_replacement():
    probs = check_path_fields(["universe", "regimes", "reports", "candidates", "reports_dir", "candidate_db_dir"])
    assert len(probs) == 4
    assert any("P.universe" in p and "ARTIFACTS.universe_root" in p for p in probs)
    assert any("P.reports" in p and "P.reports_dir" in p for p in probs)
    assert check_path_fields(V2Paths.field_names()) == []
    assert "did you mean" in check_path_fields(["reports_dri"])[0]


def test_alias_map_points_to_real_names():
    from casmi.workspace.artifact_registry import ARTIFACTS_SPECS
    fields_ = set(V2Paths.field_names())
    for alias, repl in DEPRECATED_PATH_ALIASES.items():
        assert alias not in fields_, alias
        target = repl.split()[0]
        kind, name = target.split(".", 1)
        assert (kind == "P" and name in fields_) or (kind == "ARTIFACTS" and name in ARTIFACTS_SPECS), (alias, repl)


def test_artifact_names_are_checked(ws):
    cfg, P, A = ws
    assert check_artifact_names(A, ["universe_root", "validation_regimes"]) == []
    probs = check_artifact_names(A, ["universe_index"])
    assert probs and "did you mean" in probs[0]


def test_validate_inputs_names_the_producer(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    t = validate_inputs({"ok": (tmp_path / "ok.txt", True, "nb10"), "opt": (tmp_path / "nope", False, "nb12")})
    assert t.set_index("logical_name").loc["ok", "exists"] and not t.set_index("logical_name").loc["opt", "exists"]
    with pytest.raises(MissingDependencyError, match="produced by: 12_colab"):
        validate_inputs({"universe": (tmp_path / "missing.json", True, "12_colab_candidate_universe")})


def test_signature_check_detects_obsolete_parameters():
    t, probs = check_signatures({"casmi.candidates.universe:finalize_universe": ["out_root", "bucket_list", "build_formula_index"]})
    assert probs == [] and bool(t["ok"].all())
    _, probs = check_signatures({"casmi.candidates.universe:finalize_universe": ["out_root", "formula_dir", "manifest_dir"]})
    assert probs and "formula_dir" in probs[0] and "Fix:" in probs[0]
    _, probs = check_signatures({"casmi.candidates.universe:universe_c2_mode": []})
    assert probs and "cannot be imported" in probs[0]


def test_preflight_fails_fast_listing_every_problem(ws):
    cfg, P, A = ws
    with pytest.raises(MissingDependencyError) as e:
        preflight(cfg, P, A, "13_colab_candidate_recall", NOTEBOOK_API_VERSION, requires=("validation_regimes", "universe_manifest"),
                  log=lambda *a: None)
    assert "validation_regimes" in str(e.value) and "11_colab_hidden_like_validation" in str(e.value)
    with pytest.raises(CompatibilityError) as e:
        preflight(cfg, P, A, "x", NOTEBOOK_API_VERSION, uses_paths=("universe",), uses_artifacts=("not_an_artifact",), log=lambda *a: None)
    assert "ARTIFACTS.universe_root" in str(e.value) and "not_an_artifact" in str(e.value)
    with pytest.raises(MissingDependencyError, match="coconut"):
        preflight(cfg, P, A, "12", NOTEBOOK_API_VERSION, require_external_sources=True, log=lambda *a: None)
    with pytest.raises(MissingDependencyError, match="bundle"):
        preflight(cfg, P, A, "10", NOTEBOOK_API_VERSION, check_bundle=True, log=lambda *a: None)


def _git(cwd, *args):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": os.environ["PATH"], "HOME": str(cwd), "USERPROFILE": str(cwd), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env)


def _has_git():
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.mark.skipif(not _has_git(), reason="git not available")
def test_repo_status_detects_stale_clone_and_dirty_tree(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "a.txt").write_text("1")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-q", "-m", "one")
    _git(tmp_path, "clone", "-q", str(origin), "clone")
    clone = tmp_path / "clone"
    st = repo_status(clone, fetch=True)
    assert st["is_git"] and st["branch"] == "main" and st["behind"] == 0 and check_repo(st) == []
    (origin / "a.txt").write_text("2")
    _git(origin, "commit", "-q", "-am", "two")
    st = repo_status(clone, fetch=True)
    assert st["behind"] == 1 and any("STALE SOURCE" in p for p in check_repo(st))
    (clone / "a.txt").write_text("local edit")
    st = repo_status(clone)
    assert st["dirty_files"] == ["a.txt"] and any("modified" in p for p in check_repo(st, allow_dirty=False, allow_behind=True))


def test_bootstrap_cell_is_valid_versioned_and_always_syncs():
    ast.parse(BOOTSTRAP_CELL)
    assert BOOTSTRAP_CELL.startswith(BOOTSTRAP_MARKER_PREFIX) and bootstrap_cell_version(BOOTSTRAP_CELL) == BOOTSTRAP_CELL_VERSION
    for needle in ("'fetch', 'origin', BRANCH", "'pull', '--ff-only', 'origin', BRANCH", "del sys.modules[_m]", "versions.py",
                   "ENVEDA_DRIVE_ROOT", "ENVEDA_REPO_ROOT", "ENVEDA_BOOTSTRAP_CELL", "drive.mount"):
        assert needle in BOOTSTRAP_CELL, needle
    assert '"' not in BOOTSTRAP_CELL and "\\" not in BOOTSTRAP_CELL
    assert bootstrap_cell_version("# >>> CASMI BOOTSTRAP (generated by scripts/stamp_notebooks.py -- do not edit by hand)") == "casmi-v2-bootstrap-1"
    assert bootstrap_cell_version("print(1)") is None


def test_bootstrap_refuses_without_the_generated_cell(monkeypatch):
    from casmi.workspace.bootstrap import bootstrap
    monkeypatch.delenv("ENVEDA_BOOTSTRAP_CELL", raising=False)
    with pytest.raises(CompatibilityError, match="bootstrap cell"):
        bootstrap("x", notebook_api=NOTEBOOK_API_VERSION)
    monkeypatch.setenv("ENVEDA_BOOTSTRAP_CELL", "casmi-v2-bootstrap-1")
    with pytest.raises(CompatibilityError, match="bootstrap cell"):
        bootstrap("x", notebook_api=NOTEBOOK_API_VERSION)
    monkeypatch.setenv("ENVEDA_BOOTSTRAP_CELL", BOOTSTRAP_CELL_VERSION)
    with pytest.raises(CompatibilityError, match="notebook API"):
        bootstrap("x", notebook_api="casmi-v2-notebooks-3")


def test_universe_source_summary_counts(tmp_path):
    import json

    import pandas as pd

    from casmi.candidates.universe import universe_source_summary
    (tmp_path / "buckets").mkdir()
    pd.DataFrame({"candidate_sources": [["TRAIN"], ["TRAIN", "COCONUT"], ["COCONUT"], ["PUBCHEM", "COCONUT"]]}).to_parquet(
        tmp_path / "buckets" / "bucket=A.parquet", index=False)
    pd.DataFrame({"candidate_sources": [["TRAIN"], ["TRAIN"]]}).to_parquet(tmp_path / "buckets" / "bucket=B.parquet", index=False)
    (tmp_path / "bucket_offsets.json").write_text(json.dumps({"offsets": {"A": 0, "B": 4}, "n_candidates": 6}))
    s = universe_source_summary(tmp_path)
    assert (s["n_candidates"], s["train_only"], s["train_and_external"], s["external_only"], s["external_any"]) == (6, 3, 1, 2, 3)
    assert s["per_source"] == {"COCONUT": 3, "PUBCHEM": 1, "TRAIN": 4}


def test_resource_monitor_records_stages():
    import numpy as np

    from casmi.workspace.resources import ResourceMonitor
    logs = []
    mon = ResourceMonitor(log=logs.append)
    with mon.stage("x", n_items=100, unit="rows"):
        np.ones(1000).sum()
    t = mon.table()
    assert list(t["stage"]) == ["x"] and t["seconds"].iloc[0] >= 0 and logs
    assert mon.progress("p", 5, 10)["done"] == 5
