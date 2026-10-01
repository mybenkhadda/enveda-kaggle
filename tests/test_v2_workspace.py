import math
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from casmi.workspace.config import (CONFIG_SCHEMA_VERSION, INPUT_KEYS, ConfigSchemaError, V2Paths, deep_merge, external_source_specs, input_path,
                                    load_v2_config, resolve_paths)
from casmi.workspace.environment import choose_amp_dtype, recommend_batch_size
from casmi.workspace.experiments import (EXPERIMENT_COLUMNS, latest_runs, leaderboard, load_experiments, log_experiment, records_dir,
                                         write_leaderboard)

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "casmi_v2_colab.yaml"
EXPECTED_FIELDS = ("drive_root", "repo_root", "raw_data_dir", "processed_dir", "interim_dir", "external_dir", "bundle_dir", "candidate_db_dir",
                   "embeddings_dir", "checkpoints_dir", "predictions_dir", "reports_dir", "cache_dir", "experiments_dir", "scratch_dir")


def _posix(p):
    return str(p).replace("\\", "/")


def test_v2paths_contract_is_the_authoritative_field_set():
    assert tuple(f.name for f in fields(V2Paths)) == EXPECTED_FIELDS
    for alias in ("universe", "regimes", "reports", "candidates", "formula_index", "candidate_manifests", "scratch_root", "runs", "bundle"):
        assert alias not in V2Paths.field_names()


def test_default_roots():
    cfg, paths = load_v2_config(CFG_PATH, environ={})
    assert cfg["schema_version"] == CONFIG_SCHEMA_VERSION
    assert _posix(paths.drive_root) == "/content/drive/MyDrive/EnvedaCASMI"
    assert _posix(paths.repo_root) == "/content/Enveda"                    # source is a git clone, not on Drive
    assert _posix(paths.scratch_dir) == "/content/enveda_work"
    assert _posix(paths.bundle_dir).endswith("/EnvedaCASMI/bundle")
    assert _posix(paths.candidate_db_dir).endswith("/EnvedaCASMI/candidates")   # = canonical universe root
    assert _posix(paths.experiments_dir).endswith("/EnvedaCASMI/runs")
    assert _posix(paths.external_dir).endswith("/EnvedaCASMI/data/external")


def test_env_var_overrides(tmp_path):
    cfg, paths = load_v2_config(CFG_PATH, environ={"ENVEDA_DRIVE_ROOT": str(tmp_path / "d"), "ENVEDA_SCRATCH_DIR": str(tmp_path / "w")})
    assert paths.drive_root == tmp_path / "d" and paths.scratch_dir == tmp_path / "w"
    assert paths.reports_dir == tmp_path / "d" / "reports"
    assert input_path(cfg, paths, "dev_queries") == tmp_path / "d" / "data" / "processed" / "dev_queries.parquet"
    _, old = load_v2_config(CFG_PATH, environ={"ENVEDA_SCRATCH_ROOT": str(tmp_path / "old")})   # older variable still honoured
    assert old.scratch_dir == tmp_path / "old"


def test_all_inputs_come_from_data_processed():
    cfg, paths = load_v2_config(CFG_PATH, environ={})
    for k in INPUT_KEYS:
        assert cfg["inputs"][k] == f"data/processed/{k}.parquet"
        assert "interim" not in cfg["inputs"][k]
    with pytest.raises(KeyError):
        input_path(cfg, paths, "test")


def test_coconut_path_is_canonical():
    cfg, paths = load_v2_config(CFG_PATH, environ={})
    specs = external_source_specs(cfg, paths)
    coconut = [s for s in specs if "coconut" in s["template"]]
    assert len(coconut) == 1
    assert _posix(coconut[0]["file"]) == "/content/drive/MyDrive/EnvedaCASMI/data/external/coconut/coconut.csv"
    assert _posix(coconut[0]["template_path"]).endswith("/content/Enveda/configs/v6/coconut_source.json")


def test_schema_mismatch_raises_with_fix(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(CFG_PATH.read_text(encoding="utf-8").replace(CONFIG_SCHEMA_VERSION, "casmi-v2-config-2"), encoding="utf-8")
    with pytest.raises(ConfigSchemaError, match="Fix"):
        load_v2_config(p, environ={})


def test_old_alias_config_is_rejected(tmp_path):
    cfg, _ = load_v2_config(CFG_PATH, environ={})
    bad = deep_merge(cfg, {})
    bad["paths"] = {k: v for k, v in bad["paths"].items() if k != "candidate_db_dir"}
    with pytest.raises(ConfigSchemaError, match="candidate_db_dir"):
        resolve_paths(bad, environ={})


def test_repo_is_not_on_drive():
    cfg, paths = load_v2_config(CFG_PATH, environ={})
    assert not _posix(paths.repo_root).startswith(_posix(paths.drive_root))


def test_no_windows_paths_in_v2_config():
    text = CFG_PATH.read_text(encoding="utf-8")
    assert "C:\\" not in text and "OneDrive" not in text


def test_ensure_creates_tree_but_not_repo_or_bundle_content(tmp_path):
    cfg, paths = load_v2_config(CFG_PATH, environ={"ENVEDA_DRIVE_ROOT": str(tmp_path / "d"), "ENVEDA_REPO_ROOT": str(tmp_path / "repo"),
                                                   "ENVEDA_SCRATCH_DIR": str(tmp_path / "w")})
    paths.ensure()
    assert (tmp_path / "d" / "features" / "analog").is_dir() and (tmp_path / "d" / "runs" / "records").is_dir()
    assert (tmp_path / "d" / "candidates").is_dir() and (tmp_path / "d" / "validation" / "regimes").is_dir()
    assert (tmp_path / "w").is_dir() and not (tmp_path / "repo").exists()
    assert not any((tmp_path / "d" / "bundle").iterdir())


def test_posix_rooted_path_kept():
    cfg, _ = load_v2_config(CFG_PATH, environ={})
    cfg = deep_merge(cfg, {"paths": {"cache_dir": "/content/other_cache"}})
    assert _posix(resolve_paths(cfg, environ={}).cache_dir) == "/content/other_cache"


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 5}})
    assert out == {"a": {"b": 5, "c": 2}} and base == {"a": {"b": 1, "c": 2}}


def _rep(cuda=True, bf16=False, vram=16.0):
    return {"cuda_available": cuda, "bf16_supported": bf16, "vram_gb": vram, "gpu_name": "fake"}


def test_amp_policy():
    assert choose_amp_dtype("auto", _rep(cuda=False)) is None
    assert choose_amp_dtype("auto", _rep(bf16=True)) == "bf16"
    assert choose_amp_dtype("auto", _rep(bf16=False)) == "fp16"
    assert choose_amp_dtype("off", _rep(bf16=True)) is None
    with pytest.raises(RuntimeError):
        choose_amp_dtype("bf16", _rep(bf16=False))


def test_batch_size_is_conservative_and_overridable():
    assert recommend_batch_size("fingerprint", _rep(vram=16), override=7) == 7
    small = recommend_batch_size("fingerprint", _rep(vram=16))
    big = recommend_batch_size("fingerprint", _rep(vram=80))
    assert small <= big <= 1024
    assert recommend_batch_size("fingerprint", _rep(cuda=False)) < small


def test_experiment_log_is_append_only_and_never_invents_metrics(tmp_path):
    log_experiment(tmp_path, "e1", "desc", ["mass"], C2_MRR25=0.25)
    df = load_experiments(tmp_path)
    assert list(df.columns) == EXPERIMENT_COLUMNS
    row = df.iloc[0]
    assert row["C2_MRR25"] == 0.25 and math.isnan(row["C1_MRR25"]) and math.isnan(row["Top1"])
    log_experiment(tmp_path, "e1", "desc2", ["mass"], C2_MRR25=0.3, Hit1=0.1)   # legacy alias Hit1 -> Top1
    assert len(load_experiments(tmp_path)) == 2, "re-running an experiment must ADD a run, never overwrite"
    assert len(list(records_dir(tmp_path).glob("*.json"))) == 2
    latest = latest_runs(tmp_path)
    assert len(latest) == 1 and latest.iloc[0]["C2_MRR25"] == 0.3 and latest.iloc[0]["Top1"] == 0.1
    with pytest.raises(KeyError):
        log_experiment(tmp_path, "e2", "d", ["x"], made_up_metric=1.0)
    with pytest.raises(ValueError):
        log_experiment(tmp_path, "e2", "d", ["x"], decision="MAYBE")
    with pytest.raises(ValueError):
        log_experiment(tmp_path, "e2", "d", ["x"], status="GREAT")


def test_long_experiment_ids_never_collide(tmp_path):
    long_id = "x" * 300
    for _ in range(3):
        log_experiment(tmp_path, long_id, "d", "mass", MRR25=0.1)
    assert len(list(records_dir(tmp_path).glob("*.json"))) == 3


def test_record_carries_identities_and_artifacts(tmp_path):
    rec = log_experiment(tmp_path, "e", "d", "mass", notebook="13", config={"a": 1}, status="COMPLETED",
                         regime_identity={"c2_mode": "external_universe"}, universe_identity={"n_candidates": 5},
                         artifact_paths={"report": tmp_path / "r.json"}, MRR25=0.2)
    assert rec["config_hash"] and '"n_candidates": 5' in rec["universe_identity"] and "r.json" in rec["artifact_paths_json"]


def test_leaderboard_is_derived_and_excludes_protocol_invalid_runs(tmp_path):
    legacy = pd.DataFrame([{"experiment_id": "old", "date": "2026-09-01T00:00:00+00:00", "git_commit": "abc", "description": "d",
                            "channels": "mass", "config_path": None, "C1_MRR25": 0.1, "C2_MRR25": 0.2, "C3_MRR25": 0.0,
                            "composite_MRR25": 0.135, "Hit1": 0.05, "candidate_recall": 0.9, "runtime": 1.0, "notes": ""}])
    legacy.to_parquet(tmp_path / "experiments.parquet", index=False)
    log_experiment(tmp_path, "new", "d", "mass", composite_MRR25=0.2, status="COMPLETED")
    log_experiment(tmp_path, "new", "d", "mass", MRR25=0.21, fold=0, status="COMPLETED")
    log_experiment(tmp_path, "train_only_c2", "d", "mass", composite_MRR25=0.99, status="PROTOCOL_INVALID")
    df = load_experiments(tmp_path)
    assert set(df["experiment_id"]) == {"old", "new", "train_only_c2"} and len(df) == 4
    lb = write_leaderboard(tmp_path)
    assert list(lb["experiment_id"]) == ["new", "old"]                       # the protocol-invalid run never appears
    assert lb.loc[lb.experiment_id == "old", "Top1"].iloc[0] == 0.05
    assert (tmp_path / "leaderboard.parquet").exists()
    assert "train_only_c2" in set(leaderboard(tmp_path, include_invalid=True)["experiment_id"])
    assert len(load_experiments(tmp_path)) == 4                              # rebuilding the index keeps legacy rows exactly once
