import math
from pathlib import Path

import pandas as pd
import pytest

from casmi.workspace.config import deep_merge, load_v2_config, resolve_paths
from casmi.workspace.environment import choose_amp_dtype, recommend_batch_size
from casmi.workspace.experiments import EXPERIMENT_COLUMNS, load_experiments, log_experiment

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "casmi_v2_colab.yaml"


def test_env_var_overrides_drive_root(tmp_path):
    cfg, paths = load_v2_config(CFG_PATH, environ={"ENVEDA_DRIVE_ROOT": str(tmp_path / "drive")})
    assert paths.drive_root == tmp_path / "drive"
    assert paths.reports_dir == tmp_path / "drive" / "reports"
    assert paths.candidate_db_dir == tmp_path / "drive" / "candidates"
    assert str(paths.scratch_dir).replace("\\", "/") == "/content/scratch"     # POSIX-rooted scratch is never re-rooted


def test_default_drive_root_without_env():
    cfg, paths = load_v2_config(CFG_PATH, environ={})
    assert str(paths.drive_root).replace("\\", "/") == "/content/drive/MyDrive/EnvedaCASMI"


def test_no_windows_paths_in_v2_config():
    text = CFG_PATH.read_text(encoding="utf-8")
    assert "C:\\" not in text and "OneDrive" not in text


def test_absolute_path_is_kept():
    cfg = {"paths": {"drive_root": "/d", "repo_root": "/r", **{k: "x" for k in (
        "raw_data_dir", "processed_dir", "interim_dir", "external_dir", "bundle_dir", "candidate_db_dir", "embeddings_dir",
        "checkpoints_dir", "predictions_dir", "reports_dir", "cache_dir", "experiments_dir")}, "scratch_dir": "/content/scratch"}}
    p = resolve_paths(cfg, environ={})
    assert p.scratch_dir == Path("/content/scratch")
    assert p.reports_dir == Path("/d") / "x"


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


def test_experiment_log_never_invents_metrics(tmp_path):
    log_experiment(tmp_path, "e1", "desc", ["mass"], C2_MRR25=0.25)
    df = load_experiments(tmp_path)
    assert list(df.columns) == EXPERIMENT_COLUMNS
    row = df.iloc[0]
    assert row["C2_MRR25"] == 0.25 and math.isnan(row["C1_MRR25"]) and math.isnan(row["Hit1"])
    log_experiment(tmp_path, "e1", "desc2", ["mass"], C2_MRR25=0.3)          # same id replaces
    assert len(load_experiments(tmp_path)) == 1
    with pytest.raises(KeyError):
        log_experiment(tmp_path, "e2", "d", ["x"], made_up_metric=1.0)
