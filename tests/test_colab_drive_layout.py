"""Colab / Drive migration tooling -- static checks only (no script is executed, nothing is copied):
asset map consistency, copy-only safety of the PowerShell scripts, Colab path helpers, staging."""
import os
import re
import time
from pathlib import Path

import pytest
import yaml

import casmi.workspace.colab_paths as colab_paths
from casmi.workspace.colab_paths import DRIVE_TREE, REQUIRED_PROCESSED_FILES, clone_or_update_repo, missing_required
from casmi.workspace.config import load_v2_config
from casmi.workspace.staging import stage_directory, stage_file, stage_if_changed

REPO = Path(__file__).resolve().parents[1]
MAP = yaml.safe_load((REPO / "configs" / "drive_asset_map.yaml").read_text(encoding="utf-8"))
SCRIPTS = {n: (REPO / "scripts" / n).read_text(encoding="utf-8") for n in
           ("copy_colab_assets_to_drive.ps1", "create_colab_drive_architecture.ps1", "find_google_drive.ps1", "lib/DriveAssetMap.ps1")}


def _code_lines(text):
    """PowerShell lines without comments / comment-based help (so docs mentioning /MIR do not count)."""
    out, in_help = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("<#"):
            in_help = True
        if in_help:
            if "#>" in s:
                in_help = False
            continue
        out.append(re.sub(r"(^|\s)#.*$", "", line))
    return "\n".join(out)


# ---- asset map -----------------------------------------------------------------------------------

def test_python_tree_matches_asset_map():
    assert list(DRIVE_TREE) == list(MAP["directories"])


def test_required_assets_are_the_first_migration_set():
    req = {k for k, a in MAP["assets"].items() if a["required"]}
    assert req == {"frozen_bundle", "train_spectrum_metadata", "connectivity_folds", "structure_table", "dev_queries", "molecule_mass_variants"}
    dests = {MAP["assets"][k]["destination"] for k in req}
    assert {f"data/processed/{f}" for f in REQUIRED_PROCESSED_FILES} | {"bundle"} == dests


def test_bundle_copied_whole_excluding_only_transient_content():
    b = MAP["assets"]["frozen_bundle"]
    assert b["source"] == "bundle" and b["destination"] == "bundle" and b["mode"] == "directory"
    assert set(b["exclude_dirs"]) <= {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    assert b["exclude_files"] == ["*.pyc"]


def test_external_assets_are_optional_and_raw_is_opt_in():
    for k, a in MAP["assets"].items():
        if a["category"] in ("external", "raw", "metadata"):
            assert a["required"] is False, k
        assert not os.path.isabs(a["source"]) and ".." not in Path(a["source"]).parts
        assert not os.path.isabs(a["destination"]) and ".." not in Path(a["destination"]).parts


def test_no_repository_folder_on_drive():
    assert not any(d.split("/")[0] in ("repo", "src", "notebooks") for d in MAP["directories"])
    assert not any(a["destination"].split("/")[0] in ("repo", "src", "notebooks") for a in MAP["assets"].values())


# ---- PowerShell scripts: copy-only ----------------------------------------------------------------

@pytest.mark.parametrize("name", list(SCRIPTS))
def test_scripts_never_delete_or_move(name):
    code = _code_lines(SCRIPTS[name])
    for bad in ("Remove-Item", "Move-Item", "/MIR", "/PURGE", "/MOV", "/MOVE", "rmdir", "Clear-Content"):
        assert bad.lower() not in code.lower(), f"{name}: {bad}"


def test_copy_script_contract():
    s = SCRIPTS["copy_colab_assets_to_drive.ps1"]
    code = _code_lines(s)
    for flag in ("$Yes", "$SkipBundle", "$SkipExternal", "$Overwrite", "$IncludeRaw", "SupportsShouldProcess"):
        assert flag in s
    assert "'/E', '/Z', '/R:2', '/W:5'" in code
    assert "'/XC', '/XN', '/XO'" in code                       # no overwrite of existing files by default
    assert "local_to_drive_copy_manifest.json" in code
    for status in ("COPIED", "ALREADY_EXISTS", "SKIPPED", "MISSING_OPTIONAL", "ERROR"):
        assert status in code
    assert "NOT PROVIDED YET" in code
    assert "60174e39a2a3c6b4" in code and "v2-A7" in code and "V1_TL_1K_TESTSIM_STRICT" in code
    assert "Get-FileHash" not in code and "SHA256" not in code.upper().replace("SHA256S", "")


def test_architecture_script_creates_directories_only():
    code = _code_lines(SCRIPTS["create_colab_drive_architecture.ps1"])
    assert "New-Item -ItemType Directory -Force" in code
    assert "Copy-Item" not in code and "robocopy" not in code.lower()


def test_find_drive_never_guesses():
    code = _code_lines(SCRIPTS["find_google_drive.ps1"])
    assert "exit 2" in code and "-DriveRoot" in SCRIPTS["find_google_drive.ps1"]


# ---- colab paths ----------------------------------------------------------------------------------

def test_no_second_path_vocabulary():
    assert not hasattr(colab_paths, "ColabPaths"), "physical roots live in V2Paths only"
    assert "candidates/indexes" not in DRIVE_TREE            # the universe root is candidates/ itself


def test_missing_required_and_ensure_tree(tmp_path):
    cfg, P = load_v2_config(REPO / "configs" / "casmi_v2_colab.yaml",
                            environ={"ENVEDA_DRIVE_ROOT": str(tmp_path / "d"), "ENVEDA_REPO_ROOT": str(tmp_path / "repo"),
                                     "ENVEDA_SCRATCH_DIR": str(tmp_path / "w")})
    assert len(missing_required(P)) == len(REQUIRED_PROCESSED_FILES) + 1
    P.ensure()
    for f in REQUIRED_PROCESSED_FILES:
        (P.processed_dir / f).write_bytes(b"x")
    (P.bundle_dir / "config.json").write_text("{}")
    assert missing_required(P) == []
    assert not (tmp_path / "repo").exists()


def test_clone_or_update_never_echoes_token(tmp_path):
    calls = []
    cmds = clone_or_update_repo(tmp_path / "Enveda", token="SECRET", run=lambda c, check: calls.append(c))
    assert calls[0][:2] == ["git", "clone"] and "SECRET" in calls[0][-2]
    assert all("SECRET" not in part for c in cmds for part in c)
    (tmp_path / "Enveda" / ".git").mkdir(parents=True)
    calls.clear()
    clone_or_update_repo(tmp_path / "Enveda", run=lambda c, check: calls.append(c))
    assert [c[3] for c in calls] == ["fetch", "checkout", "pull"]


# ---- staging --------------------------------------------------------------------------------------

def test_stage_if_changed_reuses_and_refreshes(tmp_path):
    src = tmp_path / "drive" / "a.bin"
    src.parent.mkdir()
    src.write_bytes(b"12345")
    dst = tmp_path / "work" / "a.bin"
    assert stage_if_changed(src, dst, min_free_gb=0)[1] == "copied"
    assert stage_if_changed(src, dst, min_free_gb=0)[1] == "reused"
    src.write_bytes(b"123456789")                                  # size changed -> re-staged
    future = time.time() + 10
    os.utime(src, (future, future))
    assert stage_if_changed(src, dst, min_free_gb=0)[1] == "copied" and dst.read_bytes() == b"123456789"


def test_stage_large_file_is_not_duplicated(tmp_path):
    src = tmp_path / "big.bin"
    src.write_bytes(b"x" * 2048)
    path, action = stage_if_changed(src, tmp_path / "w" / "big.bin", min_free_gb=0, max_file_gb=1e-9)
    assert action == "not_staged_too_large" and path == src and not (tmp_path / "w" / "big.bin").exists()


def test_stage_file_mirrors_drive_layout_and_directory(tmp_path):
    drive = tmp_path / "drive"
    (drive / "data" / "processed").mkdir(parents=True)
    f = drive / "data" / "processed" / "x.parquet"
    f.write_bytes(b"abc")
    p = stage_file(f, tmp_path / "w", drive_root=drive, min_free_gb=0)
    assert p == tmp_path / "w" / "data" / "processed" / "x.parquet"
    (drive / "data" / "processed" / "__pycache__").mkdir()
    (drive / "data" / "processed" / "__pycache__" / "y.pyc").write_bytes(b"z")
    d, s = stage_directory(drive / "data" / "processed", tmp_path / "w2", rel="proc", min_free_gb=0)
    assert s["copied"] == 1 and not (d / "__pycache__").exists()
