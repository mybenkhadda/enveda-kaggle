"""Runtime repo + Colab/Kaggle migration (write-only tests; nothing here touches a real bundle).

Covers: code-only repo (no large artifacts, whitelist, import closure), requirements, no credentials, Colab / Kaggle
notebook stage order, Drive only on Colab, Kaggle offline by design, whole-bundle copy, frozen bundle code precedence,
identity (CONFIG_HASH) blocking, failed self-test blocking, explicit CPU fallback, GPU gated by parity, immutable results,
manual submission, and the preparation script. Portable: training repo (templates under runtime/) and runtime repo
(files at the root; the PowerShell script is not shipped there, so its checks skip)."""
import ast
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
IN_DEV = (ROOT / "runtime" / "runtime_files.json").exists()
TEMPLATE = ROOT / "runtime" if IN_DEV else ROOT
NOTEBOOKS = TEMPLATE / "notebooks"
INVENTORY = ROOT / "kaggle_model_inventory.json"          # repo root in both layouts


def _strip_bundle(rel):
    return rel[len("bundle/"):] if rel.startswith("bundle/") else rel
SPEC = json.loads(((ROOT / "runtime" / "runtime_files.json") if IN_DEV else (ROOT / "runtime_files.json")).read_text(encoding="utf-8"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from casmi_runtime import backends as rb, config as rc, frozen as rf, inference as ri, named_inference as rn, results as rr  # noqa: E402
from casmi_runtime.accelerator import detect_accelerator  # noqa: E402


def _checker():
    spec = importlib.util.spec_from_file_location("check_runtime_repo", ROOT / "scripts" / "check_runtime_repo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code(name):
    nb = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _order(cells, tokens):
    idx = []
    for t in tokens:
        hits = [i for i, s in enumerate(cells) if t in s]
        assert hits, f"stage token not found: {t}"
        idx.append(hits[0])
    return idx


# ---- repo content -------------------------------------------------------------------------------------------------

def test_whitelist_is_import_closed_and_covers_the_bundle_modules():
    from casmi_infer import SHARED_CASMI_MODULES
    wl = set(SPEC["casmi_modules"])
    assert set(SHARED_CASMI_MODULES) <= wl                                  # every module the bundle ships
    src = ROOT / "src"
    files = [src / m for m in wl] + list((src / "casmi_infer").glob("*.py")) + list((src / "casmi_runtime").glob("*.py"))
    for f in files:
        assert f.exists(), f
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
            for m in mods:
                if m.split(".")[0] == "casmi":
                    rel = m.replace(".", "/")
                    assert f"{rel}.py" in wl or f"{rel}/__init__.py" in wl, f"{f.name} imports {m}, outside the runtime whitelist"


def test_casmi_runtime_never_imports_inference_code_at_module_level():
    for f in (ROOT / "src" / "casmi_runtime").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in tree.body:                                                  # module level only
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
            assert not any(m.split(".")[0] in ("casmi", "casmi_infer") for m in mods), f"{f.name}: {mods}"


def test_requirements_parse_and_stay_minimal():
    ck = _checker()
    for name in ("requirements-colab.txt", "requirements-kaggle.txt"):
        reqs = ck.requirement_names(TEMPLATE / name)
        assert {"lightgbm", "numpy", "pandas", "pyarrow", "psutil"} <= set(reqs)
        assert not set(ck.TRAINING_ONLY) & set(reqs)
        assert all("==" not in r for n, r in reqs.items() if n != "lightgbm")
    kaggle = ck.requirement_names(TEMPLATE / "requirements-kaggle.txt")
    assert all("==" not in r for r in kaggle.values())                       # no exact pin: the self-test is the gate


def test_checker_does_not_gate_on_a_lightgbm_version(tmp_path):
    ck = _checker()
    base = _minimal_repo(tmp_path / "base", ck)
    for pin in ("lightgbm>=4.6", "lightgbm==4.6.0", "lightgbm"):
        (base / "requirements-kaggle.txt").write_text(f"{pin}\nnumpy>=1.26\npandas>=2.2\npyarrow>=15\n", encoding="utf-8")
        assert ck.check_repo(base) == [], pin
    (base / "requirements-kaggle.txt").write_text("numpy>=1.26\npandas>=2.2\npyarrow>=15\n", encoding="utf-8")
    assert any("lacks lightgbm" in p for p in ck.check_repo(base))           # but it must be declared


def test_gitignore_template():
    rules = {l.strip() for l in ((TEMPLATE / "gitignore.template") if IN_DEV else (ROOT / ".gitignore")).read_text(encoding="utf-8").splitlines()}
    assert set(_checker().GITIGNORE_RULES) <= rules
    assert not rules & {"*.py", "*.ipynb", "src/", "notebooks/", "scripts/"}


def _minimal_repo(base, ck):
    spec = json.loads(json.dumps(SPEC))
    (base).mkdir(parents=True)
    (base / "runtime_files.json").write_text(json.dumps(spec), encoding="utf-8")
    for rel in ck.required_files(spec):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n", encoding="utf-8")
    (base / ".gitignore").write_text("\n".join(ck.GITIGNORE_RULES) + "\n", encoding="utf-8")
    for name in ("requirements-colab.txt", "requirements-kaggle.txt"):
        (base / name).write_text("lightgbm>=4.6\nnumpy>=1.26\npandas>=2.2\npyarrow>=15\n", encoding="utf-8")
    for nb in ("00_colab_setup.ipynb", "01_colab_inference.ipynb", "03_runtime_benchmark.ipynb"):
        (base / "notebooks" / nb).write_text(json.dumps({"cells": []}), encoding="utf-8")
    # the Kaggle entry is audited statically: ship the real notebook, entry module and inventory
    shutil.copy2(NOTEBOOKS / "02_kaggle_inference.ipynb", base / "notebooks" / "02_kaggle_inference.ipynb")
    shutil.copy2(ROOT / "src" / "casmi_runtime" / "named_inference.py", base / "src" / "casmi_runtime" / "named_inference.py")
    shutil.copy2(INVENTORY, base / "kaggle_model_inventory.json")
    (base / "runtime_files.json").write_text(json.dumps(spec), encoding="utf-8")
    return base


def test_checker_accepts_clean_repo_and_rejects_every_artifact(tmp_path):
    ck = _checker()
    base = _minimal_repo(tmp_path / "base", ck)
    assert ck.check_repo(base) == []
    bad = {"bundle/ref_peaks_mz.npy": b"0", "data/test.parquet": b"0", "src/casmi/x.parquet": b"0", "models/v1_fold0.txt": b"tree",
           "outputs/run_report.json": b"{}", "src/casmi/training_only.py": b"x = 1\n", "big.bin": b"0" * 2_000_001,
           "notes.md": ("token " + "ghp_" + "B" * 30).encode(), "kaggle.json": ('{"username": "u", "key": "' + "a" * 32 + '"}').encode(),
           "bundle_copy/manifest.json": json.dumps({"bundle_format": "casmi-modeA-bundle-1"}).encode(),
           "src/casmi_runtime/extra.py": b"from casmi.not_shipped import x\n"}
    for i, (rel, content) in enumerate(bad.items()):
        d = tmp_path / f"v{i}"
        shutil.copytree(base, d)
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(content)
        assert ck.check_repo(d, max_mb=2.0), rel
    cache = tmp_path / "cache"
    shutil.copytree(base, cache)
    (cache / "src" / "casmi" / "__pycache__").mkdir(parents=True)
    (cache / "src" / "casmi" / "__pycache__" / "x.cpython-311.pyc").write_bytes(b"0")
    assert ck.check_repo(cache) == []                                       # gitignored caches are fine
    miss = tmp_path / "miss"
    shutil.copytree(base, miss)
    (miss / "notebooks" / "02_kaggle_inference.ipynb").unlink()
    assert any("02_kaggle_inference" in p for p in ck.check_repo(miss))


def _nb_with(nb_path, old, new):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["source"] = [l.replace(old, new) for l in c["source"]]
    Path(nb_path).write_text(json.dumps(nb), encoding="utf-8")


def _edit(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert old in text, old
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


def _edit_json(path, fn):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    fn(obj)
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


def test_checker_kaggle_e2e_static_audit(tmp_path):
    ck = _checker()
    base = _minimal_repo(tmp_path / "base", ck)
    closure, unresolved, problems = ck.kaggle_import_closure(base)
    assert not unresolved and not problems
    assert set(ck.KAGGLE_CLOSURE_MUST_INCLUDE) <= set(closure)
    assert closure["casmi_runtime.named_inference"] == "src/casmi_runtime/named_inference.py"
    assert "src/casmi_runtime/named_inference.py" in SPEC["runtime_core"] and "src/casmi_runtime/named_inference.py" in ck.RUNTIME_CORE
    nb, entry, inv = "notebooks/02_kaggle_inference.ipynb", "src/casmi_runtime/named_inference.py", "kaggle_model_inventory.json"
    variants = {
        "old_anchor": lambda d: _nb_with(d / nb, "run_named_inference", "run_frozen_inference"),
        "pip": lambda d: _nb_with(d / nb, "# 3. environment", "!pip install lightgbm==4.7.0\n# 3. environment"),
        "network_nb": lambda d: _nb_with(d / nb, "# 3. environment", "!wget https://example.org/x\n# 3. environment"),
        "hashing_nb": lambda d: _nb_with(d / nb, "# 3. environment", "import hashlib\n# 3. environment"),
        "no_drift_policy": lambda d: _nb_with(d / nb, "allow_feature_only_drift=True", "allow_feature_only_drift=False"),
        "verify_true": lambda d: _edit(d / entry, "Bundle(bundle_dir, verify=False)", "Bundle(bundle_dir, verify=True)"),
        "verify_bundle": lambda d: _edit(d / entry, "def _write_json(", "def _x(f, b):\n    return f.validation.verify_bundle(b)\n\n\ndef _write_json("),
        "hashing_fixture": lambda d: _edit(d / entry, "run_selftest(b, be, fixture=fixture)", "run_selftest(b, be)"),
        "missing_closure_module": lambda d: (d / "src" / "casmi_runtime" / "backends.py").unlink(),
        "network": lambda d: (d / "src" / "casmi_runtime" / "config.py").write_text("import urllib.request\n", encoding="utf-8"),
        "inventory_fold": lambda d: _edit_json(d / inv, lambda j: j["frozen_bundle"]["required"].remove("bundle/models/v1_fold4.txt")),
        "inventory_named": lambda d: _edit_json(d / inv, lambda j: j["runtime_source"]["required"].remove("runtime/src/casmi_runtime/named_inference.py")),
        "inventory_train": lambda d: _edit_json(d / inv, lambda j: j["competition"]["required"].append("train.parquet")),
        "inventory_runtime_infer": lambda d: _edit_json(d / inv, lambda j: j["runtime_source"]["required"].append("runtime/src/casmi_infer/pipeline.py")),
        "inventory_public": lambda d: _edit_json(d / inv, lambda j: j["model"].update(private=False)),
    }
    for name, mutate in variants.items():
        d = tmp_path / name
        shutil.copytree(base, d)
        mutate(d)
        assert ck.check_repo(d), name


def test_checker_bundle_and_competition_presence(tmp_path):
    ck = _checker()
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    payload = tmp_path / "payload"
    for rel in inv["payload_top_level"]["required"] + inv["runtime_source"]["required"] + inv["frozen_bundle"]["required"]:
        (payload / rel).parent.mkdir(parents=True, exist_ok=True)
        (payload / rel).write_text("x", encoding="utf-8")
    (payload / "bundle" / "code" / "casmi" / "chemistry" / "__init__.py").write_text("", encoding="utf-8")   # empty inits are legitimate
    (payload / "bundle" / "manifest.json").write_text(json.dumps({"bundle_format": "casmi-modeA-bundle-1", "files": {"config.json": "0"}}),
                                                      encoding="utf-8")
    (payload / "PACKAGE_INFO.json").write_text(json.dumps({**ck.PAYLOAD_IDENTITY, "bundle_integrity_mode": "FILENAMES_ONLY"}), encoding="utf-8")
    b = payload / "bundle"
    assert ck.check_bundle_dir(b, inv) == []
    assert ck.check_payload(payload, inv) == []
    for mutate, token in ((lambda: (payload / "runtime" / "src" / "casmi_infer").mkdir(parents=True), "casmi_infer"),
                          (lambda: (payload / "test.parquet").write_bytes(b"0"), "competition file"),
                          (lambda: (payload / "PACKAGE_INFO.json").write_text(json.dumps({"bundle_version": "v1"}), encoding="utf-8"), "PACKAGE_INFO")):
        mutate()
        assert any(token in p for p in ck.check_payload(payload, inv)), token
    (b / "models" / "v1_fold3.txt").unlink()
    assert any("v1_fold3" in p for p in ck.check_bundle_dir(b, inv))
    c = tmp_path / "comp"
    c.mkdir()
    assert len(ck.check_competition_dir(c)) == 2
    (c / "test.parquet").write_bytes(b"0")
    (c / "sample_submission.csv").write_text("molecule_id,smiles\n")
    assert ck.check_competition_dir(c) == []


def test_no_credentials_in_source():
    ck = _checker()
    roots = [ROOT / "src" / "casmi_runtime", NOTEBOOKS, ROOT / "scripts"] + ([TEMPLATE] if IN_DEV else [])
    for r in roots:
        for f in r.rglob("*"):
            if f.is_file() and f.suffix in (".py", ".ipynb", ".md", ".txt", ".ps1", ".json", ".template"):
                assert not ck.TOKEN_PATTERN.search(f.read_text(encoding="utf-8", errors="ignore")), f


# ---- notebooks -----------------------------------------------------------------------------------------------------

def test_colab_inference_stage_order_and_drive():
    c = _code("01_colab_inference.ipynb")
    stages = ["drive.mount(", "'clone'", "requirements-colab.txt", "detect_accelerator()", "shutil.copytree(DRIVE_BUNDLE, LOCAL_BUNDLE",
              "for name in COMPETITION_FILES:", "load_frozen_code(LOCAL_BUNDLE)", "verify_bundle(LOCAL_BUNDLE)", "verify_aggregator_contract(",
              "check_bundle_identity(", "v63_bundle_check.py", "run_frozen_inference(CFG)", "save_results_immutable("]
    idx = _order(c, stages)
    assert idx == sorted(idx), dict(zip(stages, idx))
    src = "\n".join(c)
    assert "userdata.get('GITHUB_TOKEN')" in src and "os.environ['GITHUB_TOKEN']" not in src and "http.extraHeader" in src
    assert not re.search(r"print\([^)]*token", src, flags=re.IGNORECASE)


def test_colab_copies_the_whole_bundle_to_local_disk():
    src = "\n".join(_code("01_colab_inference.ipynb"))
    assert "RUNTIME_DIR = Path('/content/casmi_runtime')" in src and "LOCAL_BUNDLE, LOCAL_COMPETITION = INPUT_ROOT / 'bundle'" in src
    assert "shutil.copytree(DRIVE_BUNDLE, LOCAL_BUNDLE" in src
    assert re.findall(r"DRIVE_BUNDLE / '([^']+)'", src) == ["manifest.json", "config.json", "manifest.json"]     # never file-by-file
    assert "missing = [f for f in man['files']" in src
    assert "train.parquet" not in json.dumps(json.loads((NOTEBOOKS / "01_colab_inference.ipynb").read_text(encoding="utf-8")))


def test_colab_setup_notebook_does_no_inference():
    src = "\n".join(_code("00_colab_setup.ipynb"))
    assert "drive.mount(" in src and "detect_accelerator.py" in src and "nvidia-smi ||" in src
    for tok in ("run_frozen_inference", "load_frozen_code", "copytree", "v63_bundle_check"):
        assert tok not in src


def test_kaggle_stage_order_offline_and_no_drive():
    c = _code("02_kaggle_inference.ipynb")
    stages = ["ENTRY_PARTS", "resolve_kaggle_model_inputs(", "CONFIG['calibration_temperature'] == 1.1947045372735254", "detect_accelerator()",
              "from casmi_runtime.named_inference import run_named_inference", "run_named_inference(", "allow_feature_only_drift=True",
              "casmi_infer_source", "kaggle_anchor_report.json", "KAGGLE OFFLINE ANCHOR PASS"]
    idx = _order(c, stages)
    assert idx == sorted(idx), dict(zip(stages, idx))
    raw = (NOTEBOOKS / "02_kaggle_inference.ipynb").read_text(encoding="utf-8")
    src = "\n".join(c)
    for tok in ("google.colab", "drive.mount", "/content/drive", "git", "http", "requests", "urllib", "KaggleApi", "competitions submit",
                "run_frozen_inference", "verify_bundle", "verify=True", "v63_bundle_check", "'pip'", "pip install", "'install'", ".whl",
                "4.7.0", "train.parquet", "wget", "curl ", "hashlib", "sha256_file", "kagglehub"):
        assert tok not in src, tok
    assert "runtime='kaggle'" in src and "/kaggle/working" in src and "/kaggle/input" in src
    assert "('runtime', 'src', 'casmi_runtime', 'named_inference.py')" in src                  # model found by file name, no slug
    for key in ("'bundle_version'] == 'v2-A7'", "'CONFIG_HASH'] == '60174e39a2a3c6b4'", "'freeze_status'] == 'FROZEN'",
                "self_test_downstream_exact'] is True", "submission_validation_status'] == 'PASS'"):
        assert key in src, key
    assert "feature_mismatches'] == 3" not in src                                              # 0 or any drift count is fine
    assert "google.colab" not in raw and "drive.mount" not in raw                        # not even in markdown


def test_frozen_code_precedence_in_the_run_and_notebooks():
    run_src = (ROOT / "src" / "casmi_runtime" / "inference.py").read_text(encoding="utf-8")
    run_src = run_src[run_src.index("def run_frozen_inference"):]                          # the function body, not the docstring
    assert run_src.index("load_frozen_code(") < run_src.index("Bundle(") < run_src.index("select_backend(") < run_src.index("run_inference(")
    assert run_src.index("validate_submission(") < run_src.index("write_submission(")
    named = (ROOT / "src" / "casmi_runtime" / "named_inference.py").read_text(encoding="utf-8")
    named = named[named.index("def run_named_inference"):]
    order = ["resolve_named_inputs(", "load_frozen_code(", "check_bundle_presence(", "Bundle(bundle_dir, verify=False)", "check_bundle_identity(",
             "check_frozen_identity(", "load_fixture_filenames_only(", "select_backend(", "run_inference(", "aggregate_molecules(",
             "validate_submission(", "write_submission(", '"run_report.json"', "compare_to_anchor(", "ANCHOR_REPORT, anchor)"]
    pos = [named.index(t) for t in order]
    assert pos == sorted(pos), dict(zip(order, pos))
    assert "verify_bundle(" not in named and "verify=True" not in named and "sha256_file(" not in named
    assert _checker().check_entry_module(ROOT / "src" / "casmi_runtime" / "named_inference.py") == []
    for nb in ("01_colab_inference.ipynb", "02_kaggle_inference.ipynb"):
        assert "casmi_infer_source" in "\n".join(_code(nb))


def test_benchmark_times_only_validated_backends():
    src = "\n".join(_code("03_runtime_benchmark.ipynb"))
    assert src.index("run_selftest(") < src.index("time_backend(")
    assert "VALIDATED" in src and "parity_vs_reference(" in src and "REQUIRE_GPU" in src


# ---- frozen code loader ----------------------------------------------------------------------------------------------

def _fake_pkg(root, marker):
    for pkg in ("casmi", "casmi_infer"):
        (root / pkg).mkdir(parents=True, exist_ok=True)
        (root / pkg / "__init__.py").write_text(f"WHO = {marker!r}\n", encoding="utf-8")
    for m in rf.FROZEN_MODULES:
        (root / "casmi_infer" / f"{m}.py").write_text(f"WHO = {marker!r}\n", encoding="utf-8")


def test_load_frozen_code_replaces_foreign_modules(tmp_path):
    saved_mods = {k: v for k, v in sys.modules.items() if k.split(".")[0] in ("casmi", "casmi_infer")}
    saved_path = list(sys.path)
    try:
        github = tmp_path / "github_src"
        _fake_pkg(github, "github")
        bundle = tmp_path / "bundle"
        _fake_pkg(bundle / "code", "frozen")
        for k in list(saved_mods):
            del sys.modules[k]
        sys.path.insert(0, str(github))
        import casmi_infer  # noqa: F401  -- the GitHub copy is imported first ...
        assert sys.modules["casmi_infer"].WHO == "github"
        fz = rf.load_frozen_code(bundle)                          # ... and must be replaced by the frozen one
        assert fz.casmi_infer.WHO == "frozen" and fz.pipeline.WHO == "frozen" and "casmi_infer" in fz.purged_modules
        assert Path(fz.source).resolve().is_relative_to((bundle / "code").resolve())
        with pytest.raises(rf.FrozenCodeError):
            rf.load_frozen_code(tmp_path / "no_bundle")
    finally:
        for k in [k for k in sys.modules if k.split(".")[0] in ("casmi", "casmi_infer")]:
            del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path


# ---- identity / blocking ------------------------------------------------------------------------------------------------

def _cfg(tmp_path, **kw):
    return rc.RuntimeConfig(runtime="colab", input_root=tmp_path / "in", work_dir=tmp_path / "out", **kw)


GOOD = ({"aggregator_name": "MOST_CONFIDENT_SPECTRUM", "model_id": "V1_TL_1K_TESTSIM_STRICT", "bundle_version": "v2-A7", "CONFIG_HASH": "abc",
         "calibration_temperature": 1.19}, {"bundle_version": "v2-A7-abc-V1_TL_1K_TESTSIM_STRICT"},
        {"model_id": "V1_TL_1K_TESTSIM_STRICT", "freeze_status": "FROZEN"})


def test_identity_check_blocks_wrong_bundle(tmp_path):
    c, m, i = GOOD
    assert ri.check_bundle_identity(c, m, i, _cfg(tmp_path))
    assert ri.check_bundle_identity(c, m, i, _cfg(tmp_path, expected_config_hash="abc"))
    for cc, mm, ii, kw in [(c, m, i, {"expected_config_hash": "zzz"}),                                   # wrong CONFIG_HASH
                           ({**c, "aggregator_name": "RRF"}, m, i, {}), ({**c, "model_id": "V1_TL_1K"}, m, i, {}),
                           (c, m, {**i, "freeze_status": "PROVISIONAL"}, {}), ({**c, "bundle_version": "v1"}, {"bundle_version": "v1-x"}, i, {}),
                           ({**c, "calibration_temperature": None}, m, i, {})]:
        with pytest.raises(ri.InferenceBlocked):
            ri.check_bundle_identity(cc, mm, ii, _cfg(tmp_path, **kw))


def test_runtime_config_validation(tmp_path):
    with pytest.raises(ValueError):
        rc.RuntimeConfig(runtime="aws", input_root=tmp_path / "in", work_dir=tmp_path / "out")
    with pytest.raises(ValueError):
        _cfg(tmp_path, device_preference="cpu", require_gpu=True)
    with pytest.raises(ValueError):                                          # outputs inside the input root would be re-discovered
        rc.RuntimeConfig(runtime="colab", input_root=tmp_path / "in", work_dir=tmp_path / "in" / "out")


def test_stale_outputs_block_the_run(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.work_dir.mkdir(parents=True)
    (cfg.work_dir / "submission.csv").write_text("old")
    with pytest.raises(ri.InferenceBlocked):
        ri.run_frozen_inference(cfg, log=lambda *a: None)
    assert (cfg.work_dir / "submission.csv").read_text() == "old"


def _named_inputs(root, bundle_format="casmi-modeA-bundle-1"):
    (root / "bundle-ds" / "bundle").mkdir(parents=True)
    (root / "bundle-ds" / "bundle" / "manifest.json").write_text(json.dumps({"bundle_format": bundle_format}), encoding="utf-8")
    (root / "comp").mkdir()
    for n in ("test.parquet", "sample_submission.csv", "train.parquet"):
        (root / "comp" / n).write_bytes(b"0")
    return root


def test_named_inputs_are_resolved_by_filename(tmp_path):
    root = _named_inputs(tmp_path / "in")
    (root / "bundle-ds" / "bundle" / "selftest").mkdir()
    (root / "bundle-ds" / "bundle" / "selftest" / "test.parquet").write_bytes(b"0")      # inside the bundle: ignored
    b, t, s = rn.resolve_named_inputs(root)
    assert b == root / "bundle-ds" / "bundle" and t == root / "comp" / "test.parquet" and s == root / "comp" / "sample_submission.csv"
    (root / "other").mkdir()
    (root / "other" / "test.parquet").write_bytes(b"0")
    with pytest.raises(ri.InferenceBlocked):                                  # ambiguous
        rn.resolve_named_inputs(root)
    with pytest.raises(ri.InferenceBlocked):                                  # no CASMI bundle
        rn.resolve_named_inputs(_named_inputs(tmp_path / "in2", bundle_format="other"))


def test_bundle_presence_check_blocks_an_incomplete_upload(tmp_path):
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    b = tmp_path / "bundle"
    listed = [_strip_bundle(r) for r in inv["frozen_bundle"]["required"] if r != "bundle/manifest.json"]
    for rel in listed:
        (b / rel).parent.mkdir(parents=True, exist_ok=True)
        (b / rel).write_text("x", encoding="utf-8")
    (b / "code" / "casmi" / "spectra" / "__init__.py").write_text("", encoding="utf-8")        # empty package init: legitimate
    man = {"files": {r: "0" for r in listed}}
    rep = rn.check_bundle_presence(b, man)
    assert rep["n_manifest_files"] == len(listed) and rep["bundle_integrity_mode"] == "FILENAMES_ONLY" and rep["bundle_hash_verification"] is False
    (b / "ref_meta.parquet").write_text("", encoding="utf-8")                                   # an empty DATA file is not
    with pytest.raises(ri.InferenceBlocked):
        rn.check_bundle_presence(b, man)
    (b / "ref_meta.parquet").write_text("x", encoding="utf-8")
    (b / "selftest" / "expected_probs.parquet").unlink()
    with pytest.raises(ri.InferenceBlocked):
        rn.check_bundle_presence(b, man)


def test_named_run_blocks_and_writes_only_the_anchor_report(tmp_path):
    cfg = rc.RuntimeConfig(runtime="kaggle", input_root=tmp_path / "in", work_dir=tmp_path / "out")
    cfg.input_root.mkdir(parents=True)
    with pytest.raises(ri.InferenceBlocked):
        rn.run_named_inference(cfg, log=lambda *a: None)
    anchor = json.loads((cfg.work_dir / "kaggle_anchor_report.json").read_text())
    assert anchor["anchor_status"] == "FAIL" and anchor["failed_stage"] == "resolve_inputs" and not anchor["submission_performed"]
    assert anchor["bundle_integrity_mode"] == "FILENAMES_ONLY" and anchor["fixture_hash_verification"] is False
    assert sorted(p.name for p in cfg.work_dir.iterdir()) == ["kaggle_anchor_report.json"]    # nothing else is written
    with pytest.raises(ri.InferenceBlocked):                                  # stale outputs are never overwritten
        rn.run_named_inference(cfg, log=lambda *a: None)


def _model_payload(root, name="enveda-casmi-v2-a7/sklearn/full-e2e/1"):
    payload = root / name
    for rel in ("runtime/src/casmi_runtime/named_inference.py", "bundle/config.json", "bundle/code/casmi_infer/__init__.py"):
        (payload / rel).parent.mkdir(parents=True, exist_ok=True)
        (payload / rel).write_text("x", encoding="utf-8")
    return payload


def test_kaggle_model_inputs_are_resolved_by_filename(tmp_path):
    root = tmp_path / "input"
    payload = _model_payload(root)
    (payload / "bundle" / "selftest").mkdir()
    (payload / "bundle" / "selftest" / "test.parquet").write_bytes(b"0")                       # inside the model: ignored
    (root / "competition").mkdir()
    for n in ("test.parquet", "sample_submission.csv", "train.parquet"):
        (root / "competition" / n).write_bytes(b"0")
    got = rn.resolve_kaggle_model_inputs(root)
    assert got == {"MODEL_PAYLOAD_ROOT": payload, "RUNTIME_SRC": payload / "runtime" / "src", "BUNDLE_ROOT": payload / "bundle",
                   "TEST_PATH": root / "competition" / "test.parquet", "SAMPLE_SUBMISSION_PATH": root / "competition" / "sample_submission.csv"}
    (root / "extra").mkdir()
    (root / "extra" / "sample_submission.csv").write_bytes(b"0")
    with pytest.raises(ri.InferenceBlocked):                                  # ambiguous competition file
        rn.resolve_kaggle_model_inputs(root)
    (root / "extra" / "sample_submission.csv").unlink()
    _model_payload(root, "another-model/sklearn/full-e2e/1")
    with pytest.raises(ri.InferenceBlocked):                                  # two attached models
        rn.resolve_kaggle_model_inputs(root)


_EXACT = {k: 0 for k in rb.HARD_MISMATCH_KEYS}


def test_selftest_policy_accepts_feature_only_drift_only_when_downstream_exact():
    strict = rb.apply_selftest_policy({**_EXACT, "feature_mismatches": 0, "passed": True}, True)
    assert strict["passed"] and strict["strict_passed"] and strict["downstream_exact"] and not strict["accepted_feature_only_drift"]
    drift = {**_EXACT, "feature_mismatches": 3, "passed": False}
    ok = rb.apply_selftest_policy(drift, True)
    assert ok["passed"] and not ok["strict_passed"] and ok["downstream_exact"] and ok["accepted_feature_only_drift"] and ok["feature_mismatches"] == 3
    assert not rb.apply_selftest_policy(drift, False)["passed"]               # strict mode never accepts drift
    for k in rb.HARD_MISMATCH_KEYS:                                           # ANY hard mismatch refuses
        bad = rb.apply_selftest_policy({**drift, k: 1}, True)
        assert not bad["passed"] and not bad["downstream_exact"], k
    assert not rb.apply_selftest_policy({**drift, "error": "boom"}, True)["passed"]
    missing = dict(drift)
    missing.pop("rank_mismatches")
    assert not rb.apply_selftest_policy(missing, True)["passed"]              # an absent hard key is never exact


def test_select_backend_records_drift_acceptance():
    fz, st = _frozen()
    runner = lambda b, be: {**_EXACT, "backend": be, "feature_mismatches": 3, "passed": False}
    rec = rb.select_backend(_bundle(), fz, GPU_HW, "gpu", runner=runner, log=lambda *a: None, allow_feature_only_drift=True)
    assert rec["actual_backend"] == "numba" and not rec["gpu_used"] and st["set"] == ["numba"]
    assert rec["downstream_exact"] and not rec["strict_passed"] and rec["accepted_feature_only_drift"] and rec["feature_mismatches"] == 3
    with pytest.raises(rb.BackendSelectionError):                             # the same result without the policy: STOP
        rb.select_backend(_bundle(), fz, GPU_HW, "gpu", runner=runner, log=lambda *a: None)


def test_fixture_is_loaded_without_hashing(tmp_path):
    mod = SimpleNamespace(FIXTURE_DIR="selftest", FIXTURE_FORMAT="casmi-selftest-2", FILES=rn.SELFTEST_FIXTURES)
    d = tmp_path / "selftest"
    d.mkdir()
    (d / "fixture_manifest.json").write_text(json.dumps({"format": "casmi-selftest-1"}), encoding="utf-8")
    with pytest.raises(ri.InferenceBlocked):                                  # old fixture format
        rn.load_fixture_filenames_only(tmp_path, mod)
    (d / "fixture_manifest.json").write_text(json.dumps({"format": "casmi-selftest-2", "files": {"x": "deadbeef"}}), encoding="utf-8")
    with pytest.raises(ri.InferenceBlocked):                                  # missing fixtures, reported by NAME
        rn.load_fixture_filenames_only(tmp_path, mod)


def test_anchor_comparison_and_frozen_identity():
    exp = rn.KAGGLE_ANCHOR_EXPECTED
    assert exp["n_spectra"] == 1213 and exp["n_molecules"] == 400 and exp["candidate_pair_count"] == 471173
    assert exp["similarity_evaluations"] == 2057028 and exp["submission_rows"] == 400 and exp["unsupported_adducts"] == []
    assert not {"runtime_seconds", "peak_ram", "python_version", "gpu_name", "feature_mismatches"} & set(exp)
    assert rn.compare_to_anchor(dict(exp), exp) == {}
    assert set(rn.compare_to_anchor({**exp, "n_spectra": 1212, "unsupported_adducts": ["[M+X]+"]}, exp)) == {"n_spectra", "unsupported_adducts"}
    cfg = {"bundle_version": "v2-A7", "CONFIG_HASH": "60174e39a2a3c6b4", "model_id": "V1_TL_1K_TESTSIM_STRICT",
           "aggregator_name": "MOST_CONFIDENT_SPECTRUM", "calibration_temperature": 1.1947045372735254}
    info = {"model_id": "V1_TL_1K_TESTSIM_STRICT", "freeze_status": "FROZEN"}
    contract = {"name": "MOST_CONFIDENT_SPECTRUM", "temperature": 1.1947045372735254}
    assert rn.check_frozen_identity(cfg, info, contract)
    for bad_cfg, bad_info, bad_contract in (({**cfg, "calibration_temperature": 1.19}, info, contract), ({**cfg, "CONFIG_HASH": "x"}, info, contract),
                                            (cfg, {**info, "freeze_status": "PROVISIONAL"}, contract), (cfg, info, {**contract, "name": "RRF"})):
        with pytest.raises(ri.InferenceBlocked):
            rn.check_frozen_identity(bad_cfg, bad_info, bad_contract)


def test_missing_bundle_blocks_and_records_failure(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.input_root.mkdir(parents=True)
    with pytest.raises(ri.InferenceBlocked):
        ri.run_frozen_inference(cfg, log=lambda *a: None)
    rep = json.loads((cfg.work_dir / "run_report_failed.json").read_text())
    assert rep["failed_stage"] == "discover_inputs" and not (cfg.work_dir / "submission.csv").exists()


# ---- backend selection -----------------------------------------------------------------------------------------------------

def _frozen(numba=True, gpu=False):
    state = {"set": []}
    be = SimpleNamespace(NUMBA="numba", NUMPY="numpy", numba_available=lambda: numba, set_backend=lambda b: state["set"].append(b))
    if gpu:
        be.GPU = "cuda"
        be.gpu_available = lambda: True
    return SimpleNamespace(backend=be), state


def _bundle(gpu_decl=None):
    return SimpleNamespace(config={"bundle_version": "v2-A7", **({"gpu_backend": gpu_decl} if gpu_decl else {})})


def _runner(passing):
    return lambda b, be: {"backend": be, "passed": be in passing}


GPU_HW = {"cuda_available": True, "gpu_name": "Tesla T4"}
CPU_HW = {"cuda_available": False, "gpu_name": None}


def test_cpu_fallback_is_explicit_on_gpu_hardware_without_gpu_backend():
    fz, st = _frozen()
    rec = rb.select_backend(_bundle(), fz, GPU_HW, "gpu", runner=_runner({"numba", "numpy"}), log=lambda *a: None)
    assert rec["actual_backend"] == "numba" and not rec["gpu_used"] and rec["gpu_available"]
    assert "implements no GPU backend" in rec["fallback_reason"] and st["set"] == ["numba"]


def test_numba_failure_falls_back_to_numpy_explicitly():
    fz, st = _frozen()
    logs = []
    rec = rb.select_backend(_bundle(), fz, CPU_HW, "gpu", runner=_runner({"numpy"}), log=logs.append)
    assert rec["actual_backend"] == "numpy" and rec["numpy_fallback_used"] and "numba self-test FAILED" in rec["fallback_reason"]
    assert any("FALLBACK" in l for l in logs) and st["set"] == ["numpy"]


def test_all_backends_failing_stops():
    fz, st = _frozen()
    with pytest.raises(rb.BackendSelectionError):
        rb.select_backend(_bundle(), fz, CPU_HW, "gpu", runner=_runner(set()), log=lambda *a: None)
    assert st["set"] == []                                                   # nothing activated


def test_gpu_backend_requires_implementation_declaration_hardware_and_parity():
    decl = {"name": "cuda", "parity_validated": True}
    # implemented + declared + hardware + self-test PASS -> GPU
    fz, _ = _frozen(gpu=True)
    rec = rb.select_backend(_bundle(decl), fz, GPU_HW, "gpu", runner=_runner({"cuda", "numba", "numpy"}), log=lambda *a: None)
    assert rec["actual_backend"] == "cuda" and rec["gpu_used"]
    # implemented but NOT declared parity-validated -> never tried
    tried = []
    fz, _ = _frozen(gpu=True)
    rb.select_backend(_bundle({"name": "cuda", "parity_validated": False}), fz, GPU_HW, "gpu",
                      runner=lambda b, be: tried.append(be) or {"passed": True}, log=lambda *a: None)
    assert "cuda" not in tried
    # declared but the self-test fails on GPU -> CPU, reason recorded
    fz, _ = _frozen(gpu=True)
    rec = rb.select_backend(_bundle(decl), fz, GPU_HW, "gpu", runner=_runner({"numba", "numpy"}), log=lambda *a: None)
    assert rec["actual_backend"] == "numba" and "cuda self-test FAILED" in rec["fallback_reason"]
    # no GPU hardware -> not tried
    fz, _ = _frozen(gpu=True)
    rec = rb.select_backend(_bundle(decl), fz, CPU_HW, "gpu", runner=_runner({"cuda", "numba", "numpy"}), log=lambda *a: None)
    assert not rec["gpu_used"] and "no GPU visible" in rec["fallback_reason"]


def test_require_gpu_stops_without_validated_gpu():
    fz, _ = _frozen()
    with pytest.raises(rb.BackendSelectionError):
        rb.select_backend(_bundle(), fz, GPU_HW, "gpu", require_gpu=True, runner=_runner({"numba", "numpy"}), log=lambda *a: None)


def test_accelerator_detection_never_fails_without_gpu(monkeypatch):
    import casmi_runtime.accelerator as acc
    monkeypatch.setattr(acc.shutil, "which", lambda name: None)
    info = detect_accelerator()
    assert info["device"] == "cpu" and info["cuda_available"] is False and info["lightgbm_prediction_device"] == "cpu"


# ---- results / submission ----------------------------------------------------------------------------------------------------

def test_result_folders_are_immutable(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "submission.csv").write_text("molecule_id,smiles\n")
    (out / "run_report.json").write_text("{}")
    d = rr.save_results_immutable(out, tmp_path / "results", "abcdef1234567890", {"x": 1}, stamp="20260101T000000Z")
    assert sorted(p.name for p in d.iterdir()) == ["provenance.json", "run_report.json", "submission.csv"]
    with pytest.raises(FileExistsError):
        rr.save_results_immutable(out, tmp_path / "results", "abcdef1234567890", stamp="20260101T000000Z")
    prov = json.loads((d / "provenance.json").read_text())
    assert set(prov["files_sha256"]) == {"submission.csv", "run_report.json"}


def test_submission_stays_manual():
    texts = [f.read_text(encoding="utf-8") for f in (ROOT / "src" / "casmi_runtime").glob("*.py")]
    texts += [(NOTEBOOKS / n).read_text(encoding="utf-8") for n in ("00_colab_setup.ipynb", "01_colab_inference.ipynb", "02_kaggle_inference.ipynb",
                                                                    "03_runtime_benchmark.ipynb")]
    for t in texts:
        assert "KaggleApi" not in t and "competitions submit" not in t and "kaggle.api" not in t


# ---- preparation script (training repo only) ---------------------------------------------------------------------------------

def test_prepare_script_is_spec_driven_and_never_runs_git():
    ps1 = ROOT / "scripts" / "prepare_runtime_repo.ps1"
    if not ps1.exists():
        pytest.skip("the preparation script lives in the training repo only")
    ps = ps1.read_text(encoding="utf-8")
    assert "runtime\\runtime_files.json" in ps and "foreach ($m in $Spec.casmi_modules)" in ps
    assert 'throw "$OutDir already exists' in ps and "/XD __pycache__" in ps
    body = ps.split("$next = @'")[0]
    assert not re.search(r"^\s*(&\s*)?(git|gh)\s", body, flags=re.MULTILINE)
    assert "gh repo create <REPO>" in ps and SPEC["repo_name"] == "enveda-casmi-runtime"
    srcs = [e["src"] for e in SPEC["copy_files"]] + [t["src"] for t in SPEC["copy_trees"]]
    for d in ("bundle/", "bundle_archive/", "data/", "outputs/", "kaggle_dryrun/"):              # runtime data directories
        assert not any(s.startswith(d) for s in srcs), d
    for n in ("11_00", "11_01", "11_02", "MOL_DEV", "host_"):                                    # development notebooks / artifacts
        assert not any(n in s.split("/")[-1] for s in srcs), n
    for s in srcs + [f"src/{m}" for m in SPEC["casmi_modules"]]:
        assert (ROOT / s).exists(), s
