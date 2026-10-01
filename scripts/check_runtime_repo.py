"""Pre-upload safety + completeness check of the RUNTIME-SOURCE package (GitHub repo == Kaggle runtime-source dataset;
code only -- the bundle lives on Drive / a Kaggle Dataset). STATIC ONLY: nothing is imported or executed.

    python scripts/check_runtime_repo.py --repo colab_repo [--max-mb 50]     # from the training repo root
    python scripts/check_runtime_repo.py                                     # from inside the runtime repo
    python scripts/check_runtime_repo.py --repo colab_repo --bundle bundle --competition data   # + the other two Kaggle inputs
    python scripts/check_runtime_repo.py --repo colab_repo --payload kaggle_model_build/payload   # + the built Kaggle Model payload

Fails (exit 1) on: a file > --max-mb; any .parquet / .npy / other binary artifact; a bundle / data / outputs / results
directory; a LightGBM fold file or OOF table; a CASMI bundle manifest; anything that looks like a credential; a missing
required / runtime-core file, notebook or requirement; src/casmi holding modules outside the runtime whitelist (training
code leaked); a casmi / casmi_infer / casmi_runtime import that does not resolve inside the repo; bad .gitignore or
requirements; a malformed notebook; the Kaggle E2E import closure (AST, from notebooks/02_kaggle_inference.ipynb through
casmi_runtime.named_inference) having an unresolved or undeclared repo module; the Kaggle notebook or named_inference
using the retired anchor path (run_frozen_inference, verify_bundle, Bundle(verify=True), v63_bundle_check, pip / wheels,
Drive, clone, network, hashing, automatic submission); an incomplete kaggle_model_inventory.json. LightGBM must be DECLARED,
but no exact version is required: parity is decided by the bundle's functional self-test at run time.
With --bundle / --competition / --payload, also checks the local frozen bundle, the competition folder and the built
Kaggle Model payload against kaggle_model_inventory.json (filename presence only, no hashing).
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

SPEC_NAME = "runtime_files.json"
INVENTORY_NAME = "kaggle_model_inventory.json"
KAGGLE_NOTEBOOK = "notebooks/02_kaggle_inference.ipynb"
KAGGLE_ENTRY_MODULE = "casmi_runtime.named_inference"
KAGGLE_ENTRY_FUNCTION = "run_named_inference"
FORBIDDEN_DIRS = {"bundle", "bundle_archive", "data", "outputs", "kaggle_dryrun", "colab_runtime", "casmi_runtime_data", "results",
                  ".ipynb_checkpoints"}
FORBIDDEN_SUFFIXES = {".parquet", ".npy", ".npz", ".feather", ".pkl", ".joblib", ".pt", ".h5", ".whl"}
FORBIDDEN_NAMES = re.compile(r"(_fold\d+\.txt$|^fold_\d+\.txt$|^oof_rows|^train\.|^host_holdout|^client_secret.*\.json$|^credentials\.json$"
                             r"|^token\.json$|^kaggle\.json$|^\.env$)")
TOKEN_PATTERN = re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}"
                           r"|x-access-token:[A-Za-z0-9_]{20,}|\"key\"\s*:\s*\"[0-9a-f]{32}\"|\"client_secret\"\s*:\s*\"[^\"]{10,}\""
                           r"|\"refresh_token\"\s*:\s*\"[^\"]{10,}\")")
GITIGNORE_RULES = ("__pycache__/", "*.pyc", ".ipynb_checkpoints/", ".pytest_cache/", "bundle/", "bundle_archive/", "data/", "outputs/",
                   "colab_runtime/", "kaggle_dryrun/", "*.parquet", "*.npy", "*.npz", "*.log", ".env")
REQUIREMENT_LINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?\s*((==|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!-]+\s*(,\s*(==|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!-]+\s*)*)?$")
REQUIRED_PACKAGES = ("lightgbm", "numpy", "pandas", "pyarrow")          # declared; versions are NOT gated (self-test is)
TRAINING_ONLY = ("scikit-learn", "scipy", "matplotlib", "tqdm", "pyyaml", "joblib", "torch", "tensorflow")
RUNTIME_CORE = (
    "src/casmi_runtime/__init__.py", "src/casmi_runtime/accelerator.py", "src/casmi_runtime/backends.py", "src/casmi_runtime/benchmark.py",
    "src/casmi_runtime/config.py", "src/casmi_runtime/frozen.py", "src/casmi_runtime/inference.py", "src/casmi_runtime/named_inference.py",
    "src/casmi_runtime/results.py",
    "src/casmi_infer/__init__.py", "src/casmi_infer/adducts.py", "src/casmi_infer/aggregation.py", "src/casmi_infer/backend.py",
    "src/casmi_infer/compat.py", "src/casmi_infer/features.py", "src/casmi_infer/mass_search.py", "src/casmi_infer/pipeline.py",
    "src/casmi_infer/ranker.py", "src/casmi_infer/reference_index.py", "src/casmi_infer/selftest.py", "src/casmi_infer/similarity.py",
    "src/casmi_infer/spectrum.py", "src/casmi_infer/submission.py", "src/casmi_infer/validation.py",
    KAGGLE_NOTEBOOK, "requirements-kaggle.txt", "KAGGLE_RUN_GUIDE.md", SPEC_NAME, INVENTORY_NAME)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".ipynb", ".json", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".ps1", ".template", ""}
PACKAGES = ("casmi", "casmi_infer", "casmi_runtime")
# the modules the Kaggle entry must reach before the frozen bundle code takes over (casmi_runtime.frozen hands off)
KAGGLE_CLOSURE_MUST_INCLUDE = ("casmi_runtime.named_inference", "casmi_runtime.accelerator", "casmi_runtime.backends",
                               "casmi_runtime.frozen", "casmi_runtime.inference", "casmi_runtime.config")
NETWORK_MODULES = ("requests", "urllib", "urllib3", "http", "socket", "httpx", "aiohttp", "ftplib", "google", "googleapiclient",
                   "pydrive", "kaggle", "git")
# code-cell substrings the recommended Kaggle notebook must never contain
KAGGLE_FORBIDDEN_CODE = ("run_frozen_inference", "verify_bundle(", "verify=True", "v63_bundle_check", "'pip'", "pip install",
                         ".whl", "drive.mount", "google.colab", "/content/drive", "git clone", "'clone'", "http://", "https://",
                         "import requests", "urllib", "KaggleApi", "kaggle.api", "competitions submit", "4.7.0", "train.parquet",
                         "wget", "curl ", "hashlib", "sha256_file", "load_fixture(", "run_selftest_with_fallback", "kagglehub")
KAGGLE_REQUIRED_CODE = (f"from {KAGGLE_ENTRY_MODULE} import {KAGGLE_ENTRY_FUNCTION}", f"{KAGGLE_ENTRY_FUNCTION}(", "runtime='kaggle'",
                        "/kaggle/input", "/kaggle/working", "named_inference.py", "resolve_kaggle_model_inputs(",
                        "allow_feature_only_drift=True", "KAGGLE OFFLINE ANCHOR PASS", "kaggle_anchor_report.json")
# the entry module must not call the frozen hashing / strict-only helpers itself
ENTRY_FORBIDDEN_CALLS = ("verify_bundle", "sha256_file", "load_fixture", "run_selftest_with_fallback", "run_frozen_inference")
BUNDLE_MUST_LIST = ([f"models/v1_fold{k}.txt" for k in range(5)]
                    + [f"selftest/{f}" for f in ("queries.parquet", "expected_features.parquet", "expected_scores.parquet",
                                                 "expected_ranks.parquet", "expected_probs.parquet", "expected_molecules.parquet",
                                                 "expected_molecule_ranking.parquet", "fixture_manifest.json")]
                    + ["aggregation/calibration.json", "aggregation/selected_aggregator.json", "aggregation/aggregator_lock.json",
                       "ref_peaks_mz.npy", "ref_peaks_int.npy", "ref_peaks_offsets.npy", "ref_index_ids.npy", "ref_index_offsets.npy",
                       "ref_meta.parquet", "_ref_peaks_mz.f8", "_ref_peaks_int.f8", "structures.parquet", "structures_mass.npy",
                       "connectivities.parquet", "config.json", "manifest.json"]
                    + [f"code/casmi_infer/{m}.py" for m in ("__init__", "adducts", "aggregation", "backend", "compat", "features",
                                                             "mass_search", "pipeline", "ranker", "reference_index", "selftest",
                                                             "similarity", "spectrum", "submission", "validation")])
BUNDLE_MUST_LIST = [f"bundle/{r}" for r in BUNDLE_MUST_LIST]          # inventory paths are relative to the model payload root
COMPETITION_REQUIRED = ("test.parquet", "sample_submission.csv")
PAYLOAD_IDENTITY = {"bundle_version": "v2-A7", "model_id": "V1_TL_1K_TESTSIM_STRICT", "CONFIG_HASH": "60174e39a2a3c6b4"}


def parse_requirements(path):
    out = []
    for i, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not REQUIREMENT_LINE.match(line):
            raise ValueError(f"{Path(path).name} line {i} not parsable: {raw!r}")
        out.append(line)
    return out


def requirement_names(path):
    return {re.split(r"[<>=!~\[ ]", r)[0].lower(): r for r in parse_requirements(path)}


def load_spec(repo):
    return json.loads((Path(repo) / SPEC_NAME).read_text(encoding="utf-8"))


def required_files(spec):
    return sorted({*(f"src/{m}" for m in spec["casmi_modules"]), *(e["dst"] for e in spec["copy_files"]), *RUNTIME_CORE,
                   *spec.get("runtime_core", ())})


def unresolved_imports(repo):
    """[(file, module)] for every casmi / casmi_infer / casmi_runtime import (any nesting level) that has no file in the repo."""
    src = Path(repo) / "src"
    bad = []
    for f in sorted(list(src.rglob("*.py")) + list((Path(repo) / "scripts").glob("*.py")) + list((Path(repo) / "tests").glob("*.py"))):
        if "__pycache__" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append((str(f.relative_to(repo)), f"SyntaxError: {e}"))
            continue
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods.append(n.module)
                mods += [f"{n.module}.{a.name}" for a in n.names if (src / n.module.replace(".", "/") / a.name).is_dir()]
        for m in mods:
            if m.split(".")[0] not in PACKAGES:
                continue
            p = src / m.replace(".", "/")
            if not (p.with_suffix(".py").is_file() or (p / "__init__.py").is_file()):
                bad.append((str(f.relative_to(repo)).replace("\\", "/"), m))
    return bad


# ---------------------------------------------------------------------------------------------
# static E2E import closure (AST only -- nothing is imported)
# ---------------------------------------------------------------------------------------------

def notebook_code(nb_path):
    """[(cell_index, source)] of the code cells; raises ValueError on a malformed notebook."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    if not isinstance(nb.get("cells"), list):
        raise ValueError(f"{Path(nb_path).name}: no 'cells' list")
    out = []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code":
            src = c.get("source", "")
            out.append((i, "".join(src) if isinstance(src, list) else str(src)))
    return out


def _strip_magics(src):
    return "\n".join("" if l.lstrip().startswith(("!", "%")) else l for l in src.splitlines())


def _module_file(src_root, mod):
    p = Path(src_root) / mod.replace(".", "/")
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    return None


def _imported_modules(tree, src_root, current=None, is_pkg=False):
    """Absolute module names imported anywhere in `tree` (module level and inside functions), incl. parent packages."""
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                if not current:
                    continue
                base = current.split(".") if is_pkg else current.split(".")[:-1]
                base = base[:len(base) - (n.level - 1)] if n.level > 1 else base
                mod = ".".join(base + ([n.module] if n.module else []))
            else:
                mod = n.module
            if not mod:
                continue
            mods.append(mod)
            mods += [f"{mod}.{a.name}" for a in n.names if _module_file(src_root, f"{mod}.{a.name}") is not None]
    out = []
    for m in mods:
        parts = m.split(".")
        out += [".".join(parts[:k]) for k in range(1, len(parts) + 1)]
    return list(dict.fromkeys(out))


def kaggle_import_closure(repo, notebook=KAGGLE_NOTEBOOK):
    """(closure {module: relpath}, unresolved [(importer, module)], problems) from the Kaggle notebook's code cells,
    following only repo-owned packages. casmi_runtime.frozen hands inference to <bundle>/code by importlib -- those
    dynamic imports are the frozen boundary and are deliberately not followed."""
    repo = Path(repo)
    src_root = repo / "src"
    problems, unresolved, closure = [], [], {}
    try:
        cells = notebook_code(repo / notebook)
    except (ValueError, OSError) as e:
        return closure, unresolved, [f"{notebook}: malformed or unreadable ({e})"]
    queue = []
    for i, src in cells:
        try:
            tree = ast.parse(_strip_magics(src))
        except SyntaxError as e:
            problems.append(f"{notebook} cell {i}: SyntaxError {e}")
            continue
        queue += [(f"{notebook}[cell {i}]", m) for m in _imported_modules(tree, src_root) if m.split(".")[0] in PACKAGES]
    while queue:
        importer, mod = queue.pop(0)
        if mod in closure:
            continue
        f = _module_file(src_root, mod)
        if f is None:
            unresolved.append((importer, mod))
            continue
        rel = f.relative_to(repo).as_posix()
        closure[mod] = rel
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            problems.append(f"{rel}: SyntaxError {e}")
            continue
        queue += [(rel, m) for m in _imported_modules(tree, src_root, mod, f.name == "__init__.py") if m.split(".")[0] in PACKAGES]
    return closure, unresolved, problems


def check_kaggle_entry(repo, spec):
    """Static E2E audit of the recommended Kaggle path. Returns (problems, closure)."""
    repo = Path(repo)
    problems = []
    nb = repo / KAGGLE_NOTEBOOK
    if not nb.is_file():
        return [f"missing Kaggle notebook {KAGGLE_NOTEBOOK}"], {}
    try:
        cells = notebook_code(nb)
        raw = nb.read_text(encoding="utf-8")
    except (ValueError, OSError) as e:
        return [f"{KAGGLE_NOTEBOOK} is malformed: {e}"], {}
    code = "\n".join(s for _, s in cells)
    problems += [f"{KAGGLE_NOTEBOOK} lacks {tok!r}" for tok in KAGGLE_REQUIRED_CODE if tok not in code]
    problems += [f"{KAGGLE_NOTEBOOK} uses a forbidden / retired construct: {tok!r}" for tok in KAGGLE_FORBIDDEN_CODE if tok in code]
    problems += [f"{KAGGLE_NOTEBOOK} mentions {tok!r} (Colab / Drive only)" for tok in ("google.colab", "drive.mount") if tok in raw]

    closure, unresolved, cprobs = kaggle_import_closure(repo)
    problems += cprobs
    problems += [f"Kaggle import closure: {imp} imports {m}, which is not in the runtime source" for imp, m in unresolved]
    problems += [f"Kaggle import closure does not reach {m}" for m in KAGGLE_CLOSURE_MUST_INCLUDE if m not in closure]
    declared = set(RUNTIME_CORE) | set(spec.get("runtime_core", ()))
    problems += [f"Kaggle import closure file not declared in runtime_core: {rel}" for rel in closure.values() if rel not in declared]
    for mod, rel in closure.items():
        tree = ast.parse((repo / rel).read_text(encoding="utf-8"))
        net = [m for m in _imported_modules(tree, repo / "src", mod, rel.endswith("__init__.py")) if m.split(".")[0] in NETWORK_MODULES]
        problems += [f"{rel}: network / Drive / VCS import {m!r} in the Kaggle path" for m in net]

    entry = repo / "src" / (KAGGLE_ENTRY_MODULE.replace(".", "/") + ".py")
    if entry.is_file():
        problems += check_entry_module(entry)
    return problems, closure


def check_entry_module(path):
    """named_inference: defines the entry, opens the bundle with verify=False, never calls verify_bundle /
    Bundle(verify=True) / pip / network, never imports casmi / casmi_infer itself."""
    problems = []
    rel = Path(path).name
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    if not any(isinstance(n, ast.FunctionDef) and n.name == KAGGLE_ENTRY_FUNCTION for n in tree.body):
        problems.append(f"{rel} defines no {KAGGLE_ENTRY_FUNCTION}()")
    bundle_calls = selftest_calls = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else None
            kw = {k.arg: k.value for k in n.keywords}
            if fn in ENTRY_FORBIDDEN_CALLS:
                problems.append(f"{rel}: calls {fn}() -- not allowed in the FILENAMES_ONLY Kaggle anchor (line {n.lineno})")
            if fn == "run_selftest":
                selftest_calls += 1
                if "fixture" not in kw:
                    problems.append(f"{rel}: run_selftest() without fixture= would use the frozen sha256 fixture loader (line {n.lineno})")
            if "verify" in kw and not (isinstance(kw["verify"], ast.Constant) and kw["verify"].value is False):
                problems.append(f"{rel}: a call passes verify other than False (line {n.lineno})")
            if fn == "Bundle":
                bundle_calls += 1
                if not (isinstance(kw.get("verify"), ast.Constant) and kw["verify"].value is False):
                    problems.append(f"{rel}: Bundle(...) without verify=False (line {n.lineno})")
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in ("pip", "install"):
            problems.append(f"{rel}: pip / install string literal (line {n.lineno})")
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            if any(m.split(".")[0] in ("casmi", "casmi_infer") for m in names):
                problems.append(f"{rel}: imports casmi / casmi_infer directly (inference must come from load_frozen_code)")
    if bundle_calls == 0:
        problems.append(f"{rel}: never opens the frozen Bundle")
    if selftest_calls == 0:
        problems.append(f"{rel}: never runs the functional self-test (run_selftest)")
    return problems


# ---------------------------------------------------------------------------------------------
# inventory + the other two Kaggle inputs
# ---------------------------------------------------------------------------------------------

def check_inventory(repo, spec):
    repo = Path(repo)
    p = repo / INVENTORY_NAME
    if not p.is_file():
        return [f"missing {INVENTORY_NAME}"]
    try:
        inv = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{INVENTORY_NAME} is not valid JSON: {e}"]
    problems = [f"{INVENTORY_NAME} lacks {k}.required" for k in ("payload_top_level", "runtime_source", "frozen_bundle", "competition")
                if not isinstance((inv.get(k) or {}).get("required"), list)]
    if problems:
        return problems
    # payload "runtime/<rel>" == "<rel>" in this runtime repo
    rs = inv["runtime_source"]["required"]
    problems += [f"{INVENTORY_NAME}: runtime_source path outside runtime/: {r}" for r in rs if not r.startswith("runtime/")]
    problems += [f"{INVENTORY_NAME}: runtime_source file missing from the package: {r}" for r in rs
                 if r.startswith("runtime/") and not (repo / r[len("runtime/"):]).is_file()]
    must_ship = [r for r in RUNTIME_CORE if r.startswith("src/casmi_runtime/")] + [KAGGLE_NOTEBOOK, "requirements-kaggle.txt", "KAGGLE_RUN_GUIDE.md"]
    problems += [f"{INVENTORY_NAME}: runtime_source lacks runtime/{r}" for r in must_ship if f"runtime/{r}" not in rs]
    problems += [f"{INVENTORY_NAME}: the model payload must not ship {r} (bundle/code is the only inference code)"
                 for r in rs if r.startswith(("runtime/src/casmi_infer/", "runtime/src/casmi/"))]
    srcs = inv["runtime_source"].get("training_repo_sources") or {}
    problems += [f"{INVENTORY_NAME}: no training_repo_sources entry for {r}" for r in rs if r not in srcs]
    fb = set(inv["frozen_bundle"]["required"])
    problems += [f"{INVENTORY_NAME}: frozen_bundle.required lacks {r}" for r in BUNDLE_MUST_LIST if r not in fb]
    problems += [f"{INVENTORY_NAME}: frozen_bundle path outside bundle/: {r}" for r in fb if not r.startswith("bundle/")]
    comp = inv["competition"]
    if sorted(comp["required"]) != sorted(COMPETITION_REQUIRED) or comp.get("separate_from_model") is not True:
        problems.append(f"{INVENTORY_NAME}: competition must be separate_from_model with required exactly {list(COMPETITION_REQUIRED)}, got {comp}")
    model = inv.get("model") or {}
    problems += [f"{INVENTORY_NAME}: model.{k} {model.get(k)!r} != {v!r}" for k, v in PAYLOAD_IDENTITY.items() if model.get(k) != v]
    if model.get("private") is not True:
        problems.append(f"{INVENTORY_NAME}: model.private must be true")
    return problems


def check_bundle_dir(bundle, inventory):
    """Presence check of the frozen bundle folder to upload (no hashing): inventory + every manifest file."""
    b = Path(bundle)
    if not (b / "manifest.json").is_file():
        return [f"bundle {b}: no manifest.json"]
    man = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    listed = [r[len("bundle/"):] if r.startswith("bundle/") else r for r in inventory["frozen_bundle"]["required"]]
    need = sorted(set(listed) | set((man.get("files") or {}).keys()))
    problems = [f"bundle {b}: missing {r}" for r in need if not (b / r).is_file()]
    problems += [f"bundle {b}: empty file {r}" for r in need
                 if (b / r).is_file() and (b / r).stat().st_size == 0 and Path(r).name != "__init__.py"]    # empty package inits are fine
    problems += [f"bundle {b}: training artifact must not be uploaded: {p.relative_to(b).as_posix()}"
                 for p in b.rglob("*") if p.is_file() and re.search(r"^(train\.parquet|oof_rows|host_holdout)", p.name)]
    return problems


def check_competition_dir(comp):
    c = Path(comp)
    problems = []
    for name in COMPETITION_REQUIRED:
        hits = [p for p in c.rglob(name) if p.is_file()]
        if len(hits) != 1:
            problems.append(f"competition {c}: expected exactly one {name}, found {len(hits)}")
    return problems


def check_payload(payload, inventory):
    """Filename-only check of a built Kaggle Model payload (kaggle_model_build/payload) -- no hashing."""
    p = Path(payload)
    if not p.is_dir():
        return [f"payload {p}: not a directory"]
    problems = []
    required = (list(inventory["payload_top_level"]["required"]) + list(inventory["runtime_source"]["required"])
                + list(inventory["frozen_bundle"]["required"]))
    problems += [f"payload: missing {r}" for r in required if not (p / r).is_file()]
    problems += [f"payload: missing directory {d}" for d in inventory["frozen_bundle"].get("required_directories", []) if not (p / d).is_dir()]
    problems += [f"payload: forbidden {f}" for f in inventory["runtime_source"].get("forbidden", []) if (p / f).exists()]
    if (p / "bundle").is_dir():
        problems += check_bundle_dir(p / "bundle", inventory)
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(p).as_posix()
        if f.name in inventory["competition"].get("forbidden_in_payload", COMPETITION_REQUIRED):
            problems.append(f"payload: competition file must stay outside the model: {rel}")
        if "__pycache__" in f.parts or ".ipynb_checkpoints" in f.parts or ".git" in f.parts or f.suffix == ".pyc":
            problems.append(f"payload: cache / VCS file: {rel}")
    info_path = p / "PACKAGE_INFO.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        problems += [f"PACKAGE_INFO.json {k} {info.get(k)!r} != {v!r}" for k, v in PAYLOAD_IDENTITY.items() if info.get(k) != v]
        if info.get("bundle_integrity_mode") != "FILENAMES_ONLY":
            problems.append(f"PACKAGE_INFO.json bundle_integrity_mode {info.get('bundle_integrity_mode')!r} != 'FILENAMES_ONLY'")
    return problems


# ---------------------------------------------------------------------------------------------

def check_repo(repo, max_mb=50.0):
    repo = Path(repo).resolve()
    if not repo.is_dir():
        return [f"not a directory: {repo}"]
    if not (repo / SPEC_NAME).is_file():
        return [f"missing {SPEC_NAME}"]
    spec = load_spec(repo)
    problems = [f"missing required file: {rel}" for rel in required_files(spec) if not (repo / rel).is_file()]
    skipped = {".git", "__pycache__", ".pytest_cache"}
    files = [p for p in repo.rglob("*") if p.is_file() and not (skipped & set(p.relative_to(repo).parts)) and p.suffix != ".pyc"]
    allowed_casmi = {f"src/{m}" for m in spec["casmi_modules"]}
    for p in files:
        rel = p.relative_to(repo).as_posix()
        if set(p.relative_to(repo).parts[:-1]) & FORBIDDEN_DIRS:
            problems.append(f"forbidden directory in repo: {rel}")
        if p.suffix.lower() in FORBIDDEN_SUFFIXES or FORBIDDEN_NAMES.search(p.name):
            problems.append(f"runtime/training artifact or credential file in repo: {rel}")
        if p.stat().st_size > max_mb * 1e6:
            problems.append(f"file larger than {max_mb:g} MB: {rel} ({p.stat().st_size / 1e6:.1f} MB)")
        if rel.startswith("src/casmi/") and rel not in allowed_casmi:
            problems.append(f"src/casmi module outside the runtime whitelist (training code?): {rel}")
        if p.name == "manifest.json":
            try:
                if str(json.loads(p.read_text(encoding="utf-8")).get("bundle_format", "")).startswith("casmi-modeA-bundle"):
                    problems.append(f"a CASMI bundle manifest is in the repo: {rel}")
            except (ValueError, UnicodeDecodeError):
                pass
        if p.suffix.lower() in TEXT_SUFFIXES and p.stat().st_size < 5e6:
            if TOKEN_PATTERN.search(p.read_text(encoding="utf-8", errors="ignore")):
                problems.append(f"something that looks like a credential is in {rel}")
    gi = repo / ".gitignore"
    if gi.is_file():
        rules = {l.strip() for l in gi.read_text(encoding="utf-8").splitlines()}
        problems += [f".gitignore lacks rule {r}" for r in GITIGNORE_RULES if r not in rules]
        leaks = [r for r in rules if r in ("*.py", "*.ipynb", "src/", "notebooks/", "scripts/")]
        problems += [f".gitignore would drop source/notebooks: {r}" for r in leaks]
    for name in ("requirements-colab.txt", "requirements-kaggle.txt"):
        f = repo / name
        if not f.is_file():
            continue
        try:
            reqs = requirement_names(f)
        except ValueError as e:
            problems.append(str(e))
            continue
        problems += [f"{name} lacks {n}" for n in REQUIRED_PACKAGES if n not in reqs]
        problems += [f"{name} lists a training-only package: {n}" for n in TRAINING_ONLY if n in reqs]
    for nb in (repo / "notebooks").glob("*.ipynb") if (repo / "notebooks").is_dir() else []:
        try:
            notebook_code(nb)
        except ValueError as e:
            problems.append(f"{nb.name} is not a valid notebook: {e}")
    problems += [f"unresolved import in {f}: {m}" for f, m in unresolved_imports(repo)]
    problems += check_kaggle_entry(repo, spec)[0]
    problems += check_inventory(repo, spec)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--max-mb", type=float, default=50.0)
    ap.add_argument("--bundle", help="local frozen bundle folder to upload (presence check against the inventory + manifest)")
    ap.add_argument("--competition", help="local folder holding test.parquet + sample_submission.csv")
    ap.add_argument("--payload", help="built Kaggle Model payload folder (kaggle_model_build/payload)")
    ap.add_argument("--show-closure", action="store_true", help="print the Kaggle E2E import closure")
    args = ap.parse_args(argv)
    problems = check_repo(args.repo, args.max_mb)
    repo = Path(args.repo).resolve()
    if args.show_closure and (repo / SPEC_NAME).is_file():
        for mod, rel in kaggle_import_closure(repo)[0].items():
            print(f"CLOSURE: {mod:40s} {rel}")
        print("CLOSURE: -> frozen boundary: casmi_runtime.frozen.load_frozen_code imports <bundle>/code/casmi_infer")
    inv_path = repo / INVENTORY_NAME
    inventory = json.loads(inv_path.read_text(encoding="utf-8")) if inv_path.is_file() else None
    if args.bundle:
        problems += check_bundle_dir(args.bundle, inventory) if inventory else [f"--bundle needs {INVENTORY_NAME} in the repo"]
    if args.competition:
        problems += check_competition_dir(args.competition)
    if args.payload:
        problems += check_payload(args.payload, inventory) if inventory else [f"--payload needs {INVENTORY_NAME} in the repo"]
    for p in problems:
        print("PROBLEM:", p)
    print("RUNTIME REPO CHECK:", "PASS" if not problems else f"FAIL ({len(problems)} problems)", "|", repo)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
