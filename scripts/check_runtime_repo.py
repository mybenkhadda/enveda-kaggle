"""Pre-push safety check of the GitHub RUNTIME repo (code only; the bundle lives on Drive / a Kaggle Dataset).

    python scripts/check_runtime_repo.py --repo colab_repo [--max-mb 50]     # from the training repo root
    python scripts/check_runtime_repo.py                                     # from inside the runtime repo

Fails (exit 1) on: a file > --max-mb; any .parquet / .npy / other binary artifact; a bundle / data / outputs / results
directory; a LightGBM fold file or OOF table; a CASMI bundle manifest; anything that looks like a credential; a missing
required file / notebook / requirement; src/casmi holding modules outside the runtime whitelist (training code
leaked); a casmi / casmi_infer / casmi_runtime import that does not resolve inside the repo; bad .gitignore or
requirements. Read-only.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

SPEC_NAME = "runtime_files.json"
FORBIDDEN_DIRS = {"bundle", "bundle_archive", "data", "outputs", "kaggle_dryrun", "colab_runtime", "casmi_runtime_data", "results",
                  ".ipynb_checkpoints"}
FORBIDDEN_SUFFIXES = {".parquet", ".npy", ".npz", ".feather", ".pkl", ".joblib", ".pt", ".h5", ".whl"}
FORBIDDEN_NAMES = re.compile(r"(_fold\d+\.txt$|^fold_\d+\.txt$|^oof_rows|^train\.|^host_holdout)")
TOKEN_PATTERN = re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}"
                           r"|x-access-token:[A-Za-z0-9_]{20,}|\"key\"\s*:\s*\"[0-9a-f]{32}\")")
GITIGNORE_RULES = ("__pycache__/", "*.pyc", ".ipynb_checkpoints/", ".pytest_cache/", "bundle/", "bundle_archive/", "data/", "outputs/",
                   "colab_runtime/", "kaggle_dryrun/", "*.parquet", "*.npy", "*.npz", "*.log", ".env")
REQUIREMENT_LINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?\s*((==|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!-]+\s*(,\s*(==|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!-]+\s*)*)?$")
TRAINING_ONLY = ("scikit-learn", "scipy", "matplotlib", "tqdm", "pyyaml", "joblib", "torch", "tensorflow")
RUNTIME_CORE = ("src/casmi_infer/__init__.py", "src/casmi_infer/pipeline.py", "src/casmi_infer/selftest.py", "src/casmi_infer/validation.py",
                "src/casmi_infer/aggregation.py", "src/casmi_infer/backend.py", "src/casmi_runtime/__init__.py", "src/casmi_runtime/config.py",
                "src/casmi_runtime/accelerator.py", "src/casmi_runtime/backends.py", "src/casmi_runtime/frozen.py",
                "src/casmi_runtime/inference.py", "src/casmi_runtime/named_inference.py",
                "src/casmi_runtime/results.py", "src/casmi_runtime/benchmark.py")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".ipynb", ".json", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".ps1", ".template", ""}
PACKAGES = ("casmi", "casmi_infer", "casmi_runtime")


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
    return sorted({*(f"src/{m}" for m in spec["casmi_modules"]), *(e["dst"] for e in spec["copy_files"]), *RUNTIME_CORE})


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
            problems.append(f"runtime/training artifact in repo: {rel}")
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
        problems += [f"{name} lacks {n}" for n in ("lightgbm", "numpy", "pandas", "pyarrow") if n not in reqs]
        problems += [f"{name} lists a training-only package: {n}" for n in TRAINING_ONLY if n in reqs]
    for nb in (repo / "notebooks").glob("*.ipynb") if (repo / "notebooks").is_dir() else []:
        try:
            json.loads(nb.read_text(encoding="utf-8"))
        except ValueError as e:
            problems.append(f"{nb.name} is not valid JSON: {e}")
    problems += [f"unresolved import in {f}: {m}" for f, m in unresolved_imports(repo)]
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--max-mb", type=float, default=50.0)
    args = ap.parse_args(argv)
    problems = check_repo(args.repo, args.max_mb)
    for p in problems:
        print("PROBLEM:", p)
    print("RUNTIME REPO CHECK:", "PASS" if not problems else f"FAIL ({len(problems)} problems)", "|", Path(args.repo).resolve())
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
