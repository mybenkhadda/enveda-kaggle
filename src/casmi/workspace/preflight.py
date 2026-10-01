"""Preflight: fail in seconds, with an actionable message, instead of 30 minutes into a notebook.

Checks (all run; `preflight()` raises ONE PreflightError listing every problem as `ERROR: ... / Fix: ...`):

  versions    config schema, paths API, artifact-registry API, notebook API and the generated bootstrap-cell version
              (casmi.workspace.versions) -- the notebook/source skew that produced `AttributeError: P.universe`
  repo        repository exists, branch, HEAD sha, dirty files, commits behind origin/<branch> (stale clone)
  paths       every `P.<field>` the notebook uses is a field of `dataclasses.fields(V2Paths)`; deprecated aliases get
              their replacement (config.DEPRECATED_PATH_ALIASES)
  artifacts   every `ARTIFACTS.<name>` the notebook declares exists in the registry; required ones exist on disk
              (logical name, path, exists, size, mtime, producer)
  signatures  project functions accept the parameters the notebook passes (`inspect.signature`)
  sources     external candidate source files (reported; required only where a notebook says so)
  bundle      frozen bundle identity LABELS (bundle_version / CONFIG_HASH / model_id; never hashes files)

Nothing here creates, modifies or deletes data. Nothing falls back to an incompatible API.
"""
import difflib
import importlib
import inspect
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from casmi.workspace.config import CONFIG_SCHEMA_VERSION, DEPRECATED_PATH_ALIASES, PATHS_API_VERSION, V2Paths, external_source_specs
from casmi.workspace.versions import ARTIFACT_REGISTRY_API_VERSION, BOOTSTRAP_CELL_VERSION, NOTEBOOK_API_VERSION

RESTART_FIX = ("restart the runtime, re-run the bootstrap cell (it fetches + fast-forwards /content/Enveda to GitHub main), "
               "and open the notebook from the same commit (GitHub main)")


class PreflightError(RuntimeError):
    """One or more preflight checks failed (the message lists all of them)."""


class CompatibilityError(PreflightError):
    """Notebook and source code disagree (versions, path fields, artifact names, function signatures)."""


class MissingDependencyError(PreflightError):
    """A required input, upstream artifact, external source or bundle is missing."""


def problem(message, fix):
    return f"ERROR: {message}\n    Fix: {fix}"


# ---------------------------------------------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------------------------------------------

def check_versions(notebook_api, bootstrap_cell_version=None, cfg=None, registry=None):
    """Problems for any version the notebook expects that the imported source does not provide."""
    out = []
    if notebook_api != NOTEBOOK_API_VERSION:
        out.append(problem(f"notebook expects notebook API {notebook_api!r} but the imported source exposes {NOTEBOOK_API_VERSION!r} "
                           f"(paths API {PATHS_API_VERSION}, artifact registry API {ARTIFACT_REGISTRY_API_VERSION})", RESTART_FIX))
    if bootstrap_cell_version is not None and bootstrap_cell_version != BOOTSTRAP_CELL_VERSION:
        out.append(problem(f"this notebook's bootstrap cell is {bootstrap_cell_version!r} but the source generates {BOOTSTRAP_CELL_VERSION!r}",
                           "use the notebook from GitHub main (its bootstrap cell is re-stamped by scripts/stamp_notebooks.py)"))
    if cfg is not None and cfg.get("schema_version") != CONFIG_SCHEMA_VERSION:
        out.append(problem(f"config schema {cfg.get('schema_version')!r} != expected {CONFIG_SCHEMA_VERSION!r}", RESTART_FIX))
    if registry is not None and getattr(registry, "API_VERSION", None) != ARTIFACT_REGISTRY_API_VERSION:
        out.append(problem(f"notebook expects artifact registry API {ARTIFACT_REGISTRY_API_VERSION!r} but the imported registry exposes "
                           f"{getattr(registry, 'API_VERSION', None)!r}", RESTART_FIX))
    return out


# ---------------------------------------------------------------------------------------------------------------
# repository
# ---------------------------------------------------------------------------------------------------------------

def _git(repo_root, *args, run=subprocess.run, timeout=60):
    r = run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def repo_status(repo_root, branch="main", fetch=False, run=subprocess.run):
    """{'is_git', 'branch', 'sha', 'short_sha', 'subject', 'dirty_files', 'behind', 'ahead', 'remote_sha', 'fetched', 'error'}.
    `behind`/`ahead` compare HEAD with origin/<branch> as last fetched (`fetch=True` fetches first)."""
    repo_root = Path(repo_root)
    out = {"repo_root": str(repo_root), "is_git": (repo_root / ".git").exists(), "branch": None, "sha": None, "short_sha": None,
           "subject": None, "dirty_files": [], "behind": None, "ahead": None, "remote_sha": None, "fetched": False, "error": None}
    if not out["is_git"]:
        out["error"] = "not a git repository (uploaded copy?) -- source version cannot be verified"
        return out
    try:
        if fetch:
            rc, _, err = _git(repo_root, "fetch", "origin", branch, run=run, timeout=120)
            out["fetched"] = rc == 0
            if rc != 0:
                out["error"] = f"git fetch failed: {err[:200]}"
        _, out["branch"], _ = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD", run=run)
        _, out["sha"], _ = _git(repo_root, "rev-parse", "HEAD", run=run)
        out["short_sha"] = (out["sha"] or "")[:7] or None
        _, out["subject"], _ = _git(repo_root, "log", "-1", "--format=%s", run=run)
        _, st, _ = _git(repo_root, "diff", "--name-only", "HEAD", run=run)          # modified tracked files (staged or not)
        out["dirty_files"] = [line.strip() for line in st.splitlines() if line.strip()]
        rc, rsha, _ = _git(repo_root, "rev-parse", f"origin/{branch}", run=run)
        if rc == 0:
            out["remote_sha"] = rsha
            rc, counts, _ = _git(repo_root, "rev-list", "--left-right", "--count", f"HEAD...origin/{branch}", run=run)
            if rc == 0 and counts:
                a, b = counts.split()
                out["ahead"], out["behind"] = int(a), int(b)
    except (OSError, subprocess.SubprocessError) as e:
        out["error"] = f"git unavailable: {e}"
    return out


def check_repo(status, branch="main", allow_dirty=True, allow_behind=False, allow_other_branch=False):
    """Problems (list of str) for a `repo_status` dict. A non-git copy is reported as a NOTE by preflight()."""
    out = []
    if not status["is_git"]:
        return out
    if status["branch"] != branch and not allow_other_branch:
        out.append(problem(f"repository is on branch {status['branch']!r}, expected {branch!r}",
                           f"git -C {status['repo_root']} checkout {branch} (or set ENVEDA_BRANCH before the bootstrap cell)"))
    if status["behind"] and not allow_behind:
        out.append(problem(f"STALE SOURCE: HEAD {status['short_sha']} is {status['behind']} commit(s) behind origin/{branch}", RESTART_FIX))
    if status["dirty_files"] and not allow_dirty:
        out.append(problem(f"working tree has {len(status['dirty_files'])} modified tracked file(s): {status['dirty_files'][:5]}",
                           "source lives on GitHub only: discard local edits in the clone (git stash) and re-run the bootstrap cell"))
    return out


# ---------------------------------------------------------------------------------------------------------------
# paths / artifacts / signatures
# ---------------------------------------------------------------------------------------------------------------

def check_path_fields(used, paths_cls=V2Paths):
    """Problems for every name in `used` that is not a V2Paths field / method."""
    valid = set(paths_cls.field_names()) | {"as_dict", "ensure", "field_names"}
    out = []
    for name in sorted(set(used)):
        if name in valid:
            continue
        if name in DEPRECATED_PATH_ALIASES:
            out.append(problem(f"P.{name} is not a V2Paths field (deprecated alias)", f"use {DEPRECATED_PATH_ALIASES[name]}"))
        else:
            hint = difflib.get_close_matches(name, sorted(valid), n=2)
            out.append(problem(f"P.{name} does not exist in V2Paths{f' (did you mean {hint}?)' if hint else ''}",
                               f"use one of {sorted(paths_cls.field_names())} or an ARTIFACTS.<name> location"))
    return out


def check_artifact_names(registry, names):
    out = []
    for n in sorted(set(names)):
        try:
            registry.spec(n)
        except KeyError as e:
            out.append(problem(str(e).strip("\"'"), "use a name from casmi.workspace.artifact_registry (ArtifactRegistry.names())"))
    return out


def resolve_callable(dotted):
    """'pkg.module:attr.sub' (or 'pkg.module.attr') -> object."""
    if ":" in dotted:
        mod, attr = dotted.split(":", 1)
    else:
        mod, attr = dotted.rsplit(".", 1)
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def check_signatures(spec):
    """`spec`: {'module:callable': ['param', ...]} -- every listed parameter must be accepted (by name, or via **kwargs).
    Returns (table, problems)."""
    rows, out = [], []
    for dotted, params in (spec or {}).items():
        try:
            obj = resolve_callable(dotted)
        except (ImportError, AttributeError) as e:
            rows.append({"callable": dotted, "ok": False, "detail": f"missing: {e}"})
            out.append(problem(f"{dotted} cannot be imported ({e}) -- obsolete name or stale source", RESTART_FIX))
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            rows.append({"callable": dotted, "ok": True, "detail": "no introspectable signature"})
            continue
        names = set(sig.parameters)
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        missing = [p for p in params if p not in names and not var_kw]
        rows.append({"callable": dotted, "ok": not missing, "detail": str(sig) if not missing else f"missing params {missing}; actual {sig}"})
        if missing:
            out.append(problem(f"{dotted}{sig} does not accept {missing} -- the notebook expects an obsolete signature", RESTART_FIX))
    return pd.DataFrame(rows, columns=["callable", "ok", "detail"]), out


# ---------------------------------------------------------------------------------------------------------------
# inputs / upstream artifacts
# ---------------------------------------------------------------------------------------------------------------

def _small_dir(p, limit=2000):
    n = 0
    for _ in p.rglob("*"):
        n += 1
        if n > limit:
            return False
    return True


def _describe(name, path, required, producer):
    p = Path(path)
    exists = p.exists()
    size = mtime = None
    if exists:
        st = p.stat()
        if p.is_file():
            size = st.st_size
        elif _small_dir(p):                      # never walk a huge directory tree on Drive just to print a size
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return {"logical_name": name, "expected_path": str(p), "exists": exists, "kind": "dir" if p.is_dir() else "file",
            "size_mb": round(size / 1024 ** 2, 2) if size is not None else None, "mtime_utc": mtime,
            "required": bool(required), "producer": producer}


def validate_inputs(spec, raise_on_missing=True):
    """`spec`: {logical_name: (path, required, producer)} (producer optional). Returns the table; raises
    `MissingDependencyError` listing every missing REQUIRED entry with the notebook that produces it."""
    rows = []
    for name, v in spec.items():
        path, required, producer = (tuple(v) + (None,))[:3] if isinstance(v, (tuple, list)) else (v, True, None)
        rows.append(_describe(name, path, required, producer))
    table = pd.DataFrame(rows, columns=["logical_name", "expected_path", "exists", "kind", "size_mb", "mtime_utc", "required", "producer"])
    missing = table[table["required"] & ~table["exists"]] if len(table) else table
    if raise_on_missing and len(missing):
        lines = [f"  - {r.logical_name}: {r.expected_path}  (produced by: {r.producer or 'unknown'})" for r in missing.itertuples()]
        raise MissingDependencyError("missing required inputs / upstream artifacts:\n" + "\n".join(lines))
    return table


def artifact_spec(artifacts, required=(), optional=()):
    """Build a `validate_inputs` spec from registry names (casmi.workspace.artifact_registry)."""
    spec = {}
    for names, req in ((required, True), (optional, False)):
        for n in names:
            spec[n] = (artifacts.path(n), req, artifacts.spec(n).producer)
    return spec


def external_sources_table(cfg, paths):
    rows = [{"template": s["template"], "file": str(s["file"]), "exists": s["exists"]} for s in external_source_specs(cfg, paths)]
    return pd.DataFrame(rows, columns=["template", "file", "exists"])


# ---------------------------------------------------------------------------------------------------------------
# everything
# ---------------------------------------------------------------------------------------------------------------

def preflight(cfg, paths, artifacts, notebook, notebook_api, uses_paths=(), uses_artifacts=(), requires=(), optional=(),
              signatures=None, bootstrap_cell_version=None, require_external_sources=False, check_bundle=False, branch="main",
              fetch=False, allow_dirty=True, allow_behind=False, log=print):
    """Run every check, print a compact report, raise ONE PreflightError (subclass by the first failing family) listing
    every problem. Returns the report dict on success."""
    from casmi.workspace.artifact_registry import bundle_identity, check_bundle_identity

    status = repo_status(paths.repo_root, branch=branch, fetch=fetch)
    families = [("versions", check_versions(notebook_api, bootstrap_cell_version, cfg, artifacts), CompatibilityError),
                ("repository", check_repo(status, branch=branch, allow_dirty=allow_dirty, allow_behind=allow_behind), PreflightError),
                ("path fields", check_path_fields(uses_paths), CompatibilityError),
                ("artifact names", check_artifact_names(artifacts, list(uses_artifacts) + list(requires) + list(optional)), CompatibilityError)]
    sig_table, sig_problems = check_signatures(signatures)
    families.append(("function signatures", sig_problems, CompatibilityError))
    known = [n for n in list(requires) + list(optional) if n in artifacts.names()]
    inputs = validate_inputs(artifact_spec(artifacts, [n for n in requires if n in known], [n for n in optional if n in known]),
                             raise_on_missing=False)
    missing = inputs[inputs["required"] & ~inputs["exists"]] if len(inputs) else inputs
    families.append(("upstream artifacts", [problem(f"missing {r.logical_name}: {r.expected_path}", f"run {r.producer} first")
                                            for r in missing.itertuples()], MissingDependencyError))
    sources = external_sources_table(cfg, paths)
    if require_external_sources and not sources["exists"].any():
        families.append(("external candidate sources", [problem(
            f"no external candidate source file exists ({', '.join(sources['file'])})",
            "upload a COCONUT export to Drive data/external/coconut/coconut.csv (columns: configs/v6/coconut_source.json)")],
            MissingDependencyError))
    if check_bundle:
        families.append(("frozen bundle", [problem(p, "copy the frozen v2-A7 bundle with scripts/copy_colab_assets_to_drive.ps1")
                                           for p in check_bundle_identity(artifacts)], MissingDependencyError))

    log(f"[preflight] {notebook} | source {status.get('short_sha') or '?'} on {status.get('branch')} ({status.get('subject') or 'no git'})"
        f" | notebook API {NOTEBOOK_API_VERSION} | paths {PATHS_API_VERSION} | registry {ARTIFACT_REGISTRY_API_VERSION}")
    if status.get("dirty_files"):
        log(f"[preflight] NOTE working tree has {len(status['dirty_files'])} modified file(s) -- results are not reproducible from a commit")
    if status.get("error"):
        log(f"[preflight] NOTE {status['error']}")
    if len(inputs):
        with pd.option_context("display.max_colwidth", 90, "display.width", 200):
            log(inputs[["logical_name", "exists", "size_mb", "mtime_utc", "required", "producer"]].to_string(index=False))
    if len(sources):
        log("[preflight] external candidate sources: " + "; ".join(f"{r.file} exists={r.exists}" for r in sources.itertuples()))

    failed = [(fam, probs, exc) for fam, probs, exc in families if probs]
    if failed:
        msg = [f"PREFLIGHT FAILED for {notebook} -- nothing was computed:"]
        for fam, probs, _ in failed:
            msg.append(f"[{fam}]")
            msg += [f"  {p}" for p in probs]
        raise failed[0][2]("\n".join(msg))
    log(f"[preflight] OK: {len(requires)} required artifact(s), {len(uses_paths)} path field(s), {len(uses_artifacts)} artifact name(s), "
        f"{len(signatures or {})} signature(s) checked")
    return {"notebook": notebook, "repo": status, "inputs": inputs, "signatures": sig_table, "external_sources": sources,
            "bundle_identity": bundle_identity(artifacts) if check_bundle else None, "notebook_api": NOTEBOOK_API_VERSION}
