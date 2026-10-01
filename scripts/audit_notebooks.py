"""Static notebook auditor -- run before calling any v2 notebook done (it never executes notebook code).

    python scripts/audit_notebooks.py                       # v2 notebooks (notebooks/1[0-9]_*.ipynb, 2[0-9]_*.ipynb)
    python scripts/audit_notebooks.py notebooks/13_*.ipynb  # explicit files
    python scripts/audit_notebooks.py --json report.json

Every finding reports: notebook, cell id / index, rule, offending token, suggested replacement. Imports and call
signatures are checked against the LIVE source tree (`src/` is put on sys.path), i.e. the code the notebook will run.

  ERROR  syntax-error            cell does not parse
  ERROR  missing-bootstrap       no generated bootstrap cell / no bootstrap(...) call
  ERROR  stale-bootstrap         bootstrap cell version != casmi.workspace.versions.BOOTSTRAP_CELL_VERSION, or edited by hand
  ERROR  notebook-api-mismatch   bootstrap(notebook_api=...) != casmi.workspace.versions.NOTEBOOK_API_VERSION
  ERROR  import-before-bootstrap code (casmi imports included) placed before the bootstrap cell
  ERROR  ad-hoc-git              clone / pull / fetch logic outside the bootstrap cell
  ERROR  deprecated-path-alias   P.<alias> from the retired alias vocabulary (P.universe, P.regimes, P.reports, ...)
  ERROR  unknown-path-field      P.<x> that is not a V2Paths field
  ERROR  undeclared-path-field   P.<x> used but not listed in bootstrap(uses_paths=...)
  ERROR  unknown-artifact        ARTIFACTS.<x> / ARTIFACTS['x'] / requires=[...] not in the ArtifactRegistry
  ERROR  undeclared-artifact     ARTIFACTS.<x> used but not listed in bootstrap(uses_artifacts=/requires=/optional=)
  ERROR  forbidden-api           retired modules / names (casmi.workspace.contract, ColabPaths, cache_signature, ...)
  ERROR  obsolete-import         `from casmi... import name` where module or name does not exist
  ERROR  signature-mismatch      a project call that would not bind (inspect.signature)
  ERROR  undefined-name          a name no cell binds
  ERROR  used-before-defined     a name first bound in a LATER cell (notebooks run top to bottom)
  ERROR  hard-coded-path         Windows drive paths, /content/..., MyDrive outside the bootstrap cell
  ERROR  visible-test-leakage    reference to the visible competition test set in a research notebook
  ERROR  missing-leakage-audit   no markdown 'LEAKAGE AUDIT' section (notebooks >= 11)
  ERROR  missing-experiment-log  no log_experiment(...) call (notebooks >= 13)
  WARN   error-output            the saved notebook contains an error output
  WARN   unused-declaration      uses_paths / uses_artifacts lists a name the notebook never touches
Exit code 1 if any ERROR.
"""
import argparse
import ast
import builtins
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from casmi.workspace.artifact_registry import ARTIFACTS_SPECS  # noqa: E402
from casmi.workspace.config import DEPRECATED_PATH_ALIASES, V2Paths  # noqa: E402
from casmi.workspace.notebook_cells import BOOTSTRAP_CELL, BOOTSTRAP_MARKER_PREFIX, bootstrap_cell_version  # noqa: E402
from casmi.workspace.versions import BOOTSTRAP_CELL_VERSION, NOTEBOOK_API_VERSION  # noqa: E402

DEFAULT_GLOBS = ("notebooks/1[0-9]_*.ipynb", "notebooks/2[0-9]_*.ipynb")
P_METHODS = {"as_dict", "ensure", "field_names"}
REGISTRY_METHODS = {"path", "spec", "table", "names", "API_VERSION"}
IPY_NAMES = {"display", "get_ipython", "In", "Out", "__file__"}
HARD_PATH = re.compile(r"(^[A-Za-z]:[\\/])|(^/content/)|(MyDrive)")
TEST_LEAK = re.compile(r"(^|[/\\_'\"])test(\.parquet|_spectrum|_molecule|_spectra)", re.IGNORECASE)
INSTANCE_FACTORIES = {"load", "from_dict", "from_env", "build", "from_config"}
GIT_WORDS = {"clone", "pull", "fetch"}
FORBIDDEN_MODULES = {"casmi.workspace.contract": "casmi.workspace.artifact_registry"}
FORBIDDEN_NAMES = {  # (module, name) -> replacement
    ("casmi.workspace.colab_paths", "ColabPaths"): "V2Paths fields + casmi.workspace.colab_paths.missing_required(P)",
    ("casmi.candidates.universe", "universe_c2_mode"): "casmi.validation.c2_protocol.assess_universe",
    ("casmi.analog.pipeline", "cache_signature"): "casmi.workspace.cache_identity (run_analog_fold namespaces caches itself)",
    ("casmi.analog.pipeline", "guard_cache_dir"): "casmi.workspace.cache_identity.open_namespace",
    ("casmi.workspace.config", "LEGACY_PATH_FIELDS"): "casmi.workspace.config.DEPRECATED_PATH_ALIASES",
}
CASMI_ROOTS = ("casmi", "casmi_infer", "casmi_runtime")


def notebook_number(path):
    m = re.match(r"(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


def strip_magics(src):
    return "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.splitlines())


def _source(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def finding(nb, cell_index, cell_id, severity, code, message, token="", replacement=""):
    return {"notebook": Path(nb).name, "cell": cell_index, "cell_id": cell_id, "severity": severity, "code": code, "token": token,
            "replacement": replacement, "message": message}


def _bound_names(tree):
    """Every name a cell binds anywhere (assignments, defs, args, imports, handlers, comprehension targets)."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, ast.MatchAs) and n.name:
            out.add(n.name)
    return out


def _loaded_names(tree):
    return {(n.id, n.lineno) for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _const_str_list(node):
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


def _bind_check(obj, call, drop_first=False):
    """None if the call binds (or cannot be checked), else an error string."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return None
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    try:
        sig.bind(*([None] * (len(call.args) + (1 if drop_first else 0))), **{k.arg: None for k in call.keywords})
    except TypeError as e:
        return f"{e} -- actual signature {sig}"
    return None


def audit_notebook(path):
    path = Path(path)
    nb = json.loads(path.read_text(encoding="utf-8"))
    num = notebook_number(path)
    research = not re.search(r"(inference|kaggle)", path.name, re.IGNORECASE)
    out = []
    cells = list(enumerate(nb.get("cells", [])))
    cid = {i: c.get("id") for i, c in cells}

    def add(i, sev, code, msg, token="", repl=""):
        out.append(finding(path, i, cid.get(i), sev, code, msg, token, repl))

    md_text = "\n".join(_source(c) for _, c in cells if c.get("cell_type") == "markdown")
    code_cells = [(i, _source(c)) for i, c in cells if c.get("cell_type") == "code"]
    for i, c in cells:
        for o in c.get("outputs", []) or []:
            if o.get("output_type") == "error":
                add(i, "WARN", "error-output", f"saved error output {o.get('ename')}: {str(o.get('evalue'))[:120]}")

    # ---- bootstrap cell ------------------------------------------------------------------------------------------
    boot = [(i, s) for i, s in code_cells if s.lstrip().startswith(BOOTSTRAP_MARKER_PREFIX)]
    boot_idx = {i for i, _ in boot}
    if not boot:
        add(None, "ERROR", "missing-bootstrap", "no generated bootstrap cell", "", "python scripts/stamp_notebooks.py <notebook>")
    else:
        bi, bs = boot[0]
        v = bootstrap_cell_version(bs)
        if v != BOOTSTRAP_CELL_VERSION:
            add(bi, "ERROR", "stale-bootstrap", f"bootstrap cell version {v!r} != {BOOTSTRAP_CELL_VERSION!r}", v or "",
                "python scripts/stamp_notebooks.py <notebook>")
        elif bs.strip() != BOOTSTRAP_CELL.strip():
            add(bi, "ERROR", "stale-bootstrap", "bootstrap cell differs from casmi.workspace.notebook_cells.BOOTSTRAP_CELL (hand-edited)",
                "", "python scripts/stamp_notebooks.py <notebook>")
        for i, s in code_cells:
            if i >= bi:
                break
            if s.strip():
                token = "casmi import" if re.search(r"^\s*(from|import)\s+casmi", s, re.M) else "code"
                add(i, "ERROR", "import-before-bootstrap", "code cell placed before the bootstrap cell", token, "move it after the bootstrap cell")

    trees = []
    for i, src in code_cells:
        try:
            trees.append((i, ast.parse(strip_magics(src))))
        except SyntaxError as e:
            add(i, "ERROR", "syntax-error", f"line {e.lineno}: {e.msg}")

    # ---- names -----------------------------------------------------------------------------------------------------
    bound_by_cell = {i: _bound_names(t) for i, t in trees}
    all_bound = set().union(*bound_by_cell.values()) if bound_by_cell else set()
    builtin_names = set(dir(builtins)) | IPY_NAMES
    seen = set()
    for i, t in trees:
        for name, line in sorted(_loaded_names(t)):
            if name in builtin_names:
                continue
            if name not in all_bound:
                add(i, "ERROR", "undefined-name", f"line {line}: {name!r} is never defined in this notebook", name)
            elif name not in seen and name not in bound_by_cell[i]:
                add(i, "ERROR", "used-before-defined", f"line {line}: {name!r} is first bound in a later cell", name)
        seen |= bound_by_cell[i]

    # ---- imports / calls / P / ARTIFACTS ---------------------------------------------------------------------------
    fields_ = set(V2Paths.field_names())
    imported, instances = {}, {}
    p_used, a_used = {}, {}
    declared_paths = declared_artifacts = None
    declared_reqs = []
    boot_call = None
    for i, t in trees:
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in CASMI_ROOTS:
                if n.module in FORBIDDEN_MODULES:
                    add(i, "ERROR", "forbidden-api", f"retired module {n.module}", n.module, FORBIDDEN_MODULES[n.module])
                    continue
                try:
                    mod = importlib.import_module(n.module)
                except Exception as e:                       # ImportError, but also a SyntaxError inside the source
                    add(i, "ERROR", "obsolete-import", f"module {n.module} cannot be imported: {e}", n.module)
                    continue
                for a in n.names:
                    if a.name == "*":
                        continue
                    if (n.module, a.name) in FORBIDDEN_NAMES:
                        add(i, "ERROR", "forbidden-api", f"retired name {n.module}.{a.name}", a.name, FORBIDDEN_NAMES[(n.module, a.name)])
                    if hasattr(mod, a.name):
                        imported[a.asname or a.name] = getattr(mod, a.name)
                        continue
                    try:
                        imported[a.asname or a.name] = importlib.import_module(f"{n.module}.{a.name}")
                    except ImportError:
                        add(i, "ERROR", "obsolete-import", f"{n.module} has no attribute {a.name!r}", a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name in FORBIDDEN_MODULES:
                        add(i, "ERROR", "forbidden-api", f"retired module {a.name}", a.name, FORBIDDEN_MODULES[a.name])
                    elif a.name.split(".")[0] in CASMI_ROOTS:
                        try:
                            importlib.import_module(a.name)
                        except Exception as e:
                            add(i, "ERROR", "obsolete-import", f"module {a.name} cannot be imported: {e}", a.name)
        for n in ast.walk(t):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call):
                f = n.value.func
                if isinstance(f, ast.Name) and inspect.isclass(imported.get(f.id)):
                    instances[n.targets[0].id] = imported[f.id]
                elif (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and inspect.isclass(imported.get(f.value.id))
                      and f.attr in INSTANCE_FACTORIES):
                    instances[n.targets[0].id] = imported[f.value.id]
        for n in ast.walk(t):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                if n.value.id == "P":
                    p_used.setdefault(n.attr, i)
                elif n.value.id == "ARTIFACTS" and n.attr not in REGISTRY_METHODS and not n.attr.startswith("_"):
                    a_used.setdefault(n.attr, i)
            elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "ARTIFACTS" \
                    and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                a_used.setdefault(n.slice.value, i)
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "ARTIFACTS" and f.attr == "path" \
                    and n.args and isinstance(n.args[0], ast.Constant):
                a_used.setdefault(n.args[0].value, i)
            if isinstance(f, ast.Name) and f.id == "bootstrap":
                boot_call = (i, n)
                for k in n.keywords:
                    if k.arg == "uses_paths":
                        declared_paths = set(_const_str_list(k.value) or [])
                    elif k.arg == "uses_artifacts":
                        declared_artifacts = set(_const_str_list(k.value) or [])
                    elif k.arg in ("requires", "optional"):
                        declared_reqs += [(x, i) for x in (_const_str_list(k.value) or [])]
                    elif k.arg == "notebook_api":
                        v = k.value.value if isinstance(k.value, ast.Constant) else None
                        if v != NOTEBOOK_API_VERSION:
                            add(i, "ERROR", "notebook-api-mismatch", f"bootstrap(notebook_api={v!r}) != source {NOTEBOOK_API_VERSION!r}",
                                str(v), NOTEBOOK_API_VERSION)
            if isinstance(f, ast.Attribute) and f.attr == "run" and isinstance(f.value, ast.Name) and f.value.id == "subprocess" \
                    and i not in boot_idx:
                words = {e.value for a in n.args if isinstance(a, (ast.List, ast.Tuple)) for e in a.elts if isinstance(e, ast.Constant)}
                if "git" in words and words & GIT_WORDS:
                    add(i, "ERROR", "ad-hoc-git", f"line {n.lineno}: git {sorted(words & GIT_WORDS)} outside the bootstrap cell", "git",
                        "remove it; the generated bootstrap cell clones / fast-forwards the repository")
            obj, drop_first, label = None, False, None
            if isinstance(f, ast.Name) and f.id in imported and callable(imported[f.id]):
                obj, label = imported[f.id], f.id
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                base, cls = imported.get(f.value.id), instances.get(f.value.id)
                if base is not None and inspect.ismodule(base):
                    if not hasattr(base, f.attr):
                        add(i, "ERROR", "obsolete-import", f"{base.__name__} has no attribute {f.attr!r}", f.attr)
                    else:
                        obj, label = getattr(base, f.attr), f"{f.value.id}.{f.attr}"
                elif base is not None and hasattr(base, f.attr):
                    obj, label = getattr(base, f.attr), f"{f.value.id}.{f.attr}"
                    drop_first = inspect.isclass(base) and inspect.isfunction(inspect.getattr_static(base, f.attr, None))
                elif cls is not None:
                    if not hasattr(cls, f.attr):
                        add(i, "ERROR", "signature-mismatch", f"{f.value.id} ({cls.__name__}) has no method {f.attr!r}", f.attr)
                    else:
                        obj, label = getattr(cls, f.attr), f"{f.value.id}.{f.attr}"
                        drop_first = inspect.isfunction(inspect.getattr_static(cls, f.attr, None))
            if obj is not None and callable(obj):
                err = _bind_check(obj, n, drop_first=drop_first)
                if err:
                    add(i, "ERROR", "signature-mismatch", f"line {n.lineno}: {label}(...) -> {err}", label)
        for n in ast.walk(t):
            if i in boot_idx or not (isinstance(n, ast.Constant) and isinstance(n.value, str)) or "\n" in n.value:
                continue
            if HARD_PATH.search(n.value):
                add(i, "ERROR", "hard-coded-path", f"line {n.lineno}: hard-coded path", n.value[:80], "P.<root> / ARTIFACTS.<name>")
            if research and TEST_LEAK.search(n.value):
                add(i, "ERROR", "visible-test-leakage", f"line {n.lineno}: the visible competition test set is never a research input",
                    n.value[:80], "use dev_queries (training spectra) through the regime table")

    for name, i in sorted(p_used.items()):
        if name in fields_ or name in P_METHODS:
            continue
        if name in DEPRECATED_PATH_ALIASES:
            add(i, "ERROR", "deprecated-path-alias", f"P.{name} is a retired alias", f"P.{name}", DEPRECATED_PATH_ALIASES[name])
        else:
            add(i, "ERROR", "unknown-path-field", f"P.{name} is not a V2Paths field", f"P.{name}", f"one of {sorted(fields_)}")
    checked = sorted(a_used.items(), key=lambda kv: str(kv[0])) + declared_reqs + [(x, None) for x in sorted(declared_artifacts or [])]
    for name, i in checked:
        if name not in ARTIFACTS_SPECS:
            add(i, "ERROR", "unknown-artifact", f"artifact {name!r} is not in the ArtifactRegistry", str(name),
                "a name from casmi.workspace.artifact_registry.ARTIFACTS_SPECS")
    if boot and boot_call is None:
        add(None, "ERROR", "missing-bootstrap", "no bootstrap(...) call (casmi.workspace.bootstrap)", "", "CTX = bootstrap(NOTEBOOK, ...)")
    if boot_call is not None:
        bi = boot_call[0]
        if declared_paths is None:
            add(bi, "ERROR", "undeclared-path-field", "bootstrap(...) has no literal uses_paths=(...)")
        else:
            for name in sorted(set(p_used) - P_METHODS - declared_paths):
                add(p_used[name], "ERROR", "undeclared-path-field", f"P.{name} used but not declared in uses_paths", f"P.{name}")
            for name in sorted(declared_paths - set(p_used)):
                add(bi, "WARN", "unused-declaration", f"uses_paths declares {name!r} but the notebook never uses P.{name}", name)
        declared_all = (declared_artifacts or set()) | {x for x, _ in declared_reqs}
        if declared_artifacts is None:
            add(bi, "ERROR", "undeclared-artifact", "bootstrap(...) has no literal uses_artifacts=(...)")
        else:
            for name in sorted(set(a_used) - declared_all):
                add(a_used[name], "ERROR", "undeclared-artifact", f"ARTIFACTS.{name} used but not declared in bootstrap(...)", str(name))
            for name in sorted(declared_artifacts - set(a_used)):
                add(bi, "WARN", "unused-declaration", f"uses_artifacts declares {name!r} but the notebook never uses it", name)
    if num >= 11 and "LEAKAGE AUDIT" not in md_text.upper():
        add(None, "ERROR", "missing-leakage-audit", "no markdown 'LEAKAGE AUDIT' section", "", "## LEAKAGE AUDIT + assertions")
    if num >= 13 and research and "log_experiment(" not in "\n".join(s for _, s in code_cells):
        add(None, "ERROR", "missing-experiment-log", "notebook never calls log_experiment(...)", "", "casmi.workspace.experiments.log_experiment")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--json", help="write findings to this JSON file")
    ap.add_argument("--quiet", action="store_true", help="only print the summary line")
    a = ap.parse_args(argv)
    files = [Path(p) for p in a.paths] if a.paths else sorted({p for g in DEFAULT_GLOBS for p in REPO.glob(g)})
    findings = []
    for f in files:
        findings += audit_notebook(f)
    n_err = sum(f["severity"] == "ERROR" for f in findings)
    n_warn = sum(f["severity"] == "WARN" for f in findings)
    if not a.quiet:
        for f in findings:
            repl = f" -> {f['replacement']}" if f["replacement"] else ""
            tok = f" [{f['token']}]" if f["token"] else ""
            print(f"{f['severity']:5s} {f['notebook']} cell {f['cell']} ({f['cell_id']}) {f['code']}{tok}: {f['message']}{repl}")
    print(f"audited {len(files)} notebook(s): {n_err} error(s), {n_warn} warning(s)")
    if a.json:
        Path(a.json).write_text(json.dumps(findings, indent=1), encoding="utf-8")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
