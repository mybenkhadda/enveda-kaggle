"""Load the FROZEN inference implementation from <bundle>/code -- never the GitHub copy.

`load_frozen_code(bundle_dir)`:
  1. removes every already-imported `casmi` / `casmi_infer` module that does NOT live under <bundle>/code (e.g. the
     GitHub `src/` copy imported for tests or setup), so no stale or newer module object can be reused;
  2. puts <bundle>/code FIRST on sys.path;
  3. imports the inference modules and asserts each file lives under <bundle>/code.
The returned namespace is the only handle the orchestration uses for inference.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

FROZEN_PACKAGES = ("casmi", "casmi_infer")
FROZEN_MODULES = ("pipeline", "selftest", "validation", "submission", "backend", "aggregation")


class FrozenCodeError(RuntimeError):
    """The frozen bundle code is missing, or something other than it would run inference."""


def _is_under(path, root):
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except (TypeError, OSError):
        return False


def purge_foreign_modules(code_dir):
    """Drop cached casmi / casmi_infer modules that do not come from `code_dir`. Returns the purged names."""
    purged = []
    for name in list(sys.modules):
        if name.split(".")[0] in FROZEN_PACKAGES:
            f = getattr(sys.modules[name], "__file__", None)
            if f is None or not _is_under(f, code_dir):
                del sys.modules[name]
                purged.append(name)
    return purged


def load_frozen_code(bundle_dir, modules=FROZEN_MODULES):
    code = Path(bundle_dir) / "code"
    if not (code / "casmi_infer" / "__init__.py").exists():
        raise FrozenCodeError(f"no frozen inference code at {code}/casmi_infer -- the bundle is incomplete")
    purged = purge_foreign_modules(code)
    while str(code) in sys.path:
        sys.path.remove(str(code))
    sys.path.insert(0, str(code))
    importlib.invalidate_caches()
    ns = {"code_dir": code, "purged_modules": purged}
    pkg = importlib.import_module("casmi_infer")
    ns["casmi_infer"] = pkg
    for m in modules:
        ns[m] = importlib.import_module(f"casmi_infer.{m}")
    loaded = {n: getattr(sys.modules[n], "__file__", None) for n in sys.modules if n.split(".")[0] in FROZEN_PACKAGES}
    foreign = {n: f for n, f in loaded.items() if f is not None and not _is_under(f, code)}
    if foreign:
        raise FrozenCodeError(f"non-frozen modules would run inference: {foreign}")
    ns["source"] = str(Path(pkg.__file__).resolve().parent)
    return SimpleNamespace(**ns)
