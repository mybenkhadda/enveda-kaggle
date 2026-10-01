"""v2 configuration loader: `configs/casmi_v2_colab.yaml` -> a plain dict + resolved `V2Paths`.

`V2Paths` is the ONE authoritative path API: it holds the PHYSICAL top-level directory roots only. Semantic
artifact locations (regime table, universe manifest, analog caches, ...) are derived from these roots in exactly
one place, `casmi.workspace.artifact_registry.ArtifactRegistry`. Notebooks use `P.<field>` for roots and
`ARTIFACTS.<name>` for artifacts -- never hand-built path vocabularies.

Path rules:
  * `drive_root`  : ENVEDA_DRIVE_ROOT, else the YAML value (/content/drive/MyDrive/EnvedaCASMI)
  * `repo_root`   : ENVEDA_REPO_ROOT, else the YAML value (/content/Enveda -- a git clone, never on Drive)
  * `scratch_dir` : ENVEDA_SCRATCH_DIR (or the older ENVEDA_SCRATCH_ROOT), else the YAML value
                    (/content/enveda_work -- disposable local SSD)
  * every other path is relative to `drive_root` unless absolute (POSIX-rooted paths count as absolute on every OS)
  * the five processed inputs live in the `inputs:` section, relative to `drive_root` (data/processed/...)
"""
import copy
import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ENV_DRIVE_ROOT = "ENVEDA_DRIVE_ROOT"
ENV_REPO_ROOT = "ENVEDA_REPO_ROOT"
ENV_SCRATCH_DIR = "ENVEDA_SCRATCH_DIR"
ENV_SCRATCH_DIR_OLD = "ENVEDA_SCRATCH_ROOT"          # accepted for runtimes that still export the older variable name
DEFAULT_CONFIG_RELPATH = Path("configs") / "casmi_v2_colab.yaml"
INPUT_KEYS = ("train_spectrum_metadata", "connectivity_folds", "structure_table", "dev_queries", "molecule_mass_variants")

# The YAML must declare exactly this schema version (bump both together when the config layout changes).
CONFIG_SCHEMA_VERSION = "casmi-v2-config-3"

# Bump PATHS_API_VERSION whenever a V2Paths field is added, renamed or removed (also update
# casmi.workspace.versions.NOTEBOOK_API_VERSION and every notebook in the same commit).
PATHS_API_VERSION = "casmi-v2-paths-4"

# Names that some notebooks/code briefly used as V2Paths attributes (paths API generation 3, never on GitHub main)
# -> what to use instead. ERROR HINTS ONLY: these names are never resolved, so a notebook using one fails in
# preflight / scripts/audit_notebooks.py with this replacement in the message.
DEPRECATED_PATH_ALIASES = {
    "scratch_root": "P.scratch_dir",
    "bundle": "P.bundle_dir",
    "data_raw": "P.raw_data_dir",
    "data_processed": "P.processed_dir",
    "external_coconut": "ARTIFACTS.external_coconut_dir",
    "external_pubchem": "ARTIFACTS.external_pubchem_dir",
    "candidates": "P.candidate_db_dir (or ARTIFACTS.universe_root)",
    "universe": "ARTIFACTS.universe_root",
    "formula_index": "ARTIFACTS.candidate_formula_index",
    "candidate_manifests": "ARTIFACTS.candidate_manifests_dir",
    "validation": "ARTIFACTS.validation_dir",
    "regimes": "ARTIFACTS.regimes_dir",
    "features": "ARTIFACTS.features_dir",
    "embeddings": "P.embeddings_dir",
    "checkpoints": "P.checkpoints_dir",
    "predictions": "P.predictions_dir",
    "reports": "P.reports_dir",
    "cache": "P.cache_dir",
    "runs": "P.experiments_dir",
    "exports": "ARTIFACTS.exports_dir",
}


class ConfigSchemaError(RuntimeError):
    """The YAML config does not have the schema this source expects."""


@dataclass(frozen=True)
class V2Paths:
    """Physical directory roots (see module docstring). Field set == casmi-v2-paths-4."""
    drive_root: Path
    repo_root: Path
    raw_data_dir: Path
    processed_dir: Path
    interim_dir: Path
    external_dir: Path
    bundle_dir: Path
    candidate_db_dir: Path
    embeddings_dir: Path
    checkpoints_dir: Path
    predictions_dir: Path
    reports_dir: Path
    cache_dir: Path
    experiments_dir: Path
    scratch_dir: Path

    def as_dict(self):
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def field_names(cls):
        return tuple(f.name for f in fields(cls))

    def ensure(self):
        """Create the persistent Drive tree (`colab_paths.DRIVE_TREE`, directories only, idempotent), every root
        directory and scratch. Never creates or touches `repo_root`; never creates anything inside `bundle_dir`."""
        from casmi.workspace.colab_paths import DRIVE_TREE
        for rel in DRIVE_TREE:
            (self.drive_root / rel).mkdir(parents=True, exist_ok=True)
        for f in fields(self):
            if f.name not in ("repo_root", "drive_root", "bundle_dir"):
                getattr(self, f.name).mkdir(parents=True, exist_ok=True)
        return self


ROOT_FIELDS = ("drive_root", "repo_root", "scratch_dir")


def _resolve(base, value):
    """Absolute (incl. POSIX-rooted '/content/...', also when imported on Windows) -> as is; else relative to `base`."""
    p = Path(str(value))
    return p if (p.is_absolute() or str(value).startswith("/")) else base / p


def resolve_paths(cfg, environ=None):
    env = os.environ if environ is None else environ
    p = cfg["paths"]
    missing = [f.name for f in fields(V2Paths) if f.name not in p]
    if missing:
        raise ConfigSchemaError(f"configs/casmi_v2_colab.yaml `paths:` lacks {missing}.\n"
                                f"Fix: update the repository (git pull) -- the config and src/casmi/workspace/config.py must come from the same commit.")
    kw = {"drive_root": Path(env.get(ENV_DRIVE_ROOT) or p["drive_root"]),
          "repo_root": Path(env.get(ENV_REPO_ROOT) or p["repo_root"]),
          "scratch_dir": Path(env.get(ENV_SCRATCH_DIR) or env.get(ENV_SCRATCH_DIR_OLD) or p["scratch_dir"])}
    for f in fields(V2Paths):
        if f.name not in kw:
            kw[f.name] = _resolve(kw["drive_root"], p[f.name])
    return V2Paths(**kw)


def check_config_schema(cfg):
    """Raise ConfigSchemaError unless the YAML declares CONFIG_SCHEMA_VERSION and has the `inputs:` section."""
    v = cfg.get("schema_version")
    if v != CONFIG_SCHEMA_VERSION:
        raise ConfigSchemaError(f"config schema {v!r} != expected {CONFIG_SCHEMA_VERSION!r} ({cfg.get('_config_path')}).\n"
                                f"Fix: restart the runtime, re-run the bootstrap cell so /content/Enveda is updated from GitHub main; "
                                f"configs/ and src/ must come from the same commit.")
    missing = [k for k in INPUT_KEYS if k not in (cfg.get("inputs") or {})]
    if missing:
        raise ConfigSchemaError(f"config `inputs:` lacks {missing}")
    return v


def load_v2_config(path=None, overrides=None, environ=None):
    """Returns `(cfg, paths)`. `path` defaults to `<repo_root>/configs/casmi_v2_colab.yaml` (repo root from
    ENVEDA_REPO_ROOT, else the current directory). `overrides` is a nested dict merged on top.
    Raises ConfigSchemaError on a schema mismatch (never silently falls back)."""
    env = os.environ if environ is None else environ
    if path is None:
        path = Path(env.get(ENV_REPO_ROOT, ".")) / DEFAULT_CONFIG_RELPATH
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_config_path"] = str(path)
    check_config_schema(cfg)
    return cfg, resolve_paths(cfg, environ=env)


def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def input_path(cfg, paths, name):
    """Absolute path of a named processed input from `cfg['inputs']` (relative to drive_root)."""
    if name not in INPUT_KEYS:
        raise KeyError(f"unknown input {name!r}; expected one of {INPUT_KEYS}")
    return _resolve(paths.drive_root, cfg["inputs"][name])


def external_source_specs(cfg, paths):
    """One dict per configured external candidate source: template (repo-relative), absolute data file path,
    `exists`. Never downloads or creates anything."""
    out = []
    for s in (cfg.get("universe") or {}).get("sources") or []:
        f = _resolve(paths.drive_root, s["file"])
        out.append({"template": s["template"], "template_path": _resolve(paths.repo_root, s["template"]), "file": f,
                    "exists": f.is_file()})
    return out
