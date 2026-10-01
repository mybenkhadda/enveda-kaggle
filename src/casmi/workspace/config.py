"""v2 configuration loader: `configs/casmi_v2_colab.yaml` -> a plain dict + resolved `V2Paths`.

Path rules:
  * `drive_root` comes from ENVEDA_DRIVE_ROOT when set, else the YAML value;
  * `repo_root` comes from ENVEDA_REPO_ROOT when set, else the YAML value;
  * every other path is relative to `drive_root` unless absolute;
  * `scratch_dir` is disposable (e.g. /content/scratch) -- never the only copy of a result.
"""
import copy
import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ENV_DRIVE_ROOT = "ENVEDA_DRIVE_ROOT"
ENV_REPO_ROOT = "ENVEDA_REPO_ROOT"
DEFAULT_CONFIG_RELPATH = Path("configs") / "casmi_v2_colab.yaml"
PERSISTENT_DIRS = ("raw_data_dir", "processed_dir", "interim_dir", "external_dir", "candidate_db_dir", "embeddings_dir",
                   "checkpoints_dir", "predictions_dir", "reports_dir", "cache_dir", "experiments_dir")


@dataclass(frozen=True)
class V2Paths:
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

    def ensure(self):
        """Create every persistent directory (+ scratch). Never creates `repo_root` or `bundle_dir`."""
        for name in PERSISTENT_DIRS + ("scratch_dir",):
            getattr(self, name).mkdir(parents=True, exist_ok=True)
        return self


def _resolve(base, value):
    """Absolute (incl. POSIX-rooted '/content/...', also when the code is imported on Windows) -> as is;
    otherwise relative to `base`."""
    p = Path(str(value))
    return p if (p.is_absolute() or str(value).startswith("/")) else base / p


def resolve_paths(cfg, environ=None):
    env = os.environ if environ is None else environ
    p = cfg["paths"]
    drive = Path(env.get(ENV_DRIVE_ROOT) or p["drive_root"])
    repo = Path(env.get(ENV_REPO_ROOT) or p.get("repo_root") or drive / "repo")
    kw = {"drive_root": drive, "repo_root": repo}
    for f in fields(V2Paths):
        if f.name not in kw:
            kw[f.name] = _resolve(drive, p[f.name])
    return V2Paths(**kw)


def load_v2_config(path=None, overrides=None, environ=None):
    """Returns `(cfg, paths)`. `path` defaults to `<repo_root>/configs/casmi_v2_colab.yaml` (repo root
    from ENVEDA_REPO_ROOT, else the current directory). `overrides` is a nested dict merged on top."""
    env = os.environ if environ is None else environ
    if path is None:
        path = Path(env.get(ENV_REPO_ROOT, ".")) / DEFAULT_CONFIG_RELPATH
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_config_path"] = str(path)
    return cfg, resolve_paths(cfg, environ=env)


def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def input_path(cfg, paths, name):
    """Absolute path of a named input from `cfg['inputs']` (relative to drive_root)."""
    return _resolve(paths.drive_root, cfg["inputs"][name])
