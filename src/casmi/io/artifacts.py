"""Centralized artifact persistence: every saved table is a `<name>.parquet` plus a
`<name>.meta.json` sidecar carrying provenance (inputs, config, versions, seed, shape).

This is what replaces notebook globals like `ARTIFACTS = {}` as the way results move between
pipeline stages and between notebooks -- callers get back a `Path`, not a shared mutable dict.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from casmi.io.files import file_fingerprint
from casmi.paths import get_project_paths
from casmi.utils.hashing import stable_config_hash


def _software_versions():
    versions = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "pyarrow", "rdkit", "sklearn", "scipy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


def artifact_paths(name, directory):
    directory = Path(directory)
    return directory / f"{name}.parquet", directory / f"{name}.meta.json"


def artifact_metadata_path(name, directory):
    return artifact_paths(name, directory)[1]


def save_artifact(df, name, directory, config=None, description="", input_fingerprints=None, seed=None, extra=None):
    """Write `df` to `<directory>/<name>.parquet` plus a `<name>.meta.json` sidecar.

    `config`: the pipeline config used to build this artifact (hashed, not stored verbatim
        with every field, so a config change is detectable via `artifact_is_valid`).
    `input_fingerprints`: dict of {input_name: casmi.io.files.file_fingerprint(...)} for
        every upstream file/artifact this one was built from.
    `seed`: the random seed used, if any (redundant with `config.seed` when a config is
        given, but explicit for artifacts built without one).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = artifact_paths(name, directory)

    df.to_parquet(data_path, index=False)

    meta = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "row_count": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "software_versions": _software_versions(),
        "seed": seed if seed is not None else (getattr(config, "seed", None) if config is not None else None),
        "config_hash": stable_config_hash(config) if config is not None else None,
        "input_fingerprints": input_fingerprints or {},
    }
    if extra:
        meta["extra"] = extra

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return data_path


def save_artifact_metadata(name, directory, row_count, columns, config=None, description="",
                            input_fingerprints=None, seed=None, extra=None):
    """Write just the `<name>.meta.json` sidecar for a parquet file that was written directly
    (e.g. by `casmi.spectra.streaming.process_spectrum_file`, which writes incrementally and
    doesn't go through `save_artifact`). Same provenance contract as `save_artifact`, minus
    the actual data write."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _, meta_path = artifact_paths(name, directory)

    meta = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "row_count": row_count,
        "columns": list(columns),
        "software_versions": _software_versions(),
        "seed": seed if seed is not None else (getattr(config, "seed", None) if config is not None else None),
        "config_hash": stable_config_hash(config) if config is not None else None,
        "input_fingerprints": input_fingerprints or {},
    }
    if extra:
        meta["extra"] = extra

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta_path


def _resolve_search_dirs(directory):
    if directory is not None:
        return [Path(directory)]
    paths = get_project_paths()
    return [paths.processed, paths.interim]


def load_artifact(name, directory=None):
    """Load `<name>.parquet`. If `directory` is omitted, searches the project's `processed/`
    directory first, then `interim/` (matching the pipeline's own write order: later, more
    refined artifacts live in `processed/`)."""
    for d in _resolve_search_dirs(directory):
        data_path, _ = artifact_paths(name, d)
        if data_path.exists():
            return pd.read_parquet(data_path)
    searched = ", ".join(str(d) for d in _resolve_search_dirs(directory))
    raise FileNotFoundError(f"Artifact '{name}' not found in: {searched}")


def load_artifact_metadata(name, directory=None):
    for d in _resolve_search_dirs(directory):
        _, meta_path = artifact_paths(name, d)
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
    return None


def artifact_fingerprint(name, directory=None):
    """A stable, cheap-to-compute fingerprint for artifact `name`, suitable for passing into a
    downstream artifact's `input_fingerprints`: resolved path + size + mtime of the parquet
    file (via `casmi.io.files.file_fingerprint`, so it never hashes file content) PLUS the
    upstream artifact's own `config_hash` and `row_count` from its `.meta.json` when available
    -- so a downstream cache invalidates not only when the file itself changes on disk, but
    also when it was rebuilt under a different config even if that happened to leave size/mtime
    coincidentally similar. Returns `"missing:<name>"` if the artifact doesn't exist (never
    raises -- a missing upstream artifact should show up as an honest cache miss, not a
    crash)."""
    for d in _resolve_search_dirs(directory):
        data_path, _ = artifact_paths(name, d)
        if data_path.exists():
            meta = load_artifact_metadata(name, d) or {}
            return (
                f"{file_fingerprint(data_path)}"
                f"|config_hash={meta.get('config_hash')}"
                f"|row_count={meta.get('row_count')}"
            )
    return f"missing:{name}"


def artifact_is_valid(name, directory, config=None, input_fingerprints=None):
    """True if `<name>.parquet` exists AND (when `config`/`input_fingerprints` are given)
    its stored provenance matches the current config hash and every given input fingerprint --
    i.e. it's safe to skip recomputation and just `load_artifact` instead."""
    data_path, meta_path = artifact_paths(name, directory)
    if not data_path.exists() or not meta_path.exists():
        return False
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if config is not None and meta.get("config_hash") != stable_config_hash(config):
        return False
    if input_fingerprints:
        stored = meta.get("input_fingerprints", {})
        for key, fp in input_fingerprints.items():
            if stored.get(key) != fp:
                return False
    return True
