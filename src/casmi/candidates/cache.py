"""Cache-first helpers for the candidate-generation pipeline, so notebook 03 never silently
re-runs an expensive step: every call prints `[CACHE HIT]` or `[CACHE MISS]`, unlike
`casmi.pipelines.preprocessing`'s quieter logger-based equivalent (`_maybe_cached`), which this
project's own restart/rerun requirement (candidate generation must be idempotent and cheap to
re-run) calls for something more visible than a log line.
"""
from casmi.io.artifacts import artifact_fingerprint, artifact_is_valid, artifact_paths, load_artifact, save_artifact


def upstream_fingerprints(*names, directory=None):
    """`{name: artifact_fingerprint(name, directory)}` for every upstream artifact a
    candidate-generation step depends on -- the real dependency-graph edges from section 5:
    e.g. `dev_queries` depends on `train_spectrum_metadata`+`connectivity_folds`+
    `molecule_metadata`, so `cached_dataframe("dev_queries", ..., input_fingerprints=
    upstream_fingerprints("train_spectrum_metadata", "connectivity_folds",
    "molecule_metadata"))` invalidates the moment any one of those three changes, instead of
    the empty `{}` that used to make every candidate-generation artifact immortal regardless of
    upstream changes."""
    return {name: artifact_fingerprint(name, directory) for name in names}


def cached_dataframe(name, directory, build_fn, config=None, force=False, input_fingerprints=None, description=""):
    """Load `<name>` from `directory` if it's already valid for this exact config + inputs
    (and `force=False`); otherwise build it with `build_fn()`, save it, and return the fresh
    DataFrame. Always returns `(df, path, was_cached)`."""
    path, _ = artifact_paths(name, directory)
    if not force and artifact_is_valid(name, directory, config=config, input_fingerprints=input_fingerprints):
        print(f"[CACHE HIT] {name} <- {path}")
        return load_artifact(name, directory), path, True

    print(f"[CACHE MISS] {name} -- computing")
    df = build_fn()
    save_artifact(df, name, directory, config=config, description=description, input_fingerprints=input_fingerprints)
    print(f"[CACHE MISS] {name} -- saved {len(df):,} rows -> {path}")
    return df, path, False


def cached_mass_index(directory, build_fn, library_fingerprint, name="mass_index", force=False,
                       mass_col="exact_mass", key_col="connectivity_key"):
    """Same cache-first contract as `cached_dataframe`, for a `casmi.candidates.mass_index.
    MassIndex` (which persists as `.npz` + `.meta.json`, not parquet, so it can't go through
    `cached_dataframe`).

    A saved index is trusted ONLY if its sidecar's `library_fingerprint` matches the given
    `library_fingerprint` (typically `artifact_fingerprint("molecule_metadata")`) AND its
    `mass_index_version` matches `MASS_INDEX_VERSION` -- a stale upstream `molecule_metadata`,
    or an older on-disk format, both show up as an honest `[CACHE MISS]`, never a silently
    wrong index.
    """
    from pathlib import Path

    from casmi.candidates.mass_index import MASS_INDEX_VERSION, MassIndex

    npz_path = Path(directory) / f"{name}.npz"
    meta = MassIndex.read_meta(npz_path) if not force else None
    is_valid = (
        meta is not None and npz_path.exists()
        and meta.get("library_fingerprint") == library_fingerprint
        and meta.get("mass_index_version") == MASS_INDEX_VERSION
    )
    if is_valid:
        print(f"[CACHE HIT] {name} <- {npz_path} (library_fingerprint matches)")
        return MassIndex.load(npz_path), npz_path, True

    reason = "forced" if force else ("no sidecar" if meta is None else "library_fingerprint/version mismatch")
    print(f"[CACHE MISS] {name} -- building ({reason})")
    index = build_fn()
    saved_path = index.save(Path(directory) / name, library_fingerprint=library_fingerprint,
                             mass_col=mass_col, key_col=key_col)
    print(f"[CACHE MISS] {name} -- saved {len(index):,} structures -> {saved_path}")
    return index, saved_path, False
