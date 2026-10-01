"""Stage Drive files onto the Colab local SSD (/content/enveda_work) before heavy reads.

Drive (FUSE) reads are slow and repeated epochs over Drive files are slower still. Staging copies a file
once and re-uses the local copy while it is unchanged:

    unchanged  <=>  same relative name, same size, same modification time (shutil.copy2 preserves mtime)

No hashing (multi-GB files). Guards against filling the Colab disk: a copy that would leave less than
`min_free_gb` free is refused, and files above `max_file_gb` are NOT duplicated -- the Drive path is
returned instead (logged), so callers always get a readable path.
"""
import os
import shutil
from pathlib import Path

DEFAULT_SCRATCH = Path("/content/enveda_work")


def _same(src, dst, mtime_tol_s=2.0):
    if not dst.is_file():
        return False
    s, d = src.stat(), dst.stat()
    return s.st_size == d.st_size and abs(s.st_mtime - d.st_mtime) <= mtime_tol_s


def _target(src, scratch_root, rel, drive_root):
    if rel is not None:
        return Path(scratch_root) / rel
    if drive_root is not None:
        try:
            return Path(scratch_root) / Path(src).resolve().relative_to(Path(drive_root).resolve())
        except ValueError:
            pass
    return Path(scratch_root) / "staged" / Path(src).name


def stage_if_changed(src, dst, min_free_gb=5.0, max_file_gb=None, log=print):
    """Copy `src` -> `dst` unless `dst` is already identical (size + mtime). Returns `(path_to_use, action)`
    with action in {'reused', 'copied', 'not_staged_too_large', 'not_staged_low_disk'}."""
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    if _same(src, dst):
        return dst, "reused"
    size = src.stat().st_size
    if max_file_gb is not None and size > max_file_gb * 1024 ** 3:
        log(f"[staging] {src.name}: {size / 1024 ** 3:.2f} GB > max_file_gb={max_file_gb} -> reading from Drive")
        return src, "not_staged_too_large"
    dst.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst.parent).free
    if free - size < min_free_gb * 1024 ** 3:
        log(f"[staging] {src.name}: only {free / 1024 ** 3:.1f} GB free on scratch -> reading from Drive")
        return src, "not_staged_low_disk"
    tmp = dst.with_name(dst.name + ".staging")
    shutil.copy2(src, tmp)                     # preserves mtime -> later runs see 'unchanged'
    os.replace(tmp, dst)
    log(f"[staging] copied {src.name} ({size / 1024 ** 2:.1f} MB) -> {dst}")
    return dst, "copied"


def stage_file(src, scratch_root=DEFAULT_SCRATCH, rel=None, drive_root=None, **kw):
    """Stage one Drive file under `scratch_root` (mirroring its path below `drive_root` when given). Returns the
    path to read from."""
    path, _ = stage_if_changed(src, _target(src, scratch_root, rel, drive_root), **kw)
    return path


def stage_directory(src_dir, scratch_root=DEFAULT_SCRATCH, rel=None, drive_root=None, patterns=("*",), exclude_dirs=("__pycache__",),
                    **kw):
    """Stage every file of `src_dir` matching `patterns` (recursive). Returns `(local_dir, summary)`; files that
    were not staged (too large / low disk) are listed in `summary['not_staged']` with their Drive path, so
    callers can fall back per file."""
    src_dir = Path(src_dir)
    dst_dir = _target(src_dir, scratch_root, rel, drive_root)
    summary = {"reused": 0, "copied": 0, "not_staged": []}
    seen = set()
    for pat in patterns:
        for f in sorted(src_dir.rglob(pat)):
            if not f.is_file() or f in seen or any(p in exclude_dirs for p in f.relative_to(src_dir).parts[:-1]):
                continue
            seen.add(f)
            path, action = stage_if_changed(f, dst_dir / f.relative_to(src_dir), **kw)
            if action in ("reused", "copied"):
                summary[action] += 1
            else:
                summary["not_staged"].append(str(path))
    return dst_dir, summary
