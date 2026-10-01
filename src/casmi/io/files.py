"""Reusable file utilities: existence checks and lightweight (non-content-hashing)
fingerprints for cache-invalidation, used instead of hashing multi-GB parquet files.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileInfo:
    path: Path
    exists: bool
    size_bytes: int | None
    mtime: float | None


def file_info(path):
    """Cheap, stat()-only info about a file -- never reads file contents."""
    p = Path(path)
    if not p.exists():
        return FileInfo(path=p, exists=False, size_bytes=None, mtime=None)
    st = p.stat()
    return FileInfo(path=p, exists=True, size_bytes=st.st_size, mtime=st.st_mtime)


def file_fingerprint(path):
    """A lightweight fingerprint string for cache-invalidation: resolved path + size + mtime.
    Deliberately does NOT hash file content -- this project's raw files are multi-GB, and a
    full content hash would dominate every pipeline run's cost for no practical benefit over
    size+mtime (which already changes on any real re-download or re-export)."""
    info = file_info(path)
    if not info.exists:
        return f"missing:{info.path.resolve()}"
    return f"{info.path.resolve()}|{info.size_bytes}|{info.mtime:.6f}"


def validate_required_files(paths_obj, required=("train", "test", "sample_submission")):
    """Check that every required raw file named in `required` (attribute names on a
    `ProjectPaths`) exists. Returns {name: FileInfo}; raises AssertionError listing every
    missing file at once, rather than failing on the first one."""
    results = {}
    missing = []
    for name in required:
        p = getattr(paths_obj, name)
        info = file_info(p)
        results[name] = info
        if not info.exists:
            missing.append(f"{name} -> {p}")
    if missing:
        raise AssertionError("Missing required raw file(s):\n  " + "\n  ".join(missing))
    return results
