"""Local-SSD staging of large Drive inputs and safe persistence of completed outputs back to Drive.

Google Drive (FUSE) is slow for large sequential reads and very slow for many small writes. Stage A therefore:

  * copies a large external source (e.g. COCONUT csv) ONCE to the local scratch disk (`stage_external_file`); a
    local copy is reused when it exists, has the same file NAME and the same SIZE as the Drive source. No hashing
    (the multi-GB no-SHA policy); the Drive original is never modified;
  * writes chunk outputs to local scratch first and then persists them to Drive (`persist_files`): copy to a
    temporary name, atomic rename, size check. A caller writes its persistent done-marker ONLY after
    `persist_files` returned -- local scratch is never the sole copy of a completed chunk.
"""
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path


class StagingError(RuntimeError):
    pass


@dataclass
class StagedFile:
    source: str
    source_path: str
    local_path: str
    size_bytes: int
    action: str                 # "reused" | "copied" | "not_staged" (read from Drive directly)
    seconds: float
    reason: str = ""

    def as_dict(self):
        return asdict(self)

    def summary(self):
        mb = self.size_bytes / 1024 ** 2
        return (f"[staging] {self.source}: {self.action} | {mb:,.1f} MB | source {self.source_path} | local {self.local_path}"
                f" | {self.seconds:.1f}s" + (f" | {self.reason}" if self.reason else ""))


def _free_bytes(path):
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def stage_external_file(source_path, scratch_root, expected_source="EXTERNAL", subdir="external", min_free_gb=2.0, log=print):
    """Copy `source_path` to `<scratch_root>/<subdir>/<file name>` unless an identical-size local copy exists.
    Returns a `StagedFile`; `local_path` is the path to READ from (the Drive path itself when staging is impossible,
    e.g. not enough free scratch disk -- reported, never silent)."""
    src = Path(source_path)
    if not src.is_file():
        raise StagingError(f"{expected_source}: source file not found: {src}")
    size = src.stat().st_size
    dst = Path(scratch_root) / subdir / src.name
    t0 = time.time()
    if dst.is_file() and dst.stat().st_size == size:
        out = StagedFile(expected_source, str(src), str(dst), size, "reused", time.time() - t0)
    elif _free_bytes(dst.parent) - size < min_free_gb * 1024 ** 3:
        out = StagedFile(expected_source, str(src), str(src), size, "not_staged", 0.0,
                         reason=f"less than {min_free_gb} GB would remain free on scratch -- reading from Drive")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".staging")
        shutil.copyfile(src, tmp)                       # content only; the Drive original is never touched
        if tmp.stat().st_size != size:
            tmp.unlink(missing_ok=True)
            raise StagingError(f"{expected_source}: staged copy size {tmp.stat().st_size if tmp.exists() else 0} != source size {size}")
        os.replace(tmp, dst)
        out = StagedFile(expected_source, str(src), str(dst), size, "copied", time.time() - t0)
    if log:
        log(out.summary())
    return out


def persist_files(pairs, log=None):
    """`pairs`: iterable of (local_path, persistent_path). Each file is copied to `<dest>.part`, atomically renamed
    and its size verified. Returns {persistent_path: size}. Raises on any failure, so a caller that writes its
    done-marker after this call can never mark a chunk complete whose outputs did not reach persistence."""
    out = {}
    for local, dest in pairs:
        local, dest = Path(local), Path(dest)
        size = local.stat().st_size
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local.resolve() == dest.resolve():
            out[str(dest)] = size
            continue
        part = dest.with_name(dest.name + ".part")
        shutil.copyfile(local, part)
        os.replace(part, dest)
        got = dest.stat().st_size
        if got != size:
            raise StagingError(f"persisted size mismatch for {dest}: {got} != {size}")
        out[str(dest)] = size
        if log:
            log(f"[persist] {dest.name} ({size / 1024 ** 2:.1f} MB)")
    return out


def files_intact(sizes):
    """True when every {path: size} entry exists with exactly that size (done-marker validation; no hashing)."""
    for p, s in (sizes or {}).items():
        q = Path(p)
        if not q.is_file() or q.stat().st_size != int(s):
            return False
    return True
