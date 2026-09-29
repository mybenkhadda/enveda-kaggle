"""Immutable result storage: every run gets a NEW folder <results_root>/<UTC timestamp>_<CONFIG_HASH[:12]>/; an
existing folder is never reused or overwritten, and every copy is sha256-verified."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

RESULT_FILES = ("submission.csv", "run_report.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def save_results_immutable(output_dir, results_root, config_hash, provenance=None, extra_files=(), stamp=None):
    """Returns the new folder. Raises FileExistsError rather than overwrite anything."""
    output_dir, results_root = Path(output_dir), Path(results_root)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = results_root / f"{stamp}_{str(config_hash)[:12]}"
    dest.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in (*RESULT_FILES, *extra_files):
        src = output_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing run output {src}")
        shutil.copy2(src, dest / name)
        if _sha256(dest / name) != _sha256(src):
            raise IOError(f"{name}: copy differs from the source")
        copied[name] = _sha256(dest / name)
    (dest / "provenance.json").write_text(json.dumps({"saved_at_utc": stamp, "CONFIG_HASH": config_hash, "files_sha256": copied,
                                                      **(provenance or {})}, indent=2, default=str), encoding="utf-8")
    return dest
