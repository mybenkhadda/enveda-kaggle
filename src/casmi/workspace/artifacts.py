"""`metadata.json` for every v2 artifact folder (no content hashing -- the project's existing
fingerprints are size|mtime based; expensive hashes are not added here).

Fields: created_by_notebook, created_at, git_commit, config (or config subset), seed, input_paths,
schema_version, model_name, feature_list, plus free `extra`."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit(repo_root):
    """Short HEAD commit of `repo_root`, or None (no git / uploaded repo without .git)."""
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_metadata(folder, created_by_notebook, config=None, seed=None, input_paths=None, schema_version=None,
                   model_name=None, feature_list=None, repo_root=None, extra=None, filename="metadata.json"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"created_by_notebook": created_by_notebook, "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(repo_root) if repo_root else None, "config": config, "seed": seed,
            "input_paths": {k: str(v) for k, v in (input_paths or {}).items()}, "schema_version": schema_version,
            "model_name": model_name, "feature_list": list(feature_list) if feature_list is not None else None,
            **(extra or {})}
    (folder / filename).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def read_metadata(folder, filename="metadata.json"):
    p = Path(folder) / filename
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
