"""Fingerprinting for the v4a.1 resumable QCR pipeline: any change to the peak store, reference
index, query-id set, candidate pool, similarity config, protocol config, or code version must
invalidate every downstream shard (see RUN_GUIDE.md's fingerprint-invalidation contract, and
`tests/test_fingerprint.py` which exercises each input independently).
"""
import hashlib
import subprocess
from pathlib import Path

from casmi.io.files import file_fingerprint
from casmi.utils.hashing import stable_dict_hash


def _default_repo_root():
    from casmi.paths import get_project_paths
    return get_project_paths().root


def _hash_source_tree(src_dir):
    """Deterministic hash of every `*.py` file's content under `src_dir`, independent of
    filesystem iteration order (files are sorted by relative path first)."""
    src_dir = Path(src_dir)
    hasher = hashlib.sha256()
    for path in sorted(src_dir.rglob("*.py")):
        hasher.update(str(path.relative_to(src_dir)).replace("\\", "/").encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def code_version(repo_root=None):
    """The last commit touching `src/casmi`, if this checkout is inside a git repository;
    otherwise (or on any git failure) a deterministic content hash of every `src/casmi/**/*.py`
    file. Deliberately does NOT run `git status --porcelain` (this checkout's git root may be
    far above the project directory and scanning it can be slow/irrelevant) -- so this does
    NOT detect uncommitted local edits to tracked files; the source-hash fallback is what
    actually catches those (any local edit changes the file content hash)."""
    repo_root = Path(repo_root) if repo_root is not None else _default_repo_root()
    src_dir = repo_root / "src" / "casmi"
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "src/casmi"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        commit = out.stdout.strip()
        if out.returncode == 0 and commit:
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return f"nogit-{_hash_source_tree(src_dir)}"


def artifact_fingerprint(peak_store_path, reference_index, query_ids, candidate_pool_fingerprint,
                          similarity_config, protocol_config, repo_root=None):
    """One stable hash covering every input a QCR shard depends on:

    - `peak_store_path`: the raw parquet peaks are read from -- fingerprinted via size+mtime
      (`casmi.io.files.file_fingerprint`), never content-hashed (multi-GB file).
    - `reference_index`: `{connectivity_key: [spectrum_id, ...]}` -- hashed by content.
    - `query_ids`: iterable of query ids covered by this shard/run.
    - `candidate_pool_fingerprint`: caller-supplied fingerprint of the candidate pool table.
    - `similarity_config` / `protocol_config`: the config dicts governing evidence computation
      and protocol eligibility.
    - `code_version(repo_root)`.

    Returns a single hex string; changing ANY input changes it. Use `fingerprint_diff` to see
    exactly which component changed when two fingerprints differ."""
    return stable_dict_hash(_fingerprint_payload(
        peak_store_path, reference_index, query_ids, candidate_pool_fingerprint,
        similarity_config, protocol_config, repo_root,
    ), length=16)


def _fingerprint_payload(peak_store_path, reference_index, query_ids, candidate_pool_fingerprint,
                          similarity_config, protocol_config, repo_root=None):
    return {
        "peak_store": file_fingerprint(peak_store_path),
        "reference_index_hash": stable_dict_hash({str(k): sorted(v) for k, v in reference_index.items()}),
        "query_ids": sorted(str(q) for q in query_ids),
        "candidate_pool_fingerprint": candidate_pool_fingerprint,
        "similarity_config": similarity_config,
        "protocol_config": protocol_config,
        "code_version": code_version(repo_root),
    }


def fingerprint_components(peak_store_path, reference_index, query_ids, candidate_pool_fingerprint,
                            similarity_config, protocol_config, repo_root=None):
    """Per-component hashes (not the single combined fingerprint) -- lets a caller/test see
    exactly which input changed between two runs, and lets `artifact_fingerprint` and this stay
    trivially consistent (both built from the same `_fingerprint_payload`)."""
    payload = _fingerprint_payload(peak_store_path, reference_index, query_ids, candidate_pool_fingerprint,
                                    similarity_config, protocol_config, repo_root)
    return {k: stable_dict_hash(v) if isinstance(v, (dict, list)) else v for k, v in payload.items()}
