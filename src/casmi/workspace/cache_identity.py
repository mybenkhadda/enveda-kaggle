"""Identity-namespaced resumable caches.

Resumable shard caches (`part-00000.parquet` + `.done`) are resumed by CHUNK INDEX. Reusing a directory whose chunk
k was computed for other queries, other hidden truths or another config silently mixes stale rows -- and for the
analog neighbors, stale HIDING masks leak reference spectra of truths that are now hidden (C2 leakage).

Every cache namespace therefore lives in a directory NAMED by a fingerprint of everything its content depends on:

    <base>/ns-<identity_hash>/identity.json + part-*.parquet + part-*.done

A different query set / regime table / hidden set / reference library / universe / config gives a different
directory, so stale content is never resumed; `identity.json` is re-checked on every open (a mismatch or a legacy
directory raises `StaleCacheError`, never silently reused). Loaders read ONLY the namespace directories returned by
the run, never a glob over the base directory.

Fingerprints are deterministic lightweight metadata hashes (sha256 of sorted values / canonical JSON) -- this is
cache identity for LOCAL derived artifacts, not verification of the frozen deployment bundle (which is identified by
its config labels only; the no-hash policy for the bundle is unchanged).
"""
import hashlib
import json
from pathlib import Path

CACHE_SCHEMA_VERSION = "casmi-v2-cache-2"


class StaleCacheError(RuntimeError):
    """A cache directory does not belong to the requested identity (never resumed)."""


def fingerprint_values(values):
    """Order-free fingerprint of a collection of scalars (16 hex)."""
    blob = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def fingerprint_pairs(pairs):
    """Order-free fingerprint of (key, value) pairs, e.g. query_id -> regime."""
    return fingerprint_values(f"{k}\t{v}" for k, v in pairs)


def fingerprint_json(obj):
    """Fingerprint of a JSON-serialisable object (canonical: sorted keys)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def identity_hash(identity):
    return fingerprint_json({"cache_schema": CACHE_SCHEMA_VERSION, **identity})


def open_namespace(base_dir, identity, kind="cache"):
    """Directory for `identity` below `base_dir` (created on first use, with identity.json). Raises StaleCacheError
    if the directory exists with a different / missing identity.json (hash collision or legacy content)."""
    ident = {"cache_schema": CACHE_SCHEMA_VERSION, "kind": kind, **identity}
    d = Path(base_dir) / f"ns-{identity_hash(identity)}"
    f = d / "identity.json"
    if d.exists():
        if not f.is_file():
            if any(d.iterdir()):
                raise StaleCacheError(f"{d} has content but no identity.json -- refusing to resume it; delete the directory")
        else:
            old = json.loads(f.read_text(encoding="utf-8"))
            if old != json.loads(json.dumps(ident, default=str)):
                diff = sorted(k for k in set(old) | set(ident) if old.get(k) != ident.get(k))
                raise StaleCacheError(f"{d}: identity mismatch on {diff} -- refusing to resume a cache built for other inputs")
            return d
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "identity.json.tmp"
    tmp.write_text(json.dumps(ident, indent=1, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(f)
    return d


def read_identity(namespace_dir):
    f = Path(namespace_dir) / "identity.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def reference_library_identity(bundle_identity, ref_meta_path=None):
    """Identity of the frozen reference library: its config LABELS (+ ref_meta file size, a cheap guard against a
    partially copied bundle). No file content is hashed."""
    out = dict(bundle_identity or {})
    if ref_meta_path is not None and Path(ref_meta_path).is_file():
        out["ref_meta_bytes"] = Path(ref_meta_path).stat().st_size
    return out
