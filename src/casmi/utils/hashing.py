"""Stable hashing for config/dict provenance -- used to validate artifact caches: an artifact
built under one config hash should not be silently reused after the config changes.
"""
import hashlib
import json
from dataclasses import asdict, is_dataclass


def _to_jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def stable_dict_hash(d, length=12):
    """A short, deterministic hash of a (possibly nested) dict/dataclass -- key order does
    not affect the result, so equivalent configs always hash the same."""
    payload = json.dumps(_to_jsonable(d), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_config_hash(config, length=12):
    """Alias of `stable_dict_hash` for a config object (dataclass or dict) -- named
    separately so pipeline code reads as `stable_config_hash(config)` rather than the more
    generic `stable_dict_hash`."""
    return stable_dict_hash(config, length=length)
