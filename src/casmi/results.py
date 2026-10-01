"""Approved API for writing gate-evidence out of a v4a.1 stage notebook: every value a gate
condition reads must have passed through `record()`, never been assigned as a bare literal in
notebook source (see `tests/test_results_lint.py`) -- this is what makes gate evidence provably
runtime-derived rather than hand-typed to make a gate pass. Each stage owns its own
`results_<stage>.json`; `load_all_results` merges them (unprefixed -- the key namespace itself,
e.g. `c9.host.*` vs `c9.dev.*`, disambiguates stage of origin) into the flat dict
`casmi.gates.evaluate` reads.
"""
import json
from datetime import datetime, timezone
from pathlib import Path


def _results_path(stage, output_dir):
    return Path(output_dir) / f"results_{stage}.json"


def _default_output_dir():
    from casmi.paths import get_project_paths
    return get_project_paths().outputs / "v4a1"


def record(stage, key, value, evidence_path=None, metadata=None, output_dir=None):
    """Merge one runtime-computed `key -> value` into `outputs/v4a1/results_<stage>.json`.
    `key` may be dotted (e.g. `"c9.host.total_mismatches"`) and is stored as a nested path.
    `value` must already be a plain JSON-serializable Python value computed at runtime -- this
    function does not itself compute or interpret anything, it only persists. `evidence_path`:
    optional path to a persisted artifact backing this value, kept alongside it for
    traceability. `metadata`: optional small dict of non-result context. Returns the full
    (unflattened) stage results dict after the write."""
    output_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _results_path(stage, output_dir)

    results = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            results = json.load(f)

    node = results
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    entry = {"value": value, "recorded_at": datetime.now(timezone.utc).isoformat()}
    if evidence_path is not None:
        entry["evidence_path"] = str(evidence_path)
    if metadata:
        entry["metadata"] = metadata
    node[parts[-1]] = entry

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def _flatten(node, prefix=""):
    flat = {}
    for k, v in node.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "value" in v and "recorded_at" in v:
            flat[full_key] = v["value"]
        elif isinstance(v, dict):
            flat.update(_flatten(v, full_key))
        else:
            flat[full_key] = v
    return flat


def load_results(stage, output_dir=None):
    """Load one stage's `results_<stage>.json` as a flat `{"a.b.c": value}` dict, unwrapping
    the `{"value": ..., "recorded_at": ...}` envelope `record()` writes. Returns `{}` (never
    raises) if the stage hasn't been run yet -- a gate reading a key from an unrun stage must
    see it as missing (NOT_PROVEN), not crash."""
    output_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
    path = _results_path(stage, output_dir)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        nested = json.load(f)
    return _flatten(nested)


def load_all_results(stages, output_dir=None):
    """Merge several stages' flattened results into one flat `{"a.b.c": value}` dict -- what
    s4's gate evaluation reads. Keys are NOT stage-prefixed: the key namespace (e.g.
    `c9.host.*`/`c9.dev.*`, `dev.*`) is designed so stages never collide. `stages`: iterable of
    stage names (e.g. `("s1", "s2", "s3")`)."""
    merged = {}
    for stage in stages:
        merged.update(load_results(stage, output_dir=output_dir))
    return merged
