"""v5.3 protocol artifacts from EXISTING QCR (filter + aggregate only; the builder, similarities, peak
preprocessing, reference index, compat ranking and walk are untouched).

For one dataset (a v5 manifest built by `scripts/v5_build_scale_features.py`, or HOST):
    features  outputs/v5/features/protocols/<name>_<protocol>.parquet  (+ .meta.json sidecar)
    counts    outputs/v5/protocols/<name>_<protocol>_counts.parquet    (+ .meta.json sidecar)
    manifest  outputs/v5/manifests/<name>_<SUFFIX>.parquet            (training sets only; originals never touched)

Every build asserts: the walk is complete for the protocol (top-5 inside the walked prefix), zero
selected references in an excluded tier (incl. T1), and the top3_mean missingness contract.
A valid sidecar (same protocol hash, evidence fingerprint and source-manifest sha) makes a rebuild
a no-op, so the step is cheap to re-run.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from casmi.qcr.aggregate import assert_no_top3_missingness_leak
from casmi.qcr.protocols import (PROTOCOL_DEFS, assert_walk_complete, derive_protocol_manifest, features_from_shards, forbidden_tier_counts,
                                 protocol_counts_from_shards, protocol_semantic_hash)
from casmi.ranking.manifests import load_manifest, save_manifest

UNITS = {"TL_EVAL": ["TL_EVAL"], "TL_1K": ["TL_1K"], "TL_3K": ["TL_1K", "TL_3K"], "TL_10K": ["TL_1K", "TL_3K", "TL_10K"],
         "RND_1K": ["RND_1K"], "MOL_DEV": ["MOL_DEV"]}
SUFFIX = {"test_simulated_strict": "TESTSIM_STRICT", "test_simulated_relaxed": "TESTSIM_RELAXED", "mirror_aware": "MODE_A", "standard": "STANDARD_SENS"}
META_COLS = ["query_id", "connectivity_key", "fold", "source", "instrument", "adduct"]
READY, MISSING = "READY", "MISSING"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_sidecar(path, **extra):
    path = Path(path)
    rec = {"file": path.name, "sha256": _sha(path), "created_at": datetime.now(timezone.utc).isoformat(), **extra}
    path.with_suffix(".meta.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def sidecar_valid(path, **expected):
    path = Path(path)
    side = path.with_suffix(".meta.json")
    if not (path.exists() and side.exists()):
        return False
    rec = json.loads(side.read_text(encoding="utf-8"))
    return rec.get("sha256") == _sha(path) and all(rec.get(k) == v for k, v in expected.items())


def feature_path(D, name, protocol):
    return D["features"] / "protocols" / f"{name}_{protocol}.parquet"


def counts_path(D, name, protocol):
    return D["root"] / "protocols" / f"{name}_{protocol}_counts.parquet"


def derived_manifest_name(name, protocol):
    return f"{name}_{SUFFIX[protocol]}"


def qcr_status(D, name):
    units = UNITS[name]
    ok = all((D["qcr"] / u / "qcr" / "shards").exists() and any((D["qcr"] / u / "qcr" / "shards").glob("part-*.parquet"))
             and (D["features"] / "units" / f"{u}_pool.parquet").exists() for u in units)
    return READY if ok and (D["manifests"] / f"{name}.parquet").exists() else MISSING


def artifact_status(D, name, protocol, evidence_fingerprint):
    exp = {"protocol": protocol, "protocol_hash": protocol_semantic_hash(protocol), "evidence_fingerprint": evidence_fingerprint}
    return READY if sidecar_valid(feature_path(D, name, protocol), **exp) else MISSING


def _finish(D, name, protocol, feats, counts, audit, meta, evidence_fingerprint, source_sha):
    pdef = PROTOCOL_DEFS[protocol]
    viol = forbidden_tier_counts(audit, pdef)
    assert viol["t1_violation_count"] == 0 and viol["forbidden_tier_count"] == 0, f"{name}: forbidden tier selected {viol}"
    assert_no_top3_missingness_leak(feats)
    feats = feats.merge(meta, on="query_id", how="left", validate="many_to_one")
    fp, cp = feature_path(D, name, protocol), counts_path(D, name, protocol)
    fp.parent.mkdir(parents=True, exist_ok=True)
    cp.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(fp, index=False)
    counts.to_parquet(cp, index=False)
    common = {"protocol": protocol, "protocol_hash": protocol_semantic_hash(protocol), "evidence_fingerprint": evidence_fingerprint,
              "source_manifest_sha256": source_sha, "tier_audit": viol}
    write_sidecar(fp, n_rows=int(len(feats)), n_queries=int(feats["query_id"].nunique()), **common)
    write_sidecar(cp, n_rows=int(len(counts)), **common)
    return feats, counts


def build_protocol_artifacts(D, name, protocol, evidence_fingerprint, derive_training_manifest=True, force=False):
    """Features + counts (+ derived training manifest) for a v5 manifest `name` from its QCR units."""
    if qcr_status(D, name) != READY:
        raise FileNotFoundError(f"{name}: QCR / pool / manifest missing -- run scripts/v5_build_scale_features.py --manifest {name}")
    m0, m0_meta = load_manifest(name, D["manifests"])
    out = {"name": name, "protocol": protocol}
    exp = {"protocol": protocol, "protocol_hash": protocol_semantic_hash(protocol), "evidence_fingerprint": evidence_fingerprint,
           "source_manifest_sha256": m0_meta["manifest_sha256"]}
    if not force and sidecar_valid(feature_path(D, name, protocol), **exp) and sidecar_valid(counts_path(D, name, protocol), **exp):
        feats, counts = pd.read_parquet(feature_path(D, name, protocol)), pd.read_parquet(counts_path(D, name, protocol))
        out["reused"] = True
    else:
        units = UNITS[name]
        pool = pd.concat([pd.read_parquet(D["features"] / "units" / f"{u}_pool.parquet") for u in units], ignore_index=True)
        pool = pool[pool["query_id"].isin(set(m0["query_id"]))]
        qp = [p for u in units for p in sorted((D["qcr"] / u / "qcr" / "shards").glob("part-*.parquet"))]
        pp = [p for u in units for p in sorted((D["qcr"] / u / "qcr_pairs" / "shards").glob("part-*.parquet"))]
        counts = protocol_counts_from_shards(qp, pp, pool, [protocol])[protocol]
        if protocol != "mirror_aware":
            assert_walk_complete(counts)
        feats, audit = features_from_shards(qp, pool, PROTOCOL_DEFS[protocol])
        meta = m0[META_COLS].rename(columns={"connectivity_key": "true_connectivity_key"})
        feats, counts = _finish(D, name, protocol, feats, counts, audit, meta, evidence_fingerprint, m0_meta["manifest_sha256"])
        out["reused"] = False
    if derive_training_manifest:
        dname = derived_manifest_name(name, protocol)
        dm, aud = derive_protocol_manifest(m0, counts, SUFFIX[protocol])
        aud.update(unique_connectivities=int(dm["connectivity_key"].nunique()), protocol=protocol, protocol_hash=protocol_semantic_hash(protocol),
                   derived_from=name, derived_from_sha256=m0_meta["manifest_sha256"])
        path = D["manifests"] / f"{dname}.parquet"
        if path.exists():
            old, _ = load_manifest(dname, D["manifests"])
            assert set(old["query_id"]) == set(dm["query_id"]), f"persisted {dname} differs from its re-derivation"
        save_manifest(dm, dname, D["manifests"], extra=aud)
        out.update(derived_manifest=dname, audit=aud)
    assert load_manifest(name, D["manifests"])[1]["manifest_sha256"] == m0_meta["manifest_sha256"], f"original manifest {name} changed"
    out.update(n_feature_rows=int(len(feats)), n_queries=int(m0["query_id"].nunique()))
    return out


def build_host_protocol_artifacts(D, evidence_artifacts, host_q, protocol, evidence_fingerprint, force=False):
    """HOST features/counts under `protocol` from the settled HOST QCR (single parquet)."""
    from casmi.qcr.protocols import aggregate_protocol, select_protocol_refs
    exp = {"protocol": protocol, "protocol_hash": protocol_semantic_hash(protocol), "evidence_fingerprint": evidence_fingerprint,
           "source_manifest_sha256": "HOST"}
    if not force and sidecar_valid(feature_path(D, "HOST", protocol), **exp):
        return {"name": "HOST", "reused": True}
    pool = pd.read_parquet(evidence_artifacts["host_features_mirror_aware"], columns=["query_id", "candidate_connectivity_key", "abs_mass_error_ppm", "is_true_candidate"])
    counts = protocol_counts_from_shards([evidence_artifacts["host_qcr"]], [evidence_artifacts["host_qcr_pairs"]], pool, [protocol])[protocol]
    if protocol != "mirror_aware":
        assert_walk_complete(counts)
    qcr = pd.read_parquet(evidence_artifacts["host_qcr"])
    feats = aggregate_protocol(qcr, pool, PROTOCOL_DEFS[protocol])
    audit = select_protocol_refs(qcr, PROTOCOL_DEFS[protocol])[["query_id", "candidate_key", "is_true", "tier", "ref_source", "query_source"]]
    meta = host_q.rename(columns={"ingest_lib": "source", "instrument_type": "instrument", "true_connectivity_key": "connectivity_key"})[META_COLS] \
        .rename(columns={"connectivity_key": "true_connectivity_key"})
    _finish(D, "HOST", protocol, feats, counts, audit, meta, evidence_fingerprint, "HOST")
    return {"name": "HOST", "reused": False}
