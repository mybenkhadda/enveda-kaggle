"""Representative-SMILES ROUND-TRIP validation of the finalized universe:

    representative_smiles -> casmi.chemistry.connectivity.competition_connectivity_key (the EXACT project canonicalizer,
                             training tautomer caps) -> recomputed key  ==?  stored connectivity_key

A stored representative that does not round-trip would be submitted as a SMILES whose key is not the candidate's key
(the candidate could never score). Stratified by provenance: TRAIN only / COCONUT only / TRAIN+COCONUT / other.

Resumable per bucket (`parts/bucket=<b>.parquet` + `_done/bucket=<b>.json`, keyed by the universe build id), optional
deterministic per-bucket sample (stable hash of the connectivity key -- never random), optional worker pool.
Nothing here is executed at import; the user runs it from notebook 13.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROUNDTRIP_VERSION = "casmi-roundtrip-1"
FAILURE_COLUMNS = ["candidate_id", "stored_connectivity_key", "representative_smiles", "recomputed_connectivity_key", "candidate_sources",
                   "source_ids", "stratum", "error"]


def stratum_of(sources):
    s = set(map(str, sources)) if isinstance(sources, (list, tuple, set, np.ndarray)) else set()
    if s == {"TRAIN"}:
        return "TRAIN only"
    if s == {"COCONUT"}:
        return "COCONUT only"
    if s == {"TRAIN", "COCONUT"}:
        return "TRAIN+COCONUT"
    return "+".join(sorted(s)) or "none"


def project_keys(smiles_list):
    """Module-level (picklable) worker: competition keys with the TRAINING tautomer caps (the canonicalizer that built
    every stored key). Returns a list of (key or None, error or None)."""
    from casmi.candidates.standardize import training_tautomer_caps
    from casmi.chemistry.connectivity import competition_connectivity_key_detailed, get_tautomer_enumerator
    tmax, ttran = training_tautomer_caps()
    enum = get_tautomer_enumerator(tmax, ttran)
    out = []
    for s in smiles_list:
        d = competition_connectivity_key_detailed(s, enumerator=enum, compute_hit_cap=False)
        out.append((d["conn_key"], d["error"]))
    return out


def roundtrip_frame(rows, key_results):
    """`rows`: candidate_id, connectivity_key, representative_smiles, candidate_sources[, source_ids];
    `key_results`: aligned list of (recomputed key, error). Adds recomputed_connectivity_key, roundtrip_ok, stratum."""
    r = rows.reset_index(drop=True).copy()
    r["recomputed_connectivity_key"] = [k for k, _ in key_results]
    r["error"] = [e for _, e in key_results]
    r["roundtrip_ok"] = r["recomputed_connectivity_key"].astype(object).eq(r["connectivity_key"].astype(str)).to_numpy()
    r["stratum"] = r["candidate_sources"].map(stratum_of)
    return r


def _sample(df, max_rows, seed):
    if max_rows is None or len(df) <= max_rows:
        return df
    from casmi.validation.regimes import stable_unit_hash
    u = stable_unit_hash(df["connectivity_key"].astype(str).tolist(), f"{seed}|roundtrip")
    return df.iloc[np.sort(np.argsort(u, kind="stable")[:max_rows])]


def run_universe_roundtrip(universe_root, out_dir, n_jobs=1, max_per_bucket=None, seed=20261001, batch=2000, log=print):
    """Round-trip every (or a deterministic sample of every) bucket. Returns (failures DataFrame, summary dict)."""
    import pyarrow.parquet as pq
    root, out = Path(universe_root), Path(out_dir)
    man = json.loads((root / "universe_manifest.json").read_text(encoding="utf-8"))
    off = json.loads((root / "bucket_offsets.json").read_text(encoding="utf-8"))["offsets"]
    tag = {"roundtrip_version": ROUNDTRIP_VERSION, "universe_build_id": man.get("build_id"), "n_candidates": man.get("n_candidates"),
           "max_per_bucket": max_per_bucket, "seed": seed}
    pool = None
    if n_jobs and int(n_jobs) > 1:
        from casmi.workspace.parallel import make_pool
        pool = make_pool(int(n_jobs))
    counts = []
    try:
        for b, start in sorted(off.items(), key=lambda kv: kv[1]):
            marker = out / "_done" / f"bucket={b}.json"
            part = out / "parts" / f"bucket={b}.parquet"
            if marker.is_file() and part.is_file():
                rec = json.loads(marker.read_text(encoding="utf-8"))
                if rec.get("tag") == tag:
                    counts.append(rec["counts"])
                    continue
            cols = ["connectivity_key", "representative_smiles", "candidate_sources", "source_ids"]
            t = pq.read_table(root / "buckets" / f"bucket={b}.parquet", columns=cols)
            df = pd.DataFrame({c: t.column(c).to_pylist() for c in cols})
            df.insert(0, "candidate_id", np.arange(int(start), int(start) + len(df), dtype=np.int64))
            df = _sample(df, max_per_bucket, seed)
            smiles = df["representative_smiles"].tolist()
            batches = [smiles[i:i + batch] for i in range(0, len(smiles), batch)]
            res = []
            for r in (pool.imap(project_keys, batches) if pool else map(project_keys, batches)):
                res.extend(r)
            fr = roundtrip_frame(df, res)
            fails = fr[~fr["roundtrip_ok"]].rename(columns={"connectivity_key": "stored_connectivity_key"})[FAILURE_COLUMNS]
            part.parent.mkdir(parents=True, exist_ok=True)
            fails.assign(candidate_sources=fails["candidate_sources"].map(lambda s: ";".join(map(str, s or []))),
                         source_ids=fails["source_ids"].map(lambda s: ";".join(map(str, s or [])))).to_parquet(part, index=False)
            c = fr.groupby("stratum")["roundtrip_ok"].agg(["size", "sum"]).reset_index().rename(columns={"size": "n", "sum": "n_ok"})
            c = c.assign(bucket=b).to_dict(orient="records")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"tag": tag, "counts": c}, default=int), encoding="utf-8")   # after the part file
            counts.append(c)
            if log:
                log(f"[roundtrip] bucket {b}: {len(fr):,} checked | {int((~fr['roundtrip_ok']).sum()):,} failures")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return load_roundtrip(out, tag, counts)


def load_roundtrip(out_dir, tag=None, counts=None):
    out = Path(out_dir)
    parts = [pd.read_parquet(f) for f in sorted((out / "parts").glob("bucket=*.parquet"))]
    fails = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FAILURE_COLUMNS)
    c = pd.DataFrame([r for cc in (counts or []) for r in cc]) if counts else pd.DataFrame(columns=["stratum", "n", "n_ok"])
    by = c.groupby("stratum")[["n", "n_ok"]].sum() if len(c) else pd.DataFrame(columns=["n", "n_ok"])
    n, ok = int(by["n"].sum()) if len(by) else 0, int(by["n_ok"].sum()) if len(by) else 0
    summary = {"roundtrip_version": ROUNDTRIP_VERSION, "tag": tag, "n_candidates": n, "n_roundtrip_ok": ok, "n_roundtrip_fail": n - ok,
               "failure_rate": (n - ok) / n if n else None,
               "by_stratum": {s: {"n": int(r["n"]), "n_ok": int(r["n_ok"]), "n_fail": int(r["n"] - r["n_ok"]),
                                  "failure_rate": float((r["n"] - r["n_ok"]) / r["n"]) if r["n"] else None} for s, r in by.iterrows()}}
    return fails, summary
