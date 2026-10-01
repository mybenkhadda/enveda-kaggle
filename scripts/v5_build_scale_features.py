"""v5 OFFLINE QCR + mirror_aware feature build for one manifest (spec sections 10-12).

    PYTHONPATH=src PYTHONNOUSERSITE=1 python scripts/v5_build_scale_features.py --manifest TL_1K
    ... --manifest TL_EVAL          (also writes k=1/3/5 stress features)
    ... --manifest TL_3K            (builds only TL_3K \\ TL_1K; reuses TL_1K shards)
    ... --manifest TL_10K           (builds only TL_10K \\ TL_3K)
    ... --manifest RND_1K
    ... --manifest MOL_DEV          (molecule-aggregation development set for 11_02)

The v4b evidence machinery is reused UNCHANGED: `casmi.qcr.builder.build_qcr_resumable` (chunked,
fingerprinted shards, resume-by-skip, per-chunk wall time + RSS in timing.json, one chunk's peaks in
RAM at a time), `compat_sort_key` / exact lazy walk / mirror_aware, the similarity config recorded by
the cache settlement, `casmi.qcr.aggregate.aggregate_deterministic` (top-5, max + top3_mean), and the
training mass-window candidate generator. Queries are processed in sorted query_id order.

Nested manifests are built as BUILD UNITS so no query is ever built twice:
    TL_1K -> [TL_1K]; TL_3K -> [TL_1K, TL_3K (delta)]; TL_10K -> [TL_1K, TL_3K, TL_10K (delta)]
Do not edit src/casmi between builds: `code_version` (a src hash) is part of every shard
fingerprint, so an edit invalidates finished shards (they would be rebuilt, never silently reused).
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from casmi.candidates.generator import CandidateGenerator, dedupe_pool_to_connectivity  # noqa: E402
from casmi.candidates.mass_index import MassIndex  # noqa: E402
from casmi.io.artifacts import load_artifact  # noqa: E402
from casmi.paths import get_project_paths  # noqa: E402
from casmi.qcr.aggregate import aggregate_deterministic, features_without_references  # noqa: E402
from casmi.qcr.builder import build_qcr_resumable  # noqa: E402
from casmi.qcr.context import SpectrumLookups, config_hash, v4b_paths, v5_paths  # noqa: E402
from casmi.qcr.fingerprint import artifact_fingerprint as qcr_fingerprint  # noqa: E402
from casmi.qcr.settlement import load_settlement_or_refuse  # noqa: E402
from casmi.ranking.manifests import ALL_MANIFESTS, load_manifest  # noqa: E402
from casmi.spectra.reference_selection import PROTOCOLS  # noqa: E402

PROTOCOL = "mirror_aware"
UNITS = {"TL_1K": ["TL_1K"], "TL_3K": ["TL_1K", "TL_3K"], "TL_10K": ["TL_1K", "TL_3K", "TL_10K"],
         "RND_1K": ["RND_1K"], "TL_EVAL": ["TL_EVAL"], "MOL_DEV": ["MOL_DEV"]}
PARENT = {"TL_3K": "TL_1K", "TL_10K": "TL_3K"}
K_STRESS_MANIFESTS = ("TL_EVAL",)
META_COLS = ["query_id", "connectivity_key", "fold", "source", "instrument", "adduct", "neutral_mass"]


def unit_query_ids(unit, manifests_dir):
    m, _ = load_manifest(unit, manifests_dir)
    ids = set(m["query_id"])
    if unit in PARENT:
        parent, _ = load_manifest(PARENT[unit], manifests_dir)
        if not set(parent["query_id"]) <= ids:
            raise RuntimeError(f"{PARENT[unit]} is not nested in {unit}")
        ids -= set(parent["query_id"])
    return sorted(ids), m


def pool_fingerprint(pool):
    h = hashlib.sha256()
    for q, c in pool[["query_id", "candidate_connectivity_key"]].sort_values(["query_id", "candidate_connectivity_key"]).itertuples(index=False):
        h.update(f"{q}|{c}\n".encode("utf-8"))
    return h.hexdigest()[:24]


def build_pool(queries, generator, variant_to_connectivity, ppm):
    q = queries.rename(columns={"connectivity_key": "true_connectivity_key"})[["query_id", "neutral_mass", "true_connectivity_key"]]
    raw = generator.generate_dataframe(q, tolerance_ppm=ppm, query_id_col="query_id", mass_col="neutral_mass", true_key_col="true_connectivity_key")
    pool = dedupe_pool_to_connectivity(raw, variant_to_connectivity)
    return pool[["query_id", "candidate_connectivity_key", "abs_mass_error_ppm", "is_true_candidate"]].sort_values(
        ["query_id", "candidate_connectivity_key"]).reset_index(drop=True)


def aggregate_unit(shard_dir, pool, ks=(None,)):
    """Per-shard aggregation (never the full QCR in RAM). Returns {k: features}."""
    pool_by_q = {q: g for q, g in pool.groupby("query_id")}
    parts = {k: [] for k in ks}
    for shard in sorted(Path(shard_dir).glob("part-*.parquet")):
        q = pd.read_parquet(shard)
        if len(q) == 0:
            continue
        sub_pool = pd.concat([pool_by_q[i] for i in q["query_id"].unique() if i in pool_by_q], ignore_index=True)
        for k in ks:
            parts[k].append(aggregate_deterministic(q, sub_pool, PROTOCOL, k=k))
        del q
        gc.collect()
    out = {}
    for k in ks:
        f = pd.concat(parts[k], ignore_index=True) if parts[k] else features_without_references(pool.iloc[0:0])
        miss = pool.merge(f[["query_id", "candidate_connectivity_key"]], how="left", indicator=True)
        miss = miss[miss["_merge"] == "left_only"].drop(columns="_merge")
        if len(miss):
            f = pd.concat([f, features_without_references(miss)], ignore_index=True)
        assert len(f) == len(pool), f"feature rows {len(f)} != pool rows {len(pool)}"
        out[k] = f.sort_values(["query_id", "candidate_connectivity_key"]).reset_index(drop=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, choices=ALL_MANIFESTS)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args(argv)

    t_all = time.time()
    paths = get_project_paths()
    vp = v4b_paths(paths)
    D = v5_paths(paths)
    settlement = load_settlement_or_refuse(vp.settlement_json)
    sim_cfg = settlement["similarity_config"]
    cfg_hash = config_hash(sim_cfg)
    with open(paths.processed / "candidate_generation" / "candidate_generation_contract.json", encoding="utf-8") as f:
        ppm = json.load(f)["molecule_policy"]["tolerance_ppm"]
    print(f"[v5] manifest={args.manifest} units={UNITS[args.manifest]} evidence={settlement['evidence_fingerprint']} ppm={ppm}")

    mmv = pd.read_parquet(paths.interim / "candidate_generation" / "molecule_mass_variants.parquet")
    generator = CandidateGenerator(MassIndex.from_dataframe(mmv, mass_col="exact_mass", key_col="mass_variant_id"))
    variant_to_conn = mmv.set_index("mass_variant_id")["connectivity_key"]
    lookups = None
    unit_feature_paths, log_rows = [], []

    for unit in UNITS[args.manifest]:
        t0 = time.time()
        ids, m = unit_query_ids(unit, D["manifests"])
        unit_dir = D["qcr"] / unit
        feat_path = D["features"] / "units" / f"{unit}_{PROTOCOL}.parquet"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        queries = m[m["query_id"].isin(set(ids))].sort_values("query_id")
        pool = build_pool(queries, generator, variant_to_conn, ppm)
        pool_fp = pool_fingerprint(pool)
        if lookups is None:
            tm = load_artifact("train_spectrum_metadata")
            lookups = SpectrumLookups(tm)
            del tm
            gc.collect()
        fp = qcr_fingerprint(paths.train, lookups.reference_index, ids, pool_fp, sim_cfg, {"protocols": list(PROTOCOLS), "max_refs": 5})
        results = build_qcr_resumable("shards", pool, unit_dir, reference_index=lookups.reference_index, ref_meta_of=lookups.ref_meta_of,
                                      offset_of=lookups.offset_of, peak_store_path=paths.train, similarity_config=sim_cfg,
                                      config_hash=cfg_hash, fingerprint=fp, chunk_size=args.chunk_size, n_jobs=args.n_jobs)
        ks = (None, 1, 3, 5) if unit in K_STRESS_MANIFESTS else (None,)
        feats = aggregate_unit(unit_dir / "qcr" / "shards", pool, ks)
        for k, f in feats.items():
            p = feat_path if k is None else feat_path.with_name(f"{unit}_{PROTOCOL}_k{k}.parquet")
            f.to_parquet(p, index=False)
        pool.to_parquet(D["features"] / "units" / f"{unit}_pool.parquet", index=False)
        n_built = sum(not r.loaded_from_shard for r in results)
        rec = {"unit": unit, "manifest": args.manifest, "n_queries": len(ids), "n_queries_with_candidates": int(pool["query_id"].nunique()),
               "n_pool_rows": int(len(pool)), "pool_fingerprint": pool_fp, "qcr_fingerprint": fp, "n_chunks": len(results),
               "n_chunks_built": n_built, "n_chunks_resumed": len(results) - n_built,
               "max_chunk_rss_gb": float(np.nanmax([r.peak_rss_gb for r in results])) if results else None,
               "wall_seconds": time.time() - t0, "finished_at": datetime.now(timezone.utc).isoformat()}
        log_rows.append(rec)
        print(f"[v5] unit {unit}: {rec}")
        unit_feature_paths.append(feat_path)
        del pool, feats
        gc.collect()

    # manifest-level feature table = concat of its units + query metadata
    m_full, m_meta = load_manifest(args.manifest, D["manifests"])
    meta = m_full[META_COLS].rename(columns={"connectivity_key": "true_connectivity_key"})
    feats = pd.concat([pd.read_parquet(p) for p in unit_feature_paths], ignore_index=True)
    feats = feats[feats["query_id"].isin(set(m_full["query_id"]))].merge(meta, on="query_id", how="left", validate="many_to_one")
    assert feats["fold"].notna().all()
    out = D["features"] / f"{args.manifest}_{PROTOCOL}.parquet"
    feats.to_parquet(out, index=False)
    if args.manifest in K_STRESS_MANIFESTS:
        for k in (1, 3, 5):
            fk = pd.read_parquet(D["features"] / "units" / f"{args.manifest}_{PROTOCOL}_k{k}.parquet").merge(meta, on="query_id", how="left")
            fk.to_parquet(D["features"] / f"{args.manifest}_{PROTOCOL}_k{k}.parquet", index=False)
    summary = {"manifest": args.manifest, "manifest_sha256": m_meta["manifest_sha256"], "feature_path": str(out),
               "n_queries": int(len(m_full)), "n_queries_with_candidates": int(feats["query_id"].nunique()), "n_rows": int(len(feats)),
               "evidence_fingerprint": settlement["evidence_fingerprint"], "similarity_config_hash": cfg_hash, "units": log_rows,
               "wall_seconds": time.time() - t_all}
    (D["features"] / f"{args.manifest}_{PROTOCOL}.build.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with open(D["qcr"] / "build_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, default=str) + "\n")
    print(f"[v5] {args.manifest}: {summary['n_queries']:,} queries ({summary['n_queries_with_candidates']:,} with candidates), "
          f"{summary['n_rows']:,} rows -> {out}  [{summary['wall_seconds'] / 60:.1f} min]")
    return summary


if __name__ == "__main__":
    main()
