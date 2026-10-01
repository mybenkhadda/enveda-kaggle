"""v4b OFFLINE scale-feature build (spec sections 29-31): nested 1k -> 3k -> 10k V1-style training
sets, mirror-aware candidate features built from a chunked, resumable QCR.

Run ONCE, offline, AFTER 10v4b_00 has settled the cache (it refuses otherwise) and BEFORE
10v4b_01 section 13. It can take hours on the first run; it is safe to interrupt and re-run --
every finished 50-query chunk is a fingerprinted shard that is skipped on resume.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python scripts/v4b_build_scale_features.py
    PYTHONPATH=src PYTHONNOUSERSITE=1 python scripts/v4b_build_scale_features.py --strategy random --sizes 3000

What it does:
  1. Loads the settled evidence record (`outputs/v4b/cache_settlement.json`) -- refuses unless the
     decision is ACCEPT_EXISTING / REBUILT_AND_VALIDATED -- and takes the similarity config from it.
  2. Draws nested training-query sets from notebook 03's 60k `dev_queries` (same universe the
     DEV-1k was drawn from); the existing DEV-1k is always the 1k set. Strategy "test_like"
     (default) importance-weights by observable test metadata; "random" is the control.
     Selections are persisted (`query_sets_<strategy>.json`) and reused on resume.
  3. Builds QCR ONLY for the queries not already in the settled DEV-1k QCR, with
     `casmi.qcr.builder.build_qcr_resumable` (chunked, fingerprinted, bounded RAM).
  4. Aggregates mirror_aware candidate features shard-by-shard (never concatenating the full QCR)
     and writes one feature table per training set, with the query metadata the ranker needs.

Memory: one chunk's peak dicts at a time (chunk_size queries); the feature tables are
O(n_pool_rows) only.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from casmi.io.artifacts import artifact_fingerprint, load_artifact  # noqa: E402
from casmi.paths import get_project_paths  # noqa: E402
from casmi.qcr.aggregate import aggregate_deterministic, features_without_references  # noqa: E402
from casmi.qcr.builder import build_qcr_resumable  # noqa: E402
from casmi.qcr.context import SEED, SpectrumLookups, config_hash, v4b_paths  # noqa: E402
from casmi.qcr.fingerprint import artifact_fingerprint as qcr_fingerprint  # noqa: E402
from casmi.qcr.settlement import load_settlement_or_refuse  # noqa: E402
from casmi.ranking.training_sets import distribution_report, nested_scale_sets, test_like_weights  # noqa: E402
from casmi.spectra.reference_selection import PROTOCOLS  # noqa: E402

QUERY_META_COLS = ["query_id", "true_connectivity_key", "fold", "source", "instrument_type", "adduct", "ionization_mode", "neutral_mass"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=("test_like", "random"), default="test_like")
    ap.add_argument("--sizes", type=int, nargs="+", default=[3000, 10000])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--protocol", default="mirror_aware", choices=PROTOCOLS)
    args = ap.parse_args(argv)

    t_start = time.time()
    paths = get_project_paths()
    vp = v4b_paths(paths)
    settlement = load_settlement_or_refuse(vp.settlement_json)
    sim_cfg = settlement["similarity_config"]
    cfg_hash = config_hash(sim_cfg)
    arts = settlement["evidence_artifacts"]
    out_dir = vp.scale_dir / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scale] settled decision={settlement['decision']} evidence={settlement['evidence_fingerprint']} config_hash={cfg_hash}")

    # ---- 1. query universe + nested selection --------------------------------------------------
    cg = paths.interim / "candidate_generation"
    dev_queries = pd.read_parquet(cg / "dev_queries.parquet")
    dev_1k = pd.read_parquet(paths.interim / "spectral_ranking" / "dev_ranking_queries.parquet")
    test_queries = pd.read_parquet(cg / "test_queries.parquet")
    base_ids = dev_1k["query_id"].tolist()

    sel_path = out_dir / f"query_sets_{args.strategy}.json"
    sel_cfg = {"strategy": args.strategy, "sizes": sorted(args.sizes), "seed": args.seed,
               "dev_queries_fp": artifact_fingerprint("dev_queries", cg), "base_ids_n": len(base_ids)}
    if sel_path.exists() and json.loads(sel_path.read_text(encoding="utf-8"))["config"] == sel_cfg:
        sets = json.loads(sel_path.read_text(encoding="utf-8"))["sets"]
        print(f"[scale] reusing persisted query selection {sel_path.name}")
    else:
        weights = None
        if args.strategy == "test_like":
            weights, wreport = test_like_weights(dev_queries, test_queries)
            wreport.to_csv(out_dir / "test_like_weight_report.csv", index=False)
        sets = nested_scale_sets(dev_queries, base_ids, sizes=args.sizes, strategy=args.strategy, weights=weights, seed=args.seed)
        sel_path.write_text(json.dumps({"config": sel_cfg, "sets": sets}, indent=2), encoding="utf-8")
    distribution_report(sets, dev_queries, test_queries).to_csv(out_dir / f"distribution_report_{args.strategy}.csv", index=False)

    all_ids = list(dict.fromkeys(q for ids in sets.values() for q in ids))
    new_ids = sorted(set(all_ids) - set(base_ids))
    print(f"[scale] sets: { {k: len(v) for k, v in sets.items()} }  new queries needing QCR: {len(new_ids):,}")

    # ---- 2. candidate pool for the new queries (same tolerance as the DEV-1k pool) ------------
    with open(paths.processed / "candidate_generation" / "candidate_generation_contract.json", encoding="utf-8") as f:
        selected_ppm = json.load(f)["molecule_policy"]["tolerance_ppm"]
    pool_path = cg / "candidate_pool_dev_max50ppm.parquet"
    pool = pd.read_parquet(pool_path, columns=["query_id", "candidate_connectivity_key", "abs_mass_error_ppm", "is_true_candidate"],
                           filters=[("query_id", "in", new_ids)])
    pool = pool[pool["abs_mass_error_ppm"] <= selected_ppm].reset_index(drop=True)
    print(f"[scale] new-query pool rows: {len(pool):,} over {pool['query_id'].nunique():,} / {len(new_ids):,} queries")

    # ---- 3. chunked resumable QCR --------------------------------------------------------------
    train_meta = load_artifact("train_spectrum_metadata")
    lk = SpectrumLookups(train_meta)
    del train_meta
    gc.collect()
    fp = qcr_fingerprint(paths.train, lk.reference_index, new_ids, artifact_fingerprint("candidate_pool_dev_max50ppm", cg) + f"|ppm<={selected_ppm}",
                         sim_cfg, {"protocols": list(PROTOCOLS), "max_refs": 5})
    results = build_qcr_resumable(f"scale_{args.strategy}", pool, out_dir, reference_index=lk.reference_index,
                                  ref_meta_of=lk.ref_meta_of, offset_of=lk.offset_of, peak_store_path=paths.train,
                                  similarity_config=sim_cfg, config_hash=cfg_hash, fingerprint=fp,
                                  chunk_size=args.chunk_size, n_jobs=args.n_jobs)
    n_built = sum(not r.loaded_from_shard for r in results)
    print(f"[scale] QCR chunks: {len(results)} ({n_built} built, {len(results) - n_built} resumed)")

    # ---- 4. per-shard feature aggregation (never holds the full QCR) --------------------------
    shard_dir = out_dir / "qcr" / f"scale_{args.strategy}"
    pool_by_q = pool.groupby("query_id")
    parts = []
    for shard in sorted(shard_dir.glob("part-*.parquet")):
        q = pd.read_parquet(shard)
        if len(q) == 0:
            continue
        sub_pool = pd.concat([pool_by_q.get_group(i) for i in q["query_id"].unique()], ignore_index=True)
        parts.append(aggregate_deterministic(q, sub_pool, args.protocol))
        del q
        gc.collect()
    # pool rows of queries none of whose candidates has a reference never appear in any QCR shard
    new_feats = pd.concat(parts, ignore_index=True) if parts else features_without_references(pool.iloc[0:0])
    missing = pool.merge(new_feats[["query_id", "candidate_connectivity_key"]], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    if len(missing):
        new_feats = pd.concat([new_feats, features_without_references(missing)], ignore_index=True)
    assert len(new_feats) == len(pool), f"feature rows {len(new_feats)} != pool rows {len(pool)}"

    base_feats = pd.read_parquet(arts[f"dev_features_{args.protocol}"]).drop(columns=["fold"], errors="ignore")
    qmeta = dev_queries[QUERY_META_COLS]
    manifest = {"strategy": args.strategy, "protocol": args.protocol, "similarity_config_hash": cfg_hash, "qcr_fingerprint": fp,
                "evidence_fingerprint": settlement["evidence_fingerprint"], "seed": args.seed, "sets": {}}
    for name, ids in sets.items():
        ids_set = set(ids)
        feats = pd.concat([base_feats[base_feats["query_id"].isin(ids_set)], new_feats[new_feats["query_id"].isin(ids_set)]], ignore_index=True)
        feats = feats.merge(qmeta, on="query_id", how="left", validate="many_to_one")
        assert feats["fold"].notna().all(), "every training query must have a connectivity fold"
        path = out_dir / f"scale_features_{args.protocol}_{name}.parquet"
        feats.to_parquet(path, index=False)
        manifest["sets"][name] = {"path": str(path), "n_queries_selected": len(ids), "n_queries_with_candidates": int(feats["query_id"].nunique()),
                                  "n_rows": int(len(feats))}
        print(f"[scale] {name}: {len(ids):,} queries selected, {feats['query_id'].nunique():,} with candidates, {len(feats):,} rows -> {path.name}")
    manifest["elapsed_seconds"] = time.time() - t_start
    (out_dir / "scale_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[scale] done in {manifest['elapsed_seconds'] / 60:.1f} min -> {out_dir / 'scale_manifest.json'}")


if __name__ == "__main__":
    main()
