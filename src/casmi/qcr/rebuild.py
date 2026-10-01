"""The ONE clean evidence rebuild used by 10v4b_00's REBUILD_ONCE branch.

Never overwrites the v4a.1 artifacts: the rebuilt HOST/DEV QCR, pair summaries and the 4+4
protocol feature tables go to `data/processed/v4b_rebuild/`, built from an EMPTY similarity cache
(fresh output directory + a config hash that cannot collide with v4a.1 cache keys) with the
current, deterministic top-N code. The old caches are marked invalid with an `INVALIDATED.json`
sidecar (not deleted) so no v4b code path loads them.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from casmi.qcr.aggregate import aggregate_deterministic
from casmi.qcr.builder import build_qcr_resumable, load_qcr_shards
from casmi.qcr.fingerprint import artifact_fingerprint as qcr_fingerprint
from casmi.spectra.reference_selection import PROTOCOLS


def mark_cache_invalidated(cache_paths, reason):
    written = []
    for p in cache_paths:
        p = Path(p)
        if p.exists():
            side = p.with_suffix(".INVALIDATED.json")
            side.write_text(json.dumps({"invalidated_file": str(p), "reason": reason,
                                        "at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
            written.append(side)
    return written


def build_host_pool(host_holdout_spectra, candidate_generator, variant_to_connectivity, selected_ppm):
    """The HOST candidate pool exactly as v4a.1 built it (mass-window candidates, connectivity-deduped)."""
    from casmi.candidates.generator import dedupe_pool_to_connectivity
    raw = candidate_generator.generate_dataframe(host_holdout_spectra, tolerance_ppm=selected_ppm, query_id_col="query_id",
                                                 mass_col="neutral_mass", true_key_col="true_connectivity_key")
    return dedupe_pool_to_connectivity(raw, variant_to_connectivity)


def _add_compat_columns(qcr, host_like_enveda_prefix="enveda"):
    """Columns the v4a.1 monolithic QCR carried that the chunked builder does not write."""
    qcr = qcr.copy()
    qcr["is_same_source"] = qcr["ref_source"] == qcr["query_source"]
    qcr["ref_is_enveda"] = qcr["ref_source"].astype(str).str.lower().str.startswith(host_like_enveda_prefix)
    return qcr


def _add_ge3(pairs, qcr):
    pairs = pairs.copy()
    for p in PROTOCOLS:
        n_acc = qcr[qcr[f"accepted_{p}"]].groupby(["query_id", "candidate_key"]).size().rename("n")
        pairs = pairs.merge(n_acc.reset_index(), on=["query_id", "candidate_key"], how="left")
        pairs[f"has_ge3_{p}"] = pairs["n"].fillna(0) >= 3
        pairs = pairs.drop(columns="n")
    return pairs


def rebuild_dataset(dataset_name, pool_df, rebuild_dir, *, lookups, peak_store_path, similarity_config, config_hash,
                    fold_of_query=None, chunk_size=50, n_jobs=1, fresh=True):
    """Chunked QCR build from an empty cache -> canonical qcr / pairs / 4 protocol feature tables.
    `fresh=True` deletes a previous PARTIAL rebuild of this dataset (only inside `rebuild_dir`)."""
    rebuild_dir = Path(rebuild_dir)
    shard_root = rebuild_dir / "shards"
    if fresh:
        for kind in ("qcr", "qcr_pairs", "similarity_cache"):
            d = shard_root / kind / dataset_name
            if d.exists():
                shutil.rmtree(d)
    query_ids = sorted(pool_df["query_id"].unique().tolist())
    fp = qcr_fingerprint(peak_store_path, lookups.reference_index, query_ids, f"{dataset_name}-pool-rows={len(pool_df)}",
                         similarity_config, {"protocols": list(PROTOCOLS), "max_refs": 5})
    results = build_qcr_resumable(dataset_name, pool_df, shard_root, reference_index=lookups.reference_index,
                                  ref_meta_of=lookups.ref_meta_of, offset_of=lookups.offset_of, peak_store_path=peak_store_path,
                                  similarity_config=similarity_config, config_hash=config_hash, fingerprint=fp,
                                  chunk_size=chunk_size, n_jobs=n_jobs)
    qcr = _add_compat_columns(load_qcr_shards(shard_root, dataset_name, "qcr"))
    pairs = _add_ge3(load_qcr_shards(shard_root, dataset_name, "qcr_pairs"), qcr)
    out = {"qcr": rebuild_dir / f"{dataset_name}_qcr.parquet", "qcr_pairs": rebuild_dir / f"{dataset_name}_qcr_pairs.parquet"}
    rebuild_dir.mkdir(parents=True, exist_ok=True)
    qcr.to_parquet(out["qcr"], index=False)
    pairs.to_parquet(out["qcr_pairs"], index=False)
    base_pool = pool_df[["query_id", "candidate_connectivity_key", "abs_mass_error_ppm", "is_true_candidate"]]
    for p in PROTOCOLS:
        feats = aggregate_deterministic(qcr, base_pool, p)
        if fold_of_query is not None:
            feats["fold"] = feats["query_id"].map(fold_of_query)
        path = rebuild_dir / f"{dataset_name}_pair_features_{p}.parquet"
        feats.to_parquet(path, index=False)
        out[f"features_{p}"] = path
    stats = {"n_chunks": len(results), "n_similarity_computed": int(sum(r.n_similarity_computed for r in results)),
             "n_similarity_hits": int(sum(r.n_similarity_hits for r in results)),
             "n_resumed_chunks": int(sum(r.loaded_from_shard for r in results)), "fingerprint": fp,
             "n_qcr_rows": int(len(qcr)), "n_pair_rows": int(len(pairs))}
    return out, stats
