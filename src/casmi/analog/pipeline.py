"""Per-fold orchestration of the analog baseline (kept out of the notebook).

For evaluation fold f (hidden sets from `casmi.validation.regimes.hidden_sets`):
    1. allowed references = library minus every reference of a hidden (C2/C3) connectivity of fold f
    2. analog neighbors for fold-f queries (resumable cache)
    3. candidate pools from the mass index with C3 truths of fold f excluded
    4. candidate-level analog features, written as resumable parquet shards
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.analog.features import build_feature_shard, candidate_pools, write_schema
from casmi.analog.retrieval import T2_POLICY, build_neighbors
from casmi.candidates.universe import keys_to_ids, read_candidates
from casmi.validation.regimes import hidden_sets, removed_candidate_mask


def fingerprint_getter(fp_cache):
    return lambda keys, smiles_by_key: fp_cache.get(keys, lambda miss: [smiles_by_key.get(m) for m in miss])


def run_analog_fold(fold, queries, regimes, alib, searcher, index, universe_root, candidate_keys, fp_cache, cfg, neighbors_dir,
                    features_dir, chunk=1000, log=print):
    """`queries`: regime-table rows of fold f merged with `neutral_mass`. Returns a small run record."""
    hs = hidden_sets(regimes, fold)
    q = queries[queries["fold"] == fold].dropna(subset=["regime"]).reset_index(drop=True)
    allowed = ~alib.hidden_mask(hs["hidden_reference_keys"])
    self_rows = alib.rows_of_spectrum_ids(q["query_id"])
    if (self_rows < 0).any():
        raise ValueError(f"fold {fold}: {int((self_rows < 0).sum())} query ids are not library spectrum ids -- check the id mapping")
    neighbors = build_neighbors(searcher, q["query_id"], self_rows, allowed, Path(neighbors_dir) / f"fold={fold}", log=log)
    leaked = set(neighbors["analog_connectivity_key"]) & hs["hidden_reference_keys"]
    assert not leaked, f"fold {fold}: {len(leaked)} hidden connectivities appear as analogs"

    exclude = removed_candidate_mask(candidate_keys, hs["removed_structure_keys"])
    out = Path(features_dir) / f"fold={fold}"
    out.mkdir(parents=True, exist_ok=True)
    get_fp = fingerprint_getter(fp_cache)
    n_trunc_total, n_rows = 0, 0
    for ci, s in enumerate(range(0, len(q), chunk)):
        f, done = out / f"part-{ci:05d}.parquet", out / f"part-{ci:05d}.done"
        if f.exists() and done.exists():
            rec = json.loads(done.read_text(encoding="utf-8"))
            n_rows += rec["n_rows"]; n_trunc_total += rec["n_truncated"]
            continue
        qc = q.iloc[s:s + chunk].reset_index(drop=True)
        offsets, ids, ap, n_trunc = candidate_pools(index, qc["neutral_mass"].to_numpy(float), cfg["candidate_ppm"], exclude,
                                                    cfg.get("max_candidates_per_query"))
        cand_rows = read_candidates(universe_root, ids, columns=["connectivity_key", "molecular_formula", "representative_smiles"]).set_index("candidate_id")
        nb = neighbors[neighbors["query_id"].isin(set(qc["query_id"]))]
        akeys = nb["analog_connectivity_key"].astype(str).unique()
        aids = keys_to_ids(candidate_keys, akeys)
        arows = read_candidates(universe_root, aids[aids >= 0], columns=["connectivity_key", "molecular_formula", "exact_mass", "representative_smiles"])
        arows = arows.set_index(arows["connectivity_key"].astype(str))
        shard = build_feature_shard(qc, nb, offsets, ids, ap, cand_rows, arows, get_fp, cfg)
        shard.to_parquet(f, index=False)
        done.write_text(json.dumps({"n_queries": int(len(qc)), "n_rows": int(len(shard)), "n_truncated": n_trunc}), encoding="utf-8")
        n_rows += len(shard); n_trunc_total += n_trunc
        log(f"[analog features] fold {fold} chunk {ci}: {len(qc)} queries -> {len(shard)} rows (truncated pools: {n_trunc})")
    write_schema(out, extra={"fold": int(fold), "t2_policy": T2_POLICY, "candidate_ppm": cfg["candidate_ppm"],
                             "max_candidates_per_query": cfg.get("max_candidates_per_query")})
    return {"fold": int(fold), "n_queries": int(len(q)), "n_rows": int(n_rows), "n_truncated_pools": int(n_trunc_total),
            "n_neighbors": int(len(neighbors))}


def load_features(features_dir, folds=None, columns=None):
    parts = []
    for d in sorted(Path(features_dir).glob("fold=*")):
        f = int(d.name.split("=")[1])
        if folds is not None and f not in folds:
            continue
        parts += [pd.read_parquet(p, columns=columns) for p in sorted(d.glob("part-*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def per_query_ranks(features, score_col, all_queries, higher_is_better=True):
    """Truth rank per intended query under a deterministic order (score, then abs ppm ASC, then
    candidate_id ASC) via the existing `rank_eval`. Queries without candidates / truth -> NaN rank."""
    from casmi.ranking.rank_eval import per_query_metrics, rank_by_keys
    ranked = rank_by_keys(features, [(score_col, not higher_is_better), ("abs_mass_error_ppm", True), ("candidate_id", True)])
    pq = per_query_metrics(ranked, all_queries["query_id"].astype(str).tolist(), is_true_col="is_true_candidate")
    return pq.merge(all_queries[["query_id", "fold", "regime"]].astype({"query_id": str}), on="query_id", how="left")
