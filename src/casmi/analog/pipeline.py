"""Per-fold orchestration of the analog baseline (kept out of the notebook).

For evaluation fold f (hidden sets from `casmi.validation.regimes.hidden_sets`):
    1. allowed references = library minus every reference of a hidden (C2/C3) connectivity of fold f
    2. analog neighbors for fold-f queries (resumable, identity-namespaced cache)
    3. candidate pools from the mass index with C3 truths of fold f excluded
    4. candidate-level analog features, written as resumable parquet shards (identity-namespaced)

Cache identity (`casmi.workspace.cache_identity`): neighbor shards are namespaced by fold, query set, hidden
reference set, search config and reference-library identity; feature shards additionally by the query->regime
assignment, the removed (C3) structure set, the full analog config, the candidate-universe identity and the feature
schema. A change in any of them selects a NEW namespace directory -- stale shards are never resumed.
"""
import json
from pathlib import Path

import pandas as pd

from casmi.analog.features import build_feature_shard, candidate_pools, feature_schema, write_schema
from casmi.analog.retrieval import T2_POLICY, build_neighbors
from casmi.candidates.universe import keys_to_ids, read_candidates
from casmi.validation.regimes import hidden_sets, removed_candidate_mask
from casmi.workspace.cache_identity import (StaleCacheError, fingerprint_json, fingerprint_pairs, fingerprint_values, identity_hash,
                                            open_namespace, read_identity)

SEARCH_CONFIG_KEYS = ("bin_width_da", "peak_tol_da", "intensity_power", "prefilter_top_n", "rescore_top_n", "top_k_analogs",
                      "same_polarity_only", "max_precursor_delta_da")
NEIGHBOR_CHUNK = 2000


def neighbor_identity(fold, q, hs, cfg, extra=None):
    """What a fold's neighbor shards depend on (`extra`: e.g. {'reference_library': ...})."""
    return {"fold": int(fold), "chunk": NEIGHBOR_CHUNK, "n_queries": int(len(q)),
            "query_ids": fingerprint_values(q["query_id"].astype(str)),
            "hidden_reference_keys": fingerprint_values(hs["hidden_reference_keys"]),
            "n_hidden_reference_keys": len(hs["hidden_reference_keys"]),
            "search_config": fingerprint_json({k: cfg.get(k) for k in SEARCH_CONFIG_KEYS}), "t2_policy": T2_POLICY,
            **{k: v for k, v in (extra or {}).items() if k == "reference_library"}}


def feature_identity(fold, q, hs, cfg, chunk, extra=None):
    """What a fold's feature shards depend on (`extra`: reference_library, universe, notebook_api, ...)."""
    return {"fold": int(fold), "chunk": int(chunk), "n_queries": int(len(q)),
            "query_ids": fingerprint_values(q["query_id"].astype(str)),
            "query_regimes": fingerprint_pairs(zip(q["query_id"].astype(str), q["regime"].astype(str))),
            "regime_counts": {str(k): int(v) for k, v in q["regime"].value_counts().sort_index().items()},
            "hidden_reference_keys": fingerprint_values(hs["hidden_reference_keys"]),
            "removed_structure_keys": fingerprint_values(hs["removed_structure_keys"]),
            "analog_config": fingerprint_json(cfg), "feature_schema": feature_schema()["feature_hash"], "t2_policy": T2_POLICY,
            **(extra or {})}


def fingerprint_getter(fp_cache):
    return lambda keys, smiles_by_key: fp_cache.get(keys, lambda miss: [smiles_by_key.get(m) for m in miss])


def run_analog_fold(fold, queries, regimes, alib, searcher, index, universe_root, candidate_keys, fp_cache, cfg, neighbors_dir,
                    features_dir, chunk=1000, log=print, identity_extra=None):
    """`queries`: regime-table rows of fold f merged with `neutral_mass`. Returns a run record including the exact
    identity-namespace directories (`neighbors_dir`, `features_dir`) -- load features ONLY from those.
    `identity_extra`: {'reference_library': ..., 'universe': ..., 'notebook_api': ...} (see notebook 14)."""
    hs = hidden_sets(regimes, fold)
    q = queries[queries["fold"] == fold].dropna(subset=["regime"]).sort_values("query_id", kind="mergesort").reset_index(drop=True)
    nb_ident = neighbor_identity(fold, q, hs, cfg, identity_extra)
    ft_ident = feature_identity(fold, q, hs, cfg, chunk, identity_extra)
    nb_dir = open_namespace(Path(neighbors_dir) / f"fold={fold}", nb_ident, kind="analog_neighbors")
    out = open_namespace(Path(features_dir) / f"fold={fold}", ft_ident, kind="analog_features")
    allowed = ~alib.hidden_mask(hs["hidden_reference_keys"])
    self_rows = alib.rows_of_spectrum_ids(q["query_id"])
    if (self_rows < 0).any():
        raise ValueError(f"fold {fold}: {int((self_rows < 0).sum())} query ids are not library spectrum ids -- check the id mapping")
    neighbors = build_neighbors(searcher, q["query_id"], self_rows, allowed, nb_dir, chunk=NEIGHBOR_CHUNK, log=log)
    leaked = set(neighbors["analog_connectivity_key"]) & hs["hidden_reference_keys"]
    assert not leaked, f"fold {fold}: {len(leaked)} hidden connectivities appear as analogs"
    self_hits = int((neighbors["ref_spectrum_id"].astype(str) == neighbors["query_id"].astype(str)).sum()) if len(neighbors) else 0
    assert self_hits == 0, f"fold {fold}: {self_hits} queries matched their own spectrum as an analog"

    exclude = removed_candidate_mask(candidate_keys, hs["removed_structure_keys"])
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
        fp_cache.flush()                                   # one fingerprint shard per chunk, written before the chunk is marked done
        shard.to_parquet(f, index=False)
        done.write_text(json.dumps({"n_queries": int(len(qc)), "n_rows": int(len(shard)), "n_truncated": n_trunc}), encoding="utf-8")
        n_rows += len(shard); n_trunc_total += n_trunc
        log(f"[analog features] fold {fold} chunk {ci}: {len(qc)} queries -> {len(shard)} rows (truncated pools: {n_trunc})")
    write_schema(out, extra={"fold": int(fold), "t2_policy": T2_POLICY, "candidate_ppm": cfg["candidate_ppm"],
                             "max_candidates_per_query": cfg.get("max_candidates_per_query")})
    return {"fold": int(fold), "n_queries": int(len(q)), "n_rows": int(n_rows), "n_truncated_pools": int(n_trunc_total),
            "n_neighbors": int(len(neighbors)), "neighbors_dir": str(nb_dir), "features_dir": str(out),
            "features_identity": identity_hash(ft_ident), "neighbors_identity": identity_hash(nb_ident)}


def load_features(namespace_dirs, columns=None):
    """Concatenate the COMPLETED shards (part-*.parquet with a .done marker) of the given identity-namespace
    directories (the `features_dir` values returned by `run_analog_fold`). A directory without identity.json is
    rejected -- features are never read from a glob over the base directory."""
    if isinstance(namespace_dirs, (str, Path)):
        namespace_dirs = [namespace_dirs]
    parts = []
    for d in namespace_dirs:
        d = Path(d)
        if read_identity(d) is None:
            raise StaleCacheError(f"{d} is not an identity namespace (no identity.json) -- pass the features_dir returned by run_analog_fold")
        for p in sorted(d.glob("part-*.parquet")):
            if p.with_suffix(".done").exists():
                parts.append(pd.read_parquet(p, columns=columns))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def per_query_ranks(features, score_col, all_queries, higher_is_better=True):
    """Truth rank per intended query under a deterministic order (score, then abs ppm ASC, then
    candidate_id ASC) via the existing `rank_eval`. Queries without candidates / truth -> NaN rank."""
    from casmi.ranking.rank_eval import per_query_metrics, rank_by_keys
    ranked = rank_by_keys(features, [(score_col, not higher_is_better), ("abs_mass_error_ppm", True), ("candidate_id", True)])
    pq = per_query_metrics(ranked, all_queries["query_id"].astype(str).tolist(), is_true_col="is_true_candidate")
    return pq.merge(all_queries[["query_id", "fold", "regime"]].astype({"query_id": str}), on="query_id", how="left")
