"""Reusable retrieval metrics for later (modeling) notebooks. Not used for model evaluation
yet -- prepared now, with unit tests, so notebook 03+ doesn't reimplement or subtly
re-derive them.
"""
import numpy as np


def _dedupe_keep_first(seq, k):
    seen, out = set(), []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) >= k:
            break
    return out


def mrr_at_k(pred_lists, true_keys, k=25):
    """Mean reciprocal rank @k. Each prediction list is deduplicated (keeping the first,
    best-ranked occurrence) and truncated to `k` before scoring, so a repeated guess can't
    occupy multiple rank slots and predictions beyond `k` never count."""
    assert len(pred_lists) == len(true_keys), "pred_lists and true_keys must be the same length"
    if not pred_lists:
        return float("nan")
    total = 0.0
    for preds, true_key in zip(pred_lists, true_keys):
        deduped = _dedupe_keep_first(preds, k)
        try:
            rank = deduped.index(true_key) + 1
            total += 1.0 / rank
        except ValueError:
            pass
    return total / len(true_keys)


def hit_at_k(pred_lists, true_keys, k=25):
    """Fraction of queries whose true key appears within the top `k` (deduplicated)
    predictions."""
    assert len(pred_lists) == len(true_keys), "pred_lists and true_keys must be the same length"
    if not pred_lists:
        return float("nan")
    hits = 0
    for preds, true_key in zip(pred_lists, true_keys):
        if true_key in _dedupe_keep_first(preds, k):
            hits += 1
    return hits / len(true_keys)


def candidate_recall(candidate_lists, true_keys):
    """Fraction of queries whose true key appears ANYWHERE in their (untruncated) candidate
    list -- the ceiling any downstream ranker could possibly achieve, since a candidate
    generator that never proposes the true key makes ranking irrelevant for that query."""
    assert len(candidate_lists) == len(true_keys), "candidate_lists and true_keys must be the same length"
    if not candidate_lists:
        return float("nan")
    hits = sum(1 for cands, true_key in zip(candidate_lists, true_keys) if true_key in set(cands))
    return hits / len(true_keys)


# ---------------------------------------------------------------------------------------------
# v2: vectorized per-query metrics (C1 / C2 / C3 regimes)
#
# Input is ONE ROW PER INTENDED QUERY with a 1-based `truth_rank` (NaN = truth not in the list / no
# candidates) and the list size. A query is never dropped from a denominator: NaN rank counts as a
# miss everywhere (RR = 0, Hit = 0, Recall = 0). Ranks must already be connectivity-deduplicated
# (`casmi.ranking.rank_eval.per_query_metrics` produces this shape).
# ---------------------------------------------------------------------------------------------

CANDIDATE_K = (25, 100, 500)
RANKING_K = (1, 5, 10, 25)
MRR_K = 25
PROVISIONAL_COMPOSITE_WEIGHTS = {"C1": 0.15, "C2": 0.60, "C3": 0.25}


def _ranks(per_query, rank_col):
    r = per_query[rank_col].to_numpy(dtype=float)
    if np.any(np.isfinite(r) & (r < 1)):
        raise ValueError(f"{rank_col} must be 1-based (found values < 1)")
    return r


def candidate_stage_metrics(per_query, rank_col="truth_rank", pool_col="pool_size", k_values=CANDIDATE_K):
    """Candidate-generation metrics: Recall@k (truth within the first k of the candidate ORDER used by the
    generator, e.g. abs ppm), Recall@All, truth-missing rate and pool-size quantiles."""
    n = len(per_query)
    if n == 0:
        return {"n_queries": 0}
    r = _ranks(per_query, rank_col)
    found = np.isfinite(r)
    pool = per_query[pool_col].to_numpy(dtype=float)
    out = {"n_queries": int(n), "recall_all": float(found.mean()), "truth_missing_rate": float(1 - found.mean())}
    for k in k_values:
        out[f"recall_at_{k}"] = float((found & (r <= k)).mean())
    out.update(pool_size_median=float(np.median(pool)), pool_size_p90=float(np.quantile(pool, 0.90)),
               pool_size_p99=float(np.quantile(pool, 0.99)), pool_size_mean=float(pool.mean()),
               zero_pool_rate=float((pool == 0).mean()))
    return out


def ranking_metrics(per_query, rank_col="truth_rank", k_values=RANKING_K, mrr_k=MRR_K):
    """Ranking metrics over ALL queries: MRR@25, Hit@k, miss rate; mean / median truth rank over the
    queries whose truth was ranked (reported with the count they are computed on)."""
    n = len(per_query)
    if n == 0:
        return {"n_queries": 0}
    r = _ranks(per_query, rank_col)
    found = np.isfinite(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(found & (r <= mrr_k), 1.0 / r, 0.0)
    out = {"n_queries": int(n), f"mrr_at_{mrr_k}": float(rr.mean())}
    for k in k_values:
        out[f"hit_at_{k}"] = float((found & (r <= k)).mean())
    out["miss_rate"] = float(1 - found.mean())
    out["n_ranked"] = int(found.sum())
    out["mean_truth_rank"] = float(r[found].mean()) if found.any() else float("nan")
    out["median_truth_rank"] = float(np.median(r[found])) if found.any() else float("nan")
    return out


def composite_mrr(regime_mrr, weights=None):
    """PROVISIONAL composite: sum_r w_r * MRR25_r over C1/C2/C3. Not used by any core metric. A missing /
    NaN regime makes the composite NaN (never silently re-weighted). Raw regime values are returned too."""
    w = dict(PROVISIONAL_COMPOSITE_WEIGHTS if weights is None else weights)
    vals = {r: float(regime_mrr.get(r, float("nan"))) for r in w}
    comp = float(sum(w[r] * vals[r] for r in w)) if all(np.isfinite(v) for v in vals.values()) else float("nan")
    return {"label": "PROVISIONAL", "weights": w, "regime_mrr_at_25": vals, "composite_mrr_at_25": comp}
