"""v4b per-query ranking evaluation.

Everything is computed PER QUERY first (so the connectivity-level bootstrap can resample it) and
always over the FULL intended query list: a query with zero candidates, or whose true candidate
is absent from its pool, contributes RR = 0 and Hit@k = 0 -- it is never dropped from the
denominator. Two rank constructions:

    `rank_by_keys`         deterministic lexicographic ranking over explicit (column, ascending)
                           keys; NaN in any key sorts LAST; always end with a unique key
                           (e.g. `candidate_connectivity_key`) so the order is total.
    `expected_tie_metrics` analytic expectation under a uniformly random order inside exact
                           score ties (no hidden secondary key); NaN scores form one tail block.
"""
import numpy as np
import pandas as pd

from casmi.qcr.expected_rr import expected_reciprocal_rank

K_VALUES = (1, 5, 25)
MRR_K = 25


def rank_by_keys(df, keys, query_col="query_id"):
    """`keys`: list of `(column, ascending)` pairs, most significant first. Returns a copy with an
    integer `rank` column (1 = best) within each query."""
    cols = [query_col] + [c for c, _ in keys]
    asc = [True] + [bool(a) for _, a in keys]
    out = df.sort_values(cols, ascending=asc, na_position="last", kind="mergesort").copy()
    out["rank"] = out.groupby(query_col, sort=False).cumcount() + 1
    return out.sort_index()


def per_query_metrics(ranked_df, all_query_ids, query_col="query_id", is_true_col="is_true_candidate",
                      k_values=K_VALUES, mrr_k=MRR_K):
    """One row per id in `all_query_ids` (order preserved): `n_candidates`, `truth_rank` (NaN if
    the query has no candidates or its truth is absent), `rr` (@mrr_k) and `hit_at_<k>`."""
    all_query_ids = pd.Index(list(all_query_ids), name=query_col)
    if all_query_ids.has_duplicates:
        raise ValueError("all_query_ids contains duplicates")
    n_cand = ranked_df.groupby(query_col).size().reindex(all_query_ids, fill_value=0)
    truth = ranked_df.loc[ranked_df[is_true_col].astype(bool), [query_col, "rank"]]
    if truth[query_col].duplicated().any():
        raise ValueError("more than one true candidate for some query")
    truth_rank = truth.set_index(query_col)["rank"].reindex(all_query_ids).astype(float)
    out = pd.DataFrame({"n_candidates": n_cand.to_numpy(), "truth_rank": truth_rank.to_numpy()}, index=all_query_ids)
    r = out["truth_rank"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rr"] = np.where(np.isfinite(r) & (r <= mrr_k), 1.0 / r, 0.0)
    for k in k_values:
        out[f"hit_at_{k}"] = np.where(np.isfinite(r) & (r <= k), 1.0, 0.0)
    return out.reset_index()


def expected_tie_metrics(df, score_col, all_query_ids, query_col="query_id", is_true_col="is_true_candidate",
                         higher_is_better=True, k_values=K_VALUES, mrr_k=MRR_K):
    """Per-query EXPECTED RR@mrr_k and Hit@k when candidates with an exactly equal score are
    ordered uniformly at random (and no secondary key is used). The true candidate's tie block
    `[rank_min, rank_max]`: E[RR] = mean of 1/r over the block (0 beyond mrr_k); P(hit@k) =
    |block ∩ [1, k]| / |block|. NaN scores are one tie block after every scored candidate."""
    all_query_ids = pd.Index(list(all_query_ids), name=query_col)
    d = df[[query_col, score_col, is_true_col]].copy()
    s = d[score_col].astype(float)
    # map NaN to -inf (higher-is-better) so it forms one worst tie block; flip sign for lower-is-better
    d["_s"] = np.where(s.isna(), -np.inf, s if higher_is_better else -s)
    g = d.groupby(query_col)["_s"]
    d["rank_min"] = g.rank(method="min", ascending=False)
    d["rank_max"] = g.rank(method="max", ascending=False)
    t = d[d[is_true_col].astype(bool)].set_index(query_col)
    n_cand = d.groupby(query_col).size().reindex(all_query_ids, fill_value=0)
    out = pd.DataFrame({"n_candidates": n_cand.to_numpy()}, index=all_query_ids)
    lo = t["rank_min"].reindex(all_query_ids)
    hi = t["rank_max"].reindex(all_query_ids)
    out["truth_rank_min"], out["truth_rank_max"] = lo.to_numpy(), hi.to_numpy()
    out["truth_rank"] = ((lo + hi) / 2.0).to_numpy()  # expected rank, for stratified reporting only
    out["rr"] = [expected_reciprocal_rank(a, b, k=mrr_k) if np.isfinite(a) else 0.0 for a, b in zip(lo, hi)]
    for k in k_values:
        out[f"hit_at_{k}"] = [(max(0.0, min(k, b) - a + 1) / (b - a + 1)) if np.isfinite(a) else 0.0 for a, b in zip(lo, hi)]
    return out.reset_index()


def summarize(per_query, k_values=K_VALUES):
    """`{n_queries, n_with_candidates, mrr_at_25, hit_at_<k>}` -- plain means over ALL rows."""
    out = {"n_queries": int(len(per_query)), "n_with_candidates": int((per_query["n_candidates"] > 0).sum()),
           "mrr_at_25": float(per_query["rr"].mean()) if len(per_query) else float("nan")}
    for k in k_values:
        out[f"hit_at_{k}"] = float(per_query[f"hit_at_{k}"].mean()) if len(per_query) else float("nan")
    return out


def query_accounting(all_query_ids, feature_df, query_col="query_id", is_true_col="is_true_candidate"):
    """Spec section 12: every intended query accounted for."""
    ids = pd.Index(list(all_query_ids))
    n_cand = feature_df.groupby(query_col).size().reindex(ids, fill_value=0)
    has_truth = feature_df.groupby(query_col)[is_true_col].any().reindex(ids, fill_value=False)
    return {
        "n_queries_total": int(len(ids)),
        "n_queries_with_candidates": int((n_cand > 0).sum()),
        "n_queries_without_candidates": int((n_cand == 0).sum()),
        "n_queries_truth_in_pool": int(has_truth.sum()),
        "n_queries_truth_absent_from_nonempty_pool": int(((n_cand > 0) & ~has_truth).sum()),
        "n_feature_rows_outside_query_list": int((~feature_df[query_col].isin(ids)).sum()),
    }
