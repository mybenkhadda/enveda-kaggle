"""Ranking evaluation: score -> rank, then CONDITIONAL (only queries whose true candidate is in
the pair table) vs. END-TO-END (every intended query; a missing true candidate contributes 0 to
MRR/Hit@K, never dropped from the denominator) metrics -- kept explicitly separate, per the
project's own "never silently drop a query" convention (`casmi.candidates.evaluation`).
"""
import numpy as np
import pandas as pd


def rank_candidates(pair_df, query_id_col="query_id", score_col="score", ascending=False):
    """Add a `rank` column (1 = best) within each `query_id_col` group. Ties broken by pandas'
    stable `method="first"` (original row order) -- deterministic, not "generously" tied."""
    out = pair_df.copy()
    out["rank"] = out.groupby(query_id_col)[score_col].rank(method="first", ascending=ascending).astype(int)
    return out


def true_candidate_rank(ranked_pair_df, query_id_col="query_id", is_true_col="is_true_candidate"):
    """One entry per query_id whose true candidate appears in `ranked_pair_df`: its rank. A
    `pandas.Series` (NOT reindexed to every query -- callers needing the full denominator use
    `mrr_from_ranks`/`hit_at_k_from_ranks` with `.reindex(all_query_ids)` first)."""
    true_rows = ranked_pair_df[ranked_pair_df[is_true_col]]
    return true_rows.set_index(query_id_col)["rank"]


def per_query_reciprocal_rank(ranks, k=25):
    """Reciprocal rank PER QUERY (not averaged) -- 0.0 for a missing/beyond-k rank, never
    dropped. The per-query array `casmi.validation.bootstrap.paired_bootstrap_metric_difference`
    needs (its mean is exactly `mrr_from_ranks`)."""
    ranks = pd.Series(ranks)
    return ranks.apply(lambda r: 1.0 / r if pd.notna(r) and r <= k else 0.0).to_numpy()


def per_query_hit(ranks, k):
    """0/1 hit-at-k indicator PER QUERY (not averaged); its mean is exactly `hit_at_k_from_ranks`."""
    ranks = pd.Series(ranks)
    return (ranks <= k).fillna(False).to_numpy().astype(float)


def mrr_from_ranks(ranks, k=25):
    """Mean reciprocal rank @k over `ranks` (a Series/array, NaN entries count as a miss --
    reciprocal rank 0 -- and still count in the denominator, `len(ranks)`)."""
    ranks = pd.Series(ranks)
    if len(ranks) == 0:
        return float("nan")
    return float(per_query_reciprocal_rank(ranks, k=k).mean())


def hit_at_k_from_ranks(ranks, k):
    """Fraction of `ranks` (NaN = miss) at or above rank `k`."""
    ranks = pd.Series(ranks)
    if len(ranks) == 0:
        return float("nan")
    return float(per_query_hit(ranks, k).mean())


def ranking_metrics(ranked_pair_df, all_query_ids, query_id_col="query_id", is_true_col="is_true_candidate",
                     k_values=(1, 5, 10, 25)):
    """Both CONDITIONAL and END-TO-END metrics in one call. Returns
    `{"conditional": {...}, "end_to_end": {...}}`, each with `n_queries`, `mrr_at_25`,
    `hit_at_<k>` for every `k_values`, plus `median_rank` (conditional only) /
    `candidate_recall` (end-to-end only, i.e. the fraction of ALL queries whose true candidate
    was even present -- notebook 03's number, re-derived here as a sanity cross-check)."""
    ranks = true_candidate_rank(ranked_pair_df, query_id_col, is_true_col)
    all_query_ids = list(all_query_ids)

    conditional_ranks = ranks.dropna()
    conditional = {
        "n_queries": len(conditional_ranks),
        "mrr_at_25": mrr_from_ranks(conditional_ranks, k=25),
        "median_rank": float(conditional_ranks.median()) if len(conditional_ranks) else float("nan"),
        **{f"hit_at_{k}": hit_at_k_from_ranks(conditional_ranks, k) for k in k_values},
    }

    end_to_end_ranks = ranks.reindex(all_query_ids)
    end_to_end = {
        "n_queries": len(all_query_ids),
        "mrr_at_25": mrr_from_ranks(end_to_end_ranks, k=25),
        "candidate_recall": float(end_to_end_ranks.notna().mean()) if len(all_query_ids) else float("nan"),
        **{f"hit_at_{k}": hit_at_k_from_ranks(end_to_end_ranks, k) for k in k_values},
    }
    return {"conditional": conditional, "end_to_end": end_to_end}
