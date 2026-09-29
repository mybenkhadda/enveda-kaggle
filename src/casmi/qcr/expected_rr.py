"""Expected reciprocal rank under a uniformly random tie-break order -- the analytically exact
alternative to running one arbitrary tie-break and reporting its MRR as if representative. Used
by C7's tie-handling attribution to separate "tie-RULE effect" from genuine ranking change.
"""


def expected_reciprocal_rank(rank_min, rank_max, k=None):
    """For an item whose score ties it into a block occupying ranks `[rank_min, rank_max]`
    (1-based, inclusive), the expected reciprocal rank under a uniformly random ordering WITHIN
    that block is the mean of `1/r` over `r` in that range (each position equally likely).
    `k`: if given, ranks beyond `k` contribute 0 (matching MRR@k); `rank_min == rank_max`
    (no tie) reduces to the ordinary `1/rank_min`."""
    lo, hi = int(rank_min), int(rank_max)
    if hi < lo:
        raise ValueError(f"rank_max ({hi}) must be >= rank_min ({lo})")
    vals = [1.0 / r for r in range(lo, hi + 1) if k is None or r <= k]
    return sum(vals) / (hi - lo + 1)


def expected_tie_mrr(pair_df, score_col, all_query_ids, query_id_col="query_id", is_true_col="is_true_candidate", k=25):
    """Analytic expected MRR@k under uniformly-random tie-break order, computed exactly (never
    via a single random shuffle) from each true candidate's `[rank_min, rank_max]` tie block.
    Queries in `all_query_ids` with no scored true candidate contribute `RR=0` (never dropped
    from the denominator)."""
    df = pair_df.dropna(subset=[score_col]).copy()
    df["rank_min"] = df.groupby(query_id_col)[score_col].rank(method="min", ascending=False)
    df["rank_max"] = df.groupby(query_id_col)[score_col].rank(method="max", ascending=False)
    true_rows = df[df[is_true_col]].copy()
    true_rows["expected_rr"] = true_rows.apply(lambda row: expected_reciprocal_rank(row["rank_min"], row["rank_max"], k=k), axis=1)
    rr_by_query = true_rows.set_index(query_id_col)["expected_rr"].reindex(all_query_ids).fillna(0.0)
    return float(rr_by_query.mean())
