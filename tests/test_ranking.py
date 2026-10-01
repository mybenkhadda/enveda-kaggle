import math

import numpy as np
import pandas as pd

from casmi.ranking.aggregation import aggregate_scores
from casmi.ranking.evaluation import hit_at_k_from_ranks, mrr_from_ranks, rank_candidates, ranking_metrics, true_candidate_rank


def test_aggregate_scores_basic():
    out = aggregate_scores([1.0, 2.0, 3.0, 4.0], methods=("max", "mean", "median", "top3_mean"))
    assert out["max"] == 4.0
    assert out["mean"] == 2.5
    assert out["median"] == 2.5
    assert math.isclose(out["top3_mean"], (4 + 3 + 2) / 3)


def test_aggregate_scores_empty_returns_nan_never_raises():
    out = aggregate_scores([], methods=("max", "mean"))
    assert math.isnan(out["max"])
    assert math.isnan(out["mean"])


def test_aggregate_scores_unknown_method_raises():
    try:
        aggregate_scores([1.0], methods=("bogus",))
        assert False, "should reject an unknown aggregation method"
    except ValueError:
        pass


def test_rank_candidates_higher_score_is_rank_1():
    df = pd.DataFrame({"query_id": ["q1", "q1", "q1"], "score": [0.5, 0.9, 0.1]})
    ranked = rank_candidates(df)
    assert list(ranked.sort_values("score", ascending=False)["rank"]) == [1, 2, 3]


def test_rank_candidates_independent_per_query():
    df = pd.DataFrame({"query_id": ["q1", "q1", "q2"], "score": [0.1, 0.9, 0.5]})
    ranked = rank_candidates(df)
    assert ranked[ranked["query_id"] == "q2"]["rank"].iloc[0] == 1


def test_true_candidate_rank_extracts_only_true_rows():
    df = pd.DataFrame({
        "query_id": ["q1", "q1", "q2"],
        "rank": [2, 1, 1],
        "is_true_candidate": [True, False, True],
    })
    ranks = true_candidate_rank(df)
    assert ranks.to_dict() == {"q1": 2, "q2": 1}


def test_mrr_from_ranks_matches_definition():
    ranks = pd.Series([1, 2, 4])
    # 1/1 + 1/2 + 1/4 = 1.75, / 3 queries
    assert math.isclose(mrr_from_ranks(ranks, k=25), 1.75 / 3)


def test_mrr_from_ranks_missing_counts_as_zero_not_dropped():
    ranks = pd.Series([1.0, np.nan])
    assert math.isclose(mrr_from_ranks(ranks, k=25), 0.5)  # (1/1 + 0) / 2, not 1/1


def test_mrr_from_ranks_beyond_k_scores_zero():
    ranks = pd.Series([30])
    assert mrr_from_ranks(ranks, k=25) == 0.0


def test_hit_at_k_from_ranks():
    ranks = pd.Series([1, 5, 30, np.nan])
    assert hit_at_k_from_ranks(ranks, k=5) == 0.5  # ranks 1 and 5 qualify, 30 and NaN don't


def test_ranking_metrics_conditional_vs_end_to_end():
    # q1: true candidate ranked 1st. q2: true candidate present but ranked 3rd.
    # q3: true candidate ABSENT from the pair table entirely (candidate generation missed it).
    pair_df = pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2", "q2", "q4"],
        "rank": [1, 2, 3, 1, 2, 1],
        "is_true_candidate": [True, False, True, False, False, False],
    })
    metrics = ranking_metrics(pair_df, all_query_ids=["q1", "q2", "q3", "q4"])

    # conditional: only q1 (rank 1) and q2 (rank 3) count -- q3 and q4 excluded (no true row)
    assert metrics["conditional"]["n_queries"] == 2
    assert math.isclose(metrics["conditional"]["mrr_at_25"], (1.0 + 1 / 3) / 2)

    # end-to-end: all 4 queries count; q3 (missing entirely) and q4 (true candidate not in its
    # own pair rows) both contribute 0.
    assert metrics["end_to_end"]["n_queries"] == 4
    assert math.isclose(metrics["end_to_end"]["mrr_at_25"], (1.0 + 1 / 3 + 0 + 0) / 4)
    assert metrics["end_to_end"]["candidate_recall"] == 0.5  # 2 of 4 queries have their true candidate present


def test_per_query_reciprocal_rank_matches_mrr_mean():
    from casmi.ranking.evaluation import mrr_from_ranks, per_query_reciprocal_rank

    ranks = pd.Series([1, 2, np.nan, 30])
    per_query = per_query_reciprocal_rank(ranks, k=25)
    assert math.isclose(per_query.mean(), mrr_from_ranks(ranks, k=25))
    assert list(per_query) == [1.0, 0.5, 0.0, 0.0]


def test_per_query_hit_matches_hit_at_k_mean():
    from casmi.ranking.evaluation import hit_at_k_from_ranks, per_query_hit

    ranks = pd.Series([1, 5, 30, np.nan])
    per_query = per_query_hit(ranks, k=5)
    assert math.isclose(per_query.mean(), hit_at_k_from_ranks(ranks, k=5))
    assert list(per_query) == [1.0, 1.0, 0.0, 0.0]
