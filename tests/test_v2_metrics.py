"""v2 candidate / ranking metrics, PROVISIONAL composite, stratified reporting, ranker grouping and
Top-25 connectivity dedup (existing helpers)."""
import math

import numpy as np
import pandas as pd
import pytest

from casmi.validation.metrics import candidate_stage_metrics, composite_mrr, mrr_at_k, ranking_metrics
from casmi.validation.reporting import stratified_table


def test_ranking_metrics_hand_computed():
    pq = pd.DataFrame({"truth_rank": [1, 2, 30, np.nan]})
    m = ranking_metrics(pq)
    assert m["mrr_at_25"] == pytest.approx((1 + 0.5 + 0 + 0) / 4)
    assert m["hit_at_1"] == 0.25 and m["hit_at_5"] == 0.5 and m["hit_at_25"] == 0.5
    assert m["miss_rate"] == 0.25 and m["n_ranked"] == 3
    assert m["median_truth_rank"] == 2 and m["mean_truth_rank"] == pytest.approx(11)


def test_rank_must_be_one_based():
    with pytest.raises(ValueError):
        ranking_metrics(pd.DataFrame({"truth_rank": [0]}))


def test_candidate_stage_metrics():
    pq = pd.DataFrame({"truth_rank": [1, 30, 200, np.nan], "pool_size": [10, 100, 1000, 0]})
    m = candidate_stage_metrics(pq, k_values=(25, 100, 500))
    assert m["recall_all"] == 0.75 and m["truth_missing_rate"] == 0.25
    assert m["recall_at_25"] == 0.25 and m["recall_at_100"] == 0.5 and m["recall_at_500"] == 0.75
    assert m["pool_size_median"] == 55 and m["zero_pool_rate"] == 0.25


def test_composite_is_provisional_and_never_reweights():
    c = composite_mrr({"C1": 0.8, "C2": 0.2, "C3": 0.0})
    assert c["label"] == "PROVISIONAL"
    assert c["composite_mrr_at_25"] == pytest.approx(0.15 * 0.8 + 0.6 * 0.2)
    assert math.isnan(composite_mrr({"C1": 0.8, "C2": 0.2})["composite_mrr_at_25"])
    assert composite_mrr({"C1": 1, "C2": 1, "C3": 1}, {"C1": 1, "C2": 0, "C3": 0})["composite_mrr_at_25"] == 1


def test_stratified_table_keeps_all_rows():
    pq = pd.DataFrame({"truth_rank": [1, 2, np.nan, 1], "regime": ["C1", "C2", "C2", "C1"], "adduct": ["a", "a", "b", "b"]})
    t = stratified_table(pq, ranking_metrics, strata=("adduct",))
    all_row = t[(t.regime == "ALL") & (t.stratum == "ALL")].iloc[0]
    assert all_row["n_queries"] == 4
    assert set(t.regime) == {"ALL", "C1", "C2"}
    assert t[(t.regime == "C2") & (t.stratum == "adduct") & (t.value == "b")]["miss_rate"].iloc[0] == 1.0


def test_top25_connectivity_dedup_in_mrr():
    # a repeated connectivity must not take two rank slots, and rank > 25 never counts
    assert mrr_at_k([["A", "A", "B"]], ["B"], k=25) == pytest.approx(0.5)
    assert mrr_at_k([[f"X{i}" for i in range(25)] + ["T"]], ["T"], k=25) == 0.0


def test_ranker_groups_are_contiguous():
    from casmi.ranking.lambdamart import make_groups
    df = pd.DataFrame({"query_id": ["b", "a", "b", "a", "c"], "x": range(5)})
    d, sizes = make_groups(df)
    assert sizes.tolist() == [2, 2, 1] and d["query_id"].tolist() == ["a", "a", "b", "b", "c"]
