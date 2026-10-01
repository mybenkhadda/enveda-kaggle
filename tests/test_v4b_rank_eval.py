"""v4b per-query evaluation: analytic expected-tie metrics vs brute-force permutations,
zero-candidate queries contribute RR=0, lexicographic ranking keys, B2b baseline ordering."""
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from casmi.ranking.baselines import b2b, b3a, b3b
from casmi.ranking.rank_eval import expected_tie_metrics, per_query_metrics, query_accounting, rank_by_keys, summarize


def _brute_expected(scores, truth_idx, k_rr=25, k_hit=(1, 5, 25)):
    """Enumerate every order consistent with 'higher score first, uniform within exact ties'
    (NaN = worst block) and average the truth's RR/hits -- independent of the analytic code."""
    s = [(-np.inf if np.isnan(v) else v) for v in scores]
    n = len(s)
    rrs, hits = [], {k: [] for k in k_hit}
    for perm in permutations(range(n)):
        if any(s[perm[i]] < s[perm[i + 1]] for i in range(n - 1)):
            continue  # not a valid tie-respecting order
        rank = perm.index(truth_idx) + 1
        rrs.append(1.0 / rank if rank <= k_rr else 0.0)
        for k in k_hit:
            hits[k].append(1.0 if rank <= k else 0.0)
    return float(np.mean(rrs)), {k: float(np.mean(v)) for k, v in hits.items()}


@pytest.mark.parametrize("scores,truth_idx", [
    ([0.5, 0.5, 0.5], 0),
    ([0.9, 0.5, 0.5, 0.5, 0.1], 2),
    ([0.3, 0.3, 0.8, 0.3], 1),
    ([np.nan, 0.2, np.nan, 0.2], 0),   # truth in the NaN tail block
    ([0.7, 0.2, 0.1], 0),              # no tie
    ([0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 5),
])
def test_expected_tie_metrics_match_brute_force(scores, truth_idx):
    df = pd.DataFrame({"query_id": "q", "s": scores, "is_true_candidate": [i == truth_idx for i in range(len(scores))]})
    pq = expected_tie_metrics(df, "s", ["q"], k_values=(1, 5, 25)).iloc[0]
    rr, hits = _brute_expected(scores, truth_idx)
    assert pq["rr"] == pytest.approx(rr, abs=1e-12)
    for k in (1, 5, 25):
        assert pq[f"hit_at_{k}"] == pytest.approx(hits[k], abs=1e-12)


def test_expected_tie_hit_at_1_block_of_three():
    df = pd.DataFrame({"query_id": "q", "s": [1.0, 1.0, 1.0], "is_true_candidate": [True, False, False]})
    pq = expected_tie_metrics(df, "s", ["q"]).iloc[0]
    assert pq["hit_at_1"] == pytest.approx(1 / 3) and pq["rr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


def test_zero_candidate_query_contributes_zero():
    df = pd.DataFrame({"query_id": ["a", "a"], "candidate_connectivity_key": ["T", "X"], "score": [2.0, 1.0],
                       "is_true_candidate": [True, False]})
    ranked = rank_by_keys(df, [("score", False), ("candidate_connectivity_key", True)])
    pq = per_query_metrics(ranked, ["a", "empty_query"])
    s = summarize(pq)
    assert s["n_queries"] == 2 and s["n_with_candidates"] == 1
    assert s["mrr_at_25"] == pytest.approx(0.5) and s["hit_at_1"] == pytest.approx(0.5)
    assert pq.set_index("query_id").loc["empty_query", "rr"] == 0.0


def test_truth_absent_from_nonempty_pool_is_zero():
    df = pd.DataFrame({"query_id": ["a"], "candidate_connectivity_key": ["X"], "score": [1.0], "is_true_candidate": [False]})
    pq = per_query_metrics(rank_by_keys(df, [("score", False)]), ["a"])
    assert pq["rr"].iloc[0] == 0.0 and np.isnan(pq["truth_rank"].iloc[0])


def test_expected_tie_zero_candidate_query():
    df = pd.DataFrame({"query_id": ["a"], "s": [1.0], "is_true_candidate": [True]})
    pq = expected_tie_metrics(df, "s", ["a", "b"])
    assert pq.set_index("query_id").loc["b", "rr"] == 0.0 and pq.set_index("query_id").loc["a", "rr"] == 1.0


def test_rank_by_keys_lexicographic_and_nan_last():
    df = pd.DataFrame({"query_id": "q", "candidate_connectivity_key": ["A", "B", "C", "D"],
                       "s": [0.5, 0.5, np.nan, 0.9], "ppm": [3.0, 1.0, 0.0, 9.0], "is_true_candidate": [False, True, False, False]})
    r = rank_by_keys(df, [("s", False), ("ppm", True), ("candidate_connectivity_key", True)]).set_index("candidate_connectivity_key")
    assert r["rank"].to_dict() == {"D": 1, "B": 2, "A": 3, "C": 4}


def test_duplicate_query_ids_rejected():
    df = pd.DataFrame({"query_id": ["a"], "rank": [1], "is_true_candidate": [True]})
    with pytest.raises(ValueError):
        per_query_metrics(df, ["a", "a"])


def test_query_accounting():
    df = pd.DataFrame({"query_id": ["a", "a", "b"], "is_true_candidate": [True, False, False]})
    acc = query_accounting(["a", "b", "c"], df)
    assert acc == {"n_queries_total": 3, "n_queries_with_candidates": 2, "n_queries_without_candidates": 1,
                   "n_queries_truth_in_pool": 1, "n_queries_truth_absent_from_nonempty_pool": 1,
                   "n_feature_rows_outside_query_list": 0}


def test_b2b_popularity_then_mass():
    df = pd.DataFrame({"query_id": "q", "candidate_connectivity_key": ["T", "D1", "D2"], "popularity_level": [5, 5, 2],
                       "abs_mass_error_ppm": [2.0, 1.0, 0.1], "is_true_candidate": [True, False, False]})
    pq = b2b(df, ["q"])
    assert pq["truth_rank"].iloc[0] == 2  # D1 ties T on popularity (>=5), wins on mass; D2 lower popularity


def test_b3a_uses_no_mass_but_b3b_does():
    df = pd.DataFrame({"query_id": "q", "candidate_connectivity_key": ["T", "D"], "peak_overlap_frac_max": [0.4, 0.4],
                       "abs_mass_error_ppm": [0.1, 9.0], "is_true_candidate": [True, False]})
    assert b3a(df, ["q"])["rr"].iloc[0] == pytest.approx(0.75)
    assert b3b(df, ["q"])["rr"].iloc[0] == pytest.approx(1.0)
