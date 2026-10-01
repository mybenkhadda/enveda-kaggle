"""Validate casmi.qcr.expected_rr.expected_reciprocal_rank (a closed-form mean-of-1/r formula)
against a genuinely independent, brute-force exhaustive permutation enumeration -- see spec
section 49/67: the analytic helper must match exhaustive enumeration for small synthetic tie
blocks, not just be internally self-consistent.
"""
from itertools import permutations

import pandas as pd
import pytest

from casmi.qcr.expected_rr import expected_reciprocal_rank, expected_tie_mrr


def _exhaustive_expected_rr(rank_min, rank_max, k=None):
    """Reference implementation used ONLY in this test: enumerates every permutation of the
    tie block's `block_size = rank_max - rank_min + 1` members (item 0 is "the true
    candidate"), and averages the true candidate's reciprocal rank across ALL permutations --
    genuinely independent of the closed-form formula under test (no shared code path)."""
    block_size = rank_max - rank_min + 1
    items = list(range(block_size))
    total, count = 0.0, 0
    for perm in permutations(items):
        position_in_block = perm.index(0)
        absolute_rank = rank_min + position_in_block
        total += (1.0 / absolute_rank) if (k is None or absolute_rank <= k) else 0.0
        count += 1
    return total / count


@pytest.mark.parametrize("rank_min,rank_max", [(1, 1), (1, 2), (1, 3), (2, 4), (3, 4), (1, 5)])
def test_expected_reciprocal_rank_matches_exhaustive_enumeration(rank_min, rank_max):
    analytic = expected_reciprocal_rank(rank_min, rank_max)
    exhaustive = _exhaustive_expected_rr(rank_min, rank_max)
    assert analytic == pytest.approx(exhaustive, abs=1e-12)


@pytest.mark.parametrize("rank_min,rank_max,k", [(1, 3, 2), (3, 6, 4), (1, 5, 0)])
def test_expected_reciprocal_rank_matches_exhaustive_enumeration_with_k_cutoff(rank_min, rank_max, k):
    analytic = expected_reciprocal_rank(rank_min, rank_max, k=k)
    exhaustive = _exhaustive_expected_rr(rank_min, rank_max, k=k)
    assert analytic == pytest.approx(exhaustive, abs=1e-12)


def test_no_tie_reduces_to_ordinary_reciprocal_rank():
    assert expected_reciprocal_rank(7, 7) == pytest.approx(1.0 / 7)


def test_rank_max_less_than_rank_min_raises():
    with pytest.raises(ValueError):
        expected_reciprocal_rank(5, 3)


def test_expected_tie_mrr_gives_zero_rr_for_queries_without_a_scored_true_candidate():
    pair_df = pd.DataFrame({
        "query_id": ["q1", "q1", "q2"],
        "score": [0.9, 0.5, 0.3],
        "is_true_candidate": [True, False, False],  # q2 has no true candidate scored at all
    })
    mrr = expected_tie_mrr(pair_df, "score", all_query_ids=["q1", "q2", "q3"])
    # q1: true candidate alone at rank 1 -> RR=1.0; q2, q3: no true candidate -> RR=0
    assert mrr == pytest.approx(1.0 / 3)


def test_expected_tie_mrr_tied_true_candidate_uses_expected_rr_of_its_block():
    pair_df = pd.DataFrame({
        "query_id": ["q1", "q1", "q1"],
        "score": [0.9, 0.9, 0.9],  # 3-way tie, true candidate is one of them
        "is_true_candidate": [True, False, False],
    })
    mrr = expected_tie_mrr(pair_df, "score", all_query_ids=["q1"])
    assert mrr == pytest.approx((1 / 1 + 1 / 2 + 1 / 3) / 3)
