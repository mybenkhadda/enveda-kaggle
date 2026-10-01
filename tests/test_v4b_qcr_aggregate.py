"""v4b QCR -> candidate features: top3_mean contract (no NaN leakage of the reference count),
censored B2b popularity ordering, and training-only reference dropout."""
import numpy as np
import pandas as pd
import pytest

from casmi.qcr.aggregate import (
    aggregate_deterministic, aggregate_reference_dropout, assert_no_top3_missingness_leak, eligible_reference_counts,
    features_without_references, popularity_level, select_dropout_references, top3_sanity_table,
)
from casmi.ranking.features import aggregate_pair_scores_from_values

METRICS = ("cosine", "modified_cosine", "peak_overlap_frac", "neutral_loss_cosine")


def _qcr(rows):
    """rows: (query, cand, ref, compat_rank, cosine, eligible, accepted, is_true)"""
    recs = []
    for q, c, r, rank, cos, elig, acc, truth in rows:
        recs.append({"query_id": q, "candidate_key": c, "ref_spectrum_id": r, "compat_rank": rank, "tier": "T4",
                     "cosine": cos, "modified_cosine": cos / 2, "peak_overlap_frac": cos / 3, "neutral_loss_cosine": cos / 4,
                     "eligible_mirror_aware": elig, "accepted_mirror_aware": acc, "is_true": truth})
    return pd.DataFrame(recs)


def _pool(pairs):
    return pd.DataFrame([{"query_id": q, "candidate_connectivity_key": c, "abs_mass_error_ppm": 1.0, "is_true_candidate": t}
                         for q, c, t in pairs])


QCR = _qcr([
    ("q1", "A", "r1", 1, 0.9, True, True, True),                                   # 1 ref
    ("q1", "B", "r2", 1, 0.2, True, True, False), ("q1", "B", "r3", 2, 0.6, True, True, False),  # 2 refs
    ("q1", "C", "r4", 1, 0.1, True, True, False), ("q1", "C", "r5", 2, 0.5, True, True, False),
    ("q1", "C", "r6", 3, 0.3, True, True, False), ("q1", "C", "r7", 4, 0.8, True, True, False),  # 4 refs
    ("q1", "E", "r8", 1, 0.7, False, False, False),                                 # walked but ineligible
])
POOL = _pool([("q1", "A", True), ("q1", "B", False), ("q1", "C", False), ("q1", "D", False), ("q1", "E", False)])


def test_top3_mean_one_two_many_refs():
    f = aggregate_deterministic(QCR, POOL).set_index("candidate_connectivity_key")
    assert f.loc["A", "cosine_top3_mean"] == pytest.approx(0.9)
    assert f.loc["B", "cosine_top3_mean"] == pytest.approx((0.2 + 0.6) / 2)
    assert f.loc["C", "cosine_top3_mean"] == pytest.approx((0.8 + 0.5 + 0.3) / 3)
    assert f.loc["C", "cosine_max"] == pytest.approx(0.8)
    assert f.loc["A", "n_reference_spectra"] == 1 and f.loc["C", "n_reference_spectra"] == 4


def test_zero_refs_nan_and_everything_else_finite():
    f = aggregate_deterministic(QCR, POOL).set_index("candidate_connectivity_key")
    for key in ("D", "E"):
        assert f.loc[key, "n_reference_spectra"] == 0 and not f.loc[key, "has_reference_spectrum"]
        assert all(np.isnan(f.loc[key, f"{m}_top3_mean"]) for m in METRICS)
    for key in ("A", "B", "C"):
        assert all(np.isfinite(f.loc[key, f"{m}_top3_mean"]) for m in METRICS)
    assert_no_top3_missingness_leak(f.reset_index())


def test_matches_reference_aggregation_implementation():
    f = aggregate_deterministic(QCR, POOL).set_index("candidate_connectivity_key")
    g = QCR[QCR["candidate_key"] == "C"]
    ref = aggregate_pair_scores_from_values(*(g[m].tolist() for m in METRICS))
    for m in METRICS:
        for agg in ("max", "top3_mean"):
            assert f.loc["C", f"{m}_{agg}"] == pytest.approx(ref[f"{m}_{agg}"], abs=1e-12)


def test_k_limited_uses_compat_order():
    f = aggregate_deterministic(QCR, POOL, k=1).set_index("candidate_connectivity_key")
    assert f.loc["C", "cosine_max"] == pytest.approx(0.1)  # compat_rank 1, not the best value
    assert f.loc["C", "n_reference_spectra"] == 1
    f3 = aggregate_deterministic(QCR, POOL, k=3).set_index("candidate_connectivity_key")
    assert f3.loc["C", "cosine_top3_mean"] == pytest.approx((0.1 + 0.5 + 0.3) / 3)


def test_pool_row_order_and_count_preserved():
    f = aggregate_deterministic(QCR, POOL)
    assert f["candidate_connectivity_key"].tolist() == POOL["candidate_connectivity_key"].tolist()


def test_missingness_assertion_catches_false_nan():
    f = aggregate_deterministic(QCR, POOL)
    bad = f.copy()
    bad.loc[bad["candidate_connectivity_key"] == "B", "cosine_top3_mean"] = np.nan  # the forbidden "<3 refs -> NaN"
    with pytest.raises(AssertionError):
        assert_no_top3_missingness_leak(bad)


def test_sanity_table_bins():
    t = top3_sanity_table(aggregate_deterministic(QCR, POOL)).set_index("ref_count_bin")
    assert t.loc["0", "nan_frac_cosine_top3_mean"] == 1.0
    assert t.loc["1", "nan_frac_cosine_top3_mean"] == 0.0 and t.loc["1", "max_absdiff_top3_vs_max_cosine"] == 0.0
    assert t.loc["2", "nan_frac_cosine_top3_mean"] == 0.0 and t.loc[">=3", "nan_frac_cosine_top3_mean"] == 0.0


def test_features_without_references():
    f = features_without_references(_pool([("q9", "Z", False)]))
    assert f["n_reference_spectra"].tolist() == [0] and np.isnan(f["cosine_max"].iloc[0])


# ---- B2b censored popularity -------------------------------------------------------------------

def test_popularity_level_censoring():
    lv = popularity_level([0, 1, 2, 3, 4, 5, 5, 7], [False, False, False, False, False, True, False, False])
    assert lv.tolist() == [0, 1, 2, 3, 4, 5, 5, 5]  # exact 7 and censored >=5 share the top level


def test_popularity_ordering_is_monotone():
    counts = [0, 1, 2, 3, 4, 5]
    sat = [False] * 5 + [True]
    lv = popularity_level(counts, sat)
    assert all(a < b for a, b in zip(lv[:-1], lv[1:]))


def test_eligible_reference_counts_exact_vs_censored():
    rows = [("q", "S", f"s{i}", i + 1, 0.5, True, i < 5, False) for i in range(5)]      # 5 accepted, not exhausted -> censored
    rows += [("q", "X", f"x{i}", i + 1, 0.5, True, i < 5, False) for i in range(7)]     # 5 accepted, exhausted -> exact 7
    rows += [("q", "T", "t0", 1, 0.5, True, True, True), ("q", "T", "t1", 2, 0.5, False, False, True)]  # exact 1
    qcr = _qcr(rows)
    pairs = pd.DataFrame({"query_id": "q", "candidate_key": ["S", "X", "T"], "exhausted": [False, True, True]})
    pool = _pool([("q", "S", False), ("q", "X", False), ("q", "T", True), ("q", "N", False)])
    c = eligible_reference_counts(qcr, pairs, pool).set_index("candidate_connectivity_key")
    assert c.loc["S", "eligible_reference_saturated"] and not c.loc["S", "eligible_reference_count_is_exact"]
    assert c.loc["X", "eligible_reference_count"] == 7 and c.loc["X", "eligible_reference_count_is_exact"]
    assert c.loc["T", "eligible_reference_count"] == 1 and c.loc["N", "eligible_reference_count"] == 0
    assert c.loc["S", "popularity_level"] == c.loc["X", "popularity_level"] == 5
    assert c.loc["T", "popularity_level"] == 1 and c.loc["N", "popularity_level"] == 0


# ---- reference dropout ---------------------------------------------------------------------------

def _dropout_qcr(n_truth=8, n_decoy=8):
    rows = [("q", "T", f"t{i}", i + 1, 0.1 * (i + 1), True, i < 5, True) for i in range(n_truth)]
    rows += [("q", "D", f"d{i}", i + 1, 0.05 * (i + 1), True, i < 5, False) for i in range(n_decoy)]
    rows += [("q", "D", "d_inel", 99, 0.99, False, False, False)]  # ineligible: must never be sampled
    return _qcr(rows)


def test_dropout_samples_only_eligible_refs():
    for seed in range(20):
        kept, log = select_dropout_references(_dropout_qcr(), seed=seed)
        assert kept["eligible_mirror_aware"].all() and "d_inel" not in set(kept["ref_spectrum_id"])
        assert (kept.groupby("candidate_key").size() == log.set_index("candidate_key")["n_sampled"]).all()


def test_dropout_deterministic_under_seed_and_row_order():
    q = _dropout_qcr()
    a, _ = select_dropout_references(q, seed=7)
    b, _ = select_dropout_references(q.sample(frac=1.0, random_state=3), seed=7)
    assert sorted(a["ref_spectrum_id"]) == sorted(b["ref_spectrum_id"])


def test_dropout_can_produce_k_1_3_5_and_respects_cap():
    ks = set()
    for seed in range(60):
        _, log = select_dropout_references(_dropout_qcr(), seed=seed)
        ks |= set(log["dropout_k"])
        assert (log["n_sampled"] == np.minimum(log["dropout_k"], log["n_eligible_in_qcr"])).all()
    assert ks == {1, 3, 5}
    _, log = select_dropout_references(_dropout_qcr(n_truth=2), seed=0, k_choices=(5,), k_probs=[1.0])
    assert log.set_index("candidate_key").loc["T", "n_sampled"] == 2


def test_dropout_is_random_not_top_compat():
    picked_rank1_only = True
    for seed in range(40):
        kept, log = select_dropout_references(_dropout_qcr(), seed=seed, k_choices=(1,), k_probs=[1.0])
        if kept.loc[kept["candidate_key"] == "T", "compat_rank"].iloc[0] != 1:
            picked_rank1_only = False
    assert not picked_rank1_only


def test_dropout_identical_for_truth_and_decoy():
    """Swapping the is_true labels must not change what gets sampled."""
    q = _dropout_qcr()
    swapped = q.assign(is_true=~q["is_true"])
    a, _ = select_dropout_references(q, seed=11)
    b, _ = select_dropout_references(swapped, seed=11)
    assert sorted(a["ref_spectrum_id"]) == sorted(b["ref_spectrum_id"])


def test_dropout_features_have_no_false_nan():
    pool = _pool([("q", "T", True), ("q", "D", False)])
    feats, _ = aggregate_reference_dropout(_dropout_qcr(), pool, seed=3)
    assert_no_top3_missingness_leak(feats)
    assert feats["n_reference_spectra"].between(1, 5).all()


def test_bad_k_probs_rejected():
    with pytest.raises(ValueError):
        select_dropout_references(_dropout_qcr(), k_choices=(1, 3, 5), k_probs=[0.5, 0.5])
