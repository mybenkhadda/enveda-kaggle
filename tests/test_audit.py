import math

import numpy as np
import pandas as pd
import pytest

from casmi.ranking.audit import (
    SPECTRAL_FEATURES, candidate_order_control, constant_spectral_feature_control,
    evaluate_missingness_shortcut, missingness_only_baseline, spectrum_mask_control, summarize_feature_missingness,
)


def _toy_mode_b_frame():
    # 2 queries x 3 candidates. Query q1's true candidate (row 0) is all-NaN (unseen-connectivity,
    # by construction). One FALSE candidate (row 2, q1's decoy) is ALSO all-NaN (simulating a
    # decoy that happens to belong to the held-out fold too) -- the exact pattern section 1.1
    # hypothesizes as a shortcut. Every other row has real spectral values.
    rows = [
        {"query_id": "q1", "candidate_connectivity_key": "TRUE1", "is_true_candidate": True, "fold": 0,
         "abs_mass_error_ppm": 2.0, "cosine_max": np.nan, "cosine_top3_mean": np.nan, "modified_cosine_max": np.nan,
         "modified_cosine_top3_mean": np.nan, "peak_overlap_frac_max": np.nan, "peak_overlap_frac_top3_mean": np.nan,
         "neutral_loss_cosine_max": np.nan, "neutral_loss_cosine_top3_mean": np.nan},
        {"query_id": "q1", "candidate_connectivity_key": "DECOY1", "is_true_candidate": False, "fold": 0,
         "abs_mass_error_ppm": 5.0, "cosine_max": 0.9, "cosine_top3_mean": 0.8, "modified_cosine_max": 0.7,
         "modified_cosine_top3_mean": 0.6, "peak_overlap_frac_max": 0.5, "peak_overlap_frac_top3_mean": 0.4,
         "neutral_loss_cosine_max": 0.3, "neutral_loss_cosine_top3_mean": 0.2},
        {"query_id": "q1", "candidate_connectivity_key": "DECOY2", "is_true_candidate": False, "fold": 0,
         "abs_mass_error_ppm": 10.0, "cosine_max": np.nan, "cosine_top3_mean": np.nan, "modified_cosine_max": np.nan,
         "modified_cosine_top3_mean": np.nan, "peak_overlap_frac_max": np.nan, "peak_overlap_frac_top3_mean": np.nan,
         "neutral_loss_cosine_max": np.nan, "neutral_loss_cosine_top3_mean": np.nan},
        {"query_id": "q2", "candidate_connectivity_key": "TRUE2", "is_true_candidate": True, "fold": 1,
         "abs_mass_error_ppm": 1.0, "cosine_max": np.nan, "cosine_top3_mean": np.nan, "modified_cosine_max": np.nan,
         "modified_cosine_top3_mean": np.nan, "peak_overlap_frac_max": np.nan, "peak_overlap_frac_top3_mean": np.nan,
         "neutral_loss_cosine_max": np.nan, "neutral_loss_cosine_top3_mean": np.nan},
        {"query_id": "q2", "candidate_connectivity_key": "DECOY3", "is_true_candidate": False, "fold": 1,
         "abs_mass_error_ppm": 3.0, "cosine_max": 0.95, "cosine_top3_mean": 0.9, "modified_cosine_max": 0.8,
         "modified_cosine_top3_mean": 0.7, "peak_overlap_frac_max": 0.6, "peak_overlap_frac_top3_mean": 0.5,
         "neutral_loss_cosine_max": 0.4, "neutral_loss_cosine_top3_mean": 0.3},
    ]
    return pd.DataFrame(rows)


class FakeModel:
    """Duck-typed stand-in for an already-trained `lgb.LGBMRanker`: scores rows independently
    from their own feature values, so it can exercise `.predict`-based controls without a real
    training dependency."""

    def __init__(self, feature_weights):
        self.feature_weights = feature_weights

    def predict(self, X):
        out = np.zeros(len(X))
        for feat, weight in self.feature_weights.items():
            out += X[feat].fillna(0.0).to_numpy() * weight
        return out


def test_spectral_features_excludes_mass():
    assert "abs_mass_error_ppm" not in SPECTRAL_FEATURES
    assert len(SPECTRAL_FEATURES) == 8


def test_summarize_feature_missingness_true_always_all_nan():
    df = _toy_mode_b_frame()
    augmented, summary = summarize_feature_missingness(df)
    assert augmented.loc[augmented["is_true_candidate"], "all_spectral_nan"].all()
    assert summary["p_all_nan_given_true"] == 1.0


def test_summarize_feature_missingness_detects_false_candidates_sharing_the_pattern():
    df = _toy_mode_b_frame()
    _, summary = summarize_feature_missingness(df)
    # 1 of 3 false candidates (DECOY2) is also all-NaN
    assert math.isclose(summary["p_all_nan_given_false"], 1 / 3)


def test_summarize_feature_missingness_fold_shortcut_flag():
    df = _toy_mode_b_frame()
    # per-query all-NaN fraction: q1 = 2/3, q2 = 1/1 -> median = 5/6, nowhere near 1/5
    _, summary = summarize_feature_missingness(df, n_folds=5, fold_tolerance=0.05)
    assert summary["fold_membership_shortcut_flag"] is False

    # a pool matching the median EXACTLY to 1/n_folds should raise the flag: 5 candidates, only
    # the true candidate (by construction) is all-NaN -> per-query fraction = 1/5
    matching = pd.DataFrame({
        "query_id": ["q1"] * 5, "is_true_candidate": [True, False, False, False, False],
    })
    for feat in SPECTRAL_FEATURES:
        matching[feat] = [np.nan, 0.5, 0.5, 0.5, 0.5]
    _, matching_summary = summarize_feature_missingness(matching, n_folds=5, fold_tolerance=0.05)
    assert matching_summary["fold_membership_shortcut_flag"] is True


def test_evaluate_missingness_shortcut_ranks_within_all_nan_subset_only():
    df = _toy_mode_b_frame()
    augmented, _ = summarize_feature_missingness(df)
    result = evaluate_missingness_shortcut(augmented)
    # within the all-NaN subset: q1 has TRUE1 (ppm=2.0, best) and DECOY2 (ppm=10.0) -> true ranks 1st
    # q2 has only TRUE2 in the subset -> ranks 1st trivially
    assert result["n_queries_total"] == 2
    assert math.isclose(result["end_to_end"]["mrr_at_25"], 1.0)


def test_evaluate_missingness_shortcut_uses_explicit_denominator_not_just_pool_queries():
    # a query entirely absent from the pair table (e.g. zero unseen-connectivity candidates)
    # must still count as a miss (RR=0) in the denominator when the caller supplies the real
    # full query list -- silently deriving all_query_ids from the pair table itself would drop
    # it and inflate the MRR, exactly the "never silently drop a query" bug this guards against.
    df = _toy_mode_b_frame()
    augmented, _ = summarize_feature_missingness(df)

    result_no_explicit_denominator = evaluate_missingness_shortcut(augmented)
    assert result_no_explicit_denominator["n_queries_total"] == 2  # only q1, q2 -- silently narrow

    full_denominator = ["q1", "q2", "q3_missing_entirely"]
    result_explicit = evaluate_missingness_shortcut(augmented, all_query_ids=full_denominator)
    assert result_explicit["n_queries_total"] == 3
    assert result_explicit["end_to_end"]["mrr_at_25"] < result_no_explicit_denominator["end_to_end"]["mrr_at_25"]


def test_constant_spectral_feature_control_collapses_spectral_only_model():
    # 4 queries, 2 candidates each. The true candidate genuinely has a higher real cosine_max
    # than its decoy (so the real-feature model finds it every time), but which ROW comes first
    # alternates -- so once constant-filling erases the real spectral gap, tie-breaking (stable,
    # by original row order) only rescues the true candidate in half the queries. A model that
    # was actually reading spectral evidence must therefore score noticeably worse once that
    # evidence is replaced by one shared constant.
    rows = []
    for i, true_first in enumerate([True, False, True, False]):
        qid = f"q{i}"
        true_row = {"query_id": qid, "is_true_candidate": True, "fold": 0, "cosine_max": 0.9}
        decoy_row = {"query_id": qid, "is_true_candidate": False, "fold": 0, "cosine_max": 0.1}
        rows.extend([true_row, decoy_row] if true_first else [decoy_row, true_row])
    df = pd.DataFrame(rows)
    for feat in SPECTRAL_FEATURES:
        if feat != "cosine_max":
            df[feat] = np.nan
    df["abs_mass_error_ppm"] = 1.0

    # a model that reads ONLY spectral evidence (zero weight on mass) -- constant-filling every
    # spectral feature to the SAME value must therefore erase all real score variation.
    model = FakeModel({"cosine_max": 1.0})
    fold_models = {0: model}
    result = constant_spectral_feature_control(fold_models, df, model_features=["cosine_max"], constant_value=0.0)
    assert math.isclose(result["real_features_mrr_at_25"], 1.0)
    assert result["constant_spectral_features_mrr_at_25"] < result["real_features_mrr_at_25"]


def test_constant_spectral_feature_control_never_uses_nan_for_the_constant():
    df = _toy_mode_b_frame()
    model = FakeModel({"cosine_max": 1.0})
    fold_models = {0: model, 1: model}
    # constant_value=0.0 (default) must leave no NaNs in the spectral columns of the scored frame
    constant_spectral_feature_control(fold_models, df, model_features=["cosine_max"], constant_value=0.0)
    assert not df[SPECTRAL_FEATURES].eq(0.0).all().all()  # original df untouched (control copies)


def test_candidate_order_control_passes_for_row_independent_model():
    df = _toy_mode_b_frame()
    model = FakeModel({"abs_mass_error_ppm": -1.0})
    result = candidate_order_control(model, df, model_features=["abs_mass_error_ppm"])
    assert result["passed"] is True
    assert result["max_abs_diff"] < 1e-9


def test_spectrum_mask_control_detects_sensitivity():
    df = _toy_mode_b_frame()
    model = FakeModel({"cosine_max": 1.0})
    result = spectrum_mask_control(model, df, model_features=["cosine_max"])
    # DECOY1 and DECOY3 have real cosine_max evidence -> masking to NaN must shift their score
    assert result["n_sampled"] == 2
    assert result["sensitive"] is True


def test_spectrum_mask_control_reports_no_sensitivity_for_mass_only_model():
    df = _toy_mode_b_frame()
    model = FakeModel({"abs_mass_error_ppm": -1.0})  # ignores spectral features entirely
    result = spectrum_mask_control(model, df, model_features=["abs_mass_error_ppm"])
    assert result["sensitive"] is False


def test_missingness_only_baseline_requires_lightgbm():
    lgb = pytest.importorskip("lightgbm")
    df = _toy_mode_b_frame()
    result = missingness_only_baseline(df, lgbm_params=dict(
        objective="lambdarank", metric="ndcg", n_estimators=5, learning_rate=0.3,
        num_leaves=4, min_child_samples=1, random_state=42, verbosity=-1,
    ))
    assert result["features_used"] == ["abs_mass_error_ppm", "spectral_missing_count", "all_spectral_nan"]
    assert 0.0 <= result["mrr_at_25"] <= 1.0
