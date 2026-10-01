import pandas as pd
import pytest

from casmi.ranking.feature_policy import (
    DEFAULT_PROHIBITED_FEATURES, SPECTRAL_RANKING_FEATURE_POLICY, FeaturePolicy, assert_no_prohibited_features,
)


def test_feature_policy_rejects_overlap_between_model_and_prohibited():
    with pytest.raises(ValueError):
        FeaturePolicy(model_features=["cosine_max"], prohibited_features=["cosine_max"])


def test_feature_policy_rejects_overlap_between_model_and_diagnostic():
    with pytest.raises(ValueError):
        FeaturePolicy(model_features=["cosine_max"], diagnostic_features=["cosine_max"])


def test_feature_policy_allows_disjoint_lists():
    policy = FeaturePolicy(model_features=["cosine_max"], diagnostic_features=["n_reference_spectra"],
                            prohibited_features=["has_reference_spectrum"])
    assert policy.model_features == ["cosine_max"]


def test_assert_no_prohibited_features_passes_when_clean():
    df = pd.DataFrame({"cosine_max": [0.5], "abs_mass_error_ppm": [1.0]})
    policy = FeaturePolicy(prohibited_features=DEFAULT_PROHIBITED_FEATURES)
    assert_no_prohibited_features(df, policy)  # should not raise


def test_assert_no_prohibited_features_raises_on_violation():
    df = pd.DataFrame({"cosine_max": [0.5], "has_reference_spectrum": [True]})
    policy = FeaturePolicy(prohibited_features=DEFAULT_PROHIBITED_FEATURES)
    with pytest.raises(AssertionError, match="has_reference_spectrum"):
        assert_no_prohibited_features(df, policy)


def test_default_prohibited_features_cover_reference_availability_fields():
    for f in ("has_reference_spectrum", "n_reference_spectra", "candidate_source", "is_train_library"):
        assert f in DEFAULT_PROHIBITED_FEATURES


def test_spectral_ranking_feature_policy_shared_by_notebooks_04_and_09():
    assert set(SPECTRAL_RANKING_FEATURE_POLICY.model_features) == {
        "abs_mass_error_ppm", "cosine_max", "cosine_top3_mean", "modified_cosine_max", "modified_cosine_top3_mean",
        "peak_overlap_frac_max", "peak_overlap_frac_top3_mean", "neutral_loss_cosine_max", "neutral_loss_cosine_top3_mean",
    }
    assert set(SPECTRAL_RANKING_FEATURE_POLICY.diagnostic_features) == {"n_reference_spectra", "has_reference_spectrum"}
    assert SPECTRAL_RANKING_FEATURE_POLICY.prohibited_features == DEFAULT_PROHIBITED_FEATURES
