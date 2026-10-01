import math

import numpy as np

from casmi.validation.bootstrap import paired_bootstrap_metric_difference, paired_bootstrap_report


def test_identical_arrays_give_zero_diff_and_tight_ci():
    values = np.array([1.0, 0.5, 0.0, 1.0, 0.5] * 20)
    result = paired_bootstrap_metric_difference(values, values, n_bootstrap=1000, seed=1)
    assert result["mean_diff"] == 0.0
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_clearly_better_method_has_positive_ci_excluding_zero():
    rng = np.random.RandomState(0)
    a = rng.uniform(0.7, 1.0, 200)  # clearly better
    b = rng.uniform(0.0, 0.3, 200)  # clearly worse
    result = paired_bootstrap_metric_difference(a, b, n_bootstrap=2000, seed=1)
    assert result["mean_diff"] > 0
    assert result["ci_low"] > 0  # CI should exclude zero given the huge, consistent gap
    assert result["prob_a_greater"] > 0.99


def test_similar_methods_have_ci_spanning_zero():
    rng = np.random.RandomState(0)
    a = rng.normal(0.5, 0.3, 30)
    b = a + rng.normal(0, 0.01, 30)  # nearly identical, tiny noise
    result = paired_bootstrap_metric_difference(a, b, n_bootstrap=2000, seed=1)
    assert result["ci_low"] < 0 < result["ci_high"]


def test_mismatched_length_raises():
    try:
        paired_bootstrap_metric_difference([1.0, 2.0], [1.0], n_bootstrap=100)
        assert False, "should reject mismatched paired lengths"
    except ValueError:
        pass


def test_empty_input_returns_nan_not_error():
    result = paired_bootstrap_metric_difference([], [], n_bootstrap=100)
    assert math.isnan(result["mean_diff"])


def test_paired_bootstrap_report_shape():
    rng = np.random.RandomState(0)
    a = rng.uniform(0.5, 1.0, 50)
    b = rng.uniform(0.0, 0.5, 50)
    report = paired_bootstrap_report("mrr_at_25", "method_a", a, "method_b", b, n_bootstrap=500, seed=1)
    assert report["metric"] == "mrr_at_25"
    assert report["significant_at_95"]
    assert report["mean_a"] > report["mean_b"]
