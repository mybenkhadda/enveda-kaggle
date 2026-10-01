import numpy as np

from casmi.validation.shift import categorical_shift, numeric_shift, shift_report


def test_numeric_shift_zero_for_identical_distributions():
    values = np.linspace(0, 100, 500)
    result = numeric_shift(values, values, name="x")
    assert result["ks_stat"] == 0.0
    assert result["wasserstein"] == 0.0


def test_numeric_shift_detects_a_real_shift():
    rng = np.random.RandomState(0)
    train = rng.normal(loc=0, scale=1, size=2000)
    test = rng.normal(loc=5, scale=1, size=500)
    result = numeric_shift(train, test, name="x")
    assert result["ks_stat"] > 0.9
    assert result["wasserstein"] > 4.0


def test_numeric_shift_handles_nans_and_empty():
    result = numeric_shift([np.nan, np.nan], [1, 2, 3], name="x")
    assert result["n_train"] == 0
    assert np.isnan(result["ks_stat"])


def test_categorical_shift_zero_for_identical_distributions():
    values = ["a", "b", "a", "c"] * 10
    result = categorical_shift(values, values, name="cat")
    assert result["jensen_shannon_distance"] == 0.0
    assert result["n_categories_test_only"] == 0


def test_categorical_shift_flags_test_only_category():
    train = ["a", "a", "b"] * 10
    test = ["a", "c", "c"] * 10
    result = categorical_shift(train, test, name="cat")
    assert result["n_categories_test_only"] == 1
    assert result["jensen_shannon_distance"] > 0.0


def test_shift_report_shape_and_sort_order():
    train_df = {
        "mass": np.random.RandomState(1).normal(300, 10, 1000),
        "adduct": ["[M+H]+"] * 900 + ["[M-H]-"] * 100,
    }
    test_df = {
        "mass": np.random.RandomState(2).normal(400, 10, 200),
        "adduct": ["[M+H]+"] * 100 + ["[M+Na]+"] * 100,
    }
    import pandas as pd
    numeric_report, categorical_report = shift_report(
        pd.DataFrame(train_df), pd.DataFrame(test_df),
        numeric_cols=["mass"], categorical_cols=["adduct"],
    )
    assert len(numeric_report) == 1
    assert len(categorical_report) == 1
    assert numeric_report.loc[0, "ks_stat"] > 0.9
