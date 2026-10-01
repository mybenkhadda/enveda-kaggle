import numpy as np
import pandas as pd

from casmi.candidates.adaptive import (
    apply_adaptive_mass_windows,
    assert_no_fold_leakage,
    evaluate_adaptive_policy_foldwise,
    fit_adaptive_mass_windows,
)


def _queries_with_error(n_per_group=50, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for adduct, scale in [("[M+H]+", 5.0), ("[M-H]-", 20.0)]:
        errors = rng.exponential(scale, n_per_group)
        for i, e in enumerate(errors):
            rows.append({"adduct": adduct, "target_abs_mass_error_ppm": e, "fold": i % 5})
    return pd.DataFrame(rows)


def test_fit_adaptive_mass_windows_uses_group_quantile():
    df = _queries_with_error()
    policy = fit_adaptive_mass_windows(df, ["adduct"], target_quantile=0.995, min_group_size=10)
    tight = policy.set_index("adduct").loc["[M+H]+", "tolerance_ppm"]
    loose = policy.set_index("adduct").loc["[M-H]-", "tolerance_ppm"]
    assert tight < loose  # the wider-scale exponential group should get a looser tolerance
    assert not policy["used_fallback"].any()  # both groups have 50 >= min_group_size=10 rows


def test_fit_adaptive_mass_windows_falls_back_for_small_groups():
    df = pd.DataFrame({"adduct": ["[M+K]+"] * 5, "target_abs_mass_error_ppm": [1.0, 2.0, 3.0, 4.0, 5.0]})
    policy = fit_adaptive_mass_windows(df, ["adduct"], min_group_size=10, fallback_ppm=42.0)
    assert policy.loc[0, "used_fallback"]
    assert policy.loc[0, "tolerance_ppm"] == 42.0


def test_fit_adaptive_mass_windows_clips_to_bounds():
    df = pd.DataFrame({"adduct": ["[M+H]+"] * 20, "target_abs_mass_error_ppm": [1000.0] * 20})
    policy = fit_adaptive_mass_windows(df, ["adduct"], min_group_size=5, max_ppm=50.0)
    assert policy.loc[0, "tolerance_ppm"] == 50.0


def test_apply_adaptive_mass_windows_unknown_group_gets_fallback():
    policy = pd.DataFrame({"adduct": ["[M+H]+"], "tolerance_ppm": [7.0]})
    queries = pd.DataFrame({"adduct": ["[M+H]+", "[M+Weird]+"]})
    out = apply_adaptive_mass_windows(queries, policy, ["adduct"], fallback_ppm=42.0)
    assert list(out["candidate_tolerance_ppm"]) == [7.0, 42.0]


def test_evaluate_adaptive_policy_foldwise_never_uses_own_fold():
    df = _queries_with_error()
    per_query, fold_policies = evaluate_adaptive_policy_foldwise(
        df, ["adduct"], target_quantile=0.995, min_group_size=10,
    )
    assert len(per_query) == len(df)
    assert set(fold_policies.keys()) == {0, 1, 2, 3, 4}
    # this is the actual leakage assertion the spec requires -- not just "it ran without error"
    assert_no_fold_leakage(fold_policies, df, ["adduct"])


def test_evaluate_adaptive_policy_foldwise_produces_different_policies_per_fold():
    df = _queries_with_error(n_per_group=30, seed=1)
    _, fold_policies = evaluate_adaptive_policy_foldwise(df, ["adduct"], min_group_size=5)
    tol_fold_0 = fold_policies[0].set_index("adduct")["tolerance_ppm"]
    tol_fold_1 = fold_policies[1].set_index("adduct")["tolerance_ppm"]
    # fold 0's policy was fit excluding fold 0's own rows, fold 1's excluding fold 1's -- since
    # they exclude DIFFERENT rows, the fitted quantiles should generally differ (not asserting
    # exact values, just that they aren't trivially identical, which would suggest both were
    # fit on the same -- i.e. the full, leaked -- data).
    assert not tol_fold_0.equals(tol_fold_1)


def test_apply_adaptive_mass_windows_preserves_original_index():
    # a non-trivial (non-RangeIndex-starting-at-0) index is exactly the shape
    # evaluate_adaptive_policy_foldwise produces internally (fold subsets of a larger frame) --
    # a merge-based implementation would silently reset this and misalign downstream columns.
    policy = pd.DataFrame({"adduct": ["[M+H]+", "[M-H]-"], "tolerance_ppm": [5.0, 20.0]})
    queries = pd.DataFrame({"adduct": ["[M-H]-", "[M+H]+", "[M+H]+"]}, index=[42, 7, 99])
    out = apply_adaptive_mass_windows(queries, policy, ["adduct"])
    assert list(out.index) == [42, 7, 99]
    assert list(out["candidate_tolerance_ppm"]) == [20.0, 5.0, 5.0]


def test_evaluate_adaptive_policy_foldwise_keeps_rows_aligned_with_their_own_data():
    # regression test for the index-scrambling bug: build queries where each row's OTHER
    # column values (not just adduct/fold) must stay attached to the same row after fold-wise
    # policy application -- a merge-then-concat-then-sort_index implementation would shuffle
    # these across folds.
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "adduct": rng.choice(["[M+H]+", "[M-H]-"], n),
        "fold": rng.randint(0, 5, n),
        "target_abs_mass_error_ppm": rng.uniform(0, 10, n),
        "row_marker": np.arange(n),  # a value that must survive perfectly aligned to its row
    })
    out, _ = evaluate_adaptive_policy_foldwise(df, ["adduct"], min_group_size=5)
    # every row_marker must map back to exactly the original row's own adduct/fold/error
    for idx in df.index:
        assert out.loc[idx, "row_marker"] == df.loc[idx, "row_marker"]
        assert out.loc[idx, "adduct"] == df.loc[idx, "adduct"]
        assert out.loc[idx, "fold"] == df.loc[idx, "fold"]
        assert out.loc[idx, "target_abs_mass_error_ppm"] == df.loc[idx, "target_abs_mass_error_ppm"]
