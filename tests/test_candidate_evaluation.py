import numpy as np
import pandas as pd

from casmi.candidates.diagnostics import (
    build_failure_table,
    candidate_density_by_group,
    error_percentile_report,
    fold_stability,
    mass_bin_report,
    subgroup_report,
    verify_true_structure_in_library,
)
from casmi.candidates.evaluation import (
    add_target_mass_error,
    classify_decision,
    query_candidate_summary,
    select_operating_tolerance,
    tolerance_sweep,
    tolerance_tradeoff_deltas,
)
from casmi.candidates.mass_index import MassIndex


def _toy_index():
    # 100 evenly spaced structures around 300 Da, 0.001 Da apart -- lets us pick queries with
    # a known, exact target mass error.
    masses = [300.0 + 0.001 * i for i in range(100)]
    keys = [f"key_{i}" for i in range(100)]
    return MassIndex(keys, masses)


def test_add_target_mass_error_zero_for_exact_match():
    queries = pd.DataFrame({"neutral_mass": [300.0], "exact_mass_true": [300.0]})
    out = add_target_mass_error(queries)
    assert out["target_abs_mass_error_ppm"].iloc[0] == 0.0


def test_add_target_mass_error_nan_for_missing_mass():
    queries = pd.DataFrame({"neutral_mass": [np.nan], "exact_mass_true": [300.0]})
    out = add_target_mass_error(queries)
    assert np.isnan(out["target_abs_mass_error_ppm"].iloc[0])


def test_query_candidate_summary_counts_and_target_error():
    index = _toy_index()
    queries = pd.DataFrame({
        "query_id": ["q1", "q2"],
        "neutral_mass": [300.0, 300.0],
        "exact_mass_true": [300.0, 300.0],
    })
    summary = query_candidate_summary(queries, index, tolerance_grid=[1, 10], true_mass_col="exact_mass_true")

    assert "candidate_count_1ppm" in summary.columns
    assert "candidate_count_10ppm" in summary.columns
    # ppm=10 at mass ~300 -> window +-0.003 Da -> catches key_0..key_3ish, more than ppm=1
    assert (summary["candidate_count_10ppm"] >= summary["candidate_count_1ppm"]).all()
    assert (summary["target_abs_mass_error_ppm"] == 0.0).all()
    assert summary["neutral_mass_supported"].all()


def test_tolerance_sweep_recall_and_denominator():
    index = _toy_index()
    # one query exactly on a library mass (recoverable at any tolerance), one query far outside
    # the library (never recoverable, but must still count in the denominator).
    queries = pd.DataFrame({
        "query_id": ["hit", "miss"],
        "neutral_mass": [300.0, 500.0],
        "exact_mass_true": [300.0, 500.0],
    })
    summary = query_candidate_summary(queries, index, tolerance_grid=[1, 50], true_mass_col="exact_mass_true")
    sweep = tolerance_sweep(summary, tolerance_grid=[1, 50])

    assert (sweep["n_queries"] == 2).all()
    # "miss" has zero target_abs_mass_error_ppm too (exact_mass_true == neutral_mass for it),
    # so both should be recoverable -- recall should be 1.0 at every tolerance here.
    assert (sweep["candidate_recall"] == 1.0).all()


def test_tolerance_sweep_never_drops_a_target_absent_query():
    index = _toy_index()
    queries = pd.DataFrame({
        "query_id": ["far"], "neutral_mass": [999.0], "exact_mass_true": [300.0],
    })
    summary = query_candidate_summary(queries, index, tolerance_grid=[50], true_mass_col="exact_mass_true")
    sweep = tolerance_sweep(summary, tolerance_grid=[50])
    assert sweep.loc[0, "n_queries"] == 1
    assert sweep.loc[0, "candidate_recall"] == 0.0
    assert sweep.loc[0, "n_target_absent"] == 1


def test_select_operating_tolerance_hits_target():
    sweep = pd.DataFrame({
        "tolerance_ppm": [1, 5, 10], "candidate_recall": [0.9, 0.996, 0.999],
        "median_candidates": [2.0, 5.0, 9.0],
    })
    result = select_operating_tolerance(sweep, target_recall=0.995)
    assert result["hit_target"]
    assert result["tolerance_ppm"] == 5


def test_select_operating_tolerance_reports_best_effort_when_unreachable():
    sweep = pd.DataFrame({
        "tolerance_ppm": [1, 5], "candidate_recall": [0.80, 0.90], "median_candidates": [2.0, 5.0],
    })
    result = select_operating_tolerance(sweep, target_recall=0.995)
    assert not result["hit_target"]
    assert result["tolerance_ppm"] == 5
    assert result["recall"] == 0.90


def test_subgroup_report_flags_low_support_without_dropping():
    index = _toy_index()
    queries = pd.DataFrame({
        "query_id": [f"q{i}" for i in range(35)],
        "neutral_mass": [300.0] * 35,
        "exact_mass_true": [300.0] * 35,
        "adduct": (["[M+H]+"] * 32) + (["[M-H]-"] * 3),
    })
    summary = query_candidate_summary(queries, index, tolerance_grid=[5], true_mass_col="exact_mass_true")
    report = subgroup_report(summary, ["adduct"], tolerance_ppm=5, min_group_size=30)

    assert len(report) == 2
    rare = report[report["adduct"] == "[M-H]-"].iloc[0]
    assert rare["low_support"]
    assert rare["n_queries"] == 3


def test_fold_stability_one_row_per_fold():
    index = _toy_index()
    queries = pd.DataFrame({
        "query_id": ["a", "b", "c", "d"],
        "neutral_mass": [300.0] * 4,
        "exact_mass_true": [300.0] * 4,
        "fold": [0, 0, 1, 1],
    })
    summary = query_candidate_summary(queries, index, tolerance_grid=[5], true_mass_col="exact_mass_true")
    report = fold_stability(summary, tolerance_ppm=5)
    assert list(report["fold"]) == [0, 1]


def test_build_failure_table_classifies_reasons():
    index = _toy_index()
    # neutral_mass 300.03 is 100ppm off true_mass 300.0 -- outside the 5ppm tolerance being
    # tested, but well under the 200ppm "extreme" cutoff, so this should be ordinary
    # outside_mass_window, not mass_reconstruction_error.
    queries = pd.DataFrame({
        "query_id": ["unsupported", "outside_window"],
        "neutral_mass": [np.nan, 300.03],
        "exact_mass_true": [300.0, 300.0],
    })
    summary = query_candidate_summary(queries, index, tolerance_grid=[5], true_mass_col="exact_mass_true")
    summary["neutral_mass_supported"] = summary["neutral_mass"].notna()
    failures = build_failure_table(summary, tolerance_ppm=5)

    reasons = failures.set_index("query_id")["failure_reason"]
    assert reasons["unsupported"] == "unsupported_adduct"
    assert reasons["outside_window"] == "outside_mass_window"


def test_build_failure_table_extreme_error_is_reconstruction_error():
    index = _toy_index()
    queries = pd.DataFrame({"query_id": ["far"], "neutral_mass": [999.0], "exact_mass_true": [300.0]})
    summary = query_candidate_summary(queries, index, tolerance_grid=[5], true_mass_col="exact_mass_true")
    summary["neutral_mass_supported"] = summary["neutral_mass"].notna()
    failures = build_failure_table(summary, tolerance_ppm=5)
    assert failures.set_index("query_id").loc["far", "failure_reason"] == "mass_reconstruction_error"


def test_build_failure_table_invalid_true_mass():
    index = _toy_index()
    queries = pd.DataFrame({"query_id": ["nomass"], "neutral_mass": [300.0], "exact_mass_true": [np.nan]})
    summary = query_candidate_summary(queries, index, tolerance_grid=[5], true_mass_col="exact_mass_true")
    summary["neutral_mass_supported"] = summary["neutral_mass"].notna()
    failures = build_failure_table(summary, tolerance_ppm=5)
    assert failures.set_index("query_id").loc["nomass", "failure_reason"] == "invalid_true_exact_mass"


def test_build_failure_table_true_structure_missing_from_library():
    index = _toy_index()
    queries = pd.DataFrame({
        "query_id": ["ghost"], "neutral_mass": [300.03], "exact_mass_true": [300.0],
        "true_connectivity_key": ["not_in_library"],
    })
    summary = query_candidate_summary(queries, index, tolerance_grid=[5], true_mass_col="exact_mass_true")
    summary["neutral_mass_supported"] = summary["neutral_mass"].notna()
    summary["true_connectivity_key"] = queries["true_connectivity_key"]
    failures = build_failure_table(summary, tolerance_ppm=5, library_keys=set(index.keys))
    assert failures.set_index("query_id").loc["ghost", "failure_reason"] == "true_structure_missing_from_library"


def test_tolerance_tradeoff_deltas_shape_and_signs():
    sweep = pd.DataFrame({
        "tolerance_ppm": [1, 5, 10],
        "candidate_recall": [0.5, 0.9, 0.95],
        "median_candidates": [2.0, 10.0, 40.0],
        "p90_candidates": [5.0, 20.0, 80.0],
    })
    deltas = tolerance_tradeoff_deltas(sweep)

    assert len(deltas) == 2  # one row per adjacent pair
    assert list(deltas["from_ppm"]) == [1, 5]
    assert list(deltas["to_ppm"]) == [5, 10]
    assert deltas.loc[0, "recall_gain_pp"] == 40.0  # 0.9 - 0.5 -> 40 percentage points
    assert deltas.loc[0, "median_candidate_increase"] == 8.0
    # the 5->10ppm step buys less recall for more candidates than 1->5ppm -- diminishing returns
    assert deltas.loc[1, "candidates_per_recall_point"] > deltas.loc[0, "candidates_per_recall_point"]


def test_classify_decision_global_validated():
    global_result = {"hit_target": True, "tolerance_ppm": 10, "recall": 0.996}
    result = classify_decision(global_result)
    assert result["decision"] == "GLOBAL_POLICY_VALIDATED"
    assert result["status"] == "frozen"


def test_classify_decision_adaptive_validated_when_global_fails():
    global_result = {"hit_target": False, "tolerance_ppm": 50, "recall": 0.978}
    adaptive_result = {"hit_target": True, "recall": 0.996}
    result = classify_decision(global_result, adaptive_result)
    assert result["decision"] == "ADAPTIVE_POLICY_VALIDATED"
    assert result["status"] == "frozen"


def test_classify_decision_target_not_achieved_never_fabricates_frozen():
    global_result = {"hit_target": False, "tolerance_ppm": 50, "recall": 0.978}
    adaptive_result = {"hit_target": False, "recall": 0.981}
    result = classify_decision(global_result, adaptive_result)
    assert result["decision"] == "TARGET_NOT_ACHIEVED"
    assert result["status"] == "provisional"
    assert result["best_adaptive_recall"] == 0.981


def test_classify_decision_target_not_achieved_without_adaptive_result():
    global_result = {"hit_target": False, "tolerance_ppm": 50, "recall": 0.978}
    result = classify_decision(global_result, adaptive_result=None)
    assert result["decision"] == "TARGET_NOT_ACHIEVED"
    assert result["status"] == "provisional"
    assert result["best_adaptive_recall"] is None


def test_verify_true_structure_in_library_passes_when_consistent():
    molecule_table = pd.DataFrame({"connectivity_key": ["a", "b"], "exact_mass": [300.0, 400.0]})
    queries = pd.DataFrame({"true_connectivity_key": ["a", "b"], "exact_mass_true": [300.0, 400.0]})
    result = verify_true_structure_in_library(queries, molecule_table)
    assert result.passed


def test_verify_true_structure_in_library_catches_missing_key():
    molecule_table = pd.DataFrame({"connectivity_key": ["a"], "exact_mass": [300.0]})
    queries = pd.DataFrame({"true_connectivity_key": ["a", "ghost"], "exact_mass_true": [300.0, 400.0]})
    result = verify_true_structure_in_library(queries, molecule_table)
    assert not result.passed
    assert "missing" in result.detail


def test_verify_true_structure_in_library_catches_mass_mismatch():
    molecule_table = pd.DataFrame({"connectivity_key": ["a"], "exact_mass": [300.0]})
    queries = pd.DataFrame({"true_connectivity_key": ["a"], "exact_mass_true": [305.0]})
    result = verify_true_structure_in_library(queries, molecule_table)
    assert not result.passed
    assert "mismatch" in result.detail


def test_error_percentile_report_basic():
    df = pd.DataFrame({
        "adduct": ["[M+H]+"] * 40 + ["[M-H]-"] * 5,
        "target_abs_mass_error_ppm": [1.0] * 35 + [100.0] * 5 + [2.0] * 5,
    })
    report = error_percentile_report(df, ["adduct"], min_group_size=10)
    common = report.set_index("adduct").loc["[M+H]+"]
    assert common["n"] == 40
    assert not common["low_support"]
    rare = report.set_index("adduct").loc["[M-H]-"]
    assert rare["low_support"]


def test_mass_bin_report_bins_and_recall():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "exact_mass_true": rng.uniform(100, 900, 500),
        "target_abs_mass_error_ppm": rng.uniform(0, 10, 500),
    })
    report = mass_bin_report(df, n_bins=5, tolerance_grid=(5, 20))
    assert len(report) <= 5
    assert report["n"].sum() == 500
    assert "recall_at_5ppm" in report.columns and "recall_at_20ppm" in report.columns


def test_candidate_density_by_group_compares_dev_and_test():
    dev_df = pd.DataFrame({"adduct": ["[M+H]+"] * 20, "count": [10] * 20})
    test_df = pd.DataFrame({"adduct": ["[M+H]+"] * 20, "count": [30] * 20})
    report = candidate_density_by_group(dev_df, test_df, ["adduct"], count_col="count", min_group_size=5)
    row = report.iloc[0]
    assert row["dev_median"] == 10
    assert row["test_median"] == 30


def test_evaluate_variable_tolerance_uses_per_row_tolerance():
    index = _toy_index()
    queries = pd.DataFrame({
        "query_id": ["tight", "loose"],
        "neutral_mass": [300.0, 300.0],
        "exact_mass_true": [300.0, 300.0],
        "candidate_tolerance_ppm": [1.0, 50.0],
    })
    from casmi.candidates.evaluation import evaluate_variable_tolerance, variable_tolerance_summary

    evaluated = evaluate_variable_tolerance(queries, index, tolerance_col="candidate_tolerance_ppm")
    tight_count = evaluated.set_index("query_id").loc["tight", "candidate_count"]
    loose_count = evaluated.set_index("query_id").loc["loose", "candidate_count"]
    assert loose_count > tight_count
    assert evaluated["target_present"].all()  # both queries sit exactly on their true mass

    summary = variable_tolerance_summary(evaluated)
    assert summary["n_queries"] == 2
    assert summary["candidate_recall"] == 1.0


def test_verify_true_structure_in_library_variants_passes_with_isotope_variant():
    from casmi.candidates.diagnostics import verify_true_structure_in_library_variants

    variants = pd.DataFrame({"connectivity_key": ["A", "A", "B"], "exact_mass": [300.0, 308.05, 500.0]})
    # this query's true spectrum came from the deuterated (308.05) variant of A -- a plain
    # median-based check would have failed this (median of 300/308.05 = 304.025), but the
    # variant-aware check finds the real match.
    queries = pd.DataFrame({"true_connectivity_key": ["A"], "exact_mass_true": [308.05]})
    result = verify_true_structure_in_library_variants(queries, variants)
    assert result.passed


def test_verify_true_structure_in_library_variants_catches_missing_connectivity():
    from casmi.candidates.diagnostics import verify_true_structure_in_library_variants

    variants = pd.DataFrame({"connectivity_key": ["A"], "exact_mass": [300.0]})
    queries = pd.DataFrame({"true_connectivity_key": ["ghost"], "exact_mass_true": [300.0]})
    result = verify_true_structure_in_library_variants(queries, variants)
    assert not result.passed
    assert "missing connectivity" in result.detail


def test_verify_true_structure_in_library_variants_catches_no_matching_mass():
    from casmi.candidates.diagnostics import verify_true_structure_in_library_variants

    variants = pd.DataFrame({"connectivity_key": ["A"], "exact_mass": [300.0]})
    queries = pd.DataFrame({"true_connectivity_key": ["A"], "exact_mass_true": [999.0]})
    result = verify_true_structure_in_library_variants(queries, variants)
    assert not result.passed
    assert "no matching mass variant" in result.detail
