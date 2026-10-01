import pandas as pd

from casmi.candidates.generator import CandidateGenerationConfig, CandidateGenerator
from casmi.candidates.mass_index import MassIndex


def _toy_index():
    return MassIndex(
        keys=["true_a", "decoy_b", "decoy_c", "far_d"],
        masses=[300.0000, 300.0005, 300.0015, 310.0000],
    )


def test_generate_single_query_closest_first():
    gen = CandidateGenerator(_toy_index())
    keys, masses = gen.generate(300.0000, tolerance_ppm=10)
    assert list(keys) == ["true_a", "decoy_b", "decoy_c"]


def test_generate_dataframe_target_present():
    queries = pd.DataFrame({
        "query_id": ["q1"], "neutral_mass": [300.0000], "true_connectivity_key": ["true_a"],
    })
    gen = CandidateGenerator(_toy_index())
    pool = gen.generate_dataframe(queries, tolerance_ppm=10)

    assert set(pool["candidate_connectivity_key"]) == {"true_a", "decoy_b", "decoy_c"}
    assert pool.loc[pool["candidate_connectivity_key"] == "true_a", "is_true_candidate"].item()
    assert not pool.loc[pool["candidate_connectivity_key"] == "decoy_b", "is_true_candidate"].item()


def test_generate_dataframe_target_absent_zero_candidates():
    queries = pd.DataFrame({"query_id": ["q1"], "neutral_mass": [999.0], "true_connectivity_key": ["true_a"]})
    gen = CandidateGenerator(_toy_index())
    pool = gen.generate_dataframe(queries, tolerance_ppm=5)
    assert len(pool) == 0


def test_generate_dataframe_skips_unsupported_null_mass():
    queries = pd.DataFrame({
        "query_id": ["q1", "q2"], "neutral_mass": [float("nan"), 300.0000],
        "true_connectivity_key": ["true_a", "true_a"],
    })
    gen = CandidateGenerator(_toy_index())
    pool = gen.generate_dataframe(queries, tolerance_ppm=10)
    assert set(pool["query_id"]) == {"q2"}


def test_generate_dataframe_no_duplicate_query_candidate_pairs():
    queries = pd.DataFrame({"query_id": ["q1", "q2"], "neutral_mass": [300.0000, 300.0010]})
    gen = CandidateGenerator(_toy_index())
    pool = gen.generate_dataframe(queries, tolerance_ppm=20)
    assert not pool[["query_id", "candidate_connectivity_key"]].duplicated().any()


def test_generate_dataframe_mass_error_ppm_sign_and_magnitude():
    queries = pd.DataFrame({"query_id": ["q1"], "neutral_mass": [300.0000]})
    gen = CandidateGenerator(_toy_index())
    pool = gen.generate_dataframe(queries, tolerance_ppm=10).set_index("candidate_connectivity_key")
    # decoy_b is heavier than the query -> positive mass error
    assert pool.loc["decoy_b", "mass_error_ppm"] > 0
    assert pool.loc["decoy_b", "abs_mass_error_ppm"] == abs(pool.loc["decoy_b", "mass_error_ppm"])


def test_generate_dataframe_without_true_key_column_has_no_truth_columns():
    queries = pd.DataFrame({"query_id": ["q1"], "neutral_mass": [300.0000]})
    gen = CandidateGenerator(_toy_index())
    pool = gen.generate_dataframe(queries, tolerance_ppm=10)
    assert "is_true_candidate" not in pool.columns
    assert "true_connectivity_key" not in pool.columns


def test_candidate_generation_config_validates_tolerance_grid():
    CandidateGenerationConfig(tolerance_grid_ppm=(1, 5, 10), max_tolerance_ppm=10)
    try:
        CandidateGenerationConfig(tolerance_grid_ppm=(1, 5, 100), max_tolerance_ppm=10)
        assert False, "should reject a grid tolerance above max_tolerance_ppm"
    except ValueError:
        pass


def test_generate_dataframe_with_tolerance_col():
    queries = pd.DataFrame({
        "query_id": ["tight", "loose"], "neutral_mass": [300.0000, 300.0000],
        "candidate_tolerance_ppm": [1.0, 10.0],
    })
    gen = CandidateGenerator(_toy_index())
    pool = gen.generate_dataframe(queries, tolerance_col="candidate_tolerance_ppm")

    tight_candidates = set(pool[pool["query_id"] == "tight"]["candidate_connectivity_key"])
    loose_candidates = set(pool[pool["query_id"] == "loose"]["candidate_connectivity_key"])
    assert tight_candidates == {"true_a"}
    assert loose_candidates == {"true_a", "decoy_b", "decoy_c"}


def test_generate_dataframe_rejects_both_or_neither_tolerance_args():
    queries = pd.DataFrame({"query_id": ["q1"], "neutral_mass": [300.0]})
    gen = CandidateGenerator(_toy_index())
    try:
        gen.generate_dataframe(queries)
        assert False, "should require exactly one of tolerance_ppm/tolerance_col"
    except ValueError:
        pass
    try:
        gen.generate_dataframe(queries, tolerance_ppm=5, tolerance_col="x")
        assert False, "should reject both tolerance_ppm and tolerance_col"
    except ValueError:
        pass


def test_dedupe_pool_to_connectivity_collapses_variants_keeping_best():
    # two mass variants of connectivity "A" both fall in the window; "A__v1" is the closer match.
    pool = pd.DataFrame({
        "query_id": ["q1", "q1", "q1"],
        "candidate_connectivity_key": ["A__v0", "A__v1", "B__v0"],  # these are mass_variant_ids pre-dedup
        "abs_mass_error_ppm": [8.0, 2.0, 5.0],
        "candidate_exact_mass": [300.0, 300.05, 500.0],
    })
    variant_to_connectivity = {"A__v0": "A", "A__v1": "A", "B__v0": "B"}
    from casmi.candidates.generator import dedupe_pool_to_connectivity

    deduped = dedupe_pool_to_connectivity(pool, variant_to_connectivity)
    assert len(deduped) == 2  # collapsed A's two variants into one row
    a_row = deduped[deduped["candidate_connectivity_key"] == "A"].iloc[0]
    assert a_row["abs_mass_error_ppm"] == 2.0  # kept the closer variant
    assert a_row["mass_variant_support_count"] == 2
    b_row = deduped[deduped["candidate_connectivity_key"] == "B"].iloc[0]
    assert b_row["mass_variant_support_count"] == 1


def test_dedupe_pool_to_connectivity_no_duplicate_query_connectivity_pairs():
    pool = pd.DataFrame({
        "query_id": ["q1", "q1", "q2"],
        "candidate_connectivity_key": ["A__v0", "A__v1", "A__v0"],
        "abs_mass_error_ppm": [3.0, 1.0, 4.0],
        "candidate_exact_mass": [300.0, 300.02, 300.0],
    })
    from casmi.candidates.generator import dedupe_pool_to_connectivity

    deduped = dedupe_pool_to_connectivity(pool, {"A__v0": "A", "A__v1": "A"})
    assert not deduped[["query_id", "candidate_connectivity_key"]].duplicated().any()


def test_dedupe_pool_to_connectivity_recomputes_is_true_candidate():
    pool = pd.DataFrame({
        "query_id": ["q1"], "candidate_connectivity_key": ["A__v0"], "abs_mass_error_ppm": [1.0],
        "candidate_exact_mass": [300.0], "true_connectivity_key": ["A"],
        "is_true_candidate": [False],  # wrong, computed against the raw variant id at generation time
    })
    from casmi.candidates.generator import dedupe_pool_to_connectivity

    deduped = dedupe_pool_to_connectivity(pool, {"A__v0": "A"})
    assert deduped["is_true_candidate"].iloc[0]


def test_dedupe_pool_to_connectivity_empty_pool():
    pool = pd.DataFrame({"query_id": [], "candidate_connectivity_key": [], "abs_mass_error_ppm": [], "candidate_exact_mass": []})
    from casmi.candidates.generator import dedupe_pool_to_connectivity

    deduped = dedupe_pool_to_connectivity(pool, {})
    assert len(deduped) == 0
    assert "mass_variant_support_count" in deduped.columns
