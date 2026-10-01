import pandas as pd

from casmi.candidates.molecule import aggregate_candidates_to_molecule, molecule_candidate_summary


def _pool():
    # molecule "m1" has 2 spectra: q1 retrieves {true, decoy_a}; q2 retrieves {true, decoy_b}.
    # molecule "m2" has 1 spectrum: q3 retrieves {other}.
    return pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2", "q3"],
        "candidate_connectivity_key": ["true", "decoy_a", "true", "decoy_b", "other"],
        "abs_mass_error_ppm": [1.0, 2.0, 1.5, 3.0, 0.5],
        "candidate_exact_mass": [300.0, 300.1, 300.0, 300.2, 500.0],
        "molecule_id": ["m1", "m1", "m1", "m1", "m2"],
    })


def _queries():
    return pd.DataFrame({
        "query_id": ["q1", "q2", "q3"],
        "molecule_id": ["m1", "m1", "m2"],
        "true_connectivity_key": ["true", "true", "true"],  # m2's true structure was never retrieved
    })


def test_aggregate_candidates_to_molecule_counts_support():
    per_mol = aggregate_candidates_to_molecule(_pool(), molecule_col="molecule_id")
    row = per_mol[(per_mol["molecule_id"] == "m1") & (per_mol["candidate_connectivity_key"] == "true")].iloc[0]
    assert row["n_supporting_spectra"] == 2  # seen by both q1 and q2

    row = per_mol[(per_mol["molecule_id"] == "m1") & (per_mol["candidate_connectivity_key"] == "decoy_a")].iloc[0]
    assert row["n_supporting_spectra"] == 1  # only q1


def test_molecule_candidate_summary_union_and_intersection():
    per_mol = aggregate_candidates_to_molecule(_pool(), molecule_col="molecule_id")
    summary = molecule_candidate_summary(per_mol, _queries(), molecule_col="molecule_id",
                                          true_key_col="true_connectivity_key").set_index("molecule_id")

    m1 = summary.loc["m1"]
    assert m1["n_spectra"] == 2
    assert m1["union_candidate_count"] == 3  # true, decoy_a, decoy_b
    assert m1["intersection_candidate_count"] == 1  # only "true" was seen by both spectra
    assert m1["target_in_union"]
    assert m1["target_in_intersection"]
    assert m1["true_candidate_support_count"] == 2
    assert m1["true_candidate_support_fraction"] == 1.0


def test_molecule_candidate_summary_target_absent():
    per_mol = aggregate_candidates_to_molecule(_pool(), molecule_col="molecule_id")
    summary = molecule_candidate_summary(per_mol, _queries(), molecule_col="molecule_id",
                                          true_key_col="true_connectivity_key").set_index("molecule_id")
    m2 = summary.loc["m2"]
    assert m2["union_candidate_count"] == 1
    assert not m2["target_in_union"]
    assert not m2["target_in_intersection"]
    assert m2["true_candidate_support_count"] == 0
    assert m2["true_candidate_support_fraction"] == 0.0


def test_molecule_candidate_summary_includes_molecules_with_zero_candidates():
    per_mol = aggregate_candidates_to_molecule(_pool(), molecule_col="molecule_id")
    queries = pd.concat([_queries(), pd.DataFrame({
        "query_id": ["q4"], "molecule_id": ["m3"], "true_connectivity_key": ["true"],
    })], ignore_index=True)
    summary = molecule_candidate_summary(per_mol, queries, molecule_col="molecule_id",
                                          true_key_col="true_connectivity_key").set_index("molecule_id")
    assert "m3" in summary.index
    assert summary.loc["m3", "union_candidate_count"] == 0
    assert not summary.loc["m3", "target_in_union"]


def test_molecule_candidate_summary_without_true_key_has_no_truth_columns():
    per_mol = aggregate_candidates_to_molecule(_pool(), molecule_col="molecule_id")
    summary = molecule_candidate_summary(per_mol, _queries(), molecule_col="molecule_id")
    assert "target_in_union" not in summary.columns
    assert "true_candidate_support_count" not in summary.columns


def test_molecule_candidate_summary_when_molecule_col_equals_true_key_col():
    # the dev-molecule use case: connectivity_key IS both the grouping key and the true target,
    # so molecule_col and true_key_col are literally the same column name.
    pool = pd.DataFrame({
        "query_id": ["q1", "q2"],
        "candidate_connectivity_key": ["true", "true"],
        "abs_mass_error_ppm": [1.0, 1.0],
        "candidate_exact_mass": [300.0, 300.0],
        "true_connectivity_key": ["true", "true"],
    })
    queries = pd.DataFrame({"query_id": ["q1", "q2"], "true_connectivity_key": ["true", "true"]})
    per_mol = aggregate_candidates_to_molecule(pool, molecule_col="true_connectivity_key")
    summary = molecule_candidate_summary(per_mol, queries, molecule_col="true_connectivity_key",
                                          true_key_col="true_connectivity_key")
    row = summary.iloc[0]
    assert row["target_in_union"]
    assert row["target_in_intersection"]
    assert row["true_candidate_support_count"] == 2


def test_molecule_tolerance_sweep_recall_increases_with_tolerance():
    from casmi.candidates.molecule import molecule_tolerance_sweep

    pool = pd.DataFrame({
        "query_id": ["q1", "q1", "q2"],
        "candidate_connectivity_key": ["true", "decoy", "true"],
        "abs_mass_error_ppm": [15.0, 2.0, 1.0],
        "candidate_exact_mass": [300.0, 300.1, 300.0],
        "true_connectivity_key": ["true", "true", "true"],
    })
    queries = pd.DataFrame({"query_id": ["q1", "q2"], "true_connectivity_key": ["true", "true"]})

    sweep = molecule_tolerance_sweep(pool, queries, molecule_col="true_connectivity_key",
                                      tolerance_grid=[5, 20], true_key_col="true_connectivity_key")
    assert len(sweep) == 2
    tight, loose = sweep.iloc[0], sweep.iloc[1]
    # at 5ppm only q2's row (1.0ppm) qualifies for "true" -- q1's own true-candidate row (15ppm) doesn't
    assert loose["union_recall"] >= tight["union_recall"]


def test_molecule_tolerance_sweep_without_true_key_has_no_recall_columns():
    from casmi.candidates.molecule import molecule_tolerance_sweep

    pool = pd.DataFrame({
        "query_id": ["q1"], "candidate_connectivity_key": ["c1"], "abs_mass_error_ppm": [1.0],
        "candidate_exact_mass": [300.0], "molecule_id": ["m1"],
    })
    queries = pd.DataFrame({"query_id": ["q1"], "molecule_id": ["m1"]})
    sweep = molecule_tolerance_sweep(pool, queries, molecule_col="molecule_id", tolerance_grid=[5])
    assert "union_recall" not in sweep.columns
    assert sweep.loc[0, "n_molecules"] == 1


def test_sample_test_like_spectra_respects_size_distribution():
    from casmi.candidates.molecule import sample_test_like_spectra

    # 3 molecules with 10 spectra each available; test distribution says "always take 2".
    rows = []
    for mol in ["m1", "m2", "m3"]:
        for i in range(10):
            rows.append({"connectivity_key": mol, "spectrum_idx": i})
    available = pd.DataFrame(rows)

    sampled = sample_test_like_spectra(available, molecule_col="connectivity_key",
                                        test_n_spectra_per_molecule=[2, 2, 2, 2], seed=1)
    counts = sampled.groupby("connectivity_key").size()
    assert (counts == 2).all()


def test_sample_test_like_spectra_caps_at_available_count():
    from casmi.candidates.molecule import sample_test_like_spectra

    available = pd.DataFrame({"connectivity_key": ["m1"] * 2, "spectrum_idx": [0, 1]})
    # test distribution demands 5, but only 2 are available -- must not crash or fabricate rows.
    sampled = sample_test_like_spectra(available, molecule_col="connectivity_key",
                                        test_n_spectra_per_molecule=[5], seed=1)
    assert len(sampled) == 2
