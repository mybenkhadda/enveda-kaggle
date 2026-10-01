"""Spectrum-level candidate pools -> molecule-level candidate sets: union, intersection, and
per-candidate support across a molecule's spectra. Shared by dev-molecule (`connectivity_key`)
and test-molecule (`molecule_id`) evaluation -- the aggregation logic doesn't care which column
plays the role of "molecule", only that a candidate pool's `query_id`s can be grouped by one.
"""
import numpy as np
import pandas as pd


def aggregate_candidates_to_molecule(candidate_pool, molecule_col, candidate_col="candidate_connectivity_key",
                                      query_id_col="query_id"):
    """One row per (molecule, candidate) pair actually observed, with `n_supporting_spectra`
    -- how many of that molecule's queries retrieved this candidate. `candidate_pool` must
    already carry `molecule_col` (join it in first if the pool only has `query_id_col`, e.g.
    `pool.merge(queries[[query_id_col, molecule_col]], on=query_id_col)`)."""
    return (
        candidate_pool.groupby([molecule_col, candidate_col])
        .agg(
            n_supporting_spectra=(query_id_col, "nunique"),
            best_abs_mass_error_ppm=("abs_mass_error_ppm", "min"),
            mean_abs_mass_error_ppm=("abs_mass_error_ppm", "mean"),
            candidate_exact_mass=("candidate_exact_mass", "first"),
        )
        .reset_index()
    )


def molecule_candidate_summary(per_molecule_candidates, queries, molecule_col, query_id_col="query_id",
                                true_key_col=None, candidate_col="candidate_connectivity_key"):
    """One row per molecule: `n_spectra`, union/intersection candidate-set sizes, and a
    seen-once vs. seen-by-multiple-spectra breakdown. When `true_key_col` is given (development
    only -- omit for test, where it's the prediction target), also reports whether the true
    candidate is in the union/intersection and its own support count/fraction. Every molecule
    in `queries` gets a row, including one with zero candidates in `per_molecule_candidates`
    (via the `n_spectra` reindex) -- never silently dropped for having an empty candidate set.
    """
    n_spectra = queries.groupby(molecule_col)[query_id_col].nunique().rename("n_spectra")
    all_molecules = n_spectra.index

    union_size = per_molecule_candidates.groupby(molecule_col)[candidate_col].nunique().reindex(all_molecules, fill_value=0).rename("union_candidate_count")

    joined = per_molecule_candidates.merge(n_spectra.rename("_n_spectra_for_mol"), left_on=molecule_col, right_index=True)
    full_support = joined[joined["n_supporting_spectra"] == joined["_n_spectra_for_mol"]]
    intersection_size = full_support.groupby(molecule_col).size().reindex(all_molecules, fill_value=0).rename("intersection_candidate_count")

    seen_once = per_molecule_candidates[per_molecule_candidates["n_supporting_spectra"] == 1]
    n_seen_once = seen_once.groupby(molecule_col).size().reindex(all_molecules, fill_value=0).rename("n_candidates_seen_once")
    seen_multi = per_molecule_candidates[per_molecule_candidates["n_supporting_spectra"] > 1]
    n_seen_multi = seen_multi.groupby(molecule_col).size().reindex(all_molecules, fill_value=0).rename("n_candidates_seen_multiple_spectra")

    out = pd.concat([n_spectra, union_size, intersection_size, n_seen_once, n_seen_multi], axis=1).reset_index()

    if true_key_col:
        if true_key_col == molecule_col:
            # development use: `molecule_col` (the grouping key) and `true_key_col` are the
            # SAME connectivity_key column -- the molecule's own id already IS its true target,
            # so the "lookup" is just the identity; going through `.set_index(molecule_col)
            # [true_key_col]` here would fail, since set_index removes that very column.
            out["true_connectivity_key"] = out[molecule_col]
            candidate_true_key = per_molecule_candidates[molecule_col]
        else:
            true_keys = queries.drop_duplicates(molecule_col).set_index(molecule_col)[true_key_col]
            out["true_connectivity_key"] = out[molecule_col].map(true_keys)
            true_key_lookup = out.set_index(molecule_col)["true_connectivity_key"]
            candidate_true_key = per_molecule_candidates[molecule_col].map(true_key_lookup)

        true_rows = per_molecule_candidates[per_molecule_candidates[candidate_col] == candidate_true_key]
        support_map = true_rows.set_index(molecule_col)["n_supporting_spectra"]

        out["true_candidate_support_count"] = out[molecule_col].map(support_map).fillna(0).astype(int)
        out["true_candidate_support_fraction"] = out["true_candidate_support_count"] / out["n_spectra"]
        out["target_in_union"] = out["true_candidate_support_count"] > 0
        out["target_in_intersection"] = out["true_candidate_support_count"] == out["n_spectra"]

    return out


def molecule_tolerance_sweep(pool_max, queries, molecule_col, tolerance_grid, query_id_col="query_id",
                              true_key_col=None, candidate_col="candidate_connectivity_key",
                              error_col="abs_mass_error_ppm"):
    """The molecule-level analog of `casmi.candidates.evaluation.tolerance_sweep`: one row per
    ppm tolerance, with union/intersection recall and candidate-count percentiles. Built by
    filtering the already-generated MAX-tolerance pool per ppm (never re-running candidate
    generation once per grid point)."""
    rows = []
    for ppm in tolerance_grid:
        pool_at_ppm = pool_max[pool_max[error_col] <= ppm]
        # `molecule_col` must already be a column on `pool_max` (e.g. `true_connectivity_key`
        # for dev, `molecule_id` for test, joined in by the caller beforehand).
        per_mol = aggregate_candidates_to_molecule(pool_at_ppm, molecule_col=molecule_col, candidate_col=candidate_col, query_id_col=query_id_col)
        summary = molecule_candidate_summary(per_mol, queries, molecule_col=molecule_col, query_id_col=query_id_col, true_key_col=true_key_col)

        row = {
            "tolerance_ppm": ppm, "n_molecules": len(summary),
            "median_union_candidates": float(summary["union_candidate_count"].median()),
            "p75_union_candidates": float(summary["union_candidate_count"].quantile(0.75)),
            "p90_union_candidates": float(summary["union_candidate_count"].quantile(0.90)),
            "p95_union_candidates": float(summary["union_candidate_count"].quantile(0.95)),
            "p99_union_candidates": float(summary["union_candidate_count"].quantile(0.99)),
            "median_intersection_candidates": float(summary["intersection_candidate_count"].median()),
            "p90_intersection_candidates": float(summary["intersection_candidate_count"].quantile(0.90)),
        }
        if true_key_col:
            row["union_recall"] = float(summary["target_in_union"].mean())
            row["intersection_recall"] = float(summary["target_in_intersection"].mean())
            row["mean_true_candidate_support_fraction"] = float(summary["true_candidate_support_fraction"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def sample_test_like_spectra(candidate_spectra, molecule_col, test_n_spectra_per_molecule, seed=42):
    """For each molecule (group of `molecule_col`) in `candidate_spectra`, draw a random subset
    whose SIZE is sampled from `test_n_spectra_per_molecule` (the empirical
    n-spectra-per-test-molecule_id distribution) -- never from any candidate-generation outcome.
    A molecule with fewer available spectra than the drawn count keeps all of them (can't
    sample more than exists). Used to build a "test-like" development evaluation whose spectrum
    multiplicity matches test's real distribution, rather than development's own (usually
    richer) multiplicity."""
    rng = np.random.RandomState(seed)
    test_counts = np.asarray(test_n_spectra_per_molecule)
    parts = []
    for _, g in candidate_spectra.groupby(molecule_col):
        n_draw = int(rng.choice(test_counts))
        n_take = min(n_draw, len(g)) if n_draw > 0 else 0
        if n_take == 0:
            continue
        parts.append(g.sample(n_take, random_state=rng.randint(0, 2**31 - 1)))
    return pd.concat(parts, ignore_index=True) if parts else candidate_spectra.iloc[0:0]
