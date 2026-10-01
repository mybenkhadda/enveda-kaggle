"""Subgroup/fold/failure diagnostics for candidate generation -- all built on top of
`casmi.candidates.evaluation.query_candidate_summary`'s compact per-query table, never the full
candidate pool.
"""
import numpy as np
import pandas as pd

MIN_GROUP_SIZE_DEFAULT = 30


def subgroup_report(query_summary, group_cols, tolerance_ppm, min_group_size=MIN_GROUP_SIZE_DEFAULT,
                     target_col="target_abs_mass_error_ppm"):
    """Candidate recall + candidate-count stats at `tolerance_ppm`, broken out by
    `group_cols` (e.g. `["adduct"]`, `["fold"]`, `["ionization_mode"]`). Every group is
    reported -- a small group is flagged `low_support` (below `min_group_size`), never
    silently dropped, so a rare adduct's bad recall stays visible instead of disappearing from
    the table."""
    count_col = f"candidate_count_{tolerance_ppm}ppm"
    has_truth = target_col in query_summary.columns

    def _agg(g):
        row = {
            "n_queries": len(g),
            "median_candidates": float(g[count_col].median()),
            "p90_candidates": float(g[count_col].quantile(0.90)),
            "zero_candidate_rate": float((g[count_col] == 0).mean()),
        }
        if has_truth:
            row["candidate_recall"] = float((g[target_col] <= tolerance_ppm).mean())
        return pd.Series(row)

    report = query_summary.groupby(list(group_cols), dropna=False).apply(_agg, include_groups=False).reset_index()
    report["low_support"] = report["n_queries"] < min_group_size
    return report.sort_values("n_queries", ascending=False).reset_index(drop=True)


def fold_stability(query_summary, tolerance_ppm, fold_col="fold", min_group_size=MIN_GROUP_SIZE_DEFAULT):
    """`subgroup_report` specialized to `fold_col` -- large recall/candidate-count swings
    across folds would suggest the mass-index candidate library isn't as fold-agnostic as
    expected (it shouldn't be: `MassIndex` is built from static structure identity, never from
    other spectra, so fold membership carries no leakage risk the way a spectral reference pool
    would -- see the closed-world note in the notebook)."""
    return subgroup_report(query_summary, [fold_col], tolerance_ppm, min_group_size=min_group_size).sort_values(fold_col)


EXTREME_ERROR_PPM_DEFAULT = 200.0


def build_failure_table(query_summary, tolerance_ppm, target_col="target_abs_mass_error_ppm",
                         supported_col="neutral_mass_supported", true_key_col="true_connectivity_key",
                         true_mass_col="exact_mass_true", library_keys=None, extreme_error_ppm=EXTREME_ERROR_PPM_DEFAULT):
    """Every query whose true candidate is absent at `tolerance_ppm`, with a best-effort
    `failure_reason` -- checked in this priority order, so each failure gets exactly one
    determinable reason instead of defaulting to a catch-all:

        unsupported_adduct           -- `neutral_mass_supported` is False (adduct didn't parse)
        true_structure_missing_from_library -- `true_key_col` isn't in `library_keys` at all
                                          (would break the closed-world assumption this whole
                                          notebook relies on; expected count is zero, but
                                          asserted rather than assumed -- see
                                          `verify_true_structure_in_library`)
        invalid_true_exact_mass      -- `true_mass_col` itself is NaN
        neutral_mass_missing         -- the query's own derived mass is NaN for a reason other
                                          than an unsupported adduct
        mass_reconstruction_error    -- the true target is off by more than
                                          `extreme_error_ppm` -- large enough that it looks
                                          like a genuine data/parsing problem, not ordinary
                                          instrument measurement noise
        outside_mass_window          -- everything else: the true structure IS in the library
                                          with a valid mass, just farther than `tolerance_ppm`
                                          away (ordinary tail of measurement error)
    """
    failed = query_summary[~(query_summary[target_col] <= tolerance_ppm)].copy()

    missing_from_library = (
        ~failed[true_key_col].isin(library_keys) if library_keys is not None and true_key_col in failed.columns
        else pd.Series(False, index=failed.index)
    )
    invalid_true_mass = failed[true_mass_col].isna() if true_mass_col in failed.columns else pd.Series(False, index=failed.index)

    reason = np.select(
        [
            ~failed[supported_col],
            missing_from_library,
            invalid_true_mass,
            failed[target_col].isna(),
            failed[target_col] > extreme_error_ppm,
        ],
        [
            "unsupported_adduct", "true_structure_missing_from_library", "invalid_true_exact_mass",
            "neutral_mass_missing", "mass_reconstruction_error",
        ],
        default="outside_mass_window",
    )
    failed["failure_reason"] = reason
    failed["tolerance_ppm"] = tolerance_ppm
    return failed


def error_percentile_report(df, group_cols, error_col="target_abs_mass_error_ppm",
                             min_group_size=MIN_GROUP_SIZE_DEFAULT, thresholds=(20.0, 50.0)):
    """Mass-RECONSTRUCTION-error percentiles broken out by `group_cols` (adduct, instrument,
    source, ion mode, ...) -- distinct from `subgroup_report`, which reports candidate
    counts/recall at one fixed tolerance. This answers "whose mass error is heavy-tailed",
    which is what actually explains a recall shortfall, not just which group has low recall."""
    def _agg(g):
        row = {
            "n": len(g),
            "median_abs_ppm": float(g[error_col].median()),
            "p95_abs_ppm": float(g[error_col].quantile(0.95)),
            "p99_abs_ppm": float(g[error_col].quantile(0.99)),
        }
        for t in thresholds:
            row[f"frac_over_{int(t)}ppm"] = float((g[error_col] > t).mean())
        return pd.Series(row)

    report = df.groupby(list(group_cols), dropna=False).apply(_agg, include_groups=False).reset_index()
    report["low_support"] = report["n"] < min_group_size
    return report.sort_values("n", ascending=False).reset_index(drop=True)


def mass_bin_report(df, mass_col="exact_mass_true", error_col="target_abs_mass_error_ppm", n_bins=10,
                     tolerance_grid=(5, 10, 20, 50)):
    """Same idea as `error_percentile_report`, binned by true molecular mass (quantile bins,
    so every bin has comparable support) instead of a categorical group -- answers "are
    high-mass queries harder to retrieve"."""
    binned = df.copy()
    binned["mass_bin"] = pd.qcut(binned[mass_col], q=n_bins, duplicates="drop")

    def _agg(g):
        row = {
            "n": len(g), "mass_min": float(g[mass_col].min()), "mass_max": float(g[mass_col].max()),
            "median_abs_ppm": float(g[error_col].median()), "p95_abs_ppm": float(g[error_col].quantile(0.95)),
            "p99_abs_ppm": float(g[error_col].quantile(0.99)),
        }
        for ppm in tolerance_grid:
            row[f"recall_at_{ppm}ppm"] = float((g[error_col] <= ppm).mean())
        return pd.Series(row)

    report = binned.groupby("mass_bin", observed=True).apply(_agg, include_groups=False).reset_index()
    return report.sort_values("mass_min").reset_index(drop=True)


def candidate_density_by_group(dev_df, test_df, group_cols, count_col, min_group_size=MIN_GROUP_SIZE_DEFAULT):
    """Dev vs. test candidate-count comparison broken out by `group_cols` -- the building block
    for diagnosing WHY test candidate density differs from dev (mass shift vs. adduct shift vs.
    something else), rather than only observing THAT it differs."""
    def _describe(df):
        g = df.groupby(list(group_cols), dropna=False)[count_col]
        return pd.DataFrame({"n": g.size(), "median": g.median(), "p90": g.quantile(0.9)})

    dev_stats = _describe(dev_df).add_prefix("dev_")
    test_stats = _describe(test_df).add_prefix("test_")
    combined = dev_stats.join(test_stats, how="outer").reset_index()
    combined["low_support"] = (combined["dev_n"].fillna(0) < min_group_size) | (combined["test_n"].fillna(0) < min_group_size)
    return combined.sort_values("dev_n", ascending=False).reset_index(drop=True)


def verify_true_structure_in_library_variants(queries, molecule_mass_variants, true_key_col="true_connectivity_key",
                                               true_mass_col="exact_mass_true", key_col="connectivity_key",
                                               mass_col="exact_mass", atol_da=1e-4):
    """Variant-aware version of `verify_true_structure_in_library`: a query's true structure
    passes if its `true_key_col` exists among `molecule_mass_variants`' connectivity_keys AND
    at least ONE of that key's mass variants matches `true_mass_col` within `atol_da` -- not a
    single deduplicated (and possibly wrong-isotope-variant) mass. Reports each failure
    category separately (never a single opaque FAIL): connectivity missing from the library
    entirely, vs. present but with no matching mass variant."""
    from casmi.validation.checks import CheckResult

    unique_queries = queries[[true_key_col, true_mass_col]].drop_duplicates(true_key_col)
    variants_by_key = molecule_mass_variants.groupby(key_col)[mass_col].apply(np.asarray)

    connectivity_missing = ~unique_queries[true_key_col].isin(variants_by_key.index)

    def _has_matching_variant(key, true_mass):
        if key not in variants_by_key.index:
            return False
        masses = variants_by_key.loc[key]
        return bool(np.any(np.abs(masses - true_mass) <= atol_da))

    mass_variant_missing = pd.Series(
        [not connectivity_missing.iloc[i] and not _has_matching_variant(row[true_key_col], row[true_mass_col])
         for i, (_, row) in enumerate(unique_queries.iterrows())],
        index=unique_queries.index,
    )

    passed = not connectivity_missing.any() and not mass_variant_missing.any()
    detail = (f"{len(unique_queries)} unique true structures checked against mass variants: "
              f"0 missing connectivity, 0 without a matching mass variant" if passed else
              f"{int(connectivity_missing.sum())} missing connectivity, "
              f"{int(mass_variant_missing.sum())} present but no matching mass variant (of {len(unique_queries)})")
    return CheckResult(name="true structures exist in library with a matching mass variant", passed=passed, detail=detail)


def verify_true_structure_in_library(queries, molecule_table, true_key_col="true_connectivity_key",
                                      true_mass_col="exact_mass_true", key_col="connectivity_key",
                                      mass_col="exact_mass", atol_da=1e-4):
    """The closed-world assumption underlying this whole notebook, checked rather than assumed:
    every dev query's `true_key_col` must exist in `molecule_table`, AND that library row's own
    mass must agree with the query's own `true_mass_col` to within `atol_da` (a mismatch would
    mean the query table and the candidate library disagree about the same molecule's mass --
    a real data-integrity bug, not a modeling result). Returns a
    `casmi.validation.checks.CheckResult`."""
    from casmi.validation.checks import CheckResult

    lib = molecule_table.set_index(key_col)[mass_col]
    merged = queries[[true_key_col, true_mass_col]].drop_duplicates(true_key_col).copy()
    merged["library_mass"] = merged[true_key_col].map(lib)

    missing = merged["library_mass"].isna()
    mismatched = (~missing) & ((merged["library_mass"] - merged[true_mass_col]).abs() > atol_da)

    passed = not missing.any() and not mismatched.any()
    detail = (f"{len(merged)} unique true structures checked: 0 missing from library, 0 mass mismatches"
               if passed else
               f"{int(missing.sum())} missing from library, {int(mismatched.sum())} mass mismatches (of {len(merged)})")
    return CheckResult(name="true structures exist in library with matching mass", passed=passed, detail=detail)
