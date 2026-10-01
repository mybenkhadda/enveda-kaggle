"""Adaptive (group-specific) mass-tolerance windows -- an ALTERNATIVE to one global ppm
tolerance, evaluated against the global baseline rather than assumed better.

Fold-safe by construction: `evaluate_adaptive_policy_foldwise` fits each fold's policy from
the OTHER folds' rows only, then applies it to the held-out fold. Fitting and scoring an
adaptive window on the exact same records would be optimistically biased -- a group's own
empirical p99.5 mass error is trivially close to that same group's queries; the only honest
test is whether a policy learned WITHOUT a query still covers it.
"""
import pandas as pd


def fit_adaptive_mass_windows(queries_with_error, group_cols, target_quantile=0.995, min_group_size=100,
                               fallback_ppm=50.0, min_ppm=1.0, max_ppm=50.0, error_col="target_abs_mass_error_ppm"):
    """Empirical per-group tolerance: the `target_quantile` of `error_col` within each group of
    `queries_with_error`, clipped to `[min_ppm, max_ppm]`. A group smaller than
    `min_group_size` falls back to `fallback_ppm` -- estimating a p99.5 from a handful of
    points isn't trustworthy. Returns a policy table: `group_cols` + `tolerance_ppm` + `n` +
    `used_fallback`."""
    g = queries_with_error.groupby(list(group_cols), dropna=False)[error_col]
    policy = g.quantile(target_quantile).rename("tolerance_ppm").reset_index()
    policy = policy.merge(g.size().rename("n").reset_index(), on=list(group_cols))
    policy["used_fallback"] = policy["n"] < min_group_size
    policy.loc[policy["used_fallback"], "tolerance_ppm"] = fallback_ppm
    policy["tolerance_ppm"] = policy["tolerance_ppm"].clip(lower=min_ppm, upper=max_ppm)
    return policy


def apply_adaptive_mass_windows(query_df, policy, group_cols, fallback_ppm=50.0):
    """Add `candidate_tolerance_ppm` to `query_df` by looking `policy` up on `group_cols`. A
    group present in `query_df` but absent from `policy` (never seen while fitting) gets
    `fallback_ppm`, never a null tolerance.

    Deliberately a `.reindex()` lookup, NOT a `.merge()`: `DataFrame.merge` resets the result to
    a fresh `RangeIndex`, silently discarding `query_df`'s original index -- harmless on its
    own, but `evaluate_adaptive_policy_foldwise` concatenates several such calls (one per fold)
    and `sort_index()`s the result, which would then interleave rows from DIFFERENT folds under
    the same (reset, duplicated) index labels. Building the output as a plain array aligned
    positionally to `query_df` and wrapping it back in a Series with `query_df`'s own index
    keeps every row where it belongs."""
    group_cols = list(group_cols)
    policy_lookup = policy.set_index(group_cols)["tolerance_ppm"]
    keys = query_df[group_cols[0]] if len(group_cols) == 1 else pd.MultiIndex.from_frame(query_df[group_cols])

    out = query_df.copy()
    tolerance = pd.Series(policy_lookup.reindex(keys).to_numpy(), index=query_df.index)
    out["candidate_tolerance_ppm"] = tolerance.fillna(fallback_ppm)
    return out


def evaluate_adaptive_policy_foldwise(query_summary_with_error, group_cols, fold_col="fold",
                                       target_quantile=0.995, min_group_size=100, fallback_ppm=50.0,
                                       min_ppm=1.0, max_ppm=50.0, error_col="target_abs_mass_error_ppm"):
    """For every fold k: fit the policy from every OTHER fold's rows, apply it to fold k, and
    carry the result forward -- never estimates a fold's tolerance from itself. Returns
    `(per_query, fold_policies)`: `per_query` is `query_summary_with_error` with an added
    `candidate_tolerance_ppm` column (index-aligned with the input, order not preserved --
    reindex by the caller if needed), `fold_policies` is `{fold: policy_df}` for inspection."""
    out_parts = []
    fold_policies = {}
    for k in sorted(query_summary_with_error[fold_col].unique()):
        train_part = query_summary_with_error[query_summary_with_error[fold_col] != k]
        holdout_part = query_summary_with_error[query_summary_with_error[fold_col] == k].copy()

        policy = fit_adaptive_mass_windows(
            train_part, group_cols, target_quantile=target_quantile, min_group_size=min_group_size,
            fallback_ppm=fallback_ppm, min_ppm=min_ppm, max_ppm=max_ppm, error_col=error_col,
        )
        fold_policies[k] = policy
        out_parts.append(apply_adaptive_mass_windows(holdout_part, policy, group_cols, fallback_ppm=fallback_ppm))

    return pd.concat(out_parts, ignore_index=False).sort_index(), fold_policies


def assert_no_fold_leakage(fold_policies, queries_with_error, group_cols, fold_col="fold"):
    """Sanity check (not just a docstring claim): for every fold k, its policy's `n` per group
    must equal that group's total count MINUS fold k's own count -- i.e. the policy really was
    fit on the complement of fold k. Raises `AssertionError` on any mismatch."""
    totals = queries_with_error.groupby(list(group_cols), dropna=False).size()
    for k, policy in fold_policies.items():
        fold_counts = (
            queries_with_error[queries_with_error[fold_col] == k]
            .groupby(list(group_cols), dropna=False).size()
        )
        for _, row in policy.iterrows():
            key = tuple(row[c] for c in group_cols) if len(group_cols) > 1 else row[group_cols[0]]
            expected_n = totals.get(key, 0) - fold_counts.get(key, 0)
            assert row["n"] == expected_n, (
                f"fold {k} policy for group {key}: n={row['n']} but complement-of-fold-{k} has "
                f"{expected_n} rows -- possible fold leakage in policy fitting"
            )
