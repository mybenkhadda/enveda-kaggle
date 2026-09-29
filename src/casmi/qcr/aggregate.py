"""Candidate-level feature construction from a persisted QCR table -- ZERO similarity calls, pure
filter/aggregate (v4b).

Three reference-selection modes, one aggregation:

    deterministic   the protocol's ACCEPTED references (compat-rank order, <=5) -- the normal
                    evaluation features; `k` optionally keeps only the first k (k-stress).
    dropout         TRAINING AUGMENTATION ONLY: per (query, candidate) row draw k from a
                    configurable distribution over {1, 3, 5}, then sample min(k, n) references
                    uniformly at random from the row's full ELIGIBLE set as available in QCR (not
                    the top-compat ones). The sampler never reads `is_true`, so truths and decoys
                    are treated identically by construction.

Aggregation contract (identical to `casmi.ranking.aggregation.aggregate_scores`):
    `<metric>_max`        max over the selected references
    `<metric>_top3_mean`  mean over the top min(3, n) selected values -- 1 ref -> that value,
                          2 refs -> mean of both, >=3 -> mean of the top 3. NEVER NaN merely because
                          fewer than 3 references exist (a NaN there would encode the reference count,
                          the Mode-B missingness shortcut in reverse). NaN only when n == 0.
"""
import numpy as np
import pandas as pd

from casmi.spectra.reference_selection import MAX_REFS_PER_PROTOCOL

SPECTRAL_METRICS = ("cosine", "modified_cosine", "peak_overlap_frac", "neutral_loss_cosine")
SPECTRAL_FEATURE_COLS = tuple(f"{m}_{a}" for m in SPECTRAL_METRICS for a in ("max", "top3_mean"))
TOP3_COLS = tuple(f"{m}_top3_mean" for m in SPECTRAL_METRICS)
DEFAULT_DROPOUT_K = (1, 3, 5)

_QCR_KEY = ["query_id", "candidate_key"]
_POOL_KEY = ["query_id", "candidate_connectivity_key"]


def _aggregate_selected(selected, pool_df):
    """`selected`: QCR rows already filtered to the references each (query, candidate) uses.
    Returns `pool_df` (one row per pool row, order preserved) with n_reference_spectra,
    has_reference_spectrum and the 8 spectral features merged on."""
    base_cols = [c for c in pool_df.columns if c not in SPECTRAL_FEATURE_COLS + ("n_reference_spectra", "has_reference_spectrum")]
    out = pool_df[base_cols].copy()
    if len(selected) == 0:
        out["n_reference_spectra"] = 0
        out["has_reference_spectrum"] = False
        for c in SPECTRAL_FEATURE_COLS:
            out[c] = np.nan
        return out

    counts = selected.groupby(_QCR_KEY, sort=False).size().rename("n_reference_spectra")
    parts = [counts]
    for m in SPECTRAL_METRICS:
        s = selected[_QCR_KEY + [m]].sort_values(_QCR_KEY + [m], ascending=[True, True, False], kind="mergesort")
        g = s.groupby(_QCR_KEY, sort=False)[m]
        parts.append(g.max().rename(f"{m}_max"))
        top = s[g.cumcount() < 3]
        parts.append(top.groupby(_QCR_KEY, sort=False)[m].mean().rename(f"{m}_top3_mean"))
    agg = pd.concat(parts, axis=1).reset_index().rename(columns={"candidate_key": "candidate_connectivity_key"})
    out = out.merge(agg, on=_POOL_KEY, how="left", validate="one_to_one")
    out["n_reference_spectra"] = out["n_reference_spectra"].fillna(0).astype(int)
    out["has_reference_spectrum"] = out["n_reference_spectra"] > 0
    return out[base_cols + ["n_reference_spectra", "has_reference_spectrum", *SPECTRAL_FEATURE_COLS]]


def features_without_references(pool_df):
    """Feature rows for pool rows that have NO walked reference at all (they never appear in QCR):
    n_reference_spectra=0 and every spectral feature NaN."""
    return _aggregate_selected(pd.DataFrame(columns=_QCR_KEY), pool_df)


def aggregate_deterministic(qcr_df, pool_df, protocol="mirror_aware", k=None):
    """Normal evaluation features: the protocol's accepted references in compat-rank order,
    optionally truncated to the first `k` (k-stress; k=None or k>=5 reproduces the full table)."""
    acc = qcr_df[qcr_df[f"accepted_{protocol}"].astype(bool)]
    if k is not None:
        acc = acc.sort_values(_QCR_KEY + ["compat_rank"], kind="mergesort")
        acc = acc[acc.groupby(_QCR_KEY, sort=False).cumcount() < int(k)]
    return _aggregate_selected(acc, pool_df)


def sample_dropout_k(n_rows, k_choices=DEFAULT_DROPOUT_K, k_probs=None, rng=None):
    """Per-row training reference budget k drawn from `k_choices` with probabilities `k_probs`
    (uniform if None)."""
    rng = np.random.default_rng(0) if rng is None else rng
    k_choices = np.asarray(k_choices, dtype=int)
    if k_probs is None:
        k_probs = np.full(len(k_choices), 1.0 / len(k_choices))
    k_probs = np.asarray(k_probs, dtype=float)
    if len(k_probs) != len(k_choices) or np.any(k_probs < 0) or not np.isclose(k_probs.sum(), 1.0):
        raise ValueError(f"k_probs must be a probability vector over k_choices, got {k_probs}")
    return rng.choice(k_choices, size=n_rows, p=k_probs)


def select_dropout_references(qcr_df, protocol="mirror_aware", k_choices=DEFAULT_DROPOUT_K, k_probs=None, seed=42):
    """TRAINING AUGMENTATION: returns the QCR rows kept after random reference dropout, plus a
    per-(query, candidate) table of the drawn budget k and the eligible-set size.

    Deterministic under `seed` for a given QCR content: rows are first put into a canonical order
    (query_id, candidate_key, ref_spectrum_id) so the draw does not depend on on-disk row order.
    Draws from `eligible_<protocol>` (the full eligible set available in QCR), NOT from
    `accepted_<protocol>`, and never looks at `is_true`."""
    elig = qcr_df[qcr_df[f"eligible_{protocol}"].astype(bool)]
    elig = elig.sort_values(_QCR_KEY + ["ref_spectrum_id"], kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = elig[_QCR_KEY].drop_duplicates().reset_index(drop=True)
    groups["dropout_k"] = sample_dropout_k(len(groups), k_choices, k_probs, rng)
    elig = elig.merge(groups, on=_QCR_KEY, how="left", validate="many_to_one")
    elig["_u"] = rng.random(len(elig))
    elig["_r"] = elig.groupby(_QCR_KEY, sort=False)["_u"].rank(method="first")
    kept = elig[elig["_r"] <= elig["dropout_k"]].drop(columns=["_u", "_r"])
    sizes = elig.groupby(_QCR_KEY, sort=False).size().rename("n_eligible_in_qcr").reset_index()
    groups = groups.merge(sizes, on=_QCR_KEY, how="left")
    groups["n_sampled"] = np.minimum(groups["dropout_k"], groups["n_eligible_in_qcr"])
    return kept, groups


def aggregate_reference_dropout(qcr_df, pool_df, protocol="mirror_aware", k_choices=DEFAULT_DROPOUT_K, k_probs=None, seed=42):
    """Training feature table under random reference dropout. Returns `(features, draw_log)`."""
    kept, draw_log = select_dropout_references(qcr_df, protocol, k_choices, k_probs, seed)
    return _aggregate_selected(kept, pool_df), draw_log


# ---------------------------------------------------------------------------------------------
# eligible reference counts (censored) -- AUDIT ONLY, never a default model feature
# ---------------------------------------------------------------------------------------------

def eligible_reference_counts(qcr_df, pairs_df, pool_df, protocol="mirror_aware", max_refs=MAX_REFS_PER_PROTOCOL):
    """Per pool row: `eligible_reference_count` (exact when known, else the lower bound `max_refs`),
    `eligible_reference_count_is_exact`, `eligible_reference_saturated` (censored at >= max_refs),
    and `popularity_level` (0..max_refs, with `max_refs` meaning ">= max_refs").

    Exactness follows `casmi.qcr.censored`: fewer than `max_refs` accepted means the walk
    exhausted the list, so the count is exact; `max_refs` accepted AND `exhausted` means the
    walked eligible count is exact (possibly > max_refs); otherwise censored. Pool rows with no
    QCR rows at all have 0 references (exact)."""
    q = qcr_df[_QCR_KEY + [f"eligible_{protocol}", f"accepted_{protocol}"]]
    agg = q.groupby(_QCR_KEY, sort=False).agg(n_eligible_walked=(f"eligible_{protocol}", "sum"),
                                              n_accepted=(f"accepted_{protocol}", "sum")).reset_index()
    agg = agg.merge(pairs_df[_QCR_KEY + ["exhausted"]], on=_QCR_KEY, how="left", validate="one_to_one")
    agg = agg.rename(columns={"candidate_key": "candidate_connectivity_key"})
    out = pool_df[_POOL_KEY].merge(agg, on=_POOL_KEY, how="left", validate="one_to_one")
    out["n_eligible_walked"] = out["n_eligible_walked"].fillna(0).astype(int)
    out["n_accepted"] = out["n_accepted"].fillna(0).astype(int)
    out["exhausted"] = out["exhausted"].fillna(True).astype(bool)
    is_exact = (out["n_accepted"] < max_refs) | out["exhausted"]
    out["eligible_reference_count_is_exact"] = is_exact
    out["eligible_reference_count"] = np.where(is_exact, out["n_eligible_walked"], max_refs).astype(int)
    out["eligible_reference_saturated"] = out["eligible_reference_count"] >= max_refs
    out["popularity_level"] = popularity_level(out["eligible_reference_count"], out["eligible_reference_saturated"], max_refs)
    return out[_POOL_KEY + ["eligible_reference_count", "eligible_reference_count_is_exact", "eligible_reference_saturated",
                             "popularity_level"]]


def popularity_level(count, saturated, max_refs=MAX_REFS_PER_PROTOCOL):
    """Censoring-safe ordinal: `>=max_refs` (saturated OR an exact count >= max_refs) > exact
    max_refs-1 > ... > exact 1 > 0. An exact 7 and a censored `>=5` get the SAME level -- the
    censored representation cannot distinguish them, so the ordering must not either."""
    count = np.asarray(count, dtype=int)
    saturated = np.asarray(saturated, dtype=bool)
    return np.where(saturated | (count >= max_refs), max_refs, np.clip(count, 0, max_refs)).astype(int)


# ---------------------------------------------------------------------------------------------
# top3_mean / missingness audits
# ---------------------------------------------------------------------------------------------

def reference_count_bin(n):
    n = np.asarray(n, dtype=int)
    return np.select([n <= 0, n == 1, n == 2], ["0", "1", "2"], default=">=3")


def top3_sanity_table(features_df, ref_count_col="n_reference_spectra"):
    """Fraction NaN of every `*_top3_mean` / `*_max` feature, per reference-count bin {0,1,2,>=3},
    plus the max |top3_mean - max| for 1-ref rows (must be 0: one ref -> top3_mean == that value)."""
    df = features_df.assign(_bin=reference_count_bin(features_df[ref_count_col]))
    rows = []
    for b, g in df.groupby("_bin", sort=True):
        rec = {"ref_count_bin": b, "n_rows": len(g)}
        for c in SPECTRAL_FEATURE_COLS:
            rec[f"nan_frac_{c}"] = float(g[c].isna().mean())
        if b == "1":
            for m in SPECTRAL_METRICS:
                rec[f"max_absdiff_top3_vs_max_{m}"] = float((g[f"{m}_top3_mean"] - g[f"{m}_max"]).abs().max())
        rows.append(rec)
    return pd.DataFrame(rows)


def assert_no_top3_missingness_leak(features_df, ref_count_col="n_reference_spectra", atol=1e-12):
    """Hard assertion (raises `AssertionError`): wherever a candidate has >=1 selected reference
    and a finite `<metric>_max`, its `<metric>_top3_mean` is finite; for exactly 1 reference,
    top3_mean equals max; for 0 references both are NaN. Together these guarantee the NaN
    pattern of the top3 features can only encode "0 vs >=1 references" (the same information
    `<metric>_max` already carries), never 1 vs 2 vs >=3."""
    n = features_df[ref_count_col].to_numpy()
    problems = []
    for m in SPECTRAL_METRICS:
        mx, t3 = features_df[f"{m}_max"].to_numpy(float), features_df[f"{m}_top3_mean"].to_numpy(float)
        bad_nan = (n >= 1) & np.isfinite(mx) & ~np.isfinite(t3)
        if bad_nan.any():
            problems.append(f"{m}_top3_mean NaN on {int(bad_nan.sum())} rows with >=1 ref and finite max")
        one = (n == 1) & np.isfinite(mx)
        if one.any() and np.nanmax(np.abs(t3[one] - mx[one])) > atol:
            problems.append(f"{m}_top3_mean != {m}_max on 1-ref rows")
        zero_not_nan = (n == 0) & (np.isfinite(t3) | np.isfinite(mx))
        if zero_not_nan.any():
            problems.append(f"{m}: finite value on {int(zero_not_nan.sum())} rows with 0 refs")
    assert not problems, "top3 missingness contract violated: " + "; ".join(problems)
    return True


def nan_fraction_table(df, feature_cols, table_name):
    return pd.DataFrame([{"table": table_name, "feature": c, "nan_frac": float(df[c].isna().mean()), "n_rows": len(df)}
                         for c in feature_cols])


def nan_pattern_by_ref_count(df, feature_cols, ref_count_col="n_reference_spectra"):
    """Crosstab of the joint NaN pattern of `feature_cols` against the ref-count bin -- for rows
    with >=1 reference there must be exactly ONE pattern (all finite), otherwise the NaN pattern
    carries reference-count information."""
    pattern = df[list(feature_cols)].isna().apply(lambda r: "".join("N" if v else "." for v in r), axis=1)
    return pd.crosstab(pattern, reference_count_bin(df[ref_count_col]))
