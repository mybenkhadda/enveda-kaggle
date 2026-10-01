"""Out-of-fold precursor mass-error calibration diagnostics.

Signed error of a query:  e_ppm = (observed_neutral_mass - true_exact_mass) / true_exact_mass * 1e6.
A systematic per-(instrument, adduct) offset widens the ppm window a retriever needs; correcting it can shrink
candidate pools without losing truths. Nothing here assumes an offset: everything is ESTIMATED.

Protocol (leakage-safe): offsets for fold f are fitted on the queries of folds != f only and applied to fold f
(`cross_fold_offsets`). Groups are hierarchical: (instrument, adduct) -> (instrument) -> global; a group with fewer
than `min_group_size` training queries falls back to its parent. A correction is only worth keeping if the held-out
before/after comparison (`calibration_effect`) shows smaller pools at equal or better recall.

    corrected_mass = observed_mass / (1 + offset_ppm * 1e-6)        (observed = true * (1 + e))
"""
import numpy as np
import pandas as pd

MAD_TO_SD = 1.4826


def signed_ppm(observed, reference):
    observed, reference = np.asarray(observed, float), np.asarray(reference, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (observed - reference) / reference * 1e6


def bootstrap_median_ci(x, n_bootstrap=200, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or not n_bootstrap:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(int(n_bootstrap), len(x)), replace=True), axis=1)
    return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def robust_summary(x, n_bootstrap=200, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "median_ppm": float("nan"), "robust_sd_ppm": float("nan"), "p05_ppm": float("nan"), "p95_ppm": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    med = float(np.median(x))
    lo, hi = bootstrap_median_ci(x, n_bootstrap, seed)
    return {"n": int(len(x)), "median_ppm": med, "robust_sd_ppm": float(MAD_TO_SD * np.median(np.abs(x - med))),
            "p05_ppm": float(np.quantile(x, 0.05)), "p95_ppm": float(np.quantile(x, 0.95)), "ci_low": lo, "ci_high": hi}


def _levels(group_cols):
    group_cols = list(group_cols)
    return [tuple(group_cols[:i]) for i in range(len(group_cols), -1, -1)]      # finest -> global ()


def _keys(df, cols):
    if not cols:
        return pd.Series(["__all__"] * len(df), index=df.index)
    return df[list(cols)].astype(str).agg("|".join, axis=1)


def fit_offsets(df, group_cols=("instrument_type", "adduct"), ppm_col="signed_ppm", min_group_size=50, n_bootstrap=200, seed=0,
                max_abs_ppm=50.0):
    """Hierarchical offset table fitted on `df` (the TRAINING folds). Rows with |e| > `max_abs_ppm` (adduct / charge
    labelling errors, not instrument error) are excluded from the fit and counted. Returns DataFrame(level, key, n,
    median_ppm, robust_sd_ppm, p05_ppm, p95_ppm, ci_low, ci_high, usable)."""
    e = df[ppm_col].astype(float)
    ok = e.abs() <= max_abs_ppm
    d = df[ok]
    rows = []
    for lv in _levels(group_cols):
        k = _keys(d, lv)
        for key, idx in k.groupby(k).groups.items():
            s = robust_summary(d.loc[idx, ppm_col], n_bootstrap, seed)
            rows.append({"level": "|".join(lv) or "global", "key": key, **s, "usable": s["n"] >= (min_group_size if lv else 1)})
    t = pd.DataFrame(rows, columns=["level", "key", "n", "median_ppm", "robust_sd_ppm", "p05_ppm", "p95_ppm", "ci_low", "ci_high",
                                    "usable"])
    t.attrs["n_excluded_gross_errors"] = int((~ok).sum())
    return t


def lookup_offsets(df, table, group_cols=("instrument_type", "adduct")):
    """Offset (ppm) per row of `df`: the finest USABLE group of `table`, else coarser, else global, else 0."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for lv in _levels(group_cols):
        level = "|".join(lv) or "global"
        t = table[(table["level"] == level) & table["usable"]].set_index("key")["median_ppm"]
        cand = _keys(df, lv).map(t)
        out = out.fillna(cand)
    return out.fillna(0.0)


def apply_offset(mass, offset_ppm):
    return np.asarray(mass, float) / (1.0 + np.asarray(offset_ppm, float) * 1e-6)


def cross_fold_offsets(df, fold_col="fold", group_cols=("instrument_type", "adduct"), ppm_col="signed_ppm", **fit_kw):
    """Out-of-fold offsets: for each fold f, fit on folds != f, look up for fold f. Returns (offset Series aligned with
    df, long table of the fitted offsets with a `held_out_fold` column)."""
    off = pd.Series(np.nan, index=df.index, dtype=float)
    tables = []
    for f in sorted(df[fold_col].dropna().unique()):
        train, test = df[df[fold_col] != f], df[df[fold_col] == f]
        t = fit_offsets(train, group_cols, ppm_col, **fit_kw)
        off.loc[test.index] = lookup_offsets(test, t, group_cols).to_numpy()
        tables.append(t.assign(held_out_fold=f))
    return off, (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame())


def calibration_effect(index, masses_before, masses_after, truth_ids, ppm_grid, query_ids=None, exclude_mask=None, k_values=(25, 100)):
    """Before / after recall + pool size on held-out queries (truth reachable in the universe), per ppm. Uses the
    same retrieval order as Gate A (`casmi.candidates.recall.recall_sweep`). Returns a long table."""
    from casmi.candidates.recall import recall_by_ppm, recall_sweep
    parts = []
    for name, m in (("before", masses_before), ("after", masses_after)):
        s = recall_sweep(index, np.asarray(m, float), np.asarray(truth_ids), list(ppm_grid), exclude_mask=exclude_mask, query_ids=query_ids)
        parts.append(recall_by_ppm(s, k_values=k_values).assign(variant=name))
    return pd.concat(parts, ignore_index=True)
