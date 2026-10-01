"""v4b transparent Mode-A baselines. Each returns PER-QUERY metrics over the full query list.

    B1   mass only                  abs_mass_error_ppm ASC, key ASC
    B1L  mass likelihood            mass_loglik DESC, abs ppm ASC, key ASC
    B2a  availability + mass        has >=1 eligible mirror-aware ref DESC, abs ppm ASC, key ASC   (AUDIT)
    B2b  reference popularity       popularity_level DESC, abs ppm ASC, key ASC                    (POPULARITY AUDIT ONLY -- NOT A PRODUCTION FEATURE)
    B3a  peak overlap, neutral ties peak_overlap_frac_max, EXPECTED metrics under random tie order -- no mass at all
    B3b  peak overlap + mass tie    peak_overlap_frac_max DESC, abs ppm ASC, key ASC  (NOT pure spectrum-only: mass is a deterministic secondary signal)
    B4   peak overlap + mass fusion z(peak_overlap_frac_max) + alpha * z(mass_loglik); alpha tuned on DEV only
    B5   modcos + mass fusion       z(modified_cosine_max) + alpha * z(mass_loglik); alpha tuned on DEV only

Missing spectral evidence (a candidate with no eligible reference) sorts LAST for B3a/B3b (a NaN
key) and is scored as 0 similarity before standardization for B4/B5 (a fused score needs a
value; 0 = "no spectral support"). The z-standardization parameters are fitted on DEV rows only
and frozen before any HOST evaluation.
"""
import numpy as np
import pandas as pd

from casmi.ranking.rank_eval import expected_tie_metrics, per_query_metrics, rank_by_keys, summarize

KEY = "candidate_connectivity_key"
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)

BASELINE_LABELS = {
    "B1": "B1 mass only",
    "B1L": "B1L mass likelihood",
    "B2a": "B2a availability + mass (AUDIT)",
    "B2b": "B2b -- POPULARITY AUDIT ONLY -- NOT A PRODUCTION FEATURE",
    "B3a": "B3a peak overlap, expected random ties (spectrum-only)",
    "B3b": "B3b peak overlap + mass tie-break (NOT pure spectrum-only)",
    "B4": "B4 peak overlap + alpha*mass likelihood",
    "B5": "B5 modified cosine + alpha*mass likelihood",
}


def run_keyed(df, keys, all_query_ids):
    return per_query_metrics(rank_by_keys(df, keys), all_query_ids)


def b1(df, ids):
    return run_keyed(df, [("abs_mass_error_ppm", True), (KEY, True)], ids)


def b1l(df, ids):
    return run_keyed(df, [("mass_loglik", False), ("abs_mass_error_ppm", True), (KEY, True)], ids)


def b2a(df, ids):
    d = df.assign(_has_ref=(df["n_reference_spectra"] > 0).astype(int))
    return run_keyed(d, [("_has_ref", False), ("abs_mass_error_ppm", True), (KEY, True)], ids)


def b2b(df, ids):
    return run_keyed(df, [("popularity_level", False), ("abs_mass_error_ppm", True), (KEY, True)], ids)


def b3a(df, ids):
    return expected_tie_metrics(df, "peak_overlap_frac_max", ids, higher_is_better=True)


def b3b(df, ids):
    return run_keyed(df, [("peak_overlap_frac_max", False), ("abs_mass_error_ppm", True), (KEY, True)], ids)


# ------------------------------------------------------------------------------------------------
# fusion (B4/B5)
# ------------------------------------------------------------------------------------------------

def fit_standardizer(dev_df, cols):
    """Mean/std per column over DEV rows (spectral NaN -> 0 first). Frozen afterwards."""
    out = {}
    for c in cols:
        v = dev_df[c].astype(float).fillna(0.0).to_numpy()
        sd = float(v.std())
        out[c] = {"mean": float(v.mean()), "std": sd if sd > 0 else 1.0}
    return out


def fused_score(df, spectral_col, alpha, standardizer, mass_col="mass_loglik"):
    s = (df[spectral_col].astype(float).fillna(0.0) - standardizer[spectral_col]["mean"]) / standardizer[spectral_col]["std"]
    m = (df[mass_col].astype(float) - standardizer[mass_col]["mean"]) / standardizer[mass_col]["std"]
    return s + alpha * m


def fusion(df, ids, spectral_col, alpha, standardizer):
    d = df.assign(_fused=fused_score(df, spectral_col, alpha, standardizer))
    return run_keyed(d, [("_fused", False), ("abs_mass_error_ppm", True), (KEY, True)], ids)


def select_alpha_dev(dev_df, dev_query_ids, fold_of_query, spectral_col, standardizer, alpha_grid=ALPHA_GRID):
    """DEV-ONLY, connectivity-safe alpha choice. `fold_of_query`: {query_id: connectivity fold}.

    Returns `(chosen_alpha, grid_table, cv_table)`:
      grid_table   DEV MRR@25 of every alpha on ALL DEV queries (all-query denominator)
      cv_table     nested estimate: for each fold f, alpha* = argmax over the OTHER folds, scored on
                   fold f -- an honest DEV estimate of the tuning procedure itself
      chosen_alpha argmax over all DEV queries; exact ties resolve to the SMALLER alpha.
    No HOST input exists in this signature by design."""
    per_alpha = {}
    for a in alpha_grid:
        pq = fusion(dev_df, dev_query_ids, spectral_col, a, standardizer)
        pq["fold"] = pq["query_id"].map(fold_of_query)
        per_alpha[a] = pq
    grid = pd.DataFrame([{"alpha": a, "dev_mrr_at_25": float(pq["rr"].mean())} for a, pq in per_alpha.items()])
    chosen = float(grid.sort_values(["dev_mrr_at_25", "alpha"], ascending=[False, True]).iloc[0]["alpha"])
    cv_rows = []
    folds = sorted(pd.Series(list(fold_of_query.values())).dropna().unique())
    for f in folds:
        inner = {a: float(pq.loc[pq["fold"] != f, "rr"].mean()) for a, pq in per_alpha.items()}
        best = sorted(inner.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        outer = per_alpha[best]
        cv_rows.append({"fold": f, "alpha_selected_on_other_folds": best,
                        "heldout_mrr_at_25": float(outer.loc[outer["fold"] == f, "rr"].mean()),
                        "n_heldout_queries": int((outer["fold"] == f).sum())})
    return chosen, grid, pd.DataFrame(cv_rows)


def run_all_baselines(df, ids, alphas, standardizer, include=("B1", "B1L", "B2a", "B2b", "B3a", "B3b", "B4", "B5")):
    """`{baseline_id: per_query_df}` for one dataset with a FROZEN `alphas` dict and standardizer."""
    fns = {
        "B1": lambda: b1(df, ids), "B1L": lambda: b1l(df, ids), "B2a": lambda: b2a(df, ids), "B2b": lambda: b2b(df, ids),
        "B3a": lambda: b3a(df, ids), "B3b": lambda: b3b(df, ids),
        "B4": lambda: fusion(df, ids, "peak_overlap_frac_max", alphas["B4"], standardizer),
        "B5": lambda: fusion(df, ids, "modified_cosine_max", alphas["B5"], standardizer),
    }
    return {b: fns[b]() for b in include}


def summary_table(per_query_by_model, dataset):
    rows = []
    for mid, pq in per_query_by_model.items():
        rows.append({"model": mid, "dataset": dataset, **summarize(pq)})
    return pd.DataFrame(rows)
