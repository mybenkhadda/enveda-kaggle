"""Molecule-level late fusion: per-spectrum candidate scores -> one ranking per molecule.

The competition scores ONE ranked list per molecule, and a molecule may have several spectra. In development, a
"molecule" is the set of evaluated query spectra that share a truth connectivity inside one fold (all spectra of a
connectivity share fold and regime by construction).

Fusion methods (over the spectra whose candidate pool contains the candidate; a candidate absent from a spectrum's
pool contributes nothing, so pools that differ slightly between adducts never inject fake zeros):

    mean      mean score
    max       max score
    lse       temperature * log(mean(exp(score / temperature)))       (soft max; T -> 0 = max, T -> inf = mean)
    rrf       sum_s 1 / (rrf_k + rank_s(candidate))                    (score-scale free)
    vote      mean over spectra of 1[candidate in the spectrum's top-k] (+ tiny mean-score tie-break)

The returned table is directly usable with `casmi.validation.metrics.ranking_metrics` (column `truth_rank`).

Protocol caveat (documented in notebook 14): the frozen C1 protocol excludes, per query, only the query spectrum and
its T1/T2 duplicates from the reference library. Sibling query spectra of the same molecule therefore remain
references for each other, which makes molecule-level C1 OPTIMISTIC. C2 / C3 are unaffected (every reference of the
truth is hidden).
"""
import numpy as np
import pandas as pd

FUSION_METHODS = ("mean", "max", "lse", "rrf", "vote")


def molecule_ids(queries, cols=("fold", "true_connectivity_key")):
    """Deterministic molecule id per query row ('<fold>|<truth key>')."""
    return queries[list(cols)].astype(str).agg("|".join, axis=1)


def _per_spectrum_rank(df, query_col, score_col, cand_col):
    d = df.sort_values([query_col, score_col, cand_col], ascending=[True, False, True], kind="mergesort")
    return d.assign(_srank=d.groupby(query_col, sort=False).cumcount() + 1)


def fuse(features, score_col, query_to_molecule, method="mean", query_col="query_id", cand_col="candidate_id",
         is_true_col="is_true_candidate", temperature=1.0, rrf_k=60, vote_k=25):
    """Candidate-level fused scores per molecule: DataFrame[molecule_id, cand_col, fused_score, n_spectra, is_true]."""
    if method not in FUSION_METHODS:
        raise ValueError(f"method must be one of {FUSION_METHODS}")
    d = features[[query_col, cand_col, score_col, is_true_col]].copy()
    d = d[np.isfinite(d[score_col].to_numpy(float))]
    d["molecule_id"] = d[query_col].map(query_to_molecule)
    if d["molecule_id"].isna().any():
        raise ValueError(f"{int(d['molecule_id'].isna().sum())} feature rows have no molecule id")
    keys = ["molecule_id", cand_col]
    if method in ("rrf", "vote"):
        d = _per_spectrum_rank(d, query_col, score_col, cand_col)
    if method == "mean":
        g = d.groupby(keys, sort=False)[score_col].mean()
    elif method == "max":
        g = d.groupby(keys, sort=False)[score_col].max()
    elif method == "lse":
        t = float(temperature)
        d["_m"] = d.groupby(keys, sort=False)[score_col].transform("max")
        d["_e"] = np.exp((d[score_col] - d["_m"]) / t)
        agg = d.groupby(keys, sort=False).agg(_s=("_e", "mean"), _m=("_m", "first"))
        g = agg["_m"] + t * np.log(agg["_s"])
    elif method == "rrf":
        d["_r"] = 1.0 / (rrf_k + d["_srank"])
        g = d.groupby(keys, sort=False)["_r"].sum()
    else:  # vote
        n_spec = d.groupby("molecule_id")[query_col].nunique()
        d["_v"] = (d["_srank"] <= vote_k).astype(float)
        agg = d.groupby(keys, sort=False).agg(_v=("_v", "sum"), _mean=(score_col, "mean"))
        mol = agg.index.get_level_values("molecule_id")
        g = agg["_v"] / n_spec.reindex(mol).to_numpy() + 1e-6 * agg["_mean"].rank(pct=True)
    out = g.rename("fused_score").reset_index()
    meta = d.groupby(keys, sort=False).agg(n_spectra=(query_col, "nunique"), is_true=(is_true_col, "max")).reset_index()
    return out.merge(meta, on=keys, how="left")


def molecule_truth_ranks(fused, molecules, cand_col="candidate_id"):
    """One row per molecule in `molecules` (DataFrame with molecule_id + any metadata, e.g. fold, regime):
    `truth_rank` (1-based, NaN if absent), `pool_size`. Order: fused_score DESC, cand_col ASC (deterministic)."""
    f = fused.sort_values(["molecule_id", "fused_score", cand_col], ascending=[True, False, True], kind="mergesort")
    f = f.assign(_rank=f.groupby("molecule_id", sort=False).cumcount() + 1)
    truth = f[f["is_true"].astype(bool)].groupby("molecule_id")["_rank"].min()
    pool = f.groupby("molecule_id").size()
    m = molecules.drop_duplicates("molecule_id").copy()
    m["truth_rank"] = m["molecule_id"].map(truth).astype(float)
    m["pool_size"] = m["molecule_id"].map(pool).fillna(0).astype(int)
    return m.reset_index(drop=True)


def evaluate_fusions(features, score_col, queries, methods=FUSION_METHODS, temperatures=(0.5, 1.0, 2.0), **kw):
    """`queries` needs query_id, fold, regime, true_connectivity_key. Returns per-molecule ranks for every
    (method[, temperature]) as one long table with a `fusion` column."""
    q = queries.copy()
    q["molecule_id"] = molecule_ids(q)
    q2m = dict(zip(q["query_id"].astype(str), q["molecule_id"]))
    mols = q.groupby("molecule_id", sort=True).agg(fold=("fold", "first"), regime=("regime", "first"),
                                                   true_connectivity_key=("true_connectivity_key", "first"),
                                                   n_spectra=("query_id", "size")).reset_index()
    feats = features.assign(query_id=features["query_id"].astype(str))
    parts = []
    for m in methods:
        for t in (temperatures if m == "lse" else (None,)):
            fused = fuse(feats, score_col, q2m, method=m, temperature=t or 1.0, **kw)
            name = f"{m}(T={t})" if t is not None else m
            parts.append(molecule_truth_ranks(fused, mols).assign(fusion=name))
    return pd.concat(parts, ignore_index=True)
