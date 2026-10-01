"""v4b near-miss analysis (absorbs the deferred C4/C5 T3-impostor question).

For every evaluated query whose truth is not at rank 1: the rank-1 wrong candidate vs the truth --
mass errors, spectral evidence, reference counts, formula equality, Morgan (r=2, 2048 bits)
Tanimoto, and whether the wrong candidate's ACCEPTED mirror-aware evidence contains a T3
near-duplicate of the query spectrum (a "T3 impostor": a DIFFERENT structure whose reference
spectrum is near-identical to the query).

Structural similarity is DIAGNOSTIC ONLY -- the truth structure is unknown at inference, so
truth-vs-decoy Tanimoto can never be a model feature.
"""
import numpy as np
import pandas as pd

SPECTRAL_COLS = ("cosine_max", "modified_cosine_max", "peak_overlap_frac_max", "neutral_loss_cosine_max")


def candidate_tier_flags(qcr_df, protocol="mirror_aware"):
    """Per (query, candidate): does its accepted evidence contain a T3 / T4 reference, and the
    best accepted cosine."""
    acc = qcr_df[qcr_df[f"accepted_{protocol}"].astype(bool)].assign(_t3=lambda d: d["tier"] == "T3")
    out = acc.groupby(["query_id", "candidate_key"]).agg(
        has_t3_accepted=("_t3", "any"), n_t3_accepted=("_t3", "sum"), best_ref_cosine=("cosine", "max"))
    return out.reset_index().rename(columns={"candidate_key": "candidate_connectivity_key"})


def build_near_miss_table(scored_df, per_query, score_col, formula_of_key, smiles_of_key, tier_flags,
                          eligible_counts, high_cosine=0.9, high_tanimoto=0.7):
    """`scored_df`: candidate rows with `rank`, `score_col`, mass + spectral features.
    `per_query`: output of `per_query_metrics` for the same ranking. Returns one row per failed
    query (truth rank > 1 or truth absent) with truth-vs-top-wrong diagnostics and category flags."""
    from casmi.chemistry.similarity import morgan_fingerprint, tanimoto_similarity

    joined = ("has_t3_accepted", "n_t3_accepted", "best_ref_cosine", "eligible_reference_count", "eligible_reference_saturated")
    d = scored_df.drop(columns=[c for c in joined if c in scored_df.columns])
    d = d.merge(tier_flags, on=["query_id", "candidate_connectivity_key"], how="left")
    d = d.merge(eligible_counts[["query_id", "candidate_connectivity_key", "eligible_reference_count", "eligible_reference_saturated"]],
                on=["query_id", "candidate_connectivity_key"], how="left")
    failed_ids = per_query.loc[per_query["truth_rank"].fillna(np.inf) > 1, "query_id"]
    top = d[(d["rank"] == 1) & d["query_id"].isin(failed_ids)].set_index("query_id")
    truth = d[d["is_true_candidate"].astype(bool) & d["query_id"].isin(failed_ids)].set_index("query_id")
    pool_size = d.groupby("query_id").size()
    truth_rank_of = per_query.set_index("query_id")["truth_rank"]
    fp_cache = {}

    def flag(v):
        return bool(v) if pd.notna(v) else False

    def fp(key):
        if key not in fp_cache:
            fp_cache[key] = morgan_fingerprint(smiles_of_key.get(key), radius=2, n_bits=2048)
        return fp_cache[key]

    rows = []
    for qid in failed_ids:
        if qid not in top.index:
            continue  # zero-candidate query: nothing to compare
        w = top.loc[qid]
        t = truth.loc[qid] if qid in truth.index else None
        rec = {"query_id": qid, "pool_size": int(pool_size.get(qid, 0)),
               "truth_rank": float(truth_rank_of.loc[qid]),
               "truth_in_pool": t is not None,
               "wrong_key": w["candidate_connectivity_key"], "wrong_score": float(w[score_col]),
               "wrong_abs_ppm": float(w["abs_mass_error_ppm"]),
               "wrong_eligible_refs": w.get("eligible_reference_count"), "wrong_has_t3": flag(w.get("has_t3_accepted")),
               "wrong_n_t3": w.get("n_t3_accepted")}
        for c in SPECTRAL_COLS:
            rec[f"wrong_{c}"] = float(w[c]) if pd.notna(w[c]) else np.nan
        if t is not None:
            rec.update({"truth_key": t["candidate_connectivity_key"], "truth_score": float(t[score_col]),
                        "truth_abs_ppm": float(t["abs_mass_error_ppm"]), "truth_eligible_refs": t.get("eligible_reference_count")})
            for c in SPECTRAL_COLS:
                rec[f"truth_{c}"] = float(t[c]) if pd.notna(t[c]) else np.nan
            rec["score_gap"] = rec["wrong_score"] - rec["truth_score"]
            rec["delta_abs_ppm"] = rec["truth_abs_ppm"] - rec["wrong_abs_ppm"]
            fw, ft = formula_of_key.get(w["candidate_connectivity_key"]), formula_of_key.get(t["candidate_connectivity_key"])
            rec["same_formula"] = bool(fw is not None and fw == ft)
            rec["tanimoto_truth_vs_wrong"] = tanimoto_similarity(fp(t["candidate_connectivity_key"]), fp(w["candidate_connectivity_key"]))
        rows.append(rec)
    nm = pd.DataFrame(rows)
    if nm.empty:
        return nm
    nm["cat_high_cosine_decoy"] = nm["wrong_cosine_max"].fillna(0) >= high_cosine
    nm["cat_t3_impostor"] = nm["wrong_has_t3"].fillna(False).astype(bool)
    nm["cat_same_formula"] = nm.get("same_formula", pd.Series(False, index=nm.index)).fillna(False).astype(bool)
    nm["cat_high_tanimoto"] = nm.get("tanimoto_truth_vs_wrong", pd.Series(np.nan, index=nm.index)).fillna(0) >= high_tanimoto
    nm["cat_rank2_near_miss"] = nm["truth_rank"] == 2
    nm["cat_truth_absent"] = ~nm["truth_in_pool"]
    return nm


def near_miss_category_summary(nm, n_queries_total):
    cats = [c for c in nm.columns if c.startswith("cat_")]
    return pd.DataFrame([{"category": c[4:], "n_failures": int(nm[c].sum()), "share_of_failures": float(nm[c].mean()) if len(nm) else np.nan,
                          "share_of_all_queries": float(nm[c].sum() / n_queries_total) if n_queries_total else np.nan} for c in cats])


def select_panel_examples(per_query, nm, scored_df, score_col, n=10, seed=42):
    """Deterministic example selection for the four panel families."""
    rng = np.random.default_rng(seed)
    out = {}
    s = scored_df.sort_values(["query_id", "rank"])
    top2 = s[s["rank"] <= 2].groupby("query_id")[score_col].apply(lambda v: float(v.iloc[0] - v.iloc[1]) if len(v) == 2 else np.nan)
    correct = per_query[per_query["truth_rank"] == 1].assign(margin=lambda d: d["query_id"].map(top2))
    out["high_confidence_correct"] = correct.sort_values(["margin", "query_id"], ascending=[False, True]).head(n)["query_id"].tolist()
    r2 = nm[nm["truth_rank"] == 2].sort_values(["score_gap", "query_id"]) if "score_gap" in nm else nm.iloc[0:0]
    out["rank2_near_miss"] = r2.head(n)["query_id"].tolist()
    if len(nm):
        big = nm[nm["pool_size"] >= nm["pool_size"].quantile(0.75)]
        out["large_pool_failure"] = big.sort_values(["pool_size", "query_id"], ascending=[False, True]).head(n)["query_id"].tolist()
        t3 = nm[nm["cat_t3_impostor"]]
        idx = rng.permutation(len(t3))[:n]
        out["t3_impostor_failure"] = t3.iloc[sorted(idx)]["query_id"].tolist()
    else:
        out["large_pool_failure"], out["t3_impostor_failure"] = [], []
    return out


def plot_example_panel(ax_row, query_peaks, truth_ref_peaks, wrong_ref_peaks, truth_smiles, wrong_smiles, title):
    """One row of a failure panel: [query vs truth best ref (mirror), query vs wrong best ref
    (mirror), truth structure, wrong structure]. `*_peaks`: {"mzs","intensities"} or None."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    def _mirror(ax, top, bottom, label):
        if top is not None:
            ax.vlines(top["mzs"], 0, np.asarray(top["intensities"]) / max(np.max(top["intensities"]), 1e-12), color="C0", lw=0.8)
        if bottom is not None and len(bottom["mzs"]):
            ax.vlines(bottom["mzs"], 0, -np.asarray(bottom["intensities"]) / max(np.max(bottom["intensities"]), 1e-12), color="C3", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(label, fontsize=8)
        ax.set_yticks([])

    _mirror(ax_row[0], query_peaks, truth_ref_peaks, "query (top) vs truth best ref (bottom)")
    _mirror(ax_row[1], query_peaks, wrong_ref_peaks, "query (top) vs wrong best ref (bottom)")
    for ax, smi, lab in ((ax_row[2], truth_smiles, "truth"), (ax_row[3], wrong_smiles, "top wrong")):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is not None:
            ax.imshow(Draw.MolToImage(mol, size=(260, 200)))
        ax.set_title(lab, fontsize=8)
        ax.axis("off")
    ax_row[0].set_ylabel(title, fontsize=7, rotation=0, ha="right", va="center")
