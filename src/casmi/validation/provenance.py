"""v5.2 provenance audit of the TRUE candidate's references (TL_EVAL STANDARD_ONLY queries first).

Tiers come from the QCR `tier` column, i.e. the output of the existing
`casmi.spectra.deduplication.classify_identity_tier` -- no second classifier exists here.
For a STANDARD_ONLY query the truth has 0 mirror_aware references, so its walk never reached 5
mirror_aware acceptances and therefore EXHAUSTED the reference list: every truth reference is in QCR
(`truth_refs_complete` records this per query; it is asserted, not assumed).

Semantic categories (labels, not experimental claims):
    T1 -> EXACT_DUPLICATE_LIKE      T2 -> NEAR_DUPLICATE_LIKE      T3/T4 -> INDEPENDENT_REFERENCE_LIKE
`INDEPENDENT_REFERENCE_LIKE` is never upgraded to "independent" without acquisition provenance.

`ingest_lib` is a library/dataset grouping. It is NOT automatically an acquisition session, a replicate,
a duplicate or a hidden-test leakage boundary -- which is exactly what this audit examines.
"""
import numpy as np
import pandas as pd

TIER_ORDER = ("T1", "T2", "T3", "T4")                 # T1 = highest leakage risk
CATEGORY = {"T1": "EXACT_DUPLICATE_LIKE", "T2": "NEAR_DUPLICATE_LIKE", "T3": "INDEPENDENT_REFERENCE_LIKE", "T4": "INDEPENDENT_REFERENCE_LIKE"}
NOT_AVAILABLE = "NOT_AVAILABLE"
SESSION_FIELD_CANDIDATES = ("session_id", "run_id", "batch_id", "file_id", "acquisition_id", "raw_file", "scan_id", "experiment_id", "sample_id")
PAIR_QCR_COLS = ["query_id", "candidate_key", "is_true", "ref_spectrum_id", "compat_rank", "tier", "mirror", "near_dup_non_mirror",
                 "query_source", "ref_source", "query_instrument", "ref_instrument", "query_adduct", "ref_adduct", "query_ion_mode", "ref_ion_mode",
                 "ref_ce", "eligible_standard", "eligible_cross_library", "eligible_mirror_aware", "eligible_near_dup_strict", "accepted_standard"]


def session_fields_available(train_meta_columns):
    """Which acquisition-session-like fields exist; NOT_AVAILABLE when none (never inferred)."""
    present = [c for c in SESSION_FIELD_CANDIDATES if c in set(train_meta_columns)]
    return present or NOT_AVAILABLE


def build_truth_reference_pairs(qcr_truth, query_ids, meta, peak_hash=None, identity_thresholds=None, session_fields=NOT_AVAILABLE):
    """One row per (query, truth reference). `qcr_truth`: truth rows of QCR (PAIR_QCR_COLS). `meta`:
    per-spectrum `precursor_mz`, `ce` indexed by spectrum id. `peak_hash`: optional Series of the
    exported T1 peak hash per spectrum id (bundle ref_meta) -- NOT_AVAILABLE otherwise."""
    p = qcr_truth[qcr_truth["query_id"].isin(set(query_ids))].copy()
    p = p.rename(columns={"candidate_key": "truth_connectivity_key", "query_ion_mode": "query_polarity", "ref_ion_mode": "ref_polarity",
                          "ref_ce": "ref_collision_energy"})
    p["same_source"] = p["query_source"].astype(str) == p["ref_source"].astype(str)
    p["query_collision_energy"] = p["query_id"].map(meta["ce"])
    p["query_precursor_mz"] = p["query_id"].map(meta["precursor_mz"])
    p["ref_precursor_mz"] = p["ref_spectrum_id"].map(meta["precursor_mz"])
    p["semantic_category"] = p["tier"].map(CATEGORY)
    p["selected_standard_top5"] = p["accepted_standard"].astype(bool)
    for proto in ("standard", "cross_library", "mirror_aware", "near_dup_strict"):
        p[f"eligible_{proto}"] = p[f"eligible_{proto}"].astype(bool)
    if peak_hash is not None:
        p["query_peak_hash"] = p["query_id"].map(peak_hash)
        p["ref_peak_hash"] = p["ref_spectrum_id"].map(peak_hash)
        p["peak_hash_equal"] = (p["query_peak_hash"] == p["ref_peak_hash"]) & p["query_peak_hash"].notna()
    else:
        p["query_peak_hash"] = p["ref_peak_hash"] = NOT_AVAILABLE
        p["peak_hash_equal"] = NOT_AVAILABLE
    th = identity_thresholds or {}
    p = p.assign(**acquisition_differences(p, th))
    p["differs_session"] = NOT_AVAILABLE if session_fields == NOT_AVAILABLE else "SEE_SESSION_FIELDS"
    return p.reset_index(drop=True)


def acquisition_differences(p, identity_thresholds):
    """Metadata differences between query and reference. A comparison with a missing side is
    'UNKNOWN' (never counted as different)."""
    def cmp(a, b):
        a, b = p[a], p[b]
        unknown = a.isna() | b.isna()
        return np.where(unknown, "UNKNOWN", np.where(a.astype(str) != b.astype(str), "DIFFERENT", "SAME"))

    out = {"differs_spectrum_id": np.where(p["query_id"] != p["ref_spectrum_id"], "DIFFERENT", "SAME"),
           "differs_adduct": cmp("query_adduct", "ref_adduct"), "differs_polarity": cmp("query_polarity", "ref_polarity"),
           "differs_instrument": cmp("query_instrument", "ref_instrument")}
    qce, rce = p["query_collision_energy"].astype(float), p["ref_collision_energy"].astype(float)
    out["differs_collision_energy"] = np.where(qce.isna() | rce.isna(), "UNKNOWN", np.where(np.abs(qce - rce) > 1e-9, "DIFFERENT", "SAME"))
    dmz = (p["query_precursor_mz"].astype(float) - p["ref_precursor_mz"].astype(float)).abs()
    for name, key in (("t2", "t2_precursor_diff_da"), ("t3", "t3_precursor_diff_da")):
        tol = identity_thresholds.get(key)
        out[f"precursor_beyond_{name}_tolerance"] = (np.where(dmz.isna(), "UNKNOWN", np.where(dmz > tol, "DIFFERENT", "SAME"))
                                                     if tol is not None else NOT_AVAILABLE)
    return out


def _highest_risk(tiers):
    for t in TIER_ORDER:
        if t in tiers:
            return t
    return "NONE"


def query_level_summary(pairs, query_ids):
    """PRIMARY table: one row per query (not pair-weighted). Tier presence among ALL truth references
    and among the STANDARD-selected top-5, same-source subsets, highest-risk tier, best non-T1/T2 tier."""
    rows = []
    by_q = {q: g for q, g in pairs.groupby("query_id")}
    for q in query_ids:
        g = by_q.get(q, pairs.iloc[0:0])
        tiers, sel = set(g["tier"]), set(g.loc[g["selected_standard_top5"], "tier"])
        same = g[g["same_source"]]
        rec = {"query_id": q, "n_truth_refs": int(len(g)), "n_same_source_truth_refs": int(len(same)),
               "n_selected_standard": int(g["selected_standard_top5"].sum())}
        for t in TIER_ORDER:
            rec[f"has_{t}"] = t in tiers
            rec[f"same_source_has_{t}"] = t in set(same["tier"])
            rec[f"selected_has_{t}"] = t in sel
        rec["selected_all_T1_T2"] = bool(sel) and sel <= {"T1", "T2"}
        rec["selected_has_T3_T4"] = bool(sel & {"T3", "T4"})
        rec["highest_risk_tier"] = _highest_risk(tiers)
        rec["best_non_T1_T2_tier"] = "T3" if "T3" in tiers else ("T4" if "T4" in tiers else "NONE")
        rows.append(rec)
    return pd.DataFrame(rows)


def tier_shares(pairs, qsum):
    """Pair-level shares (secondary) and query-level shares (primary)."""
    pair = pairs["tier"].value_counts(normalize=True).reindex(list(TIER_ORDER), fill_value=0.0).to_dict()
    sel = pairs[pairs["selected_standard_top5"]]["tier"].value_counts(normalize=True).reindex(list(TIER_ORDER), fill_value=0.0).to_dict()
    n = len(qsum)
    ql = {k: float(qsum[k].mean()) if n else float("nan") for k in
          [f"has_{t}" for t in TIER_ORDER] + [f"selected_has_{t}" for t in TIER_ORDER] + ["selected_all_T1_T2", "selected_has_T3_T4"]}
    return {"n_queries": int(n), "n_pairs": int(len(pairs)), "pair_level_tier_share": pair, "selected_pair_level_tier_share": sel,
            "query_level": ql, "highest_risk_tier_share": qsum["highest_risk_tier"].value_counts(normalize=True).to_dict() if n else {},
            "best_non_T1_T2_tier_share": qsum["best_non_T1_T2_tier"].value_counts(normalize=True).to_dict() if n else {}}


def t2_rule_inputs(qsum):
    """q_t34 / q_t12_only over queries WITH >= 1 selected standard reference (query-level)."""
    s = qsum[qsum["n_selected_standard"] > 0]
    return {"q_t34": float(s["selected_has_T3_T4"].mean()) if len(s) else 0.0,
            "q_t12_only": float(s["selected_all_T1_T2"].mean()) if len(s) else 0.0, "n_queries_with_selected_refs": int(len(s))}


def provenance_by_source(regimes, qsum):
    """Per query source: n_queries, regime shares, and (STANDARD_ONLY truths) selected-tier shares."""
    d = regimes[["query_id", "source", "regime"]].merge(qsum, on="query_id", how="left")
    rows = []
    for src, g in d.groupby("source", dropna=False, sort=True):
        so = g[g["regime"] == "STANDARD_ONLY"]
        rows.append({"source": src, "n_queries": int(len(g)), "STANDARD_ONLY_share": float((g["regime"] == "STANDARD_ONLY").mean()),
                     "MODE_A_MIRROR_share": float((g["regime"] == "MODE_A_MIRROR").mean()), "n_standard_only": int(len(so)),
                     "so_selected_has_T1_share": float(so["selected_has_T1"].mean()) if len(so) else np.nan,
                     "so_selected_has_T2_share": float(so["selected_has_T2"].mean()) if len(so) else np.nan,
                     "so_selected_has_T3_T4_share": float(so["selected_has_T3_T4"].mean()) if len(so) else np.nan})
    return pd.DataFrame(rows)


def connectivity_source_summary(truth_keys_with_source, train_meta, mirror_eligible_counts=None):
    """Per TL_EVAL truth connectivity: #libraries, #spectra, same-source / cross-source spectra
    (relative to the QUERY's source; the query itself excluded), eligible cross-source references."""
    tm = train_meta[train_meta["connectivity_key"].isin(set(truth_keys_with_source["connectivity_key"]))][
        ["train_spectrum_id", "connectivity_key", "ingest_lib"]]
    lib_counts = tm.groupby("connectivity_key")["ingest_lib"].nunique()
    rows = []
    by_key = {k: g for k, g in tm.groupby("connectivity_key")}
    for r in truth_keys_with_source.itertuples(index=False):
        g = by_key.get(r.connectivity_key, tm.iloc[0:0])
        g = g[g["train_spectrum_id"] != r.query_id]
        rows.append({"query_id": r.query_id, "connectivity_key": r.connectivity_key, "query_source": r.source,
                     "n_source_libraries": int(lib_counts.get(r.connectivity_key, 0)), "n_spectra_total": int(len(g) + 1),
                     "n_same_source_spectra": int((g["ingest_lib"] == r.source).sum()), "n_cross_source_spectra": int((g["ingest_lib"] != r.source).sum()),
                     "n_eligible_cross_source_refs_mirror_aware": (int(mirror_eligible_counts.get(r.query_id, 0)) if mirror_eligible_counts is not None else NOT_AVAILABLE)})
    out = pd.DataFrame(rows)
    out["pattern"] = np.where(out["n_cross_source_spectra"] == 0, "A_SINGLE_SOURCE_ONLY", "B_MULTI_SOURCE_BUT_EXCLUDED_OR_OTHER")
    return out
