"""Candidate feature rows, exactly the frozen model's features: the 8 spectral aggregates of the
(<= 5) selected references + `abs_mass_error_ppm`. No reference-count / availability feature.

Order of operations (training's): protocol eligibility -> compat_rank -> first 5 eligible -> the 4
similarities per reference -> `aggregate_pair_scores_from_values` (max, top3_mean over available
refs; NaN only when no reference is available).
"""
import numpy as np
import pandas as pd

from casmi.ranking.features import aggregate_pair_scores_from_values
from casmi_infer.compat import select_references
from casmi_infer.similarity import pair_scores

SPECTRAL_FEATURES = ("cosine_max", "cosine_top3_mean", "modified_cosine_max", "modified_cosine_top3_mean",
                     "peak_overlap_frac_max", "peak_overlap_frac_top3_mean", "neutral_loss_cosine_max", "neutral_loss_cosine_top3_mean")


def candidate_features(lib, cand_conn_idx, cand_abs_ppm, query_sim, qmeta, query_hash, cfg,
                       exclude_sources=frozenset(), exclude_ids=frozenset(), counters=None):
    """One row per candidate: `conn_idx`, `abs_mass_error_ppm`, the 8 spectral features, and
    audit-only columns (`n_selected_refs`, `n_refs_total`, `n_t1_excluded`) that never reach the model."""
    rows = []
    for ci, ap in zip(cand_conn_idx, cand_abs_ppm):
        sel, n_total, n_t1 = select_references(lib, int(ci), qmeta, query_hash, exclude_sources, exclude_ids, cfg["top_reference_count"])
        vals = [pair_scores(query_sim, lib.peaks(int(r)), cfg) for r in sel]
        if counters is not None:
            counters["similarity_evaluations"] = counters.get("similarity_evaluations", 0) + len(vals)
            counters["candidate_pairs"] = counters.get("candidate_pairs", 0) + 1
        agg = aggregate_pair_scores_from_values(*(zip(*vals) if vals else ([], [], [], [])))
        row = {"conn_idx": int(ci), "abs_mass_error_ppm": float(ap), "n_selected_refs": len(sel), "n_refs_total": int(n_total),
               "n_t1_excluded": n_t1}
        row.update({f: agg[f] for f in SPECTRAL_FEATURES})
        rows.append(row)
    cols = ["conn_idx", "abs_mass_error_ppm", *SPECTRAL_FEATURES, "n_selected_refs", "n_refs_total", "n_t1_excluded"]
    return pd.DataFrame(rows, columns=cols)


def model_matrix(features_df, feature_names):
    """Exactly `feature_names` (the frozen training order), float64. Raises on a missing column."""
    missing = [f for f in feature_names if f not in features_df.columns]
    if missing:
        raise KeyError(f"frozen model features missing from the inference frame: {missing}")
    return features_df[list(feature_names)].astype(np.float64)
