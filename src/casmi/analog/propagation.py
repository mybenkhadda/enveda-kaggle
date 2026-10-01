"""Structural propagation: analog spectra -> evidence for mass-retrieved candidates.

For one query with analogs a = 1..k (spectral similarity s_a = modified cosine, rank r_a, analog molecule
fingerprint F_a) and candidates c (fingerprint F_c):

    T[c, a]  = Tanimoto(F_c, F_a)                       (packed Morgan bits, `fingerprint_store`)
    w_a      = max(s_a, 0) ** p                          (`similarity_weight_power`)
    support  = T >= tau                                  (`support_tanimoto_threshold`)

Candidate features (all prefixed `analog_`, see `casmi.analog.features.ANALOG_FEATURES`):
    propagation_score        sum_a w_a T[c,a] / sum_a w_a
    max_weighted             max_a s_a T[c,a]
    structural_similarity    max_a T[c,a]
    top1_tanimoto            T[c, best analog]
    similarity_max           max s_a over SUPPORTING analogs (0 if none)
    rank                     min r_a over supporting analogs (k+1 if none)
    support_count            # supporting analog spectra
    support_molecules        # distinct supporting analog connectivities
    exact_structure_support  max s_a over analogs whose connectivity IS the candidate (0 if none)
    same_formula_support     sum_a w_a [formula_a == formula_c] / sum_a w_a
    mass_delta               query neutral mass - exact mass of the best (max s*T) analog molecule
    precursor_delta          precursor delta of that analog spectrum
Query-level context broadcast to every candidate:
    best_score (s_1), topk_mean, confidence (s_1 - s_2), same_adduct_support (share of analogs with the
    query adduct), n_analogs.
"""
import numpy as np

from casmi.chemistry.fingerprint_store import tanimoto_matrix

CANDIDATE_LEVEL = ("propagation_score", "max_weighted", "structural_similarity", "top1_tanimoto", "similarity_max", "rank",
                   "support_count", "support_molecules", "exact_structure_support", "same_formula_support", "mass_delta",
                   "precursor_delta")
QUERY_LEVEL = ("best_score", "topk_mean", "confidence", "same_adduct_support", "n_analogs")


def empty_features(n_candidates, k):
    out = {f"analog_{c}": np.zeros(n_candidates, np.float32) for c in CANDIDATE_LEVEL}
    out["analog_rank"][:] = k + 1
    out["analog_mass_delta"][:] = np.nan
    out["analog_precursor_delta"][:] = np.nan
    for c in QUERY_LEVEL:
        out[f"analog_{c}"] = np.zeros(n_candidates, np.float32)
    return out


def propagate(cand_bits, cand_valid, cand_keys, cand_formulas, analog_bits, analog_valid, analog_sims, analog_ranks, analog_keys,
              analog_formulas, analog_masses, analog_precursor_deltas, analog_same_adduct, query_neutral_mass,
              support_threshold=0.5, weight_power=2.0, top_k=10):
    """Feature arrays (one value per candidate) for ONE query. Analog arrays have length k (<= top_k)."""
    n, k = len(cand_keys), len(analog_sims)
    out = empty_features(n, top_k)
    if n == 0 or k == 0:
        return out
    s = np.clip(np.asarray(analog_sims, dtype=np.float64), 0, None)
    T = tanimoto_matrix(cand_bits, analog_bits).astype(np.float64)
    T[~np.asarray(cand_valid, bool), :] = np.nan
    T[:, ~np.asarray(analog_valid, bool)] = np.nan
    T0 = np.nan_to_num(T, nan=0.0)
    w = s ** weight_power
    wsum = w.sum()
    sup = T0 >= support_threshold
    r = np.asarray(analog_ranks, dtype=np.float64)
    akeys = np.asarray(analog_keys).astype(str)
    ckeys = np.asarray(cand_keys).astype(str)

    out["analog_propagation_score"] = ((T0 * w).sum(1) / wsum if wsum > 0 else np.zeros(n)).astype(np.float32)
    sT = T0 * s
    out["analog_max_weighted"] = sT.max(1).astype(np.float32)
    out["analog_structural_similarity"] = T0.max(1).astype(np.float32)
    out["analog_top1_tanimoto"] = T0[:, int(np.argmin(r))].astype(np.float32)
    out["analog_similarity_max"] = np.where(sup, s, 0.0).max(1).astype(np.float32)
    out["analog_rank"] = np.where(sup, r, top_k + 1).min(1).astype(np.float32)
    out["analog_support_count"] = sup.sum(1).astype(np.float32)
    uk = np.unique(akeys)
    out["analog_support_molecules"] = np.stack([sup[:, akeys == u].any(1) for u in uk], 1).sum(1).astype(np.float32)
    same_struct = ckeys[:, None] == akeys[None, :]
    out["analog_exact_structure_support"] = np.where(same_struct, s, 0.0).max(1).astype(np.float32)
    same_f = np.asarray(cand_formulas).astype(str)[:, None] == np.asarray(analog_formulas).astype(str)[None, :]
    out["analog_same_formula_support"] = ((same_f * w).sum(1) / wsum if wsum > 0 else np.zeros(n)).astype(np.float32)
    best = np.argmax(sT, axis=1)
    has = sT.max(1) > 0
    am = np.asarray(analog_masses, dtype=np.float64)
    pd_ = np.asarray(analog_precursor_deltas, dtype=np.float64)
    out["analog_mass_delta"] = np.where(has, query_neutral_mass - am[best], np.nan).astype(np.float32)
    out["analog_precursor_delta"] = np.where(has, pd_[best], np.nan).astype(np.float32)

    order = np.argsort(r)
    s_sorted = s[order]
    out["analog_best_score"][:] = s_sorted[0]
    out["analog_topk_mean"][:] = s.mean()
    out["analog_confidence"][:] = s_sorted[0] - (s_sorted[1] if k > 1 else 0.0)
    out["analog_same_adduct_support"][:] = float(np.mean(np.asarray(analog_same_adduct, bool)))
    out["analog_n_analogs"][:] = k
    return out
