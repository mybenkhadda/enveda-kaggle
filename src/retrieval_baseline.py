"""Verified candidate-retrieval baseline functions, shared by notebooks 02b and 03+.

This module exists so the retrieval pipeline has exactly ONE implementation. It re-exports the
already-correct adduct/mass and spectrum-preprocessing functions from `chemistry.py`/`spectra.py`
(so there is no second, slightly-different copy of that logic), and adds the pieces that were
previously duplicated ad hoc inside notebook code: mass-candidate generation, aggregation, and
critically the corrected hybrid mass+spectral scorer.

**Why a hybrid scorer, and what bug this fixes** (diagnosed in notebook 02b): notebook 02's
`cosine_retrieve_hier` partitioned candidates into "has a reference spectrum" (cosine-scored,
placed first) vs. "no reference spectrum" (mass-order fallback, placed after ALL cosine-scored
candidates, however weak). Under connectivity-grouped CV, a query's own true connectivity has
essentially zero reference spectra in the fold-complement pool (see notebook 02 Section 6), so
the true candidate was almost always relegated to the fallback bucket -- pushed below *any* wrong
candidate that happened to have a reference spectrum, however poor its mass fit. This explains
why the old B2-B6 scored materially worse than mass-only B1, and why B2-B6 looked "wired
identically" (the ion/adduct/CE conditioning only reshuffled the small reference-bearing set,
never the dominant partition). `hybrid_candidate_score` below replaces the hard partition with a
smooth, additive blend: a candidate's mass fit always contributes, and spectral evidence adds to
(never gates) the score.
"""
import hashlib
from pathlib import Path

import numpy as np

from . import chemistry, spectra as spec_utils, metrics as metrics_mod

MODULE_PATH = Path(__file__).resolve()


def module_version_hash():
    """Short content hash of this file, printed by notebooks so it's clear which
    implementation of the baseline was actually imported."""
    return hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()[:12]


# --- Re-exported, unchanged verified logic ------------------------------------------------
parse_adduct = chemistry.parse_adduct
formula_mass = chemistry.formula_mass
neutral_mass_from_precursor = chemistry.neutral_mass_from_precursor
preprocess_spectrum_normalize = spec_utils.normalize_intensity
fixed_bin_vector = spec_utils.bin_spectrum
fixed_bin_vectors_batch = spec_utils.bin_spectra_batch
modified_cosine = spec_utils.modified_cosine
evaluate_retrieval = metrics_mod.evaluate_retrieval
recall_at_k = metrics_mod.recall_at_k


def reciprocal_rank(ranked, true, k=25):
    for i, c in enumerate(ranked[:k]):
        if c == true:
            return 1.0 / (i + 1)
    return 0.0


def mrr_at_k(ranked_lists, true_labels, k=25):
    return metrics_mod.mrr_at_k(ranked_lists, true_labels, k)


def hit_at_k(ranked_lists, true_labels, k):
    return metrics_mod.recall_at_k(ranked_lists, true_labels, k)


def median_target_rank(ranked_lists, true_labels):
    ranks = []
    for cands, true in zip(ranked_lists, true_labels):
        try:
            ranks.append(cands.index(true) + 1)
        except ValueError:
            ranks.append(np.nan)
    return float(np.nanmedian(ranks)) if ranks else np.nan


def evaluate_rankings(ranked_lists, true_labels, ks=(1, 5, 10, 25)):
    """The project's one ranking-evaluation function (notebooks 02/02b/03 all call this one,
    not a locally redefined variant) -- MRR@25, Hit@k, and median target rank."""
    out = {"mrr_at_25": mrr_at_k(ranked_lists, true_labels, 25)}
    for k in ks:
        out[f"hit_at_{k}"] = hit_at_k(ranked_lists, true_labels, k)
    out["median_target_rank"] = median_target_rank(ranked_lists, true_labels)
    out["n_queries"] = len(true_labels)
    return out


def preprocess_spectrum(mz, intensity):
    """Deterministic MS/MS cleaning + sqrt/L2 transform. Never mutates its inputs.

    Clean arrays -> drop non-finite / negative-intensity peaks -> sort by m/z -> merge
    duplicate m/z (sum intensity) -> sqrt-transform -> L2-normalize.
    """
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    n = min(len(mz), len(intensity))
    mz, intensity = mz[:n].copy(), intensity[:n].copy()

    finite = np.isfinite(mz) & np.isfinite(intensity)
    mz, intensity = mz[finite], intensity[finite]

    keep = intensity >= 0
    mz, intensity = mz[keep], intensity[keep]
    if mz.size == 0:
        return mz, intensity

    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]

    round_mz = np.round(mz, 4)
    uniq_mz, inverse = np.unique(round_mz, return_inverse=True)
    if len(uniq_mz) != len(mz):
        merged = np.zeros(len(uniq_mz))
        np.add.at(merged, inverse, intensity)
        mz, intensity = uniq_mz, merged

    intensity = spec_utils.normalize_intensity(intensity, method="sqrt")
    norm = np.sqrt((intensity ** 2).sum())
    if norm > 0:
        intensity = intensity / norm
    return mz, intensity


def build_adduct_table(adducts):
    """One row per unique observed adduct string -> parsed mass-arithmetic coefficients.

    Correctly handles generic `[nM+A]^z` multimer/dimer adducts (n > 1 divides the inferred
    neutral mass, matching `chemistry.neutral_mass_from_precursor`) -- this is the "corrected
    multimer/dimer neutral-mass logic" notebook 02b validates against known examples.
    """
    import pandas as pd

    rows = []
    for add in sorted(set(a for a in adducts if a is not None and a == a)):  # a == a filters NaN
        parsed = chemistry.parse_adduct(add)
        rec = {"adduct": add, "supported": False, "z": np.nan, "n": np.nan,
               "charge": np.nan, "delta": np.nan, "electron_term": np.nan}
        if parsed is not None:
            delta, ok = 0.0, True
            for sign, count, frag in parsed["terms"]:
                fm = chemistry.formula_mass(frag)
                if fm is None:
                    ok = False
                    break
                delta += sign * count * fm
            if ok:
                z = parsed["z"]
                electron_term = -z * chemistry.ELECTRON_MASS if parsed["polarity"] == "+" else z * chemistry.ELECTRON_MASS
                rec.update(supported=True, z=z, n=parsed["n"],
                           charge=(z if parsed["polarity"] == "+" else -z),
                           delta=delta, electron_term=electron_term)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("adduct")


class MassCandidateIndex:
    """Binary-search candidate generator over a mass-sorted structure library.

    `structures` is a DataFrame with at least `connectivity_key` and `exact_mass`, deduplicated
    by connectivity (one row per unique structure) -- this represents the mass/structure
    candidate *universe*, independent of whether any reference spectrum exists for a structure.
    """

    def __init__(self, structures, key_col="connectivity_key", mass_col="exact_mass"):
        ref = structures.dropna(subset=[key_col, mass_col]).sort_values(mass_col)
        self.masses = ref[mass_col].to_numpy()
        self.keys = ref[key_col].to_numpy()

    def generate(self, query_mass, ppm):
        if not np.isfinite(query_mass):
            return np.array([], dtype=object), np.array([])
        tol = query_mass * ppm * 1e-6
        lo = np.searchsorted(self.masses, query_mass - tol, side="left")
        hi = np.searchsorted(self.masses, query_mass + tol, side="right")
        return self.keys[lo:hi], self.masses[lo:hi]

    def rank_by_mass(self, query_mass, ppm, widen_to=(20, 50, 100)):
        keys, masses = self.generate(query_mass, ppm)
        used_ppm = ppm
        if len(keys) == 0:
            for wider in widen_to:
                if wider <= ppm:
                    continue
                keys, masses = self.generate(query_mass, wider)
                used_ppm = wider
                if len(keys):
                    break
        if len(keys) == 0:
            return [], used_ppm, np.array([])
        ppm_err = np.abs(1e6 * (query_mass - masses) / masses)
        order = np.argsort(ppm_err, kind="stable")
        return keys[order].tolist(), used_ppm, ppm_err[order]


def aggregate_reference_scores(sims, method="max"):
    """Aggregate a candidate's per-reference-spectrum similarity scores into one score.
    Verified against synthetic ground truth in notebook 02b's aggregation audit."""
    sims = np.asarray(sims, dtype=float)
    if sims.size == 0:
        return np.nan
    if method == "max":
        return float(sims.max())
    if method == "mean":
        return float(sims.mean())
    if method == "top3":
        return float(np.sort(sims)[::-1][:3].mean())
    if method == "top5":
        return float(np.sort(sims)[::-1][:5].mean())
    raise ValueError(f"unknown aggregation method: {method}")


def mass_score_from_ppm(abs_ppm_error, ppm_scale=10.0):
    """Smooth, bounded (0, 1] mass-fit score: 1.0 for a perfect match, decaying with ppm error.
    Always defined (never NaN/partitioned away), which is exactly what fixes the notebook-02 bug:
    a candidate's mass fit contributes to its score whether or not it has a reference spectrum."""
    abs_ppm_error = np.asarray(abs_ppm_error, dtype=float)
    return np.exp(-abs_ppm_error / ppm_scale)


def hybrid_candidate_score(abs_ppm_error, spectral_score, has_reference,
                            w_mass=1.0, w_spectral=1.0, ppm_scale=10.0):
    """Additive hybrid score: mass fit ALWAYS contributes; spectral evidence ADDS to it when
    available, rather than gating candidates into a hard "has reference / doesn't" partition.

    This is the corrected replacement for notebook 02's `(ranked_with_ref + remaining)`
    partition, which silently buried the true candidate (almost always reference-less under
    grouped CV) below any weakly-scored reference-bearing wrong candidate.
    """
    abs_ppm_error = np.asarray(abs_ppm_error, dtype=float)
    spectral_score = np.asarray(spectral_score, dtype=float)
    has_reference = np.asarray(has_reference, dtype=bool)
    mass_term = mass_score_from_ppm(abs_ppm_error, ppm_scale=ppm_scale)
    spectral_term = np.where(has_reference, np.nan_to_num(spectral_score, nan=0.0), 0.0)
    return w_mass * mass_term + w_spectral * spectral_term
