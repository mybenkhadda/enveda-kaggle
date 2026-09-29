"""The four training similarity kernels: binned cosine, modified cosine, peak-overlap fraction,
neutral-loss cosine.

numpy backend: `casmi.ranking.features.compute_single_pair_scores` unchanged (the validated path).
numba backend: identical computation except the two inner loops in `casmi_infer.backend`; it must pass
the bundled self-test before it is trusted (see `casmi_infer.selftest`).
"""
import numpy as np

from casmi.ranking.features import compute_single_pair_scores
from casmi.spectra.binning import bin_spectrum
from casmi.spectra.neutral_loss import neutral_loss_cosine_similarity
from casmi.spectra.similarity import modified_cosine_similarity
from casmi_infer.backend import NUMBA, active_backend, numba_kernels


def _pair_scores_numba(q, r, bin_width, peak_tol_da):
    k = numba_kernels()
    qb, qv = bin_spectrum(q["mzs"], q["intensities"], bin_width=bin_width)
    rb, rv = bin_spectrum(r["mzs"], r["intensities"], bin_width=bin_width)
    cosine = float(k["sorted_bin_dot"](qb, qv, rb, rv)) if len(qb) and len(rb) else 0.0
    mod = modified_cosine_similarity(q["mzs"], q["intensities"], q["precursor_mz"], r["mzs"], r["intensities"], r["precursor_mz"],
                                     tol_da=peak_tol_da)["score"]
    a = np.sort(np.asarray(q["mzs"], dtype=float))
    b = np.sort(np.asarray(r["mzs"], dtype=float))
    overlap = (k["overlap_count"](a, b, peak_tol_da) / len(a)) if len(a) and len(b) else 0.0
    nl = neutral_loss_cosine_similarity(q["mzs"], q["intensities"], q["precursor_mz"], r["mzs"], r["intensities"], r["precursor_mz"],
                                        bin_width=bin_width)
    return cosine, mod, float(overlap), nl


def pair_scores(query_sim_peaks, ref_sim_peaks, cfg):
    """`(cosine, modified_cosine, peak_overlap_frac, neutral_loss_cosine)` for one pair."""
    if active_backend() == NUMBA:
        return _pair_scores_numba(query_sim_peaks, ref_sim_peaks, cfg["bin_width_da"], cfg["peak_tol_da"])
    return compute_single_pair_scores(query_sim_peaks, ref_sim_peaks, bin_width=cfg["bin_width_da"], peak_tol_da=cfg["peak_tol_da"])
