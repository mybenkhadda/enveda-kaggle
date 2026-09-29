"""Sparse m/z-binned spectrum representation -- deliberately NOT a dense
(n_spectra x n_global_bins) matrix (see module docstring pattern in `casmi.candidates`: a
sparse, indexed representation is the only one that scales past a handful of spectra when most
bins are empty).
"""
import numpy as np


def bin_spectrum(mzs, intensities, bin_width=0.1, l2_normalize=True):
    """Bin `mzs`/`intensities` into `bin_width`-Da bins. Multiple peaks landing in the same bin
    have their intensities summed (not overwritten). Returns `(bin_indices, bin_values)` --
    both sorted by `bin_indices`, zeros never materialized -- so two binned spectra can be
    compared with a merge-style dot product (`casmi.spectra.similarity.binned_cosine_similarity`)
    without ever forming a dense vector.

    `l2_normalize`: if True (the default), `bin_values` is scaled so its own L2 norm is 1 --
    then `binned_cosine_similarity` reduces to a plain dot product.
    """
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if len(mzs) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    bin_idx = np.floor(mzs / bin_width).astype(np.int64)
    order = np.argsort(bin_idx)
    bin_idx = bin_idx[order]
    intensities = intensities[order]

    unique_idx, start_pos = np.unique(bin_idx, return_index=True)
    summed = np.add.reduceat(intensities, start_pos)

    if l2_normalize:
        norm = np.linalg.norm(summed)
        if norm > 0:
            summed = summed / norm
    return unique_idx, summed
