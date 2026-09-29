"""Neutral-loss representation: `precursor_mz - fragment_mz` for each fragment, physically
valid only where the loss is positive (a fragment above the precursor is measurement noise or a
multiply-charged/adduct artifact, not a real neutral loss -- see
`casmi.spectra.validation`/`preprocessing`, which already flag such fragments upstream; this
module doesn't re-filter, it assumes clean peaks in).

Two structurally different molecules can still lose the same small neutral fragments (water,
CO2, a common substituent) even when their fragment masses themselves don't line up -- the
motivation for comparing losses instead of (or alongside) raw fragment m/z.
"""
import numpy as np

from casmi.spectra.binning import bin_spectrum
from casmi.spectra.similarity import binned_cosine_similarity, peak_overlap


def neutral_losses(mzs, precursor_mz):
    """`precursor_mz - mz` for every fragment `mz` below the precursor; fragments at or above
    the precursor are dropped (see module docstring). Returns `(losses, intensities_mask)` --
    actually just the loss array; pair with the caller's own intensities filtered the same way
    via `valid_neutral_loss_mask`."""
    mzs = np.asarray(mzs, dtype=float)
    return precursor_mz - mzs


def valid_neutral_loss_mask(mzs, precursor_mz):
    return np.asarray(mzs, dtype=float) < precursor_mz


def neutral_loss_cosine_similarity(mzs_a, intensities_a, precursor_a, mzs_b, intensities_b, precursor_b, bin_width=0.1):
    """Binned-cosine similarity computed on NEUTRAL LOSSES instead of raw fragment m/z."""
    mask_a = valid_neutral_loss_mask(mzs_a, precursor_a)
    mask_b = valid_neutral_loss_mask(mzs_b, precursor_b)
    if not mask_a.any() or not mask_b.any():
        return 0.0
    losses_a = neutral_losses(np.asarray(mzs_a)[mask_a], precursor_a)
    losses_b = neutral_losses(np.asarray(mzs_b)[mask_b], precursor_b)
    bins_a, vals_a = bin_spectrum(losses_a, np.asarray(intensities_a)[mask_a], bin_width=bin_width)
    bins_b, vals_b = bin_spectrum(losses_b, np.asarray(intensities_b)[mask_b], bin_width=bin_width)
    return binned_cosine_similarity(bins_a, vals_a, bins_b, vals_b)


def neutral_loss_overlap(mzs_a, precursor_a, mzs_b, precursor_b, tol_da=0.02):
    """`casmi.spectra.similarity.peak_overlap`, computed on neutral losses instead of raw m/z."""
    mask_a = valid_neutral_loss_mask(mzs_a, precursor_a)
    mask_b = valid_neutral_loss_mask(mzs_b, precursor_b)
    if not mask_a.any() or not mask_b.any():
        return {"n_matched": 0, "frac_a_matched": 0.0, "frac_b_matched": 0.0}
    losses_a = neutral_losses(np.asarray(mzs_a)[mask_a], precursor_a)
    losses_b = neutral_losses(np.asarray(mzs_b)[mask_b], precursor_b)
    return peak_overlap(losses_a, losses_b, tol_da=tol_da)
