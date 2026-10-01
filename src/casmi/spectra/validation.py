"""Spectrum-level validity flags. Read-only: never modifies the arrays it's given -- cleanup
lives in `casmi.spectra.preprocessing`, kept deliberately separate so "what's wrong with this
spectrum" and "how do we fix it" can't accidentally become entangled in one function.
"""
import numpy as np


def validate_spectrum(mzs, intensities, precursor_mz=None, fragment_above_precursor_da=5.0, extreme_fragment_mz=5000.0):
    """Return a dict of boolean flags describing potential problems with one spectrum.

    `length_mismatch=True` means the two other length-dependent flags below it
    (nonfinite/negative/unsorted/duplicate m/z) are computed against `mzs` only, since a
    shape mismatch means intensities can't be reliably paired to m/z values.
    """
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    flags = {
        "empty": len(mzs) == 0,
        "length_mismatch": len(mzs) != len(intensities),
        "nonfinite_mz": bool(len(mzs) and (~np.isfinite(mzs)).any()),
        "nonfinite_intensity": bool(len(intensities) and (~np.isfinite(intensities)).any()),
        "negative_mz": bool(len(mzs) and (mzs < 0).any()),
        "negative_intensity": bool(len(intensities) and (intensities < 0).any()),
        "unsorted_mz": bool(len(mzs) > 1 and not np.all(np.diff(mzs) >= 0)),
        "duplicate_mz": bool(len(mzs) and len(mzs) != len(np.unique(mzs))),
        "fragment_above_precursor": False,
        "extreme_fragment_mz": bool(len(mzs) and (mzs > extreme_fragment_mz).any()),
    }
    if precursor_mz is not None and len(mzs):
        flags["fragment_above_precursor"] = bool((mzs > precursor_mz + fragment_above_precursor_da).any())
    return flags


def is_clean(flags):
    """True if `validate_spectrum`'s flags describe a spectrum with no problems at all."""
    return not any(flags.values())
