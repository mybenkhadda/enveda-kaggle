"""Query spectrum preprocessing -- the SAME functions training used (imported, not copied):

    identity representation   remove_invalid_peaks(ms2_mzs, ms2_normalized_intensities)   (T1 hash)
    similarity representation truncate_top_peaks(identity, max_peaks)  -- deterministic top-N:
                              intensity DESC, m/z ASC, re-sorted by m/z
No intensity transform beyond what the raw `ms2_normalized_intensities` already are (training
applied none); float64 throughout.
"""
import numpy as np

from casmi.spectra.deduplication import compute_peak_hash
from casmi.spectra.preprocessing import remove_invalid_peaks, truncate_top_peaks


def clean_query(mzs, intensities, precursor_mz, max_peaks):
    """Returns `(identity_peaks, similarity_peaks)` dicts in the shape training used."""
    mz, it = remove_invalid_peaks(mzs if mzs is not None else [], intensities if intensities is not None else [])
    identity = {"mzs": mz, "intensities": it, "precursor_mz": float(precursor_mz)}
    s_mz, s_it = truncate_top_peaks(mz, it, max_peaks=max_peaks)
    return identity, {"mzs": s_mz, "intensities": s_it, "precursor_mz": float(precursor_mz)}


def query_peak_hash(identity):
    """T1 key: exactly `classify_identity_tier`'s default peak-hash canonicalization."""
    return compute_peak_hash(identity["mzs"], identity["intensities"], identity["precursor_mz"])


def ce_mean(collision_energy_ev):
    """RE-IMPLEMENTATION (parity-tested against `casmi.data.metadata.summarize_collision_energy`):
    mean of the non-NaN values of a scalar/list collision energy; NaN if none."""
    if collision_energy_ev is None:
        return float("nan")
    arr = np.asarray(collision_energy_ev, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def query_meta(adduct, ionization_mode, instrument_type, collision_energy_ev, source=None):
    """Query metadata in the fields compat ordering / eligibility use. `source` is None for hidden
    test spectra (unpublished; same-source exclusion is not meaningful)."""
    return {"adduct": adduct, "ion_mode": ionization_mode, "instrument": instrument_type,
            "ce": ce_mean(collision_energy_ev), "source": source}
