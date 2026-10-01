"""Per-spectrum numeric features for EDA and (later) modeling. Deterministic and numerically
safe: an empty spectrum or a zero-precursor never raises, it just returns NaN/0 for whichever
fields are genuinely undefined.
"""
import numpy as np

FEATURE_FIELDS = [
    "n_peaks", "mz_min", "mz_max", "mz_mean", "mz_median", "mz_std", "fragment_mz_range",
    "base_peak_mz", "base_peak_intensity", "total_intensity", "top1_intensity_fraction",
    "top3_intensity_fraction", "top5_intensity_fraction", "top10_intensity_fraction",
    "spectral_entropy", "max_fragment_minus_precursor", "n_fragments_above_precursor",
    "fraction_fragments_above_precursor",
]


def _spectral_entropy(intensities):
    inten = intensities[intensities > 0]
    if inten.size == 0:
        return 0.0
    p = inten / inten.sum()
    return float(-(p * np.log(p)).sum())


def _topk_intensity_fraction(sorted_desc_intensities, total, k):
    if total <= 0:
        return 0.0
    return float(sorted_desc_intensities[:k].sum() / total)


def compute_spectrum_features(mzs, intensities, precursor_mz=None):
    """One dict of features for a single spectrum. `mzs`/`intensities` are assumed already
    valid (see `casmi.spectra.validation`/`preprocessing`) -- this function doesn't filter,
    it measures."""
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    n = len(mzs)

    if n == 0:
        feats = {k: 0.0 for k in FEATURE_FIELDS}
        feats["n_peaks"] = 0
        for k in ("mz_min", "mz_max", "mz_mean", "mz_median", "mz_std", "fragment_mz_range",
                  "base_peak_mz", "base_peak_intensity", "max_fragment_minus_precursor"):
            feats[k] = float("nan")
        return feats

    total_intensity = float(intensities.sum())
    base_idx = int(np.argmax(intensities))
    sorted_desc = np.sort(intensities)[::-1]

    n_above = int((mzs > precursor_mz).sum()) if precursor_mz is not None else 0
    max_frag_minus_precursor = float(mzs.max() - precursor_mz) if precursor_mz is not None else float("nan")

    return {
        "n_peaks": n,
        "mz_min": float(mzs.min()),
        "mz_max": float(mzs.max()),
        "mz_mean": float(mzs.mean()),
        "mz_median": float(np.median(mzs)),
        "mz_std": float(mzs.std()),
        "fragment_mz_range": float(mzs.max() - mzs.min()),
        "base_peak_mz": float(mzs[base_idx]),
        "base_peak_intensity": float(intensities[base_idx]),
        "total_intensity": total_intensity,
        "top1_intensity_fraction": _topk_intensity_fraction(sorted_desc, total_intensity, 1),
        "top3_intensity_fraction": _topk_intensity_fraction(sorted_desc, total_intensity, 3),
        "top5_intensity_fraction": _topk_intensity_fraction(sorted_desc, total_intensity, 5),
        "top10_intensity_fraction": _topk_intensity_fraction(sorted_desc, total_intensity, 10),
        "spectral_entropy": _spectral_entropy(intensities),
        "max_fragment_minus_precursor": max_frag_minus_precursor,
        "n_fragments_above_precursor": n_above,
        "fraction_fragments_above_precursor": n_above / n,
    }
