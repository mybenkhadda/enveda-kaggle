"""Pure spectrum preprocessing transforms: arrays in, arrays out, no notebook-state
dependence, no file I/O. Composable -- e.g. `normalize_max(sqrt_transform(intensities))`.
"""
import numpy as np


def sort_peaks(mzs, intensities):
    """Sort peaks by ascending m/z. A no-op copy if already sorted."""
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    order = np.argsort(mzs, kind="stable")
    return mzs[order], intensities[order]


def remove_invalid_peaks(mzs, intensities):
    """Drop peaks with a non-finite or negative m/z or intensity. Assumes `mzs`/`intensities`
    are already the same length (see `casmi.spectra.validation.validate_spectrum` for the
    length-mismatch check itself)."""
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    keep = np.isfinite(mzs) & np.isfinite(intensities) & (mzs >= 0) & (intensities >= 0)
    return mzs[keep], intensities[keep]


def merge_duplicate_mz(mzs, intensities, tol=0.0):
    """Merge peaks whose m/z values are within `tol` of each other (default 0.0: only exact
    duplicates), summing their intensities. Input need not be pre-sorted; output is sorted by
    m/z."""
    mzs, intensities = sort_peaks(mzs, intensities)
    if len(mzs) == 0:
        return mzs, intensities

    out_mz, out_inten = [mzs[0]], [intensities[0]]
    for m, i in zip(mzs[1:], intensities[1:]):
        if m - out_mz[-1] <= tol:
            out_inten[-1] += i
        else:
            out_mz.append(m)
            out_inten.append(i)
    return np.asarray(out_mz), np.asarray(out_inten)


def truncate_top_peaks(mzs, intensities, max_peaks=100):
    """Keep only the `max_peaks` most intense peaks, re-sorted by ascending m/z afterward.
    Real spectra in this dataset range from a handful of peaks up to tens of thousands (p99 is
    ~1,300; the max is >70,000) -- similarity functions that are O(n_a * n_b) in peak count
    (`casmi.spectra.similarity.modified_cosine_similarity`) need this cap to stay tractable, and
    dropping low-intensity peaks is standard MS/MS practice anyway (noise, not signal). A
    no-op if the spectrum already has `max_peaks` or fewer.

    Selection is fully deterministic (v4b): peaks are ordered by intensity DESC, then m/z ASC,
    and the first `max_peaks` are kept -- so when several peaks tie in intensity across the
    rank-`max_peaks` / rank-`max_peaks+1` boundary, the lower-m/z peaks win, independent of the
    input peak order. The previous implementation (`np.argpartition`, kept as
    `truncate_top_peaks_legacy` for the v4b cache-settlement diagnostic only) left the choice
    among boundary-tied peaks to the partition algorithm."""
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if len(mzs) <= max_peaks:
        return mzs, intensities
    order = np.lexsort((mzs, -intensities))  # last key is primary: intensity DESC, then m/z ASC
    keep = order[:max_peaks]
    return sort_peaks(mzs[keep], intensities[keep])


def truncate_top_peaks_legacy(mzs, intensities, max_peaks=100):
    """The pre-v4b top-N rule (`np.argpartition`), whose choice among intensity-tied peaks at the
    cutoff is unspecified. DIAGNOSTIC ONLY -- used by `casmi.qcr.settlement` to test whether a
    fresh-vs-cached similarity discrepancy is explained by the truncation rule. Never use it to
    build evidence."""
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if len(mzs) <= max_peaks:
        return mzs, intensities
    top_idx = np.argpartition(intensities, -max_peaks)[-max_peaks:]
    return sort_peaks(mzs[top_idx], intensities[top_idx])


def top_n_boundary_diagnostics(intensities, max_peaks=100, near_rel_tol=1e-6):
    """Is the top-`max_peaks` cut ambiguous for this spectrum? Returns
    `{n_peaks, truncated, intensity_at_n, intensity_at_n_plus_1, exact_tie, near_tie,
    n_tied_at_boundary}`: `exact_tie` means the rank-`max_peaks` and rank-`max_peaks+1`
    intensities are identical (the kept set depends on the tie rule); `near_tie` means they
    differ by at most `near_rel_tol` relative to the larger one (sensitive to float noise);
    `n_tied_at_boundary` counts all peaks sharing the rank-`max_peaks` intensity."""
    intensities = np.asarray(intensities, dtype=float)
    n = len(intensities)
    out = {"n_peaks": int(n), "truncated": bool(n > max_peaks), "intensity_at_n": float("nan"),
           "intensity_at_n_plus_1": float("nan"), "exact_tie": False, "near_tie": False, "n_tied_at_boundary": 0}
    if n <= max_peaks:
        return out
    desc = np.sort(intensities)[::-1]
    a, b = float(desc[max_peaks - 1]), float(desc[max_peaks])
    out["intensity_at_n"], out["intensity_at_n_plus_1"] = a, b
    out["exact_tie"] = bool(a == b)
    scale = max(abs(a), abs(b), 1e-300)
    out["near_tie"] = bool(abs(a - b) / scale <= near_rel_tol)
    out["n_tied_at_boundary"] = int((intensities == a).sum()) if out["exact_tie"] else 0
    return out


def normalize_max(intensities):
    """Scale so the base (most intense) peak is 1.0. Returns zeros unchanged (no division by
    zero) for an all-zero or empty spectrum."""
    intensities = np.asarray(intensities, dtype=float)
    m = intensities.max() if len(intensities) else 0.0
    return intensities / m if m > 0 else intensities


def normalize_sum(intensities):
    """Scale so total intensity is 1.0. Returns zeros unchanged for an all-zero or empty
    spectrum."""
    intensities = np.asarray(intensities, dtype=float)
    s = intensities.sum()
    return intensities / s if s > 0 else intensities


def sqrt_transform(intensities):
    """Square-root of intensity (negatives clipped to 0 first, since a valid spectrum has
    none, but this stays safe if called before `remove_invalid_peaks`)."""
    intensities = np.asarray(intensities, dtype=float)
    return np.sqrt(np.clip(intensities, 0, None))


def log1p_transform(intensities):
    """log(1 + intensity) (negatives clipped to 0 first)."""
    intensities = np.asarray(intensities, dtype=float)
    return np.log1p(np.clip(intensities, 0, None))
