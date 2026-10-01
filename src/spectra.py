"""MS/MS spectrum normalization, binning, similarity, and complexity utilities.

All functions take raw (mz, intensity) numpy arrays (or array-likes) for a single spectrum
unless noted as a "_batch" function, which takes a list of such arrays and returns a
scipy.sparse matrix -- most of this project's spectra are sparse (tens to low-hundreds of
peaks) over a wide m/z range, so dense per-spectrum vectors are wasteful at fine bin widths.
"""
import numpy as np
from scipy import sparse


def normalize_intensity(intensity, method="max"):
    """Rescale a raw intensity array. method in {'raw','max','sum','sqrt','log1p','sqrt_max'}."""
    x = np.asarray(intensity, dtype=float)
    if x.size == 0:
        return x
    if method == "raw":
        return x
    if method == "max":
        m = x.max()
        return x / m if m > 0 else x
    if method == "sum":
        s = x.sum()
        return x / s if s > 0 else x
    if method == "sqrt":
        return np.sqrt(np.clip(x, 0, None))
    if method == "log1p":
        return np.log1p(np.clip(x, 0, None))
    if method == "sqrt_max":
        y = np.sqrt(np.clip(x, 0, None))
        m = y.max()
        return y / m if m > 0 else y
    raise ValueError(f"unknown normalization method: {method}")


def bin_spectrum(mz, intensity, bin_width=1.0, mz_max=2000.0):
    """Dense binned vector for a single spectrum. Fine for exploratory single-spectrum plots;
    use bin_spectra_batch for many spectra at fine resolution."""
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    n_bins = int(mz_max / bin_width) + 1
    vec = np.zeros(n_bins, dtype=float)
    if mz.size == 0:
        return vec
    idx = np.clip((mz / bin_width).astype(int), 0, n_bins - 1)
    np.add.at(vec, idx, intensity)
    return vec


def bin_spectra_batch(mz_list, intensity_list, bin_width=1.0, mz_max=2000.0, l2_normalize=True):
    """Binned spectra for many spectra at once, as a scipy.sparse CSR matrix (n_spectra x n_bins).

    Memory-efficient for fine bin widths (e.g. 0.01 Da over 2000 Da = 200k columns) because
    each row only has as many nonzeros as the spectrum has peaks (after binning collisions).
    """
    n_bins = int(mz_max / bin_width) + 1
    rows, cols, vals = [], [], []
    for i, (mz, inten) in enumerate(zip(mz_list, intensity_list)):
        mz = np.asarray(mz, dtype=float)
        inten = np.asarray(inten, dtype=float)
        if mz.size == 0:
            continue
        idx = np.clip((mz / bin_width).astype(int), 0, n_bins - 1)
        rows.append(np.full(idx.shape, i))
        cols.append(idx)
        vals.append(inten)
    if not rows:
        return sparse.csr_matrix((len(mz_list), n_bins))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(len(mz_list), n_bins)).tocsr()
    mat.sum_duplicates()
    if l2_normalize:
        norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        mat = sparse.diags(1.0 / norms) @ mat
    return mat.tocsr()


def sparse_cosine_pairs(mat, i, j):
    """Cosine similarity between rows i and j of an already L2-normalized sparse matrix."""
    return float(mat.getrow(i).multiply(mat.getrow(j)).sum())


def peak_matching_cosine(mz1, i1, mz2, i2, tolerance_da=0.01):
    """Greedy peak-matching cosine similarity: peaks within `tolerance_da` are paired, unmatched
    peaks contribute zero to the corresponding side. A simple, dependency-free stand-in for a
    full modified-cosine spectral match."""
    mz1 = np.asarray(mz1, dtype=float)
    i1 = np.asarray(i1, dtype=float)
    mz2 = np.asarray(mz2, dtype=float)
    i2 = np.asarray(i2, dtype=float)
    if mz1.size == 0 or mz2.size == 0:
        return 0.0
    order1 = np.argsort(mz1)
    order2 = np.argsort(mz2)
    mz1, i1 = mz1[order1], i1[order1]
    mz2, i2 = mz2[order2], i2[order2]

    pairs = []
    used2 = np.zeros(len(mz2), dtype=bool)
    p2 = 0
    for a in range(len(mz1)):
        while p2 < len(mz2) and mz2[p2] < mz1[a] - tolerance_da:
            p2 += 1
        scan = p2
        best_j, best_d = -1, tolerance_da
        while scan < len(mz2) and mz2[scan] <= mz1[a] + tolerance_da:
            if not used2[scan] and abs(mz2[scan] - mz1[a]) <= best_d:
                best_j, best_d = scan, abs(mz2[scan] - mz1[a])
            scan += 1
        if best_j >= 0:
            used2[best_j] = True
            pairs.append((a, best_j))

    if not pairs:
        return 0.0
    num = sum(i1[a] * i2[b] for a, b in pairs)
    den = np.sqrt((i1 ** 2).sum()) * np.sqrt((i2 ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def modified_cosine(mz1, i1, mz2, i2, precursor1, precursor2, tolerance_da=0.01):
    """Simplified modified-cosine: peaks may match either directly, or after shifting by the
    precursor mass difference (i.e. the same neutral loss). Not a full CMS/GNPS implementation,
    but captures the key idea that a shifted fragment can still indicate structural similarity."""
    mz1 = np.asarray(mz1, dtype=float)
    i1 = np.asarray(i1, dtype=float)
    mz2 = np.asarray(mz2, dtype=float)
    i2 = np.asarray(i2, dtype=float)
    if mz1.size == 0 or mz2.size == 0:
        return 0.0
    shift = precursor1 - precursor2
    used2 = np.zeros(len(mz2), dtype=bool)
    pairs = []
    for a in range(len(mz1)):
        candidates = np.where((~used2) & ((np.abs(mz2 - mz1[a]) <= tolerance_da) |
                                           (np.abs(mz2 + shift - mz1[a]) <= tolerance_da)))[0]
        if candidates.size:
            b = candidates[np.argmax(i2[candidates])]
            used2[b] = True
            pairs.append((a, b))
    if not pairs:
        return 0.0
    num = sum(i1[a] * i2[b] for a, b in pairs)
    den = np.sqrt((i1 ** 2).sum()) * np.sqrt((i2 ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def spectral_entropy(intensity, normalize=True):
    """Shannon entropy of a spectrum's intensity distribution; 0 for a single dominant peak,
    ln(n_peaks) for perfectly flat intensities. `normalize` divides by ln(n_peaks) so the
    result is in [0, 1] and comparable across spectra with different peak counts."""
    x = np.asarray(intensity, dtype=float)
    x = x[x > 0]
    if x.size == 0:
        return 0.0
    p = x / x.sum()
    h = float(-(p * np.log(p)).sum())
    if normalize:
        return h / np.log(x.size) if x.size > 1 else 0.0
    return h


def topk_intensity_fraction(intensity, k):
    x = np.asarray(intensity, dtype=float)
    if x.size == 0:
        return np.nan
    total = x.sum()
    if total <= 0:
        return np.nan
    top = np.sort(x)[::-1][:k]
    return float(top.sum() / total)


def neutral_losses(mz, precursor_mz):
    mz = np.asarray(mz, dtype=float)
    return precursor_mz - mz


def compute_spectrum_features(mz, intensity, precursor_mz=None):
    """Scalar summary features for one spectrum, matching the EDA's per-spectrum feature table."""
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    n = mz.size
    out = {
        "n_peaks": n, "mz_min": np.nan, "mz_max": np.nan, "mz_range": np.nan,
        "base_peak_mz": np.nan, "precursor_to_base_peak_delta": np.nan,
        "mean_fragment_mz": np.nan, "median_fragment_mz": np.nan, "std_fragment_mz": np.nan,
        "intensity_entropy": np.nan, "top1_intensity_fraction": np.nan,
        "top3_intensity_fraction": np.nan, "top5_intensity_fraction": np.nan,
        "top10_intensity_fraction": np.nan, "fragment_coverage_ratio": np.nan,
    }
    if n == 0:
        return out
    base_idx = int(np.argmax(intensity))
    out.update({
        "mz_min": float(mz.min()),
        "mz_max": float(mz.max()),
        "mz_range": float(mz.max() - mz.min()),
        "base_peak_mz": float(mz[base_idx]),
        "mean_fragment_mz": float(mz.mean()),
        "median_fragment_mz": float(np.median(mz)),
        "std_fragment_mz": float(mz.std()),
        "intensity_entropy": spectral_entropy(intensity, normalize=True),
        "top1_intensity_fraction": topk_intensity_fraction(intensity, 1),
        "top3_intensity_fraction": topk_intensity_fraction(intensity, 3),
        "top5_intensity_fraction": topk_intensity_fraction(intensity, 5),
        "top10_intensity_fraction": topk_intensity_fraction(intensity, 10),
    })
    if precursor_mz is not None and precursor_mz > 0:
        out["precursor_to_base_peak_delta"] = float(precursor_mz - mz[base_idx])
        out["fragment_coverage_ratio"] = float(mz.max() / precursor_mz)
    return out
