"""Inference compute backend: "numba" (optional JIT kernels) or "numpy" (the training functions).

The numba backend replaces ONLY two inner loops whose arithmetic ports one-to-one:
  * peak overlap -- the sorted two-pointer match count of `casmi.spectra.similarity.peak_overlap`
    (integer counting: bit-identical);
  * binned cosine -- the dot product over shared bins of two L2-normalized sparse spectra produced by
    the training `bin_spectrum` (same values, summation order may differ from BLAS by ~1e-16).
Binning, modified cosine and neutral-loss cosine always run the training NumPy code.

Selection: `set_backend()` / `use_backend()` override; else env `CASMI_INFER_BACKEND`; else numba if
importable, numpy otherwise. The Kaggle notebook proves the active backend with the bundled
self-test (`casmi_infer.selftest`) and falls back to numpy if numba fails parity.
"""
import os
from contextlib import contextmanager

NUMBA, NUMPY = "numba", "numpy"
_FORCED = None
_KERNELS = None


def numba_available():
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def active_backend():
    if _FORCED is not None:
        return _FORCED
    env = os.environ.get("CASMI_INFER_BACKEND", "").strip().lower()
    if env in (NUMBA, NUMPY):
        return env if env == NUMPY or numba_available() else NUMPY
    return NUMBA if numba_available() else NUMPY


def set_backend(name):
    global _FORCED
    if name not in (NUMBA, NUMPY, None):
        raise ValueError(f"unknown backend {name!r}")
    if name == NUMBA and not numba_available():
        raise RuntimeError("numba backend requested but numba is not importable")
    _FORCED = name


@contextmanager
def use_backend(name):
    global _FORCED
    prev = _FORCED
    set_backend(name)
    try:
        yield name
    finally:
        _FORCED = prev


def numba_kernels():
    """Lazily JIT-compiled kernels (cached)."""
    global _KERNELS
    if _KERNELS is None:
        import numba

        @numba.njit(cache=False)
        def overlap_count(a_sorted, b_sorted, tol):
            i = j = n = 0
            while i < a_sorted.shape[0] and j < b_sorted.shape[0]:
                diff = a_sorted[i] - b_sorted[j]
                if abs(diff) <= tol:
                    n += 1
                    i += 1
                    j += 1
                elif diff < 0:
                    i += 1
                else:
                    j += 1
            return n

        @numba.njit(cache=False)
        def sorted_bin_dot(bins_a, vals_a, bins_b, vals_b):
            i = j = 0
            s = 0.0
            while i < bins_a.shape[0] and j < bins_b.shape[0]:
                if bins_a[i] == bins_b[j]:
                    s += vals_a[i] * vals_b[j]
                    i += 1
                    j += 1
                elif bins_a[i] < bins_b[j]:
                    i += 1
                else:
                    j += 1
            return s

        _KERNELS = {"overlap_count": overlap_count, "sorted_bin_dot": sorted_bin_dot}
    return _KERNELS
