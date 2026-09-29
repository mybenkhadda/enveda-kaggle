"""Generic multi-value aggregation, shared by two distinct places this project needs it:
reducing several REFERENCE spectra down to one score per candidate, and (later) reducing
several QUERY spectra of one molecule down to one score per candidate. Same handful of
reducers either way.
"""
import numpy as np

DEFAULT_METHODS = ("max", "mean", "median", "top3_mean")


def aggregate_scores(values, methods=DEFAULT_METHODS):
    """One dict of `{method: value}` for every requested method, computed over `values`. Empty
    input returns NaN for every method (never raises) -- a candidate with zero reference
    spectra still gets a row, just with an undefined spectral score, so mass-only ranking stays
    possible for it."""
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return {m: float("nan") for m in methods}

    sorted_desc = np.sort(values)[::-1]
    out = {}
    for m in methods:
        if m == "max":
            out[m] = float(values.max())
        elif m == "mean":
            out[m] = float(values.mean())
        elif m == "median":
            out[m] = float(np.median(values))
        elif m.startswith("top") and m.endswith("_mean"):
            k = int(m[3:-5])
            out[m] = float(sorted_desc[:k].mean())
        else:
            raise ValueError(f"unknown aggregation method: {m!r}")
    return out
