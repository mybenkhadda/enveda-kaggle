"""Dtype-safe boolean coercion.

`series.fillna(False).astype(bool)` on an OBJECT column (e.g. True/False mixed with NaN after a left merge)
triggers pandas' "Downcasting object dtype arrays on .fillna is deprecated" FutureWarning. `as_bool` gives the
SAME values without the deprecated downcast:

    bool dtype            -> copied unchanged
    nullable "boolean"    -> <NA> replaced by `missing`, then numpy bool
    object / numeric      -> missing (None / NaN / NA / NaT) replaced by `missing`, every other value -> bool(value)
                             (exactly what `fillna(missing).astype(bool)` produced: bool("False") is True, as before)

Warnings are never suppressed globally -- the deprecated operation is simply not performed.
"""
import numpy as np
import pandas as pd


def as_bool(values, missing=False):
    """Boolean Series (numpy bool dtype) from `values`, missing entries -> `missing`. Index and name are kept."""
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if s.dtype == bool:
        return s.copy()
    if isinstance(s.dtype, pd.BooleanDtype):
        return pd.Series(s.to_numpy(dtype=bool, na_value=bool(missing)), index=s.index, name=s.name)
    arr = s.to_numpy(dtype=object)
    na = pd.isna(arr)
    out = np.empty(len(arr), dtype=bool)
    out[na] = bool(missing)
    if (~na).any():
        out[~na] = np.array([bool(v) for v in arr[~na]], dtype=bool)
    return pd.Series(out, index=s.index, name=s.name)
