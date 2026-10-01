"""Lightweight memory diagnostics for pandas DataFrames.

Deliberately conservative: high-cardinality molecular identifiers (SMILES, InChIKeys,
connectivity keys) are left as strings unless a specific column is proven to benefit from
categorical/numeric downcasting, per the project rule against blind dtype optimization.
"""
import numpy as np
import pandas as pd


def dataframe_memory_mb(df):
    """Deep memory usage of `df` in MB (accounts for actual string/object payload size, not
    just pointer size)."""
    return float(df.memory_usage(deep=True).sum() / 1e6)


def optimize_dataframe_dtypes(df, categorical_cols=None, downcast_int_cols=None, downcast_float_cols=None):
    """Return a copy of `df` with explicitly-named columns downcast -- never a blanket
    "optimize everything" pass, since that risks silently truncating precision on columns
    (like exact masses) where it matters.

    `categorical_cols`: low-cardinality string columns (e.g. adduct, ionization_mode, source)
        to convert to pandas `category` dtype.
    `downcast_int_cols` / `downcast_float_cols`: columns to downcast via
        `pd.to_numeric(..., downcast=...)`.
    """
    out = df.copy()
    for col in categorical_cols or []:
        if col in out.columns:
            out[col] = out[col].astype("category")
    for col in downcast_int_cols or []:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in downcast_float_cols or []:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def array_memory_mb(arr):
    """Memory usage of a single numpy array in MB."""
    return float(np.asarray(arr).nbytes / 1e6)
