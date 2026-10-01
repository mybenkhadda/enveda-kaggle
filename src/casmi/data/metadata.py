"""Scalar train/test metadata only -- peak arrays (`ms2_mzs`/`ms2_normalized_intensities`)
are never loaded here; that's `casmi.spectra.streaming`'s job.
"""
import numpy as np

from casmi.data.schema import TEST_REQUIRED_COLUMNS, TRAIN_REQUIRED_COLUMNS
from casmi.io.parquet import load_columns
from casmi.utils.memory import optimize_dataframe_dtypes

_PEAK_COLUMNS = {"ms2_mzs", "ms2_normalized_intensities"}


def load_train_metadata(paths, max_rows=None):
    """Non-peak train columns, plus a minted `train_spectrum_id` ("train_<row offset>") --
    train has no official spectrum id. Does NOT create a `molecule_id`: the train prediction
    unit is `connectivity_key` (see `casmi.data.aggregation.build_molecule_metadata`), not a
    fabricated id.
    """
    columns = [c for c in TRAIN_REQUIRED_COLUMNS if c not in _PEAK_COLUMNS]
    df = load_columns(paths.train, columns, max_rows=max_rows).reset_index(drop=True)
    df.insert(0, "train_spectrum_id", "train_" + df.index.astype(str))
    return df


def load_test_metadata(paths, max_rows=None):
    """Non-peak test columns. `spectrum_id`/`molecule_id` are preserved exactly as provided
    -- never regenerated or renamed."""
    columns = [c for c in TEST_REQUIRED_COLUMNS if c not in _PEAK_COLUMNS]
    return load_columns(paths.test, columns, max_rows=max_rows).reset_index(drop=True)


def _ce_array(v):
    if v is None:
        return np.array([], dtype=float)
    arr = np.asarray(v, dtype=float)
    return arr[~np.isnan(arr)]


def summarize_collision_energy(df, ce_col="collision_energy_ev"):
    """Return a copy of `df` with `ce_mean`/`ce_min`/`ce_max`/`ce_n_steps`/`ce_is_ramped`
    derived from the list-valued `ce_col` (collision energy is sometimes a single value,
    sometimes a stepped/ramped acquisition stored as a list, sometimes entirely missing --
    all three cases are handled, never assumed to be a scalar)."""
    ce_arrays = df[ce_col].apply(_ce_array)
    out = df.copy()
    out["ce_mean"] = ce_arrays.apply(lambda a: float(a.mean()) if len(a) else np.nan)
    out["ce_min"] = ce_arrays.apply(lambda a: float(a.min()) if len(a) else np.nan)
    out["ce_max"] = ce_arrays.apply(lambda a: float(a.max()) if len(a) else np.nan)
    out["ce_n_steps"] = ce_arrays.apply(len)
    out["ce_is_ramped"] = out["ce_n_steps"] > 1
    return out


def optimize_metadata_dtypes(df):
    """Downcast only the columns known to be low-cardinality categoricals; molecular
    identifiers and free-text columns are left as strings (see
    `casmi.utils.memory.optimize_dataframe_dtypes` for why blanket optimization is avoided)."""
    categorical_candidates = [
        c for c in ("adduct", "adduct_orig", "ionization_mode", "instrument_type",
                     "ingest_lib", "collision_energy_orig_units")
        if c in df.columns
    ]
    return optimize_dataframe_dtypes(df, categorical_cols=categorical_candidates)
