"""All direct PyArrow parquet access lives here -- other modules read parquet data only
through these functions, never via `pyarrow.parquet` directly, so the "never load millions of
peak arrays into pandas at once" rule has exactly one place it can be violated (and tested).
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_parquet_schema(path):
    """The pyarrow schema of `path`, without reading any row data."""
    return pq.ParquetFile(path).schema_arrow


def get_parquet_columns(path):
    """Column names of `path`, without reading any row data."""
    return list(get_parquet_schema(path).names)


def count_rows(path):
    """Row count of `path` from parquet footer metadata only."""
    return pq.ParquetFile(path).metadata.num_rows


def row_group_bounds(path):
    """Global [start, end) row offset of every row group in `path`, from metadata only.
    Used to seek directly to the row group(s) containing a set of target row offsets without
    scanning the rest of the file."""
    pf = pq.ParquetFile(path)
    counts = np.array([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if len(counts) else np.array([], dtype=np.int64)
    return starts, starts + counts


def load_columns(path, columns, max_rows=None):
    """Load only `columns` from `path` into a pandas DataFrame.

    If `max_rows` is given, reads only as many *leading row groups* as needed to cover it
    (never the whole file) and truncates to exactly `max_rows` -- intended for smoke-scale
    development runs, not for a representative sample (rows are in on-disk order, and this
    dataset's row groups are contiguous per-source blocks; for a representative subsample,
    combine this with an explicit stratified selection instead).
    """
    pf = pq.ParquetFile(path)
    if max_rows is None:
        return pf.read(columns=columns).to_pandas()
    batches, n = [], 0
    for rg in range(pf.metadata.num_row_groups):
        if n >= max_rows:
            break
        tbl = pf.read_row_group(rg, columns=columns)
        batches.append(tbl)
        n += tbl.num_rows
    return pa.concat_tables(batches).slice(0, max_rows).to_pandas()


def load_rows_by_offset(path, columns, offsets):
    """Load only `columns` for the specific global row `offsets` out of `path`, by reading
    only the row group(s) that actually contain one of those offsets (parquet metadata tells
    us which, so uninvolved row groups are never touched). This is the memory-bounded way to
    pull a scattered sample of peak arrays out of a multi-GB file: at most one row group's
    worth of the requested columns is ever resident in memory at a time.

    Returns a DataFrame with an extra `_row_offset` column so callers can re-associate rows
    with whatever they used to pick the offsets in the first place.
    """
    offsets = np.asarray(sorted(offsets))
    pf = pq.ParquetFile(path)
    starts, ends = row_group_bounds(path)
    frames = []
    for rg in range(pf.metadata.num_row_groups):
        lo, hi = starts[rg], ends[rg]
        mask = (offsets >= lo) & (offsets < hi)
        if not mask.any():
            continue
        local_idx = offsets[mask] - lo
        tbl = pf.read_row_group(rg, columns=columns).to_pandas()
        sub = tbl.iloc[local_idx].copy()
        sub["_row_offset"] = offsets[mask]
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=list(columns) + ["_row_offset"])
    return pd.concat(frames, ignore_index=True)


def scan_batches(path, columns, batch_size=50_000):
    """Yield `(batch_index, pandas.DataFrame)` pairs, `batch_size` rows at a time, for
    `columns` of `path`. This is the entry point for any pass over millions of spectra:
    nothing outside the current batch is ever resident in memory.

    Example:
        for i, batch in scan_batches(path, columns=["ms2_mzs", "precursor_mz"], batch_size=50_000):
            ...
    """
    pf = pq.ParquetFile(path)
    for i, record_batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=list(columns))):
        yield i, record_batch.to_pandas()
