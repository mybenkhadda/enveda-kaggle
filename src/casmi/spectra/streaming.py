"""Scalable parquet-to-features processing: the only place that turns millions of raw peak
arrays into a features table, and it never holds more than one batch of them in memory.

    PyArrow scanner -> batch -> row-level validation/features -> chunk DataFrame -> write
    incrementally

This is the "row-level, readable baseline" version -- correct and bounded-memory first;
vectorizing the inner loop is a possible later optimization if throughput ever becomes the
bottleneck, not a correctness concern today.
"""
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from casmi.io.parquet import count_rows, scan_batches
from casmi.spectra.features import compute_spectrum_features
from casmi.spectra.preprocessing import remove_invalid_peaks
from casmi.spectra.validation import validate_spectrum
from casmi.utils.logging import get_logger

logger = get_logger(__name__)


def _process_batch(batch_df, mz_col, intensity_col, precursor_col, id_cols, row_offset,
                    fragment_above_precursor_da, extreme_fragment_mz):
    records = []
    for i, row in enumerate(batch_df.to_dict("records")):
        mzs, intensities = row[mz_col], row[intensity_col]
        precursor_mz = row.get(precursor_col) if precursor_col else None

        flags = validate_spectrum(mzs, intensities, precursor_mz, fragment_above_precursor_da, extreme_fragment_mz)
        clean_mzs, clean_intensities = (
            (mzs, intensities) if flags["length_mismatch"] else remove_invalid_peaks(mzs, intensities)
        )
        if flags["length_mismatch"]:
            clean_mzs, clean_intensities = [], []
        feats = compute_spectrum_features(clean_mzs, clean_intensities, precursor_mz=precursor_mz)

        record = {"row_index": row_offset + i}
        for c in (id_cols or []):
            record[c] = row[c]
        record.update({f"flag_{k}": v for k, v in flags.items()})
        record.update(feats)
        records.append(record)
    return records


def process_spectrum_file(path, output_path, mz_col, intensity_col, precursor_col=None,
                           id_cols=None, batch_size=50_000, fragment_above_precursor_da=5.0,
                           extreme_fragment_mz=5000.0, float_dtype="float32", show_progress=True):
    """Stream `path`, compute validation flags + features for every row, and write the
    combined table incrementally to `output_path`.

    `id_cols`: extra columns (e.g. `spectrum_id`, `molecule_id` for test) carried through
    unchanged. A `row_index` column (the physical row offset in `path`) is always included,
    since train has no official spectrum id to carry through instead.
    """
    columns = [mz_col, intensity_col] + ([precursor_col] if precursor_col else []) + list(id_cols or [])
    total_rows = count_rows(path)
    n_batches = math.ceil(total_rows / batch_size) if total_rows else 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    row_offset = 0
    iterator = scan_batches(path, columns=columns, batch_size=batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc=f"process_spectrum_file({Path(path).name})")

    for _, batch_df in iterator:
        records = _process_batch(batch_df, mz_col, intensity_col, precursor_col, id_cols,
                                  row_offset, fragment_above_precursor_da, extreme_fragment_mz)
        chunk_df = pd.DataFrame.from_records(records)
        for c in chunk_df.select_dtypes(include=["float64"]).columns:
            chunk_df[c] = chunk_df[c].astype(float_dtype)

        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        row_offset += len(batch_df)

    if writer is not None:
        writer.close()
    else:
        logger.warning("process_spectrum_file: %s had 0 rows -- writing an empty output", path)
        pd.DataFrame(columns=["row_index"] + list(id_cols or [])).to_parquet(output_path, index=False)

    logger.info("process_spectrum_file: wrote %d rows -> %s", row_offset, output_path)
    return output_path
