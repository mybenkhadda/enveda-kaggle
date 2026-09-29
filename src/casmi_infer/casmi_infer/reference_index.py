"""Exported reference library: metadata table, connectivity -> reference CSR index, and the
similarity-representation peak store (memory-mapped, so RAM stays bounded).

Peak store = EXACTLY training's similarity representation: `remove_invalid_peaks` then deterministic
`truncate_top_peaks(max_peaks)` (intensity DESC, m/z ASC, re-sorted by m/z), float64, no further
transform. Precursor m/z and the T1 peak hash (identity representation) live in `ref_meta`.
"""
from pathlib import Path

import numpy as np
import pandas as pd


class ReferenceLibrary:
    def __init__(self, bundle_dir, mmap=True):
        d = Path(bundle_dir)
        mode = "r" if mmap else None
        self.meta = pd.read_parquet(d / "ref_meta.parquet")
        if not (self.meta["ref_row"].to_numpy() == np.arange(len(self.meta))).all():
            raise ValueError("ref_meta must be ordered by ref_row = 0..N-1")
        self.index_offsets = np.load(d / "ref_index_offsets.npy", mmap_mode=mode)
        self.index_ids = np.load(d / "ref_index_ids.npy", mmap_mode=mode)
        self.peaks_offsets = np.load(d / "ref_peaks_offsets.npy", mmap_mode=mode)
        self.peaks_mz = np.load(d / "ref_peaks_mz.npy", mmap_mode=mode)
        self.peaks_int = np.load(d / "ref_peaks_int.npy", mmap_mode=mode)
        m = self.meta
        # plain arrays for vectorized compat ordering / eligibility
        self.spectrum_id = m["ref_spectrum_id"].astype(str).to_numpy()
        self.sid_rank = m["sid_rank"].to_numpy(np.int64)
        self.source = m["source"].astype(object).to_numpy()
        self.ce = m["collision_energy"].to_numpy(float)
        self.precursor_mz = m["precursor_mz"].to_numpy(float)
        self.peak_hash = m["peak_hash"].astype(object).to_numpy()
        self._vocab = {}
        for col in ("adduct", "polarity", "instrument"):
            codes, uniques = pd.factorize(m[col], use_na_sentinel=True)   # missing -> -1 (never "same")
            setattr(self, f"{col}_code", codes.astype(np.int64))
            self._vocab[col] = {v: i for i, v in enumerate(uniques)}

    def code(self, col, value):
        """Integer code of a query value in the reference vocabulary; -1 for missing, -2 for a
        value no reference has (both compare unequal to every reference code)."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return -1
        return self._vocab[col].get(value, -2)

    def refs_of(self, conn_idx):
        return np.asarray(self.index_ids[self.index_offsets[conn_idx]:self.index_offsets[conn_idx + 1]], dtype=np.int64)

    def peaks(self, ref_row):
        a, b = int(self.peaks_offsets[ref_row]), int(self.peaks_offsets[ref_row + 1])
        return {"mzs": np.asarray(self.peaks_mz[a:b], dtype=np.float64), "intensities": np.asarray(self.peaks_int[a:b], dtype=np.float64),
                "precursor_mz": float(self.precursor_mz[ref_row])}
