"""Persistent `(query_dataset, query_spectrum_id, reference_spectrum_id, config_hash) ->
(tier, cosine, modified_cosine, peak_overlap_frac, neutral_loss_cosine)` cache.

The identity-tier/spectral-similarity computation for a given (query, reference) pair is
expensive (peak matching, binning) and must never be repeated -- neither within one run (this
class memoizes in-process, and counts hits/computes so tests can verify "compute once" directly)
nor across warm reruns of the same notebook (the cache persists to a parquet file, keyed
including `config_hash`, so a config change shows up as a cache MISS, never a silently stale
hit). `query_dataset` (e.g. `"host"` / `"dev"`) is part of the key because HOST and DEV
spectrum ids are not guaranteed to be disjoint.
"""
from pathlib import Path

import pandas as pd


class SimilarityCache:
    def __init__(self, config_hash):
        self.config_hash = config_hash
        self._store = {}
        self.n_computed = 0
        self.n_hits = 0

    def _key(self, query_dataset, query_spectrum_id, reference_spectrum_id):
        return (query_dataset, query_spectrum_id, reference_spectrum_id, self.config_hash)

    def get_or_compute(self, query_dataset, query_spectrum_id, reference_spectrum_id, compute_fn):
        """`compute_fn()` (no args) is called ONLY on a cache miss and must return
        `(tier, cosine, modified_cosine, peak_overlap_frac, neutral_loss_cosine)`."""
        key = self._key(query_dataset, query_spectrum_id, reference_spectrum_id)
        cached = self._store.get(key)
        if cached is not None:
            self.n_hits += 1
            return cached
        value = compute_fn()
        self._store[key] = value
        self.n_computed += 1
        return value

    def __len__(self):
        return len(self._store)

    def __contains__(self, key_tuple):
        query_dataset, query_spectrum_id, reference_spectrum_id = key_tuple
        return self._key(query_dataset, query_spectrum_id, reference_spectrum_id) in self._store

    def load(self, path):
        """Load a previously `save`d cache -- rows whose `config_hash` doesn't match this
        instance's are silently ignored (a stale entry from an old config is not a hit)."""
        path = Path(path)
        if not path.exists():
            return 0
        df = pd.read_parquet(path)
        df = df[df["config_hash"] == self.config_hash]
        for row in df.itertuples(index=False):
            key = (row.query_dataset, row.query_spectrum_id, row.reference_spectrum_id, row.config_hash)
            self._store[key] = (row.tier, row.cosine, row.modified_cosine, row.peak_overlap_frac, row.neutral_loss_cosine)
        return len(df)

    def save(self, path):
        rows = [
            {
                "query_dataset": k[0], "query_spectrum_id": k[1], "reference_spectrum_id": k[2], "config_hash": k[3],
                "tier": v[0], "cosine": v[1], "modified_cosine": v[2], "peak_overlap_frac": v[3], "neutral_loss_cosine": v[4],
            }
            for k, v in self._store.items()
        ]
        df = pd.DataFrame(rows, columns=["query_dataset", "query_spectrum_id", "reference_spectrum_id", "config_hash",
                                          "tier", "cosine", "modified_cosine", "peak_overlap_frac", "neutral_loss_cosine"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path
