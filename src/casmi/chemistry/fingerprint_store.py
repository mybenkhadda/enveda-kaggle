"""Packed Morgan fingerprints + vectorized Tanimoto + a resumable on-disk cache keyed by connectivity_key.

Reuses `casmi.chemistry.similarity.morgan_fingerprint` (same RDKit generator, same radius / n_bits
semantics). Fingerprints are stored packed (uint8, n_bits/8 bytes) so 1M structures x 2048 bits = 256 MB.
Computed once per connectivity and never recomputed: the cache appends shards and reloads them.

Tanimoto on packed bits uses popcount (`np.bitwise_count` on NumPy >= 2, else a 256-entry table);
an all-zero / invalid fingerprint gives Tanimoto NaN against anything.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.chemistry.similarity import morgan_fingerprint

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount_u8(a):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    return _POP8[a]


def pack_fingerprints(smiles, radius=2, n_bits=2048):
    """`(bits uint8[n, n_bits/8], valid bool[n])` -- invalid SMILES -> zero row, valid=False."""
    from rdkit import DataStructs
    if n_bits % 8:
        raise ValueError("n_bits must be a multiple of 8")
    out = np.zeros((len(smiles), n_bits // 8), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    arr = np.zeros(n_bits, dtype=np.uint8)
    for i, s in enumerate(smiles):
        fp = morgan_fingerprint(s, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = np.packbits(arr)
        valid[i] = True
    return out, valid


def tanimoto_matrix(a, b, chunk=256):
    """Tanimoto between packed fingerprint sets a (n, B) and b (m, B) -> float32 (n, m)."""
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    pa = popcount_u8(a).sum(1, dtype=np.int32)
    pb = popcount_u8(b).sum(1, dtype=np.int32)
    out = np.empty((len(a), len(b)), dtype=np.float32)
    for s in range(0, len(a), chunk):
        inter = popcount_u8(a[s:s + chunk, None, :] & b[None, :, :]).sum(2, dtype=np.int32)
        union = pa[s:s + chunk, None] + pb[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:s + chunk] = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return out


class FingerprintCache:
    """Append-only cache: `<root>/r{radius}_b{n_bits}/shard-NNNNN.npz` (keys S14, bits, valid).

    `get(keys, smiles_of)` returns packed bits in the order of `keys`, computing ONLY the missing ones
    (`smiles_of(missing_keys) -> list of SMILES`).

    I/O model (Drive-safe): new fingerprints go to an in-memory PENDING buffer that is searched together with the
    loaded table; `flush()` writes the buffer as ONE shard (callers flush once per resumable chunk; it also
    happens automatically every `flush_rows` new rows and in `write_meta`). Nothing is ever re-read from disk after
    construction, and a directory with more than `compact_above` shards is compacted into one shard on load --
    the previous design wrote one tiny shard per call and reloaded EVERY shard on every miss (quadratic I/O)."""

    def __init__(self, root, radius=2, n_bits=2048, flush_rows=50_000, compact_above=32):
        self.dir = Path(root) / f"r{radius}_b{n_bits}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.radius, self.n_bits, self.flush_rows, self.compact_above = radius, n_bits, flush_rows, compact_above
        self._keys = np.zeros(0, dtype="S14")
        self._bits = np.zeros((0, n_bits // 8), dtype=np.uint8)
        self._valid = np.zeros(0, dtype=bool)
        self._pending = []                                   # list of (keys, bits, valid) not yet on disk
        self._pk, self._pb, self._pv = self._keys, self._bits, self._valid   # sorted view of the pending buffer
        self._load()

    # ---- disk -----------------------------------------------------------------------------------------------
    def _shards(self):
        return sorted(self.dir.glob("shard-*.npz"))

    def _next_shard_path(self):
        idx = [int(f.stem.split("-")[1]) for f in self._shards()]
        return self.dir / f"shard-{(max(idx) + 1 if idx else 0):05d}.npz"

    def _write_shard(self, keys, bits, valid, path=None):
        path = path or self._next_shard_path()
        tmp = self.dir / f"tmp-{path.stem}.npz"             # never matches the shard-*.npz glob if interrupted
        np.savez(tmp, keys=keys, bits=bits, valid=valid)
        tmp.replace(path)
        return path

    @staticmethod
    def _sorted_unique(k, b, v):
        order = np.argsort(k, kind="stable")
        k, b, v = k[order], b[order], v[order]
        first = np.concatenate([[True], k[1:] != k[:-1]]) if len(k) else np.zeros(0, bool)   # duplicates: keep first
        return k[first], b[first], v[first]

    def _load(self):
        files = self._shards()
        ks, bs, vs = [], [], []
        for f in files:
            with np.load(f) as z:                            # close the npz handle (Windows cannot delete open files)
                ks.append(z["keys"]); bs.append(z["bits"]); vs.append(z["valid"])
        if ks:
            self._keys, self._bits, self._valid = self._sorted_unique(np.concatenate(ks), np.concatenate(bs), np.concatenate(vs))
        if len(files) > self.compact_above:
            self.compact(files)

    def compact(self, files=None):
        """Rewrite the whole cache as one NEW shard, then delete the old ones. Safe to interrupt: the worst case is
        duplicate rows across shards, which `_load` deduplicates."""
        self.flush()
        files = files if files is not None else self._shards()
        new = self._write_shard(self._keys, self._bits, self._valid)
        for f in files:
            if f != new:
                f.unlink()

    def flush(self):
        """Persist the pending buffer as one shard. Returns the number of rows written."""
        if not self._pending:
            return 0
        k, b, v = (np.concatenate(x) for x in zip(*self._pending))
        self._write_shard(k, b, v)
        self._keys, self._bits, self._valid = self._sorted_unique(np.concatenate([self._keys, k]), np.concatenate([self._bits, b]),
                                                                  np.concatenate([self._valid, v]))
        self._pending = []
        self._pk, self._pb, self._pv = self._keys[:0], self._bits[:0], self._valid[:0]
        return len(k)

    # ---- lookup ---------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self._keys) + len(self._pk)

    @staticmethod
    def _search(sorted_keys, k):
        pos = np.searchsorted(sorted_keys, k)
        posc = np.clip(pos, 0, max(len(sorted_keys) - 1, 0))
        hit = (pos < len(sorted_keys)) & (sorted_keys[posc] == k) if len(sorted_keys) else np.zeros(len(k), bool)
        return posc, hit

    def get(self, keys, smiles_of):
        k = np.asarray(pd.Series(keys, dtype=object).astype(str).to_numpy()).astype("S14")
        pos, hit = self._search(self._keys, k)
        ppos, phit = self._search(self._pk, k)
        miss = ~hit & ~phit
        if miss.any():
            missing = np.unique(k[miss])
            smiles = list(smiles_of([m.decode() for m in missing]))
            known = np.array([isinstance(s, str) and len(s) > 0 for s in smiles], dtype=bool)
            if known.any():                               # a key with NO smiles available is not cached (not "invalid forever")
                bits, valid = pack_fingerprints([s for s, ok in zip(smiles, known) if ok], self.radius, self.n_bits)
                self._pending.append((missing[known], bits, valid))
                pk, pb, pv = (np.concatenate(x) for x in zip(*self._pending))
                self._pk, self._pb, self._pv = self._sorted_unique(pk, pb, pv)
                if sum(len(p[0]) for p in self._pending) >= self.flush_rows:
                    self.flush()
                pos, hit = self._search(self._keys, k)
                ppos, phit = self._search(self._pk, k)
        bits = np.zeros((len(k), self.n_bits // 8), dtype=np.uint8)
        valid = np.zeros(len(k), dtype=bool)
        if hit.any():
            bits[hit], valid[hit] = self._bits[pos[hit]], self._valid[pos[hit]]
        only_p = phit & ~hit
        if only_p.any():
            bits[only_p], valid[only_p] = self._pb[ppos[only_p]], self._pv[ppos[only_p]]
        return bits, valid

    def meta(self):
        return {"radius": self.radius, "n_bits": self.n_bits, "n_cached": len(self), "n_shards": len(self._shards()), "dir": str(self.dir)}

    def write_meta(self):
        self.flush()
        (self.dir / "metadata.json").write_text(json.dumps(self.meta(), indent=2), encoding="utf-8")
