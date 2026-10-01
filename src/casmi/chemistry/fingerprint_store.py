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
    (`smiles_of(missing_keys) -> list of SMILES`) and persisting them as a new shard."""

    def __init__(self, root, radius=2, n_bits=2048):
        self.dir = Path(root) / f"r{radius}_b{n_bits}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.radius, self.n_bits = radius, n_bits
        self._keys = np.zeros(0, dtype="S14")
        self._bits = np.zeros((0, n_bits // 8), dtype=np.uint8)
        self._valid = np.zeros(0, dtype=bool)
        self._load()

    def _load(self):
        ks, bs, vs = [], [], []
        for f in sorted(self.dir.glob("shard-*.npz")):
            z = np.load(f)
            ks.append(z["keys"]); bs.append(z["bits"]); vs.append(z["valid"])
        if ks:
            k = np.concatenate(ks)
            order = np.argsort(k, kind="stable")
            k, b, v = k[order], np.concatenate(bs)[order], np.concatenate(vs)[order]
            first = np.concatenate([[True], k[1:] != k[:-1]])            # duplicate keys across shards: keep first
            self._keys, self._bits, self._valid = k[first], b[first], v[first]

    def __len__(self):
        return len(self._keys)

    def _locate(self, keys):
        k = np.asarray(pd.Series(keys, dtype=object).astype(str).to_numpy()).astype("S14")
        pos = np.searchsorted(self._keys, k)
        posc = np.clip(pos, 0, max(len(self._keys) - 1, 0))
        hit = (pos < len(self._keys)) & (self._keys[posc] == k) if len(self._keys) else np.zeros(len(k), bool)
        return k, posc, hit

    def get(self, keys, smiles_of):
        k, pos, hit = self._locate(keys)
        if (~hit).any():
            missing = np.unique(k[~hit])
            smiles = list(smiles_of([m.decode() for m in missing]))
            known = np.array([isinstance(s, str) and len(s) > 0 for s in smiles], dtype=bool)
            if known.any():                               # a key with NO smiles available is not cached (not "invalid forever")
                bits, valid = pack_fingerprints([s for s, ok in zip(smiles, known) if ok], self.radius, self.n_bits)
                n = len(list(self.dir.glob("shard-*.npz")))
                tmp = self.dir / f"tmp-{n:05d}.npz"      # never matches the shard-*.npz glob if interrupted
                np.savez(tmp, keys=missing[known], bits=bits, valid=valid)
                tmp.replace(self.dir / f"shard-{n:05d}.npz")
                self._load()
            k, pos, hit = self._locate(keys)
        bits = np.zeros((len(k), self.n_bits // 8), dtype=np.uint8)
        valid = np.zeros(len(k), dtype=bool)
        if hit.any():
            bits[hit] = self._bits[pos[hit]]
            valid[hit] = self._valid[pos[hit]]
        return bits, valid

    def meta(self):
        return {"radius": self.radius, "n_bits": self.n_bits, "n_cached": len(self), "dir": str(self.dir)}

    def write_meta(self):
        (self.dir / "metadata.json").write_text(json.dumps(self.meta(), indent=2), encoding="utf-8")
