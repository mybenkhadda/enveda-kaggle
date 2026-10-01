"""Compact persisted formula index for the v2 universe: formula -> candidate ids.

On disk (no pickle, no JSON dataset):
    formula_vocab.parquet            formula (sorted), formula_id, offset, count
    formula_candidate_ids.npy        int32 candidate ids grouped by formula_id (CSR payload; ids ASC per formula)
    candidate_formula_id.npy         int32 formula_id per candidate_id (-1 = unknown formula)
    formula_index.meta.json

METADATA / EVIDENCE only: like `mass_index.FormulaIndex`, it is never a hard retrieval filter until a
validated formula predictor exists. Built with pyarrow dictionary encoding (no Python set of formulas).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FORMULA_INDEX_VERSION = 1


class CompactFormulaIndex:
    def __init__(self, vocab, offsets, candidate_ids, candidate_formula_id):
        self.vocab = np.asarray(vocab, dtype=object)              # sorted formulas
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.candidate_ids = np.asarray(candidate_ids)
        self.candidate_formula_id = np.asarray(candidate_formula_id)
        self._pos = None

    @classmethod
    def build(cls, formulas, candidate_ids, n_candidates):
        """`formulas[i]` is the formula of `candidate_ids[i]` (None/NaN allowed -> unknown)."""
        import pyarrow as pa
        import pyarrow.compute as pc
        f = pa.array(pd.Series(formulas, dtype=object).where(pd.notna(pd.Series(formulas, dtype=object)), None), type=pa.string())
        ids = np.asarray(candidate_ids, dtype=np.int64)
        valid = np.asarray(pc.is_valid(f).to_numpy(zero_copy_only=False), dtype=bool)
        enc = pc.dictionary_encode(f.filter(pa.array(valid)))
        dict_vals = np.asarray(enc.dictionary.to_pylist(), dtype=object)
        sort_order = np.argsort(dict_vals.astype(str), kind="stable")          # dictionary order -> sorted vocab
        remap = np.empty(len(dict_vals), np.int64)
        remap[sort_order] = np.arange(len(dict_vals))
        codes = remap[np.asarray(enc.indices.to_numpy(zero_copy_only=False), dtype=np.int64)]
        vids = ids[valid]
        order = np.lexsort((vids, codes))
        counts = np.bincount(codes, minlength=len(dict_vals))
        offsets = np.concatenate([[0], np.cumsum(counts)])
        cand_fid = np.full(int(n_candidates), -1, dtype=np.int32)
        cand_fid[vids] = codes.astype(np.int32)
        return cls(dict_vals[sort_order], offsets, vids[order].astype(np.int32), cand_fid)

    def __len__(self):
        return len(self.vocab)

    def formula_id(self, formula):
        if self._pos is None:
            self._pos = pd.Index(self.vocab.astype(str))
        loc = self._pos.get_indexer([str(formula)])[0]
        return int(loc)

    def lookup(self, formula):
        fid = self.formula_id(formula)
        return np.zeros(0, np.int32) if fid < 0 else self.lookup_id(fid)

    def lookup_id(self, formula_id):
        return np.asarray(self.candidate_ids[self.offsets[formula_id]:self.offsets[formula_id + 1]])

    def counts(self):
        return np.diff(self.offsets)

    def formula_frequency(self, candidate_ids):
        """Number of universe candidates sharing each candidate's formula (0 for unknown formula) -- the
        raw ingredient of a formula-frequency prior."""
        fid = self.candidate_formula_id[np.asarray(candidate_ids, dtype=np.int64)]
        c = self.counts()
        return np.where(fid >= 0, c[np.clip(fid, 0, None)], 0)

    def save(self, out_dir, extra=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"formula": self.vocab.astype(str), "formula_id": np.arange(len(self.vocab), dtype=np.int32),
                      "offset": self.offsets[:-1], "count": self.counts()}).to_parquet(out_dir / "formula_vocab.parquet", index=False)
        np.save(out_dir / "formula_candidate_ids.npy", self.candidate_ids.astype(np.int32))
        np.save(out_dir / "candidate_formula_id.npy", self.candidate_formula_id.astype(np.int32))
        meta = {"formula_index_version": FORMULA_INDEX_VERSION, "n_formulas": int(len(self)), "n_candidates": int(len(self.candidate_formula_id)),
                "n_indexed": int(len(self.candidate_ids)), "hard_filter": False, "saved_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
        (out_dir / "formula_index.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, out_dir, mmap=True):
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "formula_index.meta.json").read_text(encoding="utf-8"))
        if meta.get("formula_index_version") != FORMULA_INDEX_VERSION:
            raise ValueError("formula index version mismatch -- rebuild it")
        v = pd.read_parquet(out_dir / "formula_vocab.parquet")
        mode = "r" if mmap else None
        offsets = np.concatenate([v["offset"].to_numpy(np.int64), [int(v["offset"].iloc[-1] + v["count"].iloc[-1]) if len(v) else 0]])
        return cls(v["formula"].to_numpy(object), offsets, np.load(out_dir / "formula_candidate_ids.npy", mmap_mode=mode),
                   np.load(out_dir / "candidate_formula_id.npy", mmap_mode=mode))
