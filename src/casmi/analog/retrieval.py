"""Analog retrieval over the frozen reference library.

Stage 1 (prefilter, whole library at once): two L2-normalized binned representations per spectrum --
fragments and neutral losses (`precursor - mz`, mz < precursor) -- in `bin_width` bins with intensities
summed per bin, i.e. EXACTLY `casmi.spectra.binning.bin_spectrum` (+ `neutral_loss.neutral_losses`) but built
vectorized as one scipy CSR matrix. Query x library cosine = one sparse x dense product per query batch
(`device='cuda'` runs it with torch sparse CSR). Neutral losses make shifted analogs reachable (a
substituent change shifts fragments but keeps many losses).

Stage 2 (rescore): the union of the per-channel top-N is rescored with the EXISTING
`casmi.spectra.similarity.modified_cosine_similarity` (precursor-shift-aware, `peak_tol_da`).

Exclusions per query (never optional): the query's own library row, T1 duplicates (same identity
`peak_hash` -- the production reading; T2 needs identity peaks the bundle does not ship, exactly as in
frozen inference), and every reference of a fold-hidden connectivity (C2/C3, `allowed_mask`). Configurable:
same polarity only, |precursor delta| <= max_precursor_delta_da.
"""
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.spectra.similarity import modified_cosine_similarity

NEIGHBOR_COLUMNS = ["query_id", "analog_rank", "ref_row", "ref_spectrum_id", "analog_connectivity_key", "modified_cosine", "n_matched",
                    "cosine", "nl_cosine", "precursor_delta", "ref_adduct", "same_adduct"]
T2_POLICY = "T1 excluded by identity peak_hash; T2 not applied (identity peaks not shipped in the bundle -- same as production inference)"


@dataclass(frozen=True)
class AnalogConfig:
    bin_width_da: float = 0.1
    peak_tol_da: float = 0.02
    intensity_power: float = 1.0
    max_mz: float = 5000.0
    prefilter_top_n: int = 200
    rescore_top_n: int = 50
    top_k_analogs: int = 10
    same_polarity_only: bool = True
    max_precursor_delta_da: float = 300.0
    query_batch_size: int = 64
    device: str = "auto"

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    @property
    def n_bins(self):
        return int(np.floor(self.max_mz / self.bin_width_da)) + 1


# ---------------------------------------------------------------------------------------------
# vectorized binning (== bin_spectrum, row-wise)
# ---------------------------------------------------------------------------------------------

def binned_csr(offsets, mz, intensity, precursor, cfg, neutral_loss=False):
    """CSR (n_spectra x n_bins, float32), rows L2-normalized, from CSR peaks. `offsets` may be any
    contiguous slice of a global offsets array (it is re-based). Empty rows stay all-zero."""
    import scipy.sparse as sp
    offsets = np.asarray(offsets, dtype=np.int64)
    n = len(offsets) - 1
    a, b = int(offsets[0]), int(offsets[-1])
    x = np.asarray(mz[a:b], dtype=np.float64)
    v = np.asarray(intensity[a:b], dtype=np.float64)
    row = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
    if neutral_loss:
        prec = np.asarray(precursor, dtype=np.float64)[row]
        keep = x < prec                                       # == neutral_loss.valid_neutral_loss_mask
        x = prec - x
    else:
        keep = np.ones(len(x), bool)
    if cfg.intensity_power != 1.0:
        v = np.power(np.clip(v, 0, None), cfg.intensity_power)
    keep &= np.isfinite(x) & np.isfinite(v) & (x >= 0) & (x < cfg.max_mz)
    col = np.floor(x[keep] / cfg.bin_width_da).astype(np.int64)
    m = sp.coo_matrix((v[keep], (row[keep], col)), shape=(n, cfg.n_bins)).tocsr()
    m.sum_duplicates()                                        # peaks in the same bin are SUMMED (bin_spectrum)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    inv = np.where(norms > 0, 1.0 / np.where(norms > 0, norms, 1.0), 0.0)
    return (sp.diags(inv) @ m).astype(np.float32).tocsr()


# ---------------------------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------------------------

class AnalogLibrary:
    """The frozen bundle reference library + per-row arrays the search needs (connectivity per row,
    polarity / adduct, precursor, identity peak hash)."""

    def __init__(self, bundle_dir):
        from casmi_infer.reference_index import ReferenceLibrary
        self.bundle_dir = Path(bundle_dir)
        self.lib = ReferenceLibrary(bundle_dir, mmap=True)
        conn = pd.read_parquet(self.bundle_dir / "connectivities.parquet", columns=["connectivity_key"])
        self.conn_keys = conn["connectivity_key"].astype(str).to_numpy()
        n_refs, n_conn = len(self.lib.meta), len(self.conn_keys)
        off = np.asarray(self.lib.index_offsets, dtype=np.int64)
        if len(off) != n_conn + 1:
            raise ValueError(f"index offsets ({len(off)}) do not match connectivities ({n_conn}) + 1")
        ref_conn = np.full(n_refs, -1, dtype=np.int64)
        ref_conn[np.asarray(self.lib.index_ids, dtype=np.int64)] = np.repeat(np.arange(n_conn), np.diff(off))
        if (ref_conn < 0).any():
            raise ValueError(f"{int((ref_conn < 0).sum())} reference rows without a connectivity")
        self.ref_conn_idx = ref_conn
        self.ref_keys = self.conn_keys[ref_conn]
        self.n_refs = n_refs
        self.precursor = self.lib.precursor_mz
        self.polarity_code = self.lib.polarity_code
        self.adduct = self.lib.meta["adduct"].astype(object).to_numpy()
        self.spectrum_id = self.lib.spectrum_id
        h_codes, self.hash_uniques = pd.factorize(self.lib.meta["peak_hash"].astype(object))
        self.hash_code = h_codes.astype(np.int64)
        order = np.argsort(self.hash_code, kind="stable")
        counts = np.bincount(self.hash_code[self.hash_code >= 0], minlength=len(self.hash_uniques))
        self._hash_rows = order[np.sum(self.hash_code < 0):]
        self._hash_off = np.concatenate([[0], np.cumsum(counts)])
        self._hash_index = pd.Index(self.hash_uniques)
        self._sid_index = pd.Index(self.spectrum_id)

    def rows_of_spectrum_ids(self, spectrum_ids):
        """Library row per spectrum id (-1 when absent)."""
        return self._sid_index.get_indexer(pd.Index(pd.Series(spectrum_ids).astype(str)))

    def rows_with_hash(self, peak_hash):
        c = self._hash_index.get_indexer([peak_hash])[0]
        return np.zeros(0, np.int64) if c < 0 else self._hash_rows[self._hash_off[c]:self._hash_off[c + 1]]

    def peaks(self, row):
        return self.lib.peaks(int(row))

    def hidden_mask(self, hidden_keys):
        """Bool over reference rows: True = reference of a hidden connectivity (C2/C3 of the fold)."""
        if not hidden_keys:
            return np.zeros(self.n_refs, bool)
        hidden_conn = np.flatnonzero(pd.Index(self.conn_keys).isin(list(hidden_keys)))
        return np.isin(self.ref_conn_idx, hidden_conn)

    def binned_matrices(self, cfg, cache_dir=None, chunk_rows=250_000, log=print):
        """(fragment CSR, neutral-loss CSR) over all reference rows; cached as .npz when `cache_dir`."""
        import scipy.sparse as sp
        tag = f"bw{cfg.bin_width_da}_p{cfg.intensity_power}_mz{cfg.max_mz}"
        if cache_dir is not None:
            d = Path(cache_dir) / f"analog_library_{tag}"
            ff, fn = d / "fragments.npz", d / "neutral_losses.npz"
            if ff.exists() and fn.exists():
                log(f"[analog library] cache hit {d}")
                return sp.load_npz(ff).tocsr(), sp.load_npz(fn).tocsr()
        off = np.asarray(self.lib.peaks_offsets, dtype=np.int64)
        parts_f, parts_n = [], []
        for s in range(0, self.n_refs, chunk_rows):
            e = min(s + chunk_rows, self.n_refs)
            o = off[s:e + 1]
            prec = self.precursor[s:e]
            parts_f.append(binned_csr(o, self.lib.peaks_mz, self.lib.peaks_int, prec, cfg, neutral_loss=False))
            parts_n.append(binned_csr(o, self.lib.peaks_mz, self.lib.peaks_int, prec, cfg, neutral_loss=True))
            log(f"[analog library] binned rows {e}/{self.n_refs}")
        Lf, Ln = sp.vstack(parts_f).tocsr(), sp.vstack(parts_n).tocsr()
        if cache_dir is not None:
            d.mkdir(parents=True, exist_ok=True)
            sp.save_npz(d / "fragments.npz", Lf)
            sp.save_npz(d / "neutral_losses.npz", Ln)
            (d / "metadata.json").write_text(json.dumps({"config": asdict(cfg), "n_refs": int(self.n_refs), "nnz_fragments": int(Lf.nnz),
                                                         "nnz_neutral_losses": int(Ln.nnz), "bundle_dir": str(self.bundle_dir),
                                                         "created_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        return Lf, Ln


# ---------------------------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------------------------

def _resolve_device(device):
    if device in ("cpu", "cuda"):
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class AnalogSearcher:
    def __init__(self, alib, Lf, Ln, cfg=AnalogConfig()):
        self.alib, self.cfg = alib, cfg
        self.Lf, self.Ln = Lf, Ln
        self.device = _resolve_device(cfg.device)
        if self.device == "cuda":
            import torch
            self._torch = torch
            self._Lf_t, self._Ln_t = self._to_torch(Lf), self._to_torch(Ln)

    def _to_torch(self, m):
        t = self._torch
        return t.sparse_csr_tensor(t.from_numpy(m.indptr.astype(np.int64)), t.from_numpy(m.indices.astype(np.int64)),
                                   t.from_numpy(m.data.astype(np.float32)), size=m.shape).to("cuda")

    def _scores(self, L, L_t, Q):
        """Dense (n_queries x n_refs) float32 cosine of query rows Q (CSR) against library L."""
        Qd = Q.T.toarray().astype(np.float32)                 # n_bins x n_q (small)
        if self.device == "cuda":
            t = self._torch
            with t.inference_mode():
                S = t.sparse.mm(L_t, t.from_numpy(Qd).to("cuda", non_blocking=True))
                return S.T.contiguous().cpu().numpy()
        return np.ascontiguousarray(np.asarray(L @ Qd, dtype=np.float32).T)

    def base_mask(self, allowed_mask):
        """Fold-level ALLOWED mask (not hidden) split by polarity code, computed once per fold."""
        allowed = np.asarray(allowed_mask, bool)
        if not self.cfg.same_polarity_only:
            return {None: allowed}
        return {int(p): allowed & (self.alib.polarity_code == p) for p in np.unique(self.alib.polarity_code)}

    def search_batch(self, q, base):
        """`q`: dict of arrays for a query batch -- query_id, offsets/mz/intensity (CSR peaks), precursor,
        polarity_code, adduct, peak_hash, self_row (-1 if none). Returns the neighbor DataFrame."""
        cfg = self.cfg
        Qf = binned_csr(q["offsets"], q["mz"], q["intensity"], q["precursor"], cfg, neutral_loss=False)
        Qn = binned_csr(q["offsets"], q["mz"], q["intensity"], q["precursor"], cfg, neutral_loss=True)
        Sf = self._scores(self.Lf, getattr(self, "_Lf_t", None), Qf)
        Sn = self._scores(self.Ln, getattr(self, "_Ln_t", None), Qn)
        rows = []
        prec_lib = self.alib.precursor
        for i in range(len(q["query_id"])):
            if cfg.same_polarity_only:
                m = base.get(int(q["polarity_code"][i]))
                if m is None:                                  # polarity unseen in the library: no analogs
                    continue
            else:
                m = base[None]
            m = m & (np.abs(prec_lib - q["precursor"][i]) <= cfg.max_precursor_delta_da)
            excl = list(self.alib.rows_with_hash(q["peak_hash"][i])) if q["peak_hash"][i] is not None else []
            if q["self_row"][i] >= 0:
                excl.append(int(q["self_row"][i]))
            if excl:
                m = m.copy()
                m[np.asarray(excl, dtype=np.int64)] = False
            cand = set()
            for S in (Sf, Sn):
                s = np.where(m, S[i], -np.inf)
                n = min(cfg.prefilter_top_n, int(m.sum()))
                if n <= 0:
                    continue
                top = np.argpartition(-s, n - 1)[:n]
                cand.update(top[np.isfinite(s[top]) & (s[top] > 0)].tolist())
            if not cand:
                continue
            cand = np.array(sorted(cand), dtype=np.int64)
            pre = np.maximum(Sf[i, cand], Sn[i, cand])
            keep = cand[np.lexsort((cand, -pre))][:cfg.rescore_top_n]
            a, b = int(q["offsets"][i]), int(q["offsets"][i + 1])
            qm, qi = np.asarray(q["mz"][a:b], float), np.asarray(q["intensity"][a:b], float)
            for r in keep:
                pk = self.alib.peaks(r)
                mc = modified_cosine_similarity(qm, qi, float(q["precursor"][i]), pk["mzs"], pk["intensities"], pk["precursor_mz"], tol_da=cfg.peak_tol_da)
                rows.append((q["query_id"][i], int(r), mc["score"], mc["n_matched"], float(Sf[i, r]), float(Sn[i, r])))
        if not rows:
            return pd.DataFrame(columns=NEIGHBOR_COLUMNS)
        d = pd.DataFrame(rows, columns=["query_id", "ref_row", "modified_cosine", "n_matched", "cosine", "nl_cosine"])
        d = d.sort_values(["query_id", "modified_cosine", "ref_row"], ascending=[True, False, True], kind="mergesort")
        d["analog_rank"] = d.groupby("query_id").cumcount() + 1
        d = d[d["analog_rank"] <= cfg.top_k_analogs].copy()
        qprec = pd.Series(q["precursor"], index=q["query_id"])
        qadd = pd.Series(q["adduct"], index=q["query_id"])
        d["ref_spectrum_id"] = self.alib.spectrum_id[d["ref_row"].to_numpy()]
        d["analog_connectivity_key"] = self.alib.ref_keys[d["ref_row"].to_numpy()]
        d["precursor_delta"] = d["query_id"].map(qprec).to_numpy() - prec_lib[d["ref_row"].to_numpy()]
        d["ref_adduct"] = self.alib.adduct[d["ref_row"].to_numpy()]
        d["same_adduct"] = (d["ref_adduct"].astype(str).to_numpy() == d["query_id"].map(qadd).astype(str).to_numpy())
        return d[NEIGHBOR_COLUMNS].reset_index(drop=True)


def library_query_batch(alib, query_ids, self_rows):
    """Query dict for DEV queries that ARE library spectra: peaks / precursor / polarity / adduct / hash come
    from their own library row (identical preprocessing by construction)."""
    rows = np.asarray(self_rows, dtype=np.int64)
    if (rows < 0).any():
        raise ValueError(f"{int((rows < 0).sum())} dev queries are not library rows -- use raw_query_batch")
    off = np.asarray(alib.lib.peaks_offsets, dtype=np.int64)
    counts = off[rows + 1] - off[rows]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    mz = np.concatenate([np.asarray(alib.lib.peaks_mz[off[r]:off[r + 1]]) for r in rows]) if len(rows) else np.zeros(0)
    it = np.concatenate([np.asarray(alib.lib.peaks_int[off[r]:off[r + 1]]) for r in rows]) if len(rows) else np.zeros(0)
    return {"query_id": np.asarray(query_ids).astype(str), "offsets": offsets, "mz": mz, "intensity": it,
            "precursor": alib.precursor[rows], "polarity_code": alib.polarity_code[rows], "adduct": alib.adduct[rows],
            "peak_hash": np.array([alib.hash_uniques[c] if c >= 0 else None for c in alib.hash_code[rows]], dtype=object),
            "self_row": rows}


def raw_query_batch(alib, query_ids, mzs_list, ints_list, precursors, ion_modes, adducts, max_peaks=100):
    """Query dict for RAW spectra (e.g. hidden test): the same cleaning as training / inference
    (`remove_invalid_peaks`, deterministic `truncate_top_peaks`), T1 hash from the identity peaks."""
    from casmi.spectra.deduplication import compute_peak_hash
    from casmi.spectra.preprocessing import remove_invalid_peaks, truncate_top_peaks
    mz_parts, it_parts, hashes, counts = [], [], [], []
    for mzs, its, p in zip(mzs_list, ints_list, precursors):
        im, ii = remove_invalid_peaks(mzs if mzs is not None else [], its if its is not None else [])
        hashes.append(compute_peak_hash(im, ii, p))
        sm, si = truncate_top_peaks(im, ii, max_peaks=max_peaks)
        mz_parts.append(sm); it_parts.append(si); counts.append(len(sm))
    return {"query_id": np.asarray(query_ids).astype(str), "offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            "mz": np.concatenate(mz_parts) if mz_parts else np.zeros(0), "intensity": np.concatenate(it_parts) if it_parts else np.zeros(0),
            "precursor": np.asarray(precursors, float), "polarity_code": np.array([alib.lib.code("polarity", m) for m in ion_modes]),
            "adduct": np.asarray(adducts, dtype=object), "peak_hash": np.asarray(hashes, dtype=object), "self_row": np.full(len(mz_parts), -1)}


def build_neighbors(searcher, query_ids, self_rows, allowed_mask, out_dir, chunk=2000, log=print):
    """Resumable neighbor cache for one fold: `<out_dir>/part-NNNNN.parquet` per query chunk (+ done
    markers). Returns the concatenated neighbor table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = searcher.base_mask(allowed_mask)
    qids, rows = np.asarray(query_ids).astype(str), np.asarray(self_rows, dtype=np.int64)
    parts = []
    for ci, s in enumerate(range(0, len(qids), chunk)):
        f = out_dir / f"part-{ci:05d}.parquet"
        done = out_dir / f"part-{ci:05d}.done"
        if f.exists() and done.exists():
            parts.append(pd.read_parquet(f))
            continue
        res = []
        for b in range(s, min(s + chunk, len(qids)), searcher.cfg.query_batch_size):
            e = min(b + searcher.cfg.query_batch_size, s + chunk, len(qids))
            res.append(searcher.search_batch(library_query_batch(searcher.alib, qids[b:e], rows[b:e]), base))
        d = pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=NEIGHBOR_COLUMNS)
        d.to_parquet(f, index=False)
        done.write_text(json.dumps({"n_queries": int(min(s + chunk, len(qids)) - s), "n_rows": int(len(d))}), encoding="utf-8")
        parts.append(d)
        log(f"[analog neighbors] chunk {ci}: queries {s}-{min(s + chunk, len(qids))} -> {len(d)} rows")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=NEIGHBOR_COLUMNS)
