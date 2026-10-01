"""Exact-mass candidate index: given a neutral mass and a ppm/Da tolerance, which structures
are plausible? A sorted-array + binary-search index -- deliberately the simplest thing that
works; a few hundred thousand structures need nothing fancier than `np.searchsorted`.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Bumped whenever the on-disk .npz/.meta.json layout or the search semantics change, so a
# cache built by an older version is never silently trusted (`cached_mass_index` in
# `casmi.candidates.cache` checks this against the current value before treating a saved
# index as a cache hit).
MASS_INDEX_VERSION = 2


class MassIndex:
    def __init__(self, keys, masses):
        """`keys`, `masses`: parallel arrays/sequences (typically `connectivity_key` and
        `exact_mass` from a molecule table). Rows with a null mass are dropped -- a candidate
        generator has nothing to search for them anyway."""
        keys = np.asarray(keys)
        masses = np.asarray(masses, dtype=float)
        valid = ~np.isnan(masses)
        order = np.argsort(masses[valid])
        self.keys = keys[valid][order]
        self.masses = masses[valid][order]

    @classmethod
    def from_dataframe(cls, molecule_table, mass_col="exact_mass", key_col="connectivity_key"):
        return cls(molecule_table[key_col], molecule_table[mass_col])

    # `from_molecule_table` predates `from_dataframe` (added when the candidate-generation
    # design was formalized) -- both names are kept so neither call site has to change.
    from_molecule_table = from_dataframe

    def __len__(self):
        return len(self.masses)

    def _window(self, neutral_mass, tolerance_ppm, tolerance_da):
        if (tolerance_ppm is None) == (tolerance_da is None):
            raise ValueError("give exactly one of tolerance_ppm or tolerance_da")
        tol_da = tolerance_da if tolerance_da is not None else abs(neutral_mass) * tolerance_ppm * 1e-6
        lo = np.searchsorted(self.masses, neutral_mass - tol_da, side="left")
        hi = np.searchsorted(self.masses, neutral_mass + tol_da, side="right")
        return lo, hi

    def query(self, neutral_mass, tolerance_ppm=None, tolerance_da=None):
        """Every key within tolerance of `neutral_mass`, closest first. Exactly one of
        `tolerance_ppm` / `tolerance_da` must be given."""
        lo, hi = self._window(neutral_mass, tolerance_ppm, tolerance_da)
        cand_keys = self.keys[lo:hi]
        cand_masses = self.masses[lo:hi]
        order = np.argsort(np.abs(cand_masses - neutral_mass))
        return cand_keys[order]

    def query_with_masses(self, neutral_mass, tolerance_ppm=None, tolerance_da=None):
        """Like `query`, but also returns each candidate's own mass (needed to report
        `candidate_exact_mass`/`mass_error_ppm` per candidate -- `query` alone only exposes
        the key). Returns `(keys, masses)`, both closest-first."""
        lo, hi = self._window(neutral_mass, tolerance_ppm, tolerance_da)
        cand_keys = self.keys[lo:hi]
        cand_masses = self.masses[lo:hi]
        order = np.argsort(np.abs(cand_masses - neutral_mass))
        return cand_keys[order], cand_masses[order]

    def query_many(self, neutral_masses, tolerance_ppm=None, tolerance_da=None):
        """`query` for an array of neutral masses; returns a list of key-arrays, same order as
        the input."""
        return [
            self.query(m, tolerance_ppm=tolerance_ppm, tolerance_da=tolerance_da)
            for m in neutral_masses
        ]

    def candidate_count(self, neutral_mass, tolerance_ppm=None, tolerance_da=None):
        lo, hi = self._window(neutral_mass, tolerance_ppm, tolerance_da)
        return int(hi - lo)

    def save(self, path, library_fingerprint=None, mass_col="exact_mass", key_col="connectivity_key"):
        """Persist the sorted arrays as `<path>.npz` + a `<path>.meta.json` sidecar. The
        sidecar carries everything `cached_mass_index` (`casmi.candidates.cache`) needs to
        decide whether a later run may trust this file instead of rebuilding: which upstream
        molecule table it was built from (`library_fingerprint`), which columns, under which
        `MASS_INDEX_VERSION`, and at what dtype -- so a cache built from a stale
        `molecule_metadata`, or by an older/incompatible version of this class, is never
        silently reused."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=self.keys, masses=self.masses)
        meta = {
            "mass_index_version": MASS_INDEX_VERSION,
            "n_structures": len(self),
            "mass_min": float(self.masses.min()) if len(self) else None,
            "mass_max": float(self.masses.max()) if len(self) else None,
            "mass_col": mass_col,
            "key_col": key_col,
            "masses_dtype": str(self.masses.dtype),
            "library_fingerprint": library_fingerprint,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path):
        path = Path(path)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        data = np.load(npz_path, allow_pickle=True)
        index = cls.__new__(cls)
        index.keys = data["keys"]
        index.masses = data["masses"]
        return index

    @staticmethod
    def read_meta(path):
        """Peek at `<path>.meta.json` without loading the (potentially large) `.npz` arrays --
        what `cached_mass_index` checks before deciding CACHE HIT vs. CACHE MISS. Returns
        `None` if no sidecar exists."""
        path = Path(path)
        meta_path = path if path.suffix == ".json" else path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)


def validate_against_brute_force(mass_index, molecule_table, tolerance_ppms=(1, 5, 10, 20, 50),
                                  mass_col="exact_mass", key_col="connectivity_key", n_samples=200, seed=42):
    """Cross-check `mass_index` against an independent brute-force DataFrame filter, over a
    deterministic random sample of query masses at each of `tolerance_ppms`. Asserts EXACT
    candidate-key-set equality at every (query, tolerance) pair -- not just "index is
    non-empty". Also explicitly exercises the boundary cases a binary-search implementation
    could get subtly wrong: the query's own exact mass (exact hit), a mass with no nearby
    library entries (no hit), a mass far outside the library range (no hit, no IndexError),
    and duplicate library masses (multiple hits at the same mass).

    Returns a `casmi.validation.checks.CheckResult`.
    """
    from casmi.validation.checks import CheckResult

    rng = np.random.RandomState(seed)
    masses = molecule_table[mass_col].dropna().to_numpy(dtype=float)
    keys = molecule_table[key_col].to_numpy()

    query_masses = list(rng.choice(masses, size=min(n_samples, len(masses)), replace=False))
    # boundary cases, appended (not sampled): exact hit, far outside range, and (if present) a
    # duplicated mass.
    query_masses.append(float(masses[0]))
    query_masses.append(float(masses.max()) + 10_000.0)
    dup_mass = pd.Series(masses).value_counts()
    dup_mass = dup_mass[dup_mass > 1]
    if len(dup_mass):
        query_masses.append(float(dup_mass.index[0]))

    mismatches = []
    for q in query_masses:
        for ppm in tolerance_ppms:
            tol_da = abs(q) * ppm * 1e-6
            brute_force = set(keys[np.abs(masses - q) <= tol_da])
            indexed = set(mass_index.query(q, tolerance_ppm=ppm))
            if brute_force != indexed:
                mismatches.append({
                    "query_mass": q, "tolerance_ppm": ppm,
                    "n_brute_force": len(brute_force), "n_indexed": len(indexed),
                    "only_in_brute_force": len(brute_force - indexed), "only_in_indexed": len(indexed - brute_force),
                })

    n_checked = len(query_masses) * len(tolerance_ppms)
    return CheckResult(
        name="mass index matches brute force",
        passed=not mismatches,
        detail=(f"{n_checked} (query, ppm) pairs checked, 0 mismatches"
                if not mismatches else f"{len(mismatches)}/{n_checked} mismatches, e.g. {mismatches[0]}"),
    )


# ---------------------------------------------------------------------------------------------
# v6: OPEN candidate universe index (TRAIN + external sources), retrieval identical to inference
# ---------------------------------------------------------------------------------------------

OPEN_INDEX_VERSION = 1


class OpenMassIndex:
    """Mass-sorted index over MASS VARIANTS of the unified universe (a connectivity may own several
    masses). Rows are ordered by (mass ASC, connectivity_key ASC) -- a total, deterministic order.
    Retrieval uses the inference arithmetic (`casmi_infer.mass_search.MassSearch`):

        tol_da = |m| * ppm * 1e-6;  window = searchsorted(left, m - tol) : searchsorted(right, m + tol)
        one row per connectivity (min abs ppm), ordered abs ppm ASC then connectivity_key ASC.

    Persisted as `mass_index.npy` (float64 masses) + `mass_index_keys.npy` (unicode keys, no pickle)."""

    def __init__(self, keys, masses):
        keys = np.asarray(keys).astype(str)
        masses = np.asarray(masses, dtype=np.float64)
        ok = np.isfinite(masses)
        keys, masses = keys[ok], masses[ok]
        order = np.lexsort((keys, masses))
        self.keys, self.masses = keys[order], masses[order]
        self.key_table = np.unique(self.keys)                           # sorted -> conn_idx order == key order
        from casmi_infer.mass_search import MassSearch
        self._search = MassSearch(self.masses, np.searchsorted(self.key_table, self.keys), self.key_table)

    @classmethod
    def from_variants(cls, variants, key_col="connectivity_key", mass_col="exact_mass"):
        return cls(variants[key_col].to_numpy(), variants[mass_col].to_numpy())

    def __len__(self):
        return len(self.masses)

    def search(self, neutral_mass, ppm):
        """`(keys, abs_ppm)` for every connectivity within `ppm` of `neutral_mass` (deterministic order)."""
        if neutral_mass is None or not np.isfinite(neutral_mass) or neutral_mass <= 0:
            return np.zeros(0, dtype=self.key_table.dtype), np.zeros(0)
        lo, hi = self._search.window(neutral_mass, ppm)
        ci, ap = self._search._dedupe(np.arange(lo, hi), neutral_mass)
        return self.key_table[ci], ap

    def save(self, out_dir, extra=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "mass_index.npy", self.masses)
        np.save(out_dir / "mass_index_keys.npy", self.keys)
        meta = {"open_index_version": OPEN_INDEX_VERSION, "n_rows": int(len(self)), "n_connectivities": int(len(self.key_table)),
                "order": "mass ASC, connectivity_key ASC", "saved_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
        (out_dir / "mass_index.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, out_dir):
        out_dir = Path(out_dir)
        return cls(np.load(out_dir / "mass_index_keys.npy"), np.load(out_dir / "mass_index.npy"))


def brute_force_open_search(keys, masses, neutral_mass, ppm):
    """Independent reference implementation: full scan, inclusive window, min abs ppm per connectivity,
    ordered (abs ppm, key)."""
    keys, masses = np.asarray(keys).astype(str), np.asarray(masses, dtype=np.float64)
    tol = abs(neutral_mass) * ppm * 1e-6
    hit = (masses >= neutral_mass - tol) & (masses <= neutral_mass + tol)
    best = {}
    for k, m in zip(keys[hit], masses[hit]):
        ap = abs(1e6 * (m - neutral_mass) / neutral_mass)
        best[k] = min(ap, best.get(k, np.inf))
    items = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))
    return np.array([k for k, _ in items], dtype=str), np.array([v for _, v in items], dtype=float)


def open_index_parity(index, keys, masses, query_masses, ppms=(5, 10, 50)):
    """Rows of (query, ppm, n_index, n_brute, keys_equal, ppm_max_abs_diff); every row must be equal."""
    rows = []
    for q in query_masses:
        for ppm in ppms:
            a_k, a_p = index.search(q, ppm)
            b_k, b_p = brute_force_open_search(keys, masses, q, ppm)
            eq = len(a_k) == len(b_k) and bool(np.array_equal(a_k, b_k))
            rows.append({"query_mass": float(q), "ppm": ppm, "n_index": int(len(a_k)), "n_brute": int(len(b_k)), "keys_equal": eq,
                         "ppm_max_abs_diff": float(np.max(np.abs(a_p - b_p))) if eq and len(a_k) else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# v2: id-based, memory-mappable index for the LARGE external universe (COCONUT / PubChem)
# ---------------------------------------------------------------------------------------------

CANDIDATE_INDEX_VERSION = 1


class CandidateMassIndex:
    """Variant-level mass index over integer `candidate_id`s (the key-sorted universe row number, so id
    order == connectivity_key order -- the same tie convention as inference `conn_idx`).

    Arithmetic (identical to `OpenMassIndex` / inference `MassSearch`):
        tol_da = |m| * ppm * 1e-6, inclusive window, one row per candidate (min abs ppm),
        ordered abs ppm ASC then candidate_id ASC.

    Storage: two .npy arrays (float64 masses, int32 ids) sorted by (mass, id) -- `load(mmap=True)` keeps
    RAM bounded. Keys/SMILES live in the universe table, never in the index."""

    def __init__(self, masses, candidate_ids, n_candidates=None):
        masses = np.asarray(masses, dtype=np.float64)
        ids = np.asarray(candidate_ids, dtype=np.int64)
        ok = np.isfinite(masses)
        masses, ids = masses[ok], ids[ok]
        order = np.lexsort((ids, masses))
        self.masses = masses[order]
        self.ids = ids[order].astype(np.int32)
        self.n_candidates = int(n_candidates if n_candidates is not None else (ids.max() + 1 if len(ids) else 0))
        self._by_id = None

    def __len__(self):
        return len(self.masses)

    # ---- persistence -------------------------------------------------------------------------
    def save(self, out_dir, extra=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "mass_index_masses.npy", self.masses)
        np.save(out_dir / "mass_index_candidate_ids.npy", self.ids)
        meta = {"candidate_index_version": CANDIDATE_INDEX_VERSION, "n_rows": int(len(self)), "n_candidates": self.n_candidates,
                "order": "mass ASC, candidate_id ASC", "ppm_rule": "tol_da=|m|*ppm*1e-6 inclusive; min abs ppm per candidate",
                "saved_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
        (out_dir / "mass_index.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, out_dir, mmap=True):
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "mass_index.meta.json").read_text(encoding="utf-8"))
        if meta.get("candidate_index_version") != CANDIDATE_INDEX_VERSION:
            raise ValueError(f"mass index version {meta.get('candidate_index_version')} != {CANDIDATE_INDEX_VERSION} -- rebuild it")
        idx = cls.__new__(cls)
        mode = "r" if mmap else None
        idx.masses = np.load(out_dir / "mass_index_masses.npy", mmap_mode=mode)
        idx.ids = np.load(out_dir / "mass_index_candidate_ids.npy", mmap_mode=mode)
        idx.n_candidates = int(meta["n_candidates"])
        idx._by_id = None
        return idx

    # ---- windows -----------------------------------------------------------------------------
    def windows(self, neutral_masses, ppm):
        """Vectorized `(lo, hi)` row windows for an array of neutral masses (NaN / <= 0 -> empty)."""
        m = np.asarray(neutral_masses, dtype=np.float64)
        ppm = np.broadcast_to(np.asarray(ppm, dtype=np.float64), m.shape)
        tol = np.abs(m) * ppm * 1e-6
        good = np.isfinite(m) & (m > 0)
        lo = np.searchsorted(self.masses, np.where(good, m - tol, np.inf), side="left")
        hi = np.searchsorted(self.masses, np.where(good, m + tol, -np.inf), side="right")
        return lo, np.maximum(hi, lo)

    def variant_counts(self, neutral_masses, ppm):
        """Variant ROWS per window (upper bound of the unique-candidate count; exact when every candidate
        has one mass). Fully vectorized."""
        lo, hi = self.windows(neutral_masses, ppm)
        return hi - lo

    def _dedupe(self, lo, hi, m, exclude_mask=None):
        ids = np.asarray(self.ids[lo:hi], dtype=np.int64)
        ap = np.abs(np.asarray(self.masses[lo:hi]) - m) / m * 1e6
        if exclude_mask is not None and len(ids):
            keep = ~exclude_mask[ids]
            ids, ap = ids[keep], ap[keep]
        if not len(ids):
            return ids, ap
        order = np.lexsort((ids, ap))                      # ap ASC, id ASC -> first occurrence = min ap per id
        ids, ap = ids[order], ap[order]
        _, first = np.unique(ids, return_index=True)
        first = np.sort(first)                             # keep the (ap, id) order
        return ids[first], ap[first]

    def search_ppm(self, neutral_mass, ppm, exclude_mask=None):
        """`(candidate_ids, abs_ppm)` within `ppm` of `neutral_mass`, deterministic order."""
        lo, hi = self.windows([neutral_mass], ppm)
        if hi[0] <= lo[0]:
            return np.zeros(0, np.int64), np.zeros(0)
        return self._dedupe(int(lo[0]), int(hi[0]), float(neutral_mass), exclude_mask)

    def search_ppm_batch(self, neutral_masses, ppm, exclude_mask=None, max_per_query=None):
        """CSR result for many queries: `(offsets, candidate_ids, abs_ppm)`; query i owns
        `[offsets[i], offsets[i+1])`. Windows are vectorized; only the per-window dedupe loops.
        `max_per_query` keeps the nearest-by-ppm candidates (the caller logs truncation)."""
        m = np.asarray(neutral_masses, dtype=np.float64)
        lo, hi = self.windows(m, ppm)
        parts_i, parts_p, sizes = [], [], np.zeros(len(m), np.int64)
        for q in range(len(m)):
            if hi[q] <= lo[q]:
                continue
            ids, ap = self._dedupe(int(lo[q]), int(hi[q]), float(m[q]), exclude_mask)
            if max_per_query is not None:
                ids, ap = ids[:max_per_query], ap[:max_per_query]
            parts_i.append(ids)
            parts_p.append(ap)
            sizes[q] = len(ids)
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        cat = (lambda parts, dt: np.concatenate(parts).astype(dt) if parts else np.zeros(0, dt))
        return offsets, cat(parts_i, np.int64), cat(parts_p, np.float64)

    def unique_counts(self, neutral_masses, ppm, exclude_mask=None):
        """Exact number of unique candidates per window (dedupes each window)."""
        offsets, _, _ = self.search_ppm_batch(neutral_masses, ppm, exclude_mask=exclude_mask)
        return np.diff(offsets)

    # ---- truth diagnostics -------------------------------------------------------------------
    def _id_csr(self):
        if self._by_id is None:
            order = np.argsort(np.asarray(self.ids), kind="stable")
            counts = np.bincount(np.asarray(self.ids)[order], minlength=self.n_candidates)
            self._by_id = (np.concatenate([[0], np.cumsum(counts)]), np.asarray(self.masses)[order])
        return self._by_id

    def candidate_masses(self, candidate_id):
        off, ms = self._id_csr()
        return ms[off[candidate_id]:off[candidate_id + 1]]

    def truth_abs_ppm(self, neutral_masses, truth_ids):
        """Min abs ppm between each query mass and ANY mass variant of its truth (NaN when the truth id is
        missing / -1 / has no variant). Recall@All at tolerance p is simply `truth_abs_ppm <= p`."""
        off, ms = self._id_csr()
        m = np.asarray(neutral_masses, dtype=np.float64)
        t = np.asarray(truth_ids, dtype=np.int64)
        out = np.full(len(m), np.nan)
        for i in range(len(m)):
            if t[i] < 0 or t[i] >= self.n_candidates or not np.isfinite(m[i]) or m[i] <= 0:
                continue
            v = ms[off[t[i]]:off[t[i] + 1]]
            if len(v):
                out[i] = float(np.min(np.abs(v - m[i]) / m[i] * 1e6))
        return out

    def truth_rank(self, neutral_masses, truth_ids, ppm, exclude_mask=None):
        """1-based rank of the truth in the deduplicated mass-ordered pool at `ppm` (NaN when outside the
        window or excluded) and the pool size. Returns `(rank, pool_size)`."""
        m = np.asarray(neutral_masses, dtype=np.float64)
        t = np.asarray(truth_ids, dtype=np.int64)
        offsets, ids, _ = self.search_ppm_batch(m, ppm, exclude_mask=exclude_mask)
        rank = np.full(len(m), np.nan)
        for i in range(len(m)):
            seg = ids[offsets[i]:offsets[i + 1]]
            hit = np.flatnonzero(seg == t[i])
            if len(hit):
                rank[i] = float(hit[0] + 1)
        return rank, np.diff(offsets)


class FormulaIndex:
    """Optional formula -> connectivities lookup. METADATA ONLY: never a hard retrieval filter until a
    validated formula predictor exists (`HARD_FILTER` stays False)."""
    HARD_FILTER = False

    def __init__(self, keys, formulas):
        df = pd.DataFrame({"k": np.asarray(keys).astype(str), "f": pd.Series(formulas).astype("string")}).dropna(subset=["f"])
        self._by = {f: np.sort(g["k"].unique()) for f, g in df.groupby("f", sort=True)}

    @classmethod
    def from_unified(cls, unified):
        return cls(unified["connectivity_key"].to_numpy(), unified["formula"].to_numpy())

    def lookup(self, formula):
        return self._by.get(str(formula), np.zeros(0, dtype=str))

    def __len__(self):
        return len(self._by)
