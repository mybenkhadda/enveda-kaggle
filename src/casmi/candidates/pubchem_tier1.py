"""PubChem Tier-1 ARCHITECTURE (local metadata files only; nothing here downloads, and PubChem is never fully standardized).

    PubChem metadata (user-supplied local table, e.g. CID / exact mass / formula / charge / components / literature / patents / NP flag)
        -> cheap metadata prefilter (`tier1_prefilter`; thresholds from config, unknown thresholds DISABLED)
        -> Tier-1 table (`build_tier1_table`: cid, exact_mass, molecular_formula, evidence columns; sorted by mass)
        -> mass / formula index (`Tier1MassIndex`)
        -> query mass-window retrieval (`Tier1MassIndex.window_cids`)
        -> competition RDKit canonicalization ONLY for the retained, manageable subset (`canonicalize_subset`, which
           reuses the Stage-A engine = the training canonicalizer) -> connectivity dedup.

Tier 1 = plausible mass + organic + (optionally) literature / patent / natural-product evidence. Tier 2 = the broader
remaining PubChem space, only if Tier 1 misses too much coverage (decided from notebook 13's PubChem membership numbers).

Every column is OPTIONAL: a rule whose column is absent is SKIPPED and reported (`rules_skipped`), never guessed.
No literature / patent threshold is invented: they default to None (= rule disabled) and must be set explicitly.
`pubchem_projected_raw_count` from this module counts RAW PubChem rows (CIDs) in a mass window -- it is NOT a finalized
connectivity count (several CIDs can share a connectivity) and is labelled as such.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TIER1_VERSION = "casmi-pubchem-tier1-arch-1"
TIER1_COLUMNS = ("cid", "exact_mass", "molecular_formula", "formal_charge", "component_count", "literature_count", "patent_count",
                 "np_flag")


@dataclass
class PubChemTier1Config:
    enabled: bool = False
    path: str = ""                          # local metadata table (csv / tsv / parquet; gzip allowed for text)
    format: str = "tsv"
    delimiter: str | None = None
    chunksize: int = 1_000_000
    columns: dict = field(default_factory=lambda: {c: None for c in TIER1_COLUMNS})   # canonical name -> file column (None = absent)
    min_exact_mass: float | None = None     # None -> the universe filter range is used by the caller
    max_exact_mass: float | None = None
    organic_only: bool = True
    neutral_only: bool | None = None        # None -> rule disabled (follow the cleanup policy decision instead)
    max_component_count: int | None = None  # None -> disabled
    min_literature_count: int | None = None # None -> disabled (no threshold is invented)
    min_patent_count: int | None = None     # None -> disabled
    require_np_flag: bool = False
    evidence_mode: str = "any"              # any: literature OR patent OR NP (among ENABLED evidence rules); all: every enabled rule
    version: str = TIER1_VERSION

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        cols = {c: None for c in TIER1_COLUMNS}
        cols.update(d.pop("columns", None) or {})
        return cls(columns=cols, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "columns"})

    def identity(self):
        d = asdict(self)
        p = Path(self.path) if self.path else None
        d["source_file_name"] = p.name if p else None
        d["source_size_bytes"] = p.stat().st_size if p and p.is_file() else None
        d.pop("path", None)
        return d


def tier1_prefilter(chunk, cfg, universe_filters=None):
    """Boolean keep-mask + per-row reasons for a metadata chunk already renamed to canonical column names.
    Returns `(keep, reasons, rules_applied, rules_skipped)`."""
    from casmi.candidates.filters import is_organic_formula
    n = len(chunk)
    reasons = [[] for _ in range(n)]
    applied, skipped = [], []

    def rule(name, col, bad_mask):
        if col not in chunk.columns:
            skipped.append(f"{name}: column '{col}' absent")
            return
        applied.append(name)
        for i in np.flatnonzero(np.asarray(bad_mask, bool)):
            reasons[i].append(name)

    lo = cfg.min_exact_mass if cfg.min_exact_mass is not None else getattr(universe_filters, "min_exact_mass", None)
    hi = cfg.max_exact_mass if cfg.max_exact_mass is not None else getattr(universe_filters, "max_exact_mass", None)
    if "exact_mass" in chunk.columns and (lo is not None or hi is not None):
        m = pd.to_numeric(chunk["exact_mass"], errors="coerce")
        bad = m.isna() | ((m < lo) if lo is not None else False) | ((m > hi) if hi is not None else False)
        rule("mass_out_of_range", "exact_mass", bad)
    elif lo is not None or hi is not None:
        skipped.append("mass_out_of_range: column 'exact_mass' absent")
    if cfg.organic_only:
        organic_elements = getattr(universe_filters, "organic_elements", None)
        f = chunk["molecular_formula"] if "molecular_formula" in chunk.columns else None
        bad = None if f is None else ~f.map(lambda x: is_organic_formula(x, organic_elements) if organic_elements else is_organic_formula(x)).to_numpy(bool)
        rule("not_organic", "molecular_formula", bad if bad is not None else np.zeros(n, bool))
    if cfg.neutral_only:
        q = pd.to_numeric(chunk.get("formal_charge"), errors="coerce") if "formal_charge" in chunk.columns else None
        rule("charged", "formal_charge", (q.fillna(1) != 0).to_numpy() if q is not None else np.zeros(n, bool))
    if cfg.max_component_count is not None:
        c = pd.to_numeric(chunk.get("component_count"), errors="coerce") if "component_count" in chunk.columns else None
        rule("too_many_components", "component_count", (c.fillna(np.inf) > cfg.max_component_count).to_numpy() if c is not None else np.zeros(n, bool))
    evid = []
    if cfg.min_literature_count is not None and "literature_count" in chunk.columns:
        evid.append(pd.to_numeric(chunk["literature_count"], errors="coerce").fillna(0).to_numpy() >= cfg.min_literature_count)
        applied.append("literature_evidence")
    elif cfg.min_literature_count is not None:
        skipped.append("literature_evidence: column 'literature_count' absent")
    if cfg.min_patent_count is not None and "patent_count" in chunk.columns:
        evid.append(pd.to_numeric(chunk["patent_count"], errors="coerce").fillna(0).to_numpy() >= cfg.min_patent_count)
        applied.append("patent_evidence")
    elif cfg.min_patent_count is not None:
        skipped.append("patent_evidence: column 'patent_count' absent")
    if cfg.require_np_flag and "np_flag" in chunk.columns:
        evid.append(pd.Series(chunk["np_flag"]).astype(str).str.lower().isin(["1", "true", "yes", "y", "t"]).to_numpy())
        applied.append("np_evidence")
    elif cfg.require_np_flag:
        skipped.append("np_evidence: column 'np_flag' absent")
    if evid:
        ok = np.logical_or.reduce(evid) if cfg.evidence_mode == "any" else np.logical_and.reduce(evid)
        for i in np.flatnonzero(~ok):
            reasons[i].append("insufficient_evidence")
    keep = np.array([not r for r in reasons], dtype=bool)
    return keep, [";".join(r) for r in reasons], applied, skipped


def _iter_metadata(cfg):
    p = Path(cfg.path)
    if not p.is_file():
        raise FileNotFoundError(f"PubChem Tier-1 metadata file not found: {p} (local files only; nothing is downloaded)")
    rename = {v: k for k, v in (cfg.columns or {}).items() if v}
    if cfg.format == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(p)
        cols = [c for c in rename if c in pf.schema.names]
        for b in pf.iter_batches(batch_size=cfg.chunksize, columns=cols):
            yield b.to_pandas().rename(columns=rename)
        return
    sep = cfg.delimiter or ("\t" if cfg.format == "tsv" else ",")
    for df in pd.read_csv(p, sep=sep, dtype=str, chunksize=cfg.chunksize, usecols=lambda c: c in rename, compression="infer"):
        yield df.rename(columns=rename)


def build_tier1_table(cfg, out_path, universe_filters=None, log=print):
    """Chunked metadata pass -> Tier-1 parquet (kept rows only, sorted by exact_mass) + a JSON report of rules applied /
    skipped and counts. Architecture for a LATER user decision (BUILD_PUBCHEM_TIER1_NEXT); not run by notebook 13."""
    if not cfg.enabled:
        raise RuntimeError("PubChem Tier-1 is disabled in the config (coverage.pubchem_tier1.enabled)")
    parts, n_in, n_keep, applied, skipped = [], 0, 0, set(), set()
    for i, chunk in enumerate(_iter_metadata(cfg)):
        keep, _, a, s = tier1_prefilter(chunk, cfg, universe_filters)
        applied.update(a)
        skipped.update(s)
        parts.append(chunk[keep])
        n_in += len(chunk)
        n_keep += int(keep.sum())
        if log:
            log(f"[tier1] chunk {i}: {len(chunk):,} rows -> kept {int(keep.sum()):,} (total kept {n_keep:,} / {n_in:,})")
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(TIER1_COLUMNS))
    if "exact_mass" in t.columns:
        t["exact_mass"] = pd.to_numeric(t["exact_mass"], errors="coerce")
        t = t.sort_values("exact_mass", kind="mergesort").reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out_path, index=False)
    report = {"tier1_version": TIER1_VERSION, "identity": cfg.identity(), "n_input_rows": int(n_in), "n_tier1_rows": int(n_keep),
              "rules_applied": sorted(applied), "rules_skipped": sorted(skipped)}
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


class Tier1MassIndex:
    """Sorted exact masses (+ CIDs) of the Tier-1 table: projected RAW pool counts and window CIDs (no RDKit)."""

    def __init__(self, masses, cids=None):
        m = np.asarray(masses, dtype=np.float64)
        ok = np.isfinite(m)
        order = np.argsort(m[ok], kind="stable")
        self.masses = m[ok][order]
        self.cids = None if cids is None else np.asarray(cids, dtype=object)[ok][order]

    @classmethod
    def from_table(cls, path_or_df):
        t = pd.read_parquet(path_or_df, columns=["cid", "exact_mass"]) if not isinstance(path_or_df, pd.DataFrame) else path_or_df
        return cls(pd.to_numeric(t["exact_mass"], errors="coerce").to_numpy(), t["cid"].to_numpy() if "cid" in t else None)

    def __len__(self):
        return len(self.masses)

    def projected_counts(self, neutral_masses, ppm):
        """pubchem_projected_raw_count per query (RAW CID rows within |m| * ppm * 1e-6, inclusive)."""
        return projected_pool_counts(self.masses, neutral_masses, ppm)

    def window_cids(self, neutral_mass, ppm):
        tol = abs(neutral_mass) * ppm * 1e-6
        lo = np.searchsorted(self.masses, neutral_mass - tol, side="left")
        hi = np.searchsorted(self.masses, neutral_mass + tol, side="right")
        return self.cids[lo:hi] if self.cids is not None else np.arange(lo, hi)


def projected_pool_counts(sorted_masses, neutral_masses, ppm):
    """Vectorized raw counts of `sorted_masses` inside each query window (NaN / <= 0 masses -> 0)."""
    s = np.asarray(sorted_masses, dtype=np.float64)
    m = np.asarray(neutral_masses, dtype=np.float64)
    good = np.isfinite(m) & (m > 0)
    tol = np.abs(m) * float(ppm) * 1e-6
    lo = np.searchsorted(s, np.where(good, m - tol, np.inf), side="left")
    hi = np.searchsorted(s, np.where(good, m + tol, -np.inf), side="right")
    return np.maximum(hi - lo, 0)


def canonicalize_subset(cid_smiles, profile="universe_minimal", n_jobs=1):
    """LATE canonicalization of a manageable retained subset: `cid_smiles` (cid, smiles) -> standardized rows with the
    competition connectivity key, via the Stage-A engine (the training canonicalizer). Then dedup by connectivity_key.
    Call only on window-retained subsets -- never on all of PubChem."""
    from casmi.candidates.stage_a import StructureEngine, standardize_records_fast
    rec = pd.DataFrame({"source": "PUBCHEM", "source_id": cid_smiles["cid"].astype(str).to_numpy(), "raw_smiles": cid_smiles["smiles"].to_numpy()})
    with StructureEngine(profile, n_jobs) as eng:
        std, rej, stats = standardize_records_fast(rec, eng)
    dedup = std.sort_values(["connectivity_key", "source_id"], kind="mergesort").drop_duplicates("connectivity_key")
    return dedup.reset_index(drop=True), rej, stats
