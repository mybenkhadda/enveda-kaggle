"""Stage A of the candidate universe (external source -> standardized, filtered, bucketed chunks), optimized
WITHOUT changing the scientific identity contract.

Identity contract (unchanged; enforced by tests/test_stage_a_optimization.py):

    raw external SMILES -> casmi.chemistry.connectivity.canonicalize_smiles            (normalized / representative SMILES)
                        -> competition_connectivity_key_detailed (TautomerEnumerator.Canonicalize -> InChIKey[:14])
                        -> CalcExactMolWt / MolWt / GetFormalCharge on MolFromSmiles(raw)  (casmi.chemistry.descriptors)
                        -> molecular_formula(normalized SMILES), n_fragments(normalized SMILES)
                        -> formula prefilter / apply_filters / provenance exactly as before

What changed (performance only):

  * `universe_minimal` profile: the same calls as above, without the structural EDA descriptors (Murcko scaffolds,
    AddHs atom counts, TPSA, logP, HBD/HBA, ring / rotatable-bond / element counts) that no universe stage reads.
    The `full` profile reproduces the legacy columns exactly (legacy worker `_process_one`).
  * molecular formula computed inside the parallel workers (same function) instead of serially in the parent;
  * ONE persistent worker pool per source, batched tasks, native thread limits (no oversubscription);
  * exact raw-SMILES dedup accounting (each identical raw SMILES is canonicalized once per chunk -- as before -- and
    optionally once per session via a SQLite cache keyed by (contract, raw SMILES)); provenance is never deduplicated;
  * the source is read from a local-SSD copy; chunk outputs are written locally, persisted to Drive with a size
    check, and only THEN marked done;
  * build identity: chunk markers live under `stage_a/<build_id>/` where build_id fingerprints source file name + size,
    column mapping, chunk size, canonicalizer contract (incl. RDKit version and profile), filters and prefilter. A
    changed source / chunk size / filter / canonicalizer never reuses an old chunk.
  * progress + ETA + resource telemetry.

GPU: not used. RDKit parsing, tautomer canonicalization, canonical SMILES and InChI(Key) are CPU-only; replacing
them with a GPU chemistry stack could change the competition connectivity identity (see GPU_STAGE_A_POLICY).
"""
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from casmi.candidates.standardize import (PROFILE_COLUMNS, STANDARDIZATION_CONTRACT_VERSION, assemble_standardized, molecular_formula,
                                          prepare_records, training_tautomer_caps)

STAGE_A_SCHEMA_VERSION = "casmi-stage-a-2"
PROFILES = tuple(PROFILE_COLUMNS)
GPU_STAGE_A_POLICY = ("GPU detected but not used for Stage A: RDKit parsing, tautomer canonicalization, canonical SMILES and InChIKey are "
                      "CPU-bound, and a GPU substitute would risk changing the competition chemistry identity.")
DEVICE_TABLE = [
    ("CSV parsing", "CPU / I/O", "no meaningful benefit"),
    ("formula prefilter", "CPU", "no"),
    ("RDKit parsing", "CPU", "no"),
    ("tautomer canonicalization", "CPU", "no identity-safe GPU equivalent"),
    ("canonical SMILES", "CPU", "no"),
    ("InChI / InChIKey", "CPU", "no"),
    ("identity descriptors (mass, MW, charge, formula)", "CPU", "no"),
    ("bucket sort", "CPU", "usually no (small per chunk)"),
    ("parquet write", "CPU / I/O", "no"),
    ("later: spectrum neural models / contrastive training / embedding inference", "GPU", "yes (later notebooks)"),
]

UNIVERSE_STRUCTURE_FIELDS = ["smiles", "parse_ok", "error", "canonical_smiles", "plain_inchikey14", "connectivity_key", "tautomer_hit_cap",
                             "exact_mass", "molecular_weight", "formal_charge", "molecular_formula"]


class StageAIncompleteError(RuntimeError):
    """A configured, present source has no COMPLETE Stage-A build matching the current identity."""


def _write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_json(path):
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None


def fingerprint(obj):
    """Deterministic 16-hex fingerprint of JSON-serialisable metadata (cache identity -- not a file hash)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class StageAPerformance:
    """Performance-only knobs (`universe.performance` in configs/casmi_v2_colab.yaml). Only `standardization_profile`
    and `compute_tautomer_hit_cap` change outputs (columns / the diagnostic hit-cap flag) and are therefore part of
    the build identity; everything else (batch size, workers, cache, staging, progress) never changes a value."""
    standardization_profile: str = "universe_minimal"
    compute_tautomer_hit_cap: bool = True          # False skips only the diagnostic Enumerate() call (key unchanged)
    standardize_batch_records: int = 2000          # unique SMILES per worker task
    stage_input_to_scratch: bool = True
    local_temp_outputs: bool = True
    start_method: str | None = None                # None -> casmi.workspace.parallel.default_start_method()
    ram_per_worker_gb: float = 1.0
    cache_enabled: bool = True
    cache_dir: str = "cache/standardization"      # relative to the scratch root
    progress_every_records: int = 5000
    progress_every_seconds: float = 30.0
    telemetry: bool = True

    def __post_init__(self):
        if self.standardization_profile not in PROFILES:
            raise ValueError(f"standardization_profile must be one of {PROFILES}")
        if self.standardize_batch_records < 1:
            raise ValueError("standardize_batch_records must be >= 1")

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        cache = d.pop("canonicalization_cache", None) or {}
        prog = d.pop("progress", None) or {}
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "enabled" in cache:
            kw["cache_enabled"] = bool(cache["enabled"])
        if "dir" in cache:
            kw["cache_dir"] = cache["dir"]
        if "every_records" in prog:
            kw["progress_every_records"] = int(prog["every_records"])
        if "every_seconds" in prog:
            kw["progress_every_seconds"] = float(prog["every_seconds"])
        if prog.get("enabled") is False:
            kw["progress_every_records"] = 0
        return cls(**kw)


def canonicalizer_contract(profile, compute_hit_cap=True):
    """Everything that defines the canonicalizer's OUTPUT (part of the build identity)."""
    tmax, ttran = training_tautomer_caps()
    try:
        import rdkit
        rdkit_version = rdkit.__version__
    except ImportError:
        rdkit_version = None
    return {"standardization_contract": STANDARDIZATION_CONTRACT_VERSION, "profile": profile,
            "tautomer_hit_cap": True if profile == "full" else bool(compute_hit_cap), "tautomer_caps": [tmax, ttran],
            "rdkit_version": rdkit_version}


def stage_a_identity(source_cfg, source_file, ucfg, perf):
    """Stage-A build identity for one source. `source_file` is the PERSISTENT (Drive) file: its name and size count,
    not the local staged path. No content hashing."""
    p = Path(source_file)
    return {"stage_a_schema": STAGE_A_SCHEMA_VERSION, "source": source_cfg.canonical_source(), "source_file_name": p.name,
            "source_size_bytes": p.stat().st_size,
            "source_columns": {"format": source_cfg.format, "id_col": source_cfg.id_col, "smiles_col": source_cfg.smiles_col,
                               "name_col": source_cfg.name_col, "formula_col": source_cfg.formula_col,
                               "metadata_cols": list(source_cfg.metadata_cols or []), "delimiter": source_cfg.delimiter,
                               "smi_has_header": source_cfg.smi_has_header, "max_records": source_cfg.max_records},
            "chunk_records": int(ucfg.chunk_records), "bucket_prefix_len": int(ucfg.bucket_prefix_len),
            "canonicalizer": canonicalizer_contract(perf.standardization_profile, perf.compute_tautomer_hit_cap),
            "filters": json.loads(json.dumps(asdict(ucfg.filters), default=list)),
            "prefilter": {"by_source_formula": bool(ucfg.prefilter_by_source_formula), "mass_margin_da": float(ucfg.prefilter_mass_margin_da)}}


def build_id_of(identity):
    return fingerprint(identity)


# ---------------------------------------------------------------------------------------------------------------
# workers (module level -> picklable for spawn)
# ---------------------------------------------------------------------------------------------------------------

def structure_row_universe(smiles, max_tautomers, max_transforms, compute_hit_cap=True):
    """`universe_minimal` worker: the same identity calls as the legacy worker, minus EDA descriptors."""
    from casmi.chemistry.connectivity import canonicalize_smiles, competition_connectivity_key_detailed
    from casmi.chemistry.descriptors import compute_identity_descriptors
    row = {k: None for k in UNIVERSE_STRUCTURE_FIELDS}
    row["smiles"] = smiles
    row["parse_ok"] = False
    canon = None
    try:
        canon = canonicalize_smiles(smiles)
        if canon is None:
            row["error"] = "parse_failed"
            return row
        row["parse_ok"] = True
        row["canonical_smiles"] = canon
        conn = competition_connectivity_key_detailed(smiles, max_tautomers=max_tautomers, max_transforms=max_transforms,
                                                     compute_hit_cap=compute_hit_cap)
        row["plain_inchikey14"] = conn["plain_inchikey14"]
        row["connectivity_key"] = conn["conn_key"]
        row["tautomer_hit_cap"] = conn["hit_cap"]
        if conn["error"]:
            row["error"] = conn["error"]
        row.update(compute_identity_descriptors(smiles))
    except Exception as exc:
        row["error"] = f"unexpected_error: {exc}"
    if canon is not None:
        row["molecular_formula"] = molecular_formula(canon)          # same function the legacy parent applied to normalized SMILES
    return row


def structure_row_full(smiles, max_tautomers, max_transforms, compute_hit_cap=True):
    """`full` worker: the legacy `_process_one` unchanged (+ the formula, computed by the same function)."""
    from casmi.chemistry.structures import _process_one
    row = _process_one(smiles, max_tautomers, max_transforms)
    canon = row.get("canonical_smiles")
    row["molecular_formula"] = molecular_formula(canon) if canon else None
    return row


def profile_fields(profile):
    if profile == "full":
        from casmi.chemistry.structures import STRUCTURE_TABLE_FIELDS
        return list(STRUCTURE_TABLE_FIELDS) + ["molecular_formula"]
    return list(UNIVERSE_STRUCTURE_FIELDS)


def process_batch(payload):
    """One worker task: (profile, [smiles...], max_tautomers, max_transforms, compute_hit_cap) -> rows in input order."""
    profile, smiles, tmax, ttran, hit_cap = payload
    fn = structure_row_full if profile == "full" else structure_row_universe
    return [fn(s, tmax, ttran, hit_cap) for s in smiles]


# ---------------------------------------------------------------------------------------------------------------
# optional cross-chunk cache (correctness never depends on it)
# ---------------------------------------------------------------------------------------------------------------

class StandardizationCache:
    """SQLite store {(contract key, exact raw SMILES) -> structure row JSON}. Batched lookups / inserts (no per-molecule
    file I/O). Any SQLite error disables the cache for the run (logged) -- results are then simply recomputed."""

    def __init__(self, path, contract_key, log=print):
        self.path, self.contract, self.log, self.enabled = Path(path), str(contract_key), log, True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(str(self.path))
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.execute("PRAGMA synchronous=NORMAL")
            self.con.execute("CREATE TABLE IF NOT EXISTS std_cache (contract TEXT NOT NULL, smiles TEXT NOT NULL, row TEXT NOT NULL, "
                             "PRIMARY KEY (contract, smiles)) WITHOUT ROWID")
            self.con.commit()
        except sqlite3.Error as exc:
            self._disable(exc)

    def _disable(self, exc):
        self.enabled = False
        if self.log:
            self.log(f"[std-cache] disabled ({exc}); results are recomputed")

    def get_many(self, smiles, block=500):
        out = {}
        if not self.enabled or not smiles:
            return out
        try:
            for s in range(0, len(smiles), block):
                part = list(smiles[s:s + block])
                # only '?' placeholders are interpolated; every value is a bound parameter (no SQL injection surface)
                q = f"SELECT smiles, row FROM std_cache WHERE contract = ? AND smiles IN ({','.join('?' * len(part))})"  # nosec B608
                for k, v in self.con.execute(q, [self.contract, *part]):
                    out[k] = json.loads(v)
        except (sqlite3.Error, ValueError) as exc:
            self._disable(exc)
            return {}
        return out

    def put_many(self, rows):
        if not self.enabled or not rows:
            return
        try:
            self.con.executemany("INSERT OR REPLACE INTO std_cache VALUES (?, ?, ?)",
                                 [(self.contract, r["smiles"], json.dumps(r)) for r in rows])
            self.con.commit()
        except (sqlite3.Error, TypeError, ValueError) as exc:
            self._disable(exc)

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


def cache_contract_key(profile, compute_hit_cap):
    return fingerprint(canonicalizer_contract(profile, compute_hit_cap))


# ---------------------------------------------------------------------------------------------------------------
# structure engine (persistent pool, batches, cache)
# ---------------------------------------------------------------------------------------------------------------

class StructureEngine:
    """Canonicalizes UNIQUE raw SMILES. One pool per engine (= per source run), reused for every chunk.
    `row_fn` (tests only) replaces the RDKit worker and forces inline execution."""

    def __init__(self, profile="universe_minimal", n_jobs=1, batch_records=2000, compute_hit_cap=True, start_method=None, cache=None,
                 row_fn=None):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        self.profile, self.n_jobs, self.batch_records = profile, int(n_jobs or 1), int(batch_records)
        self.compute_hit_cap, self.start_method, self.cache, self.row_fn = compute_hit_cap, start_method, cache, row_fn
        self.tmax, self.ttran = training_tautomer_caps()
        self.fields = profile_fields(profile)
        self.pool = None
        self.last = {"n_unique": 0, "n_cache_hits": 0, "n_standardizer_calls": 0}
        self.total = {"n_unique": 0, "n_cache_hits": 0, "n_standardizer_calls": 0}

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close(terminate=exc_type is not None)

    def open(self):
        if self.pool is None and self.n_jobs > 1 and self.row_fn is None:
            from casmi.workspace.parallel import make_pool
            self.pool = make_pool(self.n_jobs, self.start_method)
        return self

    def close(self, terminate=False):
        if self.pool is not None:
            (self.pool.terminate if terminate else self.pool.close)()
            self.pool.join()
            self.pool = None
        if self.cache is not None:
            self.cache.close()

    def structure_table(self, unique_smiles, progress=None):
        """One row per input SMILES (same order), columns = profile fields."""
        smiles = list(unique_smiles)
        n = len(smiles)
        rows = [None] * n
        hits = self.cache.get_many(smiles) if self.cache is not None else {}
        for i, s in enumerate(smiles):
            if s in hits:
                rows[i] = hits[s]
        todo = [i for i in range(n) if rows[i] is None]
        if progress is not None and len(hits):
            progress.advance(len(hits), cached=True)
        batches = [todo[s:s + self.batch_records] for s in range(0, len(todo), self.batch_records)]
        payloads = [(self.profile, [smiles[i] for i in b], self.tmax, self.ttran, self.compute_hit_cap) for b in batches]
        if self.row_fn is not None:
            results = ([self.row_fn(s) for s in p[1]] for p in payloads)
        elif self.pool is None:
            results = (process_batch(p) for p in payloads)
        else:
            results = self.pool.imap(process_batch, payloads, chunksize=1)        # ordered -> deterministic association
        for b, res in zip(batches, results, strict=True):
            for i, r in zip(b, res, strict=True):           # a batch must return exactly one row per SMILES
                rows[i] = r
            if self.cache is not None:
                self.cache.put_many(res)
            if progress is not None:
                progress.advance(len(b))
        self.last = {"n_unique": n, "n_cache_hits": len(hits), "n_standardizer_calls": len(todo)}
        for k, v in self.last.items():
            self.total[k] += v
        return pd.DataFrame.from_records(rows, columns=self.fields) if n else pd.DataFrame(columns=self.fields)


IDENTITY_COLUMNS = ["source", "source_id", "raw_smiles", "normalized_smiles", "representative_smiles", "connectivity_key", "plain_inchikey14",
                    "molecular_formula", "neutral_monoisotopic_mass", "molecular_weight", "formal_charge", "n_fragments", "is_charged", "name",
                    "source_formula", "source_metadata"]


def _missing(v):
    try:
        return v is None or (isinstance(v, float) and v != v) or v is pd.NA
    except TypeError:
        return False


def identity_mismatches(left, right, columns=IDENTITY_COLUMNS, max_rows=20):
    """Exact positional comparison of two standardized (or rejected) tables on `columns`. Both paths emit rows in the
    order of the input records, so rows are compared position by position. None / NaN / NA count as equal to each other;
    everything else must be `==` (floats included: the same RDKit call must give the same double). Returns a DataFrame
    (empty = identical)."""
    rows = []
    if len(left) != len(right):
        return pd.DataFrame([{"row": None, "column": "<length>", "left": len(left), "right": len(right)}])
    lv, rv = left.reset_index(drop=True), right.reset_index(drop=True)
    for c in columns:
        if c not in lv.columns or c not in rv.columns:
            if (c in lv.columns) != (c in rv.columns):
                rows.append({"row": None, "column": c, "left": c in lv.columns, "right": c in rv.columns})
            continue
        for i, (a, b) in enumerate(zip(lv[c].tolist(), rv[c].tolist(), strict=True)):
            if _missing(a) and _missing(b):
                continue
            if _missing(a) or _missing(b) or a != b:
                rows.append({"row": i, "column": c, "left": a, "right": b})
                if len(rows) >= max_rows:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=["row", "column", "left", "right"])


def standardize_records_fast(records, engine, progress=None):
    """Same contract as `standardize_records` (shared prepare / assemble), structure rows from `engine`. Returns
    `(standardized, rejected, stats)`; every source record lands in exactly one of the two tables, with its own
    source / source_id / name / source_formula / source_metadata (provenance is never deduplicated)."""
    rec, todo, rejected = prepare_records(records)
    unique = pd.unique(pd.Series(todo["raw_smiles"], dtype="object").dropna())
    if progress is not None:
        progress.start_structures(len(unique))
    table = engine.structure_table(unique, progress)
    std, rej = assemble_standardized(rec, todo, table, rejected, PROFILE_COLUMNS[engine.profile])
    n_todo = int(len(todo))
    stats = {"n_input_records": int(len(records)), "n_blank_smiles": int(len(rec) - n_todo), "n_unique_raw_smiles": int(len(unique)),
             "n_duplicate_raw_smiles_saved": int(n_todo - len(unique)), "duplicate_fraction": float((n_todo - len(unique)) / n_todo) if n_todo else 0.0,
             "n_standardizer_calls": int(engine.last["n_standardizer_calls"]), "n_cache_hits": int(engine.last["n_cache_hits"])}
    return std, rej, stats


# ---------------------------------------------------------------------------------------------------------------
# progress / ETA / telemetry
# ---------------------------------------------------------------------------------------------------------------

def _fmt_s(s):
    if s is None or s != s or s == float("inf"):
        return "?"
    s = int(max(s, 0))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class StageAProgress:
    """Rolling progress without a full pre-scan: per-chunk structure progress every `every_records` structures or
    `every_seconds`, plus source-level ETA from processed-row throughput and a ROUGH row estimate."""

    def __init__(self, source, est_total_rows=None, every_records=5000, every_seconds=30.0, log=print, telemetry=False, telemetry_paths=()):
        self.source, self.est_total_rows, self.every_records, self.every_seconds = source, est_total_rows, every_records, every_seconds
        self.log, self.telemetry, self.telemetry_paths = log, telemetry, telemetry_paths
        self.t_source = time.time()
        self.rows_read = self.rows_processed = 0
        self.seconds_processing = 0.0
        self.chunk = None

    def _emit(self, msg):
        if self.log and self.every_records:
            self.log(msg)

    def start_chunk(self, index, n_rows):
        self.chunk = {"index": index, "n_rows": n_rows, "t0": time.time(), "n_unique": 0, "done": 0, "cached": 0, "last_emit_n": 0,
                      "last_emit_t": time.time()}

    def start_structures(self, n_unique):
        if self.chunk is not None:
            self.chunk["n_unique"] = n_unique

    def advance(self, n, cached=False):
        c = self.chunk
        if c is None:
            return
        c["done"] += n
        if cached:
            c["cached"] += n
        now = time.time()
        if (self.every_records and c["done"] - c["last_emit_n"] >= self.every_records) or now - c["last_emit_t"] >= self.every_seconds:
            c["last_emit_n"], c["last_emit_t"] = c["done"], now
            el = now - c["t0"]
            rate = (c["done"] - c["cached"]) / el if el > 0 else 0.0
            eta = (c["n_unique"] - c["done"]) / rate if rate > 0 else None
            self._emit(f"[{self.source}] chunk {c['index']}: structures {c['done']:,}/{c['n_unique']:,} (cache hits {c['cached']:,}) | "
                       f"{rate:,.1f} structures/s | chunk elapsed {_fmt_s(el)} | chunk ETA {_fmt_s(eta)}")

    def skip_chunk(self, index, n_rows):
        self.rows_read += n_rows
        self._emit(f"[{self.source}] chunk {index}: valid done-marker -> reused ({n_rows:,} rows)")

    def end_chunk(self, rec):
        c = self.chunk or {"t0": time.time()}
        el = time.time() - c["t0"]
        self.rows_read += rec["n_input"]
        self.rows_processed += rec["n_input"]
        self.seconds_processing += el
        rows_rate = self.rows_processed / self.seconds_processing if self.seconds_processing > 0 else 0.0
        remaining = (self.est_total_rows - self.rows_read) if self.est_total_rows else None
        eta = remaining / rows_rate if remaining is not None and rows_rate > 0 else None
        msg = (f"[{self.source}] chunk {rec['chunk']} done in {_fmt_s(el)}: rows read {self.rows_read:,}"
               + (f" / ~{self.est_total_rows:,} (estimate)" if self.est_total_rows else "")
               + f" | prefiltered {rec['n_prefiltered']:,} | unique raw SMILES {rec['n_unique_raw_smiles']:,}"
               f" | standardizer calls {rec['n_standardizer_calls']:,} | duplicates avoided {rec['n_duplicate_raw_smiles_saved']:,}"
               f" | kept {rec['n_kept']:,} | {rows_rate:,.1f} rows/s | source ETA {_fmt_s(eta)}")
        if self.telemetry:
            from casmi.workspace.parallel import runtime_resources
            msg += f" | {runtime_resources(self.telemetry_paths)}"
        if self.log:
            self.log(msg)


# ---------------------------------------------------------------------------------------------------------------
# sources (staging) + plan
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class SourceSpec:
    source: str
    template: str
    drive_path: str
    read_path: str
    exists: bool
    cfg: object = None
    staged: dict | None = None
    size_bytes: int | None = None


def configured_sources(cfg_yaml, drive_root, repo_root, scratch_root=None, perf=None, stage=True, log=print):
    """Every `universe.sources` entry: template loaded, Drive file resolved (path as configured, relative to drive_root),
    local-SSD copy staged when present and enabled. A missing file is reported, never an error."""
    from casmi.candidates.sources import load_source_config
    from casmi.workspace.config import _resolve
    from casmi.workspace.staging import stage_external_file
    perf = perf or StageAPerformance()
    specs = []
    for s in (cfg_yaml.get("universe") or {}).get("sources") or []:
        scfg = load_source_config(Path(repo_root) / s["template"])
        drive_path = _resolve(Path(drive_root), s["file"])
        exists = drive_path.is_file()
        read_path, staged = drive_path, None
        if exists and stage and perf.stage_input_to_scratch and scratch_root:
            st = stage_external_file(drive_path, scratch_root, scfg.canonical_source(), log=log)
            read_path, staged = Path(st.local_path), st.as_dict()
        elif not exists and log:
            log(f"[sources] NOT PROVIDED {scfg.canonical_source()}: {drive_path}")
        scfg.path = str(read_path)
        specs.append(SourceSpec(scfg.canonical_source(), s["template"], str(drive_path), str(read_path), exists, scfg, staged,
                                drive_path.stat().st_size if exists else None))
    return specs


def stage_a_plan(specs, ucfg, perf, n_jobs=None):
    """Pre-run summary (one row per source) printed by notebook 12 before Stage A starts."""
    from casmi.candidates.sources import estimate_record_count
    from casmi.workspace.environment import gpu_report
    from casmi.workspace.parallel import resolve_n_jobs
    gpu = gpu_report()
    workers = resolve_n_jobs(n_jobs if n_jobs is not None else ucfg.n_jobs, perf.ram_per_worker_gb)
    rows = []
    for s in specs:
        ident = stage_a_identity(s.cfg, s.drive_path, ucfg, perf) if s.exists else None
        rows.append({"source": s.source, "exists": s.exists, "drive_path": s.drive_path,
                     "size_mb": round(s.size_bytes / 1024 ** 2, 1) if s.size_bytes else None, "read_path": s.read_path,
                     "staging": (s.staged or {}).get("action"), "estimated_rows": estimate_record_count(s.read_path) if s.exists else None,
                     "cpu_count": os.cpu_count(), "workers": workers, "gpu": gpu.get("gpu_name") or "none",
                     "gpu_used_for_stage_a": "NO", "gpu_reason": GPU_STAGE_A_POLICY, "read_chunk_records": ucfg.chunk_records,
                     "worker_batch_records": perf.standardize_batch_records, "checkpoint_records": ucfg.chunk_records,
                     "profile": perf.standardization_profile, "tautomer_hit_cap": perf.compute_tautomer_hit_cap,
                     "build_id": build_id_of(ident) if ident else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Stage A run (resumable, local-first, persist-then-mark)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class StageAResult:
    source: str
    build_id: str
    identity: dict
    namespace: str
    summary: pd.DataFrame
    manifest: dict = field(default_factory=dict)


def stage_a_namespace(work_root, build_id):
    return Path(work_root) / "stage_a" / build_id


def _ensure_identity(ns, identity, build_id):
    f = ns / "identity.json"
    old = _read_json(f)
    if old is not None and old.get("build_id") != build_id:
        raise RuntimeError(f"{f} belongs to build {old.get('build_id')} -- refusing to reuse it for {build_id}")
    if old is None:
        _write_json_atomic(f, {"build_id": build_id, "identity": identity})


def _valid_marker(marker, ns, build_id, chunk, row_start, n_input):
    from casmi.workspace.staging import files_intact
    rec = _read_json(marker)
    if not rec or rec.get("build_id") != build_id or rec.get("chunk") != chunk or rec.get("row_start") != row_start or rec.get("n_input") != n_input:
        return None
    if not files_intact({str(ns / rel): size for rel, size in (rec.get("files") or {}).items()}):
        return None
    return rec


def _process_chunk(chunk, ucfg, engine, formula_cache, progress):
    from casmi.candidates.filters import apply_filters
    from casmi.candidates.universe import formula_prefilter
    keep, pre = formula_prefilter(chunk, ucfg, cache=formula_cache)
    std, rej, stats = standardize_records_fast(keep, engine, progress)
    kept, filt = apply_filters(std, ucfg.filters)
    fo = pd.concat([pre[["source", "source_id", "raw_smiles", "filter_reason"]],
                    filt[["source", "source_id", "raw_smiles", "connectivity_key", "filter_reason"]]], ignore_index=True)
    return kept, rej, fo, pre, filt, stats


def run_stage_a(source_cfg, work_root, scratch_root, ucfg, perf=None, source_identity_path=None, n_jobs=None, log=print, row_fn=None):
    """Stage A for one source. `source_cfg.path` = the file to READ (local staged copy); `source_identity_path` = the
    persistent Drive file whose name + size define the build identity (defaults to the read path).
    Chunk outputs go to `<work_root>/stage_a/<build_id>/{standardized,rejected,filtered_out}/<SOURCE>/chunk-NNNNN.parquet`
    and markers to `.../_done/<SOURCE>/chunk-NNNNN.json` -- a marker is written only after its files are persisted."""
    from casmi.candidates.sources import estimate_record_count, iter_source_records
    from casmi.candidates.universe import _write_bucketed
    from casmi.workspace.parallel import resolve_n_jobs
    from casmi.workspace.staging import persist_files
    perf = perf or StageAPerformance()
    source_cfg.validate()
    src = source_cfg.canonical_source()
    id_path = Path(source_identity_path or source_cfg.path)
    identity = stage_a_identity(source_cfg, id_path, ucfg, perf)
    build_id = build_id_of(identity)
    ns = stage_a_namespace(work_root, build_id)
    local_ns = Path(scratch_root) / "stage_a" / build_id if (perf.local_temp_outputs and scratch_root) else ns
    _ensure_identity(ns, identity, build_id)
    workers = resolve_n_jobs(n_jobs if n_jobs is not None else ucfg.n_jobs, perf.ram_per_worker_gb)
    cache = None
    if perf.cache_enabled and scratch_root:
        cache = StandardizationCache(Path(scratch_root) / perf.cache_dir / "std_cache.sqlite",
                                     cache_contract_key(perf.standardization_profile, perf.compute_tautomer_hit_cap), log=log)
    progress = StageAProgress(src, estimate_record_count(source_cfg.path), perf.progress_every_records, perf.progress_every_seconds,
                              log=log, telemetry=perf.telemetry, telemetry_paths=[p for p in (scratch_root,) if p])
    if log:
        log(f"[{src}] Stage A build {build_id} | profile {perf.standardization_profile} | workers {workers} | chunk {ucfg.chunk_records:,} rows"
            f" | worker batch {perf.standardize_batch_records:,} | namespace {ns}")
    source_cfg.chunksize = int(ucfg.chunk_records)
    rows, formula_cache, row_start = [], {}, 0
    with StructureEngine(perf.standardization_profile, workers, perf.standardize_batch_records, perf.compute_tautomer_hit_cap,
                         perf.start_method, cache, row_fn=row_fn) as engine:          # row_fn: tests only (inline fake worker)
        for i, chunk in enumerate(iter_source_records(source_cfg)):
            n = int(len(chunk))
            marker = ns / "_done" / src / f"chunk-{i:05d}.json"
            rec = _valid_marker(marker, ns, build_id, i, row_start, n)
            if rec is not None:
                rows.append(rec)
                progress.skip_chunk(i, n)
                row_start += n
                continue
            t0 = time.time()
            progress.start_chunk(i, n)
            kept, rej, fo, pre, filt, stats = _process_chunk(chunk, ucfg, engine, formula_cache, progress)
            rel = {"standardized": f"standardized/{src}/chunk-{i:05d}.parquet", "rejected": f"rejected/{src}/chunk-{i:05d}.parquet",
                   "filtered_out": f"filtered_out/{src}/chunk-{i:05d}.parquet"}
            _write_bucketed(kept, local_ns / rel["standardized"], ucfg.bucket_prefix_len)
            (local_ns / rel["rejected"]).parent.mkdir(parents=True, exist_ok=True)
            rej.to_parquet(local_ns / rel["rejected"], index=False)
            (local_ns / rel["filtered_out"]).parent.mkdir(parents=True, exist_ok=True)
            fo.to_parquet(local_ns / rel["filtered_out"], index=False)
            persist_files([(local_ns / r, ns / r) for r in rel.values()])
            files = {r: (ns / r).stat().st_size for r in rel.values()}
            rec = {"source": src, "chunk": i, "build_id": build_id, "row_start": row_start, "row_end": row_start + n, "n_input": n,
                   "n_prefiltered": int(len(pre)), "n_rejected": int(len(rej)), "n_filtered": int(len(filt)), "n_kept": int(len(kept)),
                   "n_unique_connectivities_in_chunk": int(kept["connectivity_key"].nunique()), **stats, "files": files,
                   "seconds": round(time.time() - t0, 2), "finished_at": datetime.now(timezone.utc).isoformat()}
            if rec["n_prefiltered"] + rec["n_rejected"] + rec["n_filtered"] + rec["n_kept"] != n:
                raise RuntimeError(f"[{src}] chunk {i}: record accounting broken -- not marking it done")
            _write_json_atomic(marker, rec)                  # ONLY after every output reached its persistent location
            if local_ns != ns:
                for r in rel.values():
                    (local_ns / r).unlink(missing_ok=True)
            rows.append(rec)
            progress.end_chunk(rec)
            row_start += n
        totals = dict(engine.total)
    summary = pd.DataFrame(rows)
    manifest = {"build_id": build_id, "identity": identity, "source": src, "complete": True, "n_chunks": len(rows),
                "n_input": int(summary["n_input"].sum()) if len(summary) else 0, "n_kept": int(summary["n_kept"].sum()) if len(summary) else 0,
                "chunks": [{k: r[k] for k in ("chunk", "row_start", "row_end", "n_input", "n_kept", "files")} for r in rows],
                "engine_totals_this_session": totals, "completed_at": datetime.now(timezone.utc).isoformat()}
    _write_json_atomic(ns / "manifest.json", manifest)
    _write_json_atomic(Path(work_root) / "stage_a" / "ACTIVE" / f"{src}.json", {"build_id": build_id, "manifest": str(ns / "manifest.json"),
                                                                                "completed_at": manifest["completed_at"]})
    return StageAResult(src, build_id, identity, str(ns), summary, manifest)


def resolve_stage_a_inputs(work_root, specs, ucfg, perf=None, require_complete=True):
    """Stage-B inputs: for every configured source whose file EXISTS, the COMPLETE Stage-A build whose identity matches
    the CURRENT source file / config (recomputed here, never taken from a stale pointer). Returns
    {'sources': {SRC: {...}}, 'std_files': [...], 'namespaces': [...]}. Raises StageAIncompleteError when a present source
    has no complete matching build (unless require_complete=False, which skips it -- logged in the result)."""
    from casmi.workspace.staging import files_intact
    perf = perf or StageAPerformance()
    out = {"sources": {}, "std_files": [], "namespaces": []}
    for s in specs:
        if not s.exists:
            out["sources"][s.source] = {"status": "not_provided", "drive_path": s.drive_path}
            continue
        identity = stage_a_identity(s.cfg, s.drive_path, ucfg, perf)
        bid = build_id_of(identity)
        ns = stage_a_namespace(work_root, bid)
        man = _read_json(ns / "manifest.json")
        ok = bool(man and man.get("complete") and man.get("build_id") == bid)
        files = {}
        if ok:
            for c in man["chunks"]:
                files.update({str(ns / rel): size for rel, size in c["files"].items()})
            ok = files_intact(files)
        if not ok:
            if require_complete:
                raise StageAIncompleteError(f"{s.source}: no complete Stage-A build {bid} for {s.drive_path} (current config / file). "
                                            f"Run Stage A for this source first (resumable).")
            out["sources"][s.source] = {"status": "incomplete", "build_id": bid}
            continue
        std = sorted(p for p in files if "/standardized/" in p.replace("\\", "/"))
        out["sources"][s.source] = {"status": "complete", "build_id": bid, "n_chunks": man["n_chunks"], "n_kept": man["n_kept"],
                                    "n_input": man["n_input"], "std_files": {p: files[p] for p in std}}
        out["std_files"] += std
        out["namespaces"].append(str(ns))
    return out
