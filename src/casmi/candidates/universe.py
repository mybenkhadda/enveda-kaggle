"""v2 candidate structure universe at PubChem scale: resumable, bucketed, one row per connectivity.

Reuses, never re-implements:
    casmi.candidates.sources.iter_source_records      local file readers (csv/tsv/parquet/smi/sdf), never downloads
    casmi.candidates.standardize.standardize_records  the TRAINING canonicalizer (tautomer-canonical InChIKey14)
    casmi.candidates.filters.apply_filters            mass / organic / neutral / single-component (reasons kept)
    casmi.candidates.merge.unify                      TRAIN + external dedup by connectivity_key, provenance

Stages (each resumable through `_done/*.json` markers; nothing needs all PubChem rows in RAM):

  A  standardize_source   source chunks -> optional formula-mass prefilter -> standardize -> filters ->
                          standardized/<SOURCE>/chunk-NNNNN.parquet, written SORTED BY BUCKET with ONE ROW
                          GROUP PER BUCKET (bucket = connectivity_key[:bucket_prefix_len]), plus
                          rejected/ (parse failures) and filtered_out/ (filter reasons).
  B  merge_bucket         per bucket: read only that bucket's row groups from every source (pyarrow filter),
                          + the TRAIN mass variants of that bucket -> `merge.unify` -> v2 schema ->
                          buckets/bucket=<b>.parquet + variants/bucket=<b>.parquet. All rows of one connectivity
                          share a bucket, so per-bucket dedup == global dedup.
  C  finalize_universe    buckets in sorted order -> global `candidate_id` (key-sorted universe row number),
                          candidate_keys.npy (S14), CandidateMassIndex, CompactFormulaIndex, universe manifest.

Universe schema (one row per connectivity; `has_reference_spectrum` / `in_train` are METADATA, forbidden as
ranker features -- `casmi.candidates.provenance.FORBIDDEN_SHORTCUT_FEATURES`):
    candidate_id, connectivity_key, representative_smiles, exact_mass, molecular_formula, molecular_weight,
    formal_charge, num_fragments, is_organic, is_single_component,
    train_present, pubchem_present, coconut_present, source_count, source_ids, n_source_records,
    pubchem_literature_count, pubchem_patent_count, natural_product_score  (priors: NA until a source provides them)
  + unify's provenance columns kept for `class2_split`: in_train, has_reference_spectrum, candidate_sources,
    open_representative_smiles, n_mass_variants.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.candidates.filters import UniverseFilterConfig, apply_filters, is_organic_formula, structure_flags
from casmi.candidates.merge import unify
from casmi.candidates.standardize import REJECT_COLUMNS

UNIVERSE_SCHEMA_VERSION = "casmi-v2-universe-1"
STAGE_B_VERSION = "casmi-stage-b-2"
UNIVERSE_MANIFEST_VERSION = 2
KNOWN_SOURCES = ("TRAIN", "COCONUT", "PUBCHEM", "LOTUS", "NPATLAS")


class StageBIncompleteError(RuntimeError):
    """Stage C was asked to finalize buckets that were not all merged for the current Stage-B identity."""


KEY_LEN = 14
V2_COLUMNS = ["connectivity_key", "representative_smiles", "exact_mass", "molecular_formula", "molecular_weight", "formal_charge",
              "num_fragments", "is_organic", "is_single_component", "train_present", "pubchem_present", "coconut_present",
              "source_count", "source_ids", "n_source_records", "pubchem_literature_count", "pubchem_patent_count",
              "natural_product_score", "in_train", "has_reference_spectrum", "candidate_sources", "open_representative_smiles",
              "n_mass_variants", "bucket"]


@dataclass
class UniverseBuildConfig:
    bucket_prefix_len: int = 1
    chunk_records: int = 200_000           # read chunk == resumable checkpoint (part of the Stage-A build identity)
    n_jobs: int | str = 1                  # int, or "auto" / -1 (casmi.workspace.parallel.resolve_n_jobs); never changes outputs
    max_source_ids_per_candidate: int = 20
    filters: UniverseFilterConfig = field(default_factory=UniverseFilterConfig)
    prefilter_mass_margin_da: float = 1.0
    prefilter_by_source_formula: bool = True

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        filt = UniverseFilterConfig.from_dict(d.pop("filters", {}))
        pre = d.pop("prefilter", {}) or {}
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "mass_margin_da" in pre:
            kw["prefilter_mass_margin_da"] = pre["mass_margin_da"]
        if "by_source_formula" in pre:
            kw["prefilter_by_source_formula"] = bool(pre["by_source_formula"])
        return cls(filters=filt, **kw)


def bucket_of(keys, prefix_len):
    return pd.Series(keys, dtype=object).astype(str).str[:prefix_len]


def _write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------
# stage A: standardize one source (resumable)
# ---------------------------------------------------------------------------------------------

def formula_prefilter(records, cfg, cache=None):
    """Cheap pre-RDKit filter from the SOURCE formula (no SMILES parsing): drop records whose source formula
    is parseable AND clearly outside the mass range (+/- margin) or not organic. Unparseable / missing
    formulas pass through to the real standardizer + filters. Returns `(keep, dropped_with_reason)`.

    `cache` (optional dict, one per source run with a FIXED cfg): formula string -> (mass, organic), so formulas that
    recur across chunks are parsed once. The same functions compute the values, so decisions are identical."""
    from casmi.chemistry.adducts import _formula_mass
    if not cfg.prefilter_by_source_formula or "source_formula" not in records or records["source_formula"].isna().all():
        return records, records.iloc[0:0].assign(filter_reason=pd.Series(dtype=str))
    f = records["source_formula"].astype(object)
    uniq = pd.unique(f.dropna())
    cache = {} if cache is None else cache
    for u in uniq:
        if u not in cache:
            cache[u] = (_formula_mass(str(u)), is_organic_formula(str(u), cfg.filters.organic_elements))
    mass = {u: cache[u][0] for u in uniq}
    org = {u: cache[u][1] for u in uniq}
    m = f.map(mass).astype(float)
    lo, hi = cfg.filters.min_exact_mass - cfg.prefilter_mass_margin_da, cfg.filters.max_exact_mass + cfg.prefilter_mass_margin_da
    out_of_range = m.notna() & ((m < lo) | (m > hi))
    not_org = f.notna() & m.notna() & ~f.map(org).fillna(True).astype(bool) if cfg.filters.organic_only else pd.Series(False, index=f.index)
    bad = out_of_range | not_org
    reason = np.where(out_of_range, "prefilter_mass_out_of_range", "prefilter_not_organic")
    return records[~bad], records[bad].assign(filter_reason=reason[bad.to_numpy()])


_TEXT_COLS = ("source", "source_id", "raw_smiles", "normalized_smiles", "representative_smiles", "connectivity_key", "plain_inchikey14",
              "molecular_formula", "murcko_scaffold", "name", "source_formula", "source_metadata", "bucket")
_FLOAT_COLS = ("neutral_monoisotopic_mass", "molecular_weight", "formal_charge", "n_fragments", "num_heavy_atoms", "num_rings",
               "num_aromatic_rings", "hbd", "hba", "tpsa", "logp")
_BOOL_COLS = ("tautomer_hit_cap", "is_charged", "is_organic", "is_single_component", "is_neutral", "in_mass_range")


def _normalize_dtypes(df, copy=True):
    """Identical Arrow schema in every shard (an all-None object column would otherwise become `null`
    type in one file and `string` in another, breaking the multi-file bucket read)."""
    d = df.copy() if copy else df
    for c in d.columns:
        if c in _TEXT_COLS:
            d[c] = d[c].astype("string")
        elif c in _FLOAT_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
        elif c in _BOOL_COLS:
            d[c] = d[c].fillna(False).astype(bool)
    return d


def _write_bucketed(df, path, prefix_len):
    """Parquet sorted by bucket, one row group per bucket (so a pyarrow filter on `bucket` reads only it).
    The Arrow table is built ONCE and written as contiguous slices (one per bucket) -- same schema, same row
    groups and same values as building one table per bucket from pandas."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    df = df.assign(bucket=bucket_of(df["connectivity_key"], prefix_len).to_numpy()).sort_values(["bucket", "connectivity_key"], kind="mergesort")
    b = df["bucket"].astype(str).to_numpy()
    df = _normalize_dtypes(df, copy=False)                        # `df` is already a fresh sorted copy
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]]) if len(b) else np.zeros(0, dtype=int)
    ends = np.r_[starts[1:], len(b)] if len(b) else np.zeros(0, dtype=int)
    with pq.ParquetWriter(tmp, table.schema) as w:
        for s, e in zip(starts, ends, strict=True):
            w.write_table(table.slice(int(s), int(e - s)))
    tmp.replace(path)


def standardize_source(source_cfg, out_dir, cfg=UniverseBuildConfig(), show_progress=True):
    """Backward-compatible Stage A entry point -> `casmi.candidates.stage_a.run_stage_a` with the LEGACY full
    standardization profile, outputs written directly to `out_dir` (no scratch). Unlike the old implementation,
    chunk markers carry the build identity, so a changed source / chunk size / config is never reused.
    Returns the per-chunk summary (DataFrame)."""
    from casmi.candidates.stage_a import StageAPerformance, run_stage_a
    perf = StageAPerformance(standardization_profile="full", stage_input_to_scratch=False, local_temp_outputs=False, cache_enabled=False,
                             progress_every_records=5000 if show_progress else 0, telemetry=False)
    return run_stage_a(source_cfg, out_dir, None, cfg, perf, log=print if show_progress else None).summary


def _mk(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


# ---------------------------------------------------------------------------------------------
# stage B: per-bucket merge (resumable)
# ---------------------------------------------------------------------------------------------

def _table_to_pandas(table):
    """pyarrow -> pandas, robust to the pandas-3 list-dtype metadata issue (retry without pandas metadata)."""
    try:
        return table.to_pandas()
    except TypeError:
        return table.to_pandas(ignore_metadata=True)


def _std_files(std_root=None, std_files=None):
    """The standardized Stage-A chunk files to read: an explicit list (current builds, preferred) or -- legacy --
    every `<std_root>/*/chunk-*.parquet`."""
    if std_files is not None:
        return [Path(f) for f in std_files]
    return sorted(Path(std_root).glob("*/chunk-*.parquet")) if std_root is not None and Path(std_root).is_dir() else []


def _read_bucket(files, bucket):
    import pyarrow.dataset as ds
    files = [Path(f) for f in files]
    if not files:
        return None
    t = ds.dataset([str(f) for f in files], format="parquet").to_table(filter=ds.field("bucket") == bucket)
    return _table_to_pandas(t)


def frame_identity(df, key_col="connectivity_key", value_col=None):
    """Deterministic lightweight metadata of an in-memory table (no hashing of file contents)."""
    if df is None:
        return None
    out = {"n_rows": int(len(df))}
    if key_col in df.columns and len(df):
        k = df[key_col].dropna().astype(str)
        out.update(n_unique_keys=int(k.nunique()), min_key=str(k.min()) if len(k) else None, max_key=str(k.max()) if len(k) else None)
    if value_col and value_col in df.columns and len(df):
        out["value_sum"] = round(float(pd.to_numeric(df[value_col], errors="coerce").sum()), 6)
    return out


def stage_b_identity(stage_a_inputs, mass_variants, train_structure_table, cfg):
    """Everything a merged bucket depends on. `stage_a_inputs`: `stage_a.resolve_stage_a_inputs(...)` output (per source:
    build id, chunk files + sizes) or, for legacy callers, {'legacy_std_files': [(name, size), ...]}."""
    srcs = {}
    for name, s in sorted((stage_a_inputs.get("sources") or {}).items()):
        srcs[name] = {k: s.get(k) for k in ("status", "build_id", "n_chunks", "n_kept")}
        if s.get("std_files"):
            srcs[name]["files"] = sorted((Path(p).name, int(sz)) for p, sz in s["std_files"].items())
    return {"stage_b_version": STAGE_B_VERSION, "universe_schema_version": UNIVERSE_SCHEMA_VERSION, "sources": srcs,
            "legacy_std_files": stage_a_inputs.get("legacy_std_files"),
            "mass_variants": frame_identity(mass_variants, "connectivity_key", "exact_mass"),
            "structure_table": frame_identity(train_structure_table, "connectivity_key", "molecular_weight"),
            "config": {"bucket_prefix_len": cfg.bucket_prefix_len, "max_source_ids_per_candidate": cfg.max_source_ids_per_candidate,
                       "filters": json.loads(json.dumps(cfg.filters.__dict__, default=list))}}


def stage_b_id_of(identity):
    import hashlib
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _bucket_marker(out_root, stage_b_id, bucket):
    return Path(out_root) / "_done" / "stage_b" / stage_b_id / f"bucket-{bucket}.json"


def _first_by_key(df, key="connectivity_key", order_cols=()):
    d = df.sort_values([key, *order_cols], kind="mergesort") if order_cols else df.sort_values(key, kind="mergesort")
    return d.drop_duplicates(key).set_index(key)


def to_v2_schema(unified, external_rows=None, train_structure_table=None, max_ids=20, prefix_len=1, filters=UniverseFilterConfig()):
    """`merge.unify` output -> v2 universe columns (descriptors: TRAIN structure table first, else the
    first external record by source order)."""
    u = unified.copy()
    desc = pd.DataFrame(index=pd.Index(u["connectivity_key"], name="connectivity_key"))
    for c in ("molecular_weight", "formal_charge", "num_fragments"):
        desc[c] = np.nan
    if external_rows is not None and len(external_rows):
        from casmi.candidates.merge import SOURCE_ORDER
        e = external_rows.assign(_o=external_rows["source"].map(SOURCE_ORDER).fillna(99))
        first = _first_by_key(e, order_cols=("_o", "source_id"))
        desc["molecular_weight"] = first["molecular_weight"].reindex(desc.index).astype(float)
        desc["formal_charge"] = first["formal_charge"].reindex(desc.index).astype(float)
        desc["num_fragments"] = first["n_fragments"].reindex(desc.index).astype(float)
    if train_structure_table is not None and len(train_structure_table):
        st = _first_by_key(train_structure_table.dropna(subset=["connectivity_key"]), order_cols=("smiles",))
        for c in ("molecular_weight", "formal_charge"):
            tv = st[c].reindex(desc.index).astype(float)
            desc[c] = np.where(u["in_train"].to_numpy() & tv.notna().to_numpy(), tv.to_numpy(), desc[c].to_numpy())
    u["molecular_formula"] = u["formula"]
    u["molecular_weight"] = desc["molecular_weight"].to_numpy()
    u["formal_charge"] = desc["formal_charge"].to_numpy()
    nf = desc["num_fragments"].to_numpy()
    u["num_fragments"] = np.where(np.isnan(nf), u["representative_smiles"].astype(str).str.count(r"\.").to_numpy() + 1, nf).astype(int)
    flags = structure_flags(pd.DataFrame({"molecular_formula": u["molecular_formula"].to_numpy(), "neutral_monoisotopic_mass": u["exact_mass"].to_numpy(),
                                          "formal_charge": u["formal_charge"].fillna(0).to_numpy(), "n_fragments": u["num_fragments"].to_numpy()}), filters)
    u["is_organic"] = flags["is_organic"].to_numpy()
    u["is_single_component"] = flags["is_single_component"].to_numpy()
    u = add_provenance_aliases(u)
    u["source_ids"] = [list(s)[:max_ids] for s in u["source_ids"]]
    for c in ("pubchem_literature_count", "pubchem_patent_count"):
        u[c] = pd.array([pd.NA] * len(u), dtype="Int64")
    u["natural_product_score"] = np.nan
    u["bucket"] = bucket_of(u["connectivity_key"], prefix_len).to_numpy()
    return u[V2_COLUMNS]


def add_provenance_aliases(u):
    """v2 provenance flags derived from `candidate_sources` (recompute after any provenance stripping)."""
    srcs = u["candidate_sources"].map(lambda s: set(s) if s is not None else set())
    u = u.copy()
    u["train_present"] = u["in_train"].astype(bool)
    u["pubchem_present"] = srcs.map(lambda s: "PUBCHEM" in s).to_numpy()
    u["coconut_present"] = srcs.map(lambda s: "COCONUT" in s).to_numpy()
    u["source_count"] = srcs.map(len).astype(int).to_numpy()
    return u


def provenance_view(rows, hidden_reference_keys):
    """C2/C3 evaluation view of universe ROWS (e.g. a candidate shortlist): hidden truths are stripped of
    TRAIN provenance via `class2_split.class2_candidate_view`, then the v2 aliases are recomputed."""
    from casmi.validation.class2_split import class2_candidate_view
    return add_provenance_aliases(class2_candidate_view(rows, hidden_reference_keys))


def merge_bucket(bucket, std_root, mass_variants, out_root, train_structure_table=None, cfg=UniverseBuildConfig(), std_files=None,
                 stage_b_id=None):
    """Stage B for one bucket. Resumes (skips) ONLY when a marker exists for the same Stage-B identity and the bucket
    files it recorded are intact; otherwise the bucket is rebuilt -- a TRAIN-only bucket can never survive into a
    universe whose Stage-A inputs changed (e.g. COCONUT added). Inputs: `std_files` (explicit list of the current
    standardized chunk files) or, legacy, every chunk under `std_root`. Returns the bucket record."""
    from casmi.workspace.staging import files_intact
    out_root = Path(out_root)
    files = _std_files(std_root, std_files)
    if stage_b_id is None:                       # legacy callers: identity from the file listing they read
        legacy = sorted((f.name if std_root is None else str(f.relative_to(std_root)).replace("\\", "/"), f.stat().st_size) for f in files)
        stage_b_id = stage_b_id_of(stage_b_identity({"legacy_std_files": legacy}, mass_variants, train_structure_table, cfg))
    done = _bucket_marker(out_root, stage_b_id, bucket)
    if done.exists():
        rec = json.loads(done.read_text(encoding="utf-8"))
        if rec.get("stage_b_id") == stage_b_id and files_intact({str(out_root / r): s for r, s in (rec.get("files") or {}).items()}):
            return rec
    ext = _read_bucket(files, bucket)
    if ext is None:
        ext = pd.DataFrame(columns=["source", "source_id", "connectivity_key", "representative_smiles", "molecular_formula",
                                    "neutral_monoisotopic_mass", "name", "molecular_weight", "formal_charge", "n_fragments"])
    mv = mass_variants[bucket_of(mass_variants["connectivity_key"], cfg.bucket_prefix_len).to_numpy() == bucket]
    st = None
    if train_structure_table is not None:
        st = train_structure_table[bucket_of(train_structure_table["connectivity_key"], cfg.bucket_prefix_len).to_numpy() == bucket]
    if not len(mv) and not len(ext):
        u, variants = pd.DataFrame(columns=V2_COLUMNS), pd.DataFrame(columns=["connectivity_key", "exact_mass", "variant_source"])
    else:
        u, variants = unify(mv, ext)
        u = to_v2_schema(u, ext, st, cfg.max_source_ids_per_candidate, cfg.bucket_prefix_len, cfg.filters)
    assert u["connectivity_key"].is_unique if len(u) else True, "bucket must hold one row per connectivity"
    _mk(out_root / "buckets")
    _mk(out_root / "variants")
    rel = {"buckets": f"buckets/bucket={bucket}.parquet", "variants": f"variants/bucket={bucket}.parquet"}
    u.to_parquet(out_root / rel["buckets"], index=False)
    variants.to_parquet(out_root / rel["variants"], index=False)
    rec = {"bucket": bucket, "stage_b_id": stage_b_id, "n_connectivities": int(len(u)), "n_variants": int(len(variants)),
           "n_train": int(u["train_present"].sum()) if len(u) else 0, "n_external_records": int(len(ext)),
           "files": {r: (out_root / r).stat().st_size for r in rel.values()}, "finished_at": datetime.now(timezone.utc).isoformat()}
    _write_json_atomic(done, rec)
    return rec


def all_buckets(std_root, mass_variants, prefix_len, std_files=None):
    """Every bucket seen in the TRAIN variants or any standardized shard (sorted)."""
    import pyarrow.parquet as pq
    b = set(bucket_of(mass_variants["connectivity_key"], prefix_len))
    for f in _std_files(std_root, std_files):
        b |= set(pq.read_table(f, columns=["bucket"]).column("bucket").unique().to_pylist())
    return sorted(b)


def run_stage_b(stage_a_inputs, mass_variants, train_structure_table, out_root, cfg=UniverseBuildConfig(), progress=None, log=print):
    """Stage B over the CURRENT Stage-A builds only (`stage_a.resolve_stage_a_inputs`). Every bucket is merged under the
    Stage-B identity of these inputs; buckets merged for any other identity are rebuilt. Returns
    {'stage_b_id', 'identity', 'buckets', 'records'}."""
    identity = stage_b_identity(stage_a_inputs, mass_variants, train_structure_table, cfg)
    sid = stage_b_id_of(identity)
    files = list(stage_a_inputs.get("std_files") or [])
    buckets = all_buckets(None, mass_variants, cfg.bucket_prefix_len, std_files=files)
    _write_json_atomic(Path(out_root) / "_done" / "stage_b" / sid / "identity.json", {"stage_b_id": sid, "identity": identity})
    if log:
        log(f"[stage B] id {sid} | {len(files)} standardized chunk file(s) | sources "
            f"{ {k: v.get('status') for k, v in (stage_a_inputs.get('sources') or {}).items()} } | {len(buckets)} buckets")
    it = progress(buckets) if progress else buckets
    recs = [merge_bucket(b, None, mass_variants, out_root, train_structure_table, cfg, std_files=files, stage_b_id=sid) for b in it]
    return {"stage_b_id": sid, "identity": identity, "buckets": buckets, "records": pd.DataFrame(recs)}


# ---------------------------------------------------------------------------------------------
# stage C: finalize (global candidate ids + indexes)
# ---------------------------------------------------------------------------------------------

def _source_counts(candidate_sources, in_train, per_source, counts):
    for s, tr in zip(candidate_sources, in_train, strict=True):
        ss = set(s) if isinstance(s, (list, tuple, np.ndarray)) else set()
        per_source.update(ss)
        ext = ss - {"TRAIN"}
        is_train = bool(tr) or "TRAIN" in ss
        counts["train"] += int(is_train)
        counts["external_any"] += int(bool(ext))
        counts["external_only"] += int(bool(ext) and not is_train)
        counts["train_and_external"] += int(bool(ext) and is_train)


def finalize_universe(out_root, bucket_list, build_formula_index=True, stage_b_id=None, build_info=None):
    """Global key-sorted `candidate_id` = bucket offset + row position (bucket files are key-sorted and
    buckets are key-prefix ranges, so the concatenation is globally key-sorted). Writes
    candidate_keys.npy, bucket_offsets.json, index/ (mass), formula_index/ and universe_manifest.json.

    `stage_b_id` (from `run_stage_b`): every bucket must carry a marker for THIS Stage-B identity, otherwise
    StageBIncompleteError (never finalize a mix of old and new buckets). The manifest records the build identity and the
    source composition (TRAIN / COCONUT / PUBCHEM counts, `external_source_present`) so the C2 protocol can be checked
    without scanning buckets. Always rebuilt from the buckets (Stage C is cheap)."""
    import pyarrow.parquet as pq
    from collections import Counter
    from casmi.candidates.formula_index import CompactFormulaIndex
    from casmi.candidates.mass_index import CandidateMassIndex
    out_root = Path(out_root)
    if stage_b_id is not None:
        missing = [b for b in bucket_list if not _bucket_marker(out_root, stage_b_id, b).exists()]
        if missing:
            raise StageBIncompleteError(f"{len(missing)} bucket(s) not merged for Stage-B identity {stage_b_id}: {missing[:10]} -- run Stage B")
    offsets, keys, masses, ids, formulas, total, prev_last = {}, [], [], [], [], 0, ""
    per_source, counts = Counter(), Counter()
    for b in sorted(bucket_list):
        p = out_root / "buckets" / f"bucket={b}.parquet"
        t = pq.read_table(p, columns=["connectivity_key", "molecular_formula", "candidate_sources", "in_train"])
        k = np.array([str(x) for x in t.column("connectivity_key").to_pylist()], dtype=object)
        _source_counts(t.column("candidate_sources").to_pylist(), t.column("in_train").to_pylist(), per_source, counts)
        formula_values = np.array(t.column("molecular_formula").to_pylist(), dtype=object)
        if len(k):
            assert (k[:-1] < k[1:]).all(), f"bucket {b} is not strictly key-sorted"
            assert k[0] > prev_last, f"bucket {b} overlaps the previous bucket (prefix ranges must be disjoint)"
            prev_last = k[-1]
        offsets[b] = total
        v = pd.read_parquet(out_root / "variants" / f"bucket={b}.parquet", columns=["connectivity_key", "exact_mass"])
        vk = v["connectivity_key"].astype(str).to_numpy()
        pos = np.searchsorted(k, vk)
        assert len(v) == 0 or ((pos < len(k)).all() and (k[np.clip(pos, 0, len(k) - 1)] == vk).all()), f"bucket {b}: variant without a structure row"
        keys.append(k.astype(f"S{KEY_LEN}"))
        masses.append(v["exact_mass"].to_numpy(np.float64))
        ids.append((pos + total).astype(np.int64))
        formulas.append(formula_values)
        total += len(k)
    all_keys = np.concatenate(keys) if keys else np.zeros(0, f"S{KEY_LEN}")
    np.save(out_root / "candidate_keys.npy", all_keys)
    _write_json_atomic(out_root / "bucket_offsets.json", {"offsets": offsets, "n_candidates": total})
    idx = CandidateMassIndex(np.concatenate(masses) if masses else [], np.concatenate(ids) if ids else [], n_candidates=total)
    mmeta = idx.save(out_root / "index", extra={"universe_schema_version": UNIVERSE_SCHEMA_VERSION})
    fmeta = None
    if build_formula_index:
        fidx = CompactFormulaIndex.build(np.concatenate(formulas) if formulas else [], np.arange(total), total)
        fmeta = fidx.save(out_root / "formula_index")
    sources = {s: {"present": per_source.get(s, 0) > 0, "n_candidates": int(per_source.get(s, 0))}
               for s in sorted(set(KNOWN_SOURCES) | set(per_source))}
    manifest = {"universe_schema_version": UNIVERSE_SCHEMA_VERSION, "manifest_version": UNIVERSE_MANIFEST_VERSION, "build_id": stage_b_id,
                "build_info": build_info, "n_candidates": total, "n_buckets": len(offsets), "sources": sources,
                "external_source_present": counts["external_any"] > 0, "n_train_candidates": int(counts["train"]),
                "n_external_any": int(counts["external_any"]), "n_external_only": int(counts["external_only"]),
                "n_train_and_external": int(counts["train_and_external"]), "mass_index": mmeta, "formula_index": fmeta,
                "finalized_at": datetime.now(timezone.utc).isoformat()}
    _write_json_atomic(out_root / "universe_manifest.json", manifest)
    return manifest


def universe_source_status(out_root):
    """C2-relevant universe status from `universe_manifest.json` alone (no bucket scan):
    'absent' | 'train_only' | 'external' | 'unknown' (manifest written before source counts existed -> re-run Stage C)."""
    p = Path(out_root) / "universe_manifest.json"
    if not p.is_file():
        return {"status": "absent", "external_source_present": False}
    m = json.loads(p.read_text(encoding="utf-8"))
    if "external_source_present" not in m:
        return {"status": "unknown", "external_source_present": None, "n_candidates": m.get("n_candidates"),
                "reason": "manifest predates source counts -- re-run Stage C (finalize_universe)"}
    return {"status": "external" if m["external_source_present"] else "train_only", "external_source_present": m["external_source_present"],
            "n_candidates": m.get("n_candidates"), "sources": m.get("sources"), "build_id": m.get("build_id")}


def load_candidate_keys(out_root, mmap=True):
    return np.load(Path(out_root) / "candidate_keys.npy", mmap_mode="r" if mmap else None)


def keys_to_ids(candidate_keys_sorted, keys):
    """candidate_id for each key (-1 when absent) -- searchsorted on the sorted S14 key array."""
    k = np.asarray(pd.Series(keys, dtype=object).astype(str).to_numpy()).astype(f"S{KEY_LEN}")
    pos = np.searchsorted(candidate_keys_sorted, k)
    pos_c = np.clip(pos, 0, max(len(candidate_keys_sorted) - 1, 0))
    hit = (pos < len(candidate_keys_sorted)) & (np.asarray(candidate_keys_sorted)[pos_c] == k) if len(candidate_keys_sorted) else np.zeros(len(k), bool)
    return np.where(hit, pos_c, -1).astype(np.int64)


def read_candidates(out_root, candidate_ids, columns=None):
    """Universe rows for a set of candidate ids (reads only the buckets involved) with `candidate_id`."""
    out_root = Path(out_root)
    off = json.loads((out_root / "bucket_offsets.json").read_text(encoding="utf-8"))["offsets"]
    bnames = sorted(off, key=lambda b: off[b])
    starts = np.array([off[b] for b in bnames])
    ids = np.unique(np.asarray(candidate_ids, dtype=np.int64))
    which = np.searchsorted(starts, ids, side="right") - 1
    parts = []
    for bi in np.unique(which):
        b = bnames[bi]
        local = ids[which == bi] - starts[bi]
        t = pd.read_parquet(out_root / "buckets" / f"bucket={b}.parquet", columns=columns)
        parts.append(t.iloc[local].assign(candidate_id=local + starts[bi]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=(columns or V2_COLUMNS) + ["candidate_id"])


def build_universe_in_memory(mass_variants, external_standardized, train_structure_table=None, cfg=UniverseBuildConfig()):
    """Small universes (tests, COCONUT-only): same logic as stages B + C without files. Returns the v2
    table with `candidate_id` (key-sorted) and the unify variants with `candidate_id`."""
    ext = external_standardized
    if len(ext):
        ext, _ = apply_filters(ext, cfg.filters)
    u, variants = unify(mass_variants, ext)
    u = to_v2_schema(u, ext, train_structure_table, cfg.max_source_ids_per_candidate, cfg.bucket_prefix_len, cfg.filters)
    u = u.sort_values("connectivity_key", kind="mergesort").reset_index(drop=True)
    u.insert(0, "candidate_id", np.arange(len(u), dtype=np.int64))
    variants = variants.assign(candidate_id=variants["connectivity_key"].map(u.set_index("connectivity_key")["candidate_id"]).astype(np.int64))
    return u, variants


def rejected_summary(out_dir):
    """Counts of rejected (parse) and filtered-out rows by source and reason (reads the small side tables).
    `out_dir`: one directory or a list of directories (e.g. the current Stage-A build namespaces)."""
    dirs = [Path(d) for d in out_dir] if isinstance(out_dir, (list, tuple)) else [Path(out_dir)]
    parts = []
    for kind, col in (("rejected", "failure_reason"), ("filtered_out", "filter_reason")):
        for f in sorted(p for d in dirs for p in (d / kind).glob("*/chunk-*.parquet")):
            t = pd.read_parquet(f, columns=["source", col])
            t[col] = t[col].astype(str).str.split(":").str[0]
            parts.append(t.groupby(["source", col]).size().rename("n").reset_index().rename(columns={col: "reason"}).assign(kind=kind))
    if not parts:
        return pd.DataFrame(columns=["kind", "source", "reason", "n"])
    return pd.concat(parts).groupby(["kind", "source", "reason"], as_index=False)["n"].sum()


__all__ = ["UniverseBuildConfig", "standardize_source", "merge_bucket", "all_buckets", "run_stage_b", "stage_b_identity", "stage_b_id_of",
           "finalize_universe", "universe_source_status", "StageBIncompleteError", "read_candidates", "keys_to_ids", "load_candidate_keys",
           "build_universe_in_memory", "provenance_view", "rejected_summary", "REJECT_COLUMNS"]
