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
from casmi.candidates.standardize import REJECT_COLUMNS, standardize_records
from casmi.io.parquet import read_parquet_safe, table_to_pandas

UNIVERSE_SCHEMA_VERSION = "casmi-v2-universe-1"
UNIVERSE_MANIFEST_VERSION = 2          # 2: manifest embeds `source_summary`; manifests without it are summarized on read
KEY_LEN = 14
V2_COLUMNS = ["connectivity_key", "representative_smiles", "exact_mass", "molecular_formula", "molecular_weight", "formal_charge",
              "num_fragments", "is_organic", "is_single_component", "train_present", "pubchem_present", "coconut_present",
              "source_count", "source_ids", "n_source_records", "pubchem_literature_count", "pubchem_patent_count",
              "natural_product_score", "in_train", "has_reference_spectrum", "candidate_sources", "open_representative_smiles",
              "n_mass_variants", "bucket"]


@dataclass
class UniverseBuildConfig:
    bucket_prefix_len: int = 1
    chunk_records: int = 200_000
    n_jobs: int = 1
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

def formula_prefilter(records, cfg):
    """Cheap pre-RDKit filter from the SOURCE formula (no SMILES parsing): drop records whose source formula
    is parseable AND clearly outside the mass range (+/- margin) or not organic. Unparseable / missing
    formulas pass through to the real standardizer + filters. Returns `(keep, dropped_with_reason)`."""
    from casmi.chemistry.adducts import _formula_mass
    if not cfg.prefilter_by_source_formula or "source_formula" not in records or records["source_formula"].isna().all():
        return records, records.iloc[0:0].assign(filter_reason=pd.Series(dtype=str))
    f = records["source_formula"].astype(object)
    uniq = pd.unique(f.dropna())
    mass = {u: _formula_mass(str(u)) for u in uniq}
    org = {u: is_organic_formula(str(u), cfg.filters.organic_elements) for u in uniq}
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


def _normalize_dtypes(df):
    """Identical Arrow schema in every shard (an all-None object column would otherwise become `null`
    type in one file and `string` in another, breaking the multi-file bucket read)."""
    d = df.copy()
    for c in d.columns:
        if c in _TEXT_COLS:
            d[c] = d[c].astype("string")
        elif c in _FLOAT_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
        elif c in _BOOL_COLS:
            d[c] = d[c].fillna(False).astype(bool)
    return d


def _write_bucketed(df, path, prefix_len):
    """Parquet sorted by bucket, one row group per bucket (so a pyarrow filter on `bucket` reads only it)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    df = df.assign(bucket=bucket_of(df["connectivity_key"], prefix_len).to_numpy()).sort_values(["bucket", "connectivity_key"], kind="mergesort")
    df = _normalize_dtypes(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pq.ParquetWriter(tmp, table.schema) as w:
        for b in df["bucket"].unique():
            w.write_table(pa.Table.from_pandas(df[df["bucket"] == b], schema=table.schema, preserve_index=False))
    tmp.replace(path)


def standardize_source(source_cfg, out_dir, cfg=UniverseBuildConfig(), show_progress=True, standardize_fn=standardize_records):
    """Stage A for one `SourceConfig`. Returns the per-chunk manifest (DataFrame)."""
    from casmi.candidates.sources import iter_source_records
    out_dir = Path(out_dir)
    src = source_cfg.canonical_source()
    source_cfg.chunksize = cfg.chunk_records
    rows = []
    for i, chunk in enumerate(iter_source_records(source_cfg)):
        done = out_dir / "_done" / f"{src}-chunk-{i:05d}.json"
        if done.exists():
            rec = json.loads(done.read_text(encoding="utf-8"))
            if rec.get("n_input") == len(chunk):
                rows.append(rec)
                continue
        keep, pre = formula_prefilter(chunk, cfg)
        std, rej = standardize_fn(keep, n_jobs=cfg.n_jobs, show_progress=False)
        kept, filt = apply_filters(std, cfg.filters)
        _write_bucketed(kept, out_dir / "standardized" / src / f"chunk-{i:05d}.parquet", cfg.bucket_prefix_len)
        rej.to_parquet(_mk(out_dir / "rejected" / src) / f"chunk-{i:05d}.parquet", index=False)
        fo = pd.concat([pre[["source", "source_id", "raw_smiles", "filter_reason"]],
                        filt[["source", "source_id", "raw_smiles", "connectivity_key", "filter_reason"]]], ignore_index=True)
        fo.to_parquet(_mk(out_dir / "filtered_out" / src) / f"chunk-{i:05d}.parquet", index=False)
        rec = {"source": src, "chunk": i, "n_input": int(len(chunk)), "n_prefiltered": int(len(pre)), "n_rejected": int(len(rej)),
               "n_filtered": int(len(filt)), "n_kept": int(len(kept)), "n_unique_connectivities_in_chunk": int(kept["connectivity_key"].nunique()),
               "finished_at": datetime.now(timezone.utc).isoformat()}
        assert rec["n_prefiltered"] + rec["n_rejected"] + rec["n_filtered"] + rec["n_kept"] == rec["n_input"], "record accounting broken"
        _write_json_atomic(done, rec)
        rows.append(rec)
        if show_progress:
            print(f"[{src}] chunk {i}: in={rec['n_input']} kept={rec['n_kept']} rejected={rec['n_rejected']} "
                  f"filtered={rec['n_filtered']} prefiltered={rec['n_prefiltered']}")
    return pd.DataFrame(rows)


def _mk(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


# ---------------------------------------------------------------------------------------------
# stage B: per-bucket merge (resumable)
# ---------------------------------------------------------------------------------------------

def _read_bucket(std_root, bucket):
    import pyarrow.dataset as ds
    files = sorted(Path(std_root).glob("*/chunk-*.parquet"))
    if not files:
        return None
    t = ds.dataset([str(f) for f in files], format="parquet").to_table(filter=ds.field("bucket") == bucket)
    return table_to_pandas(t)


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


def stage_a_signature(std_root, mass_variants, cfg=UniverseBuildConfig()):
    """Lightweight identity of everything a bucket merge reads: the standardized stage-A chunk files (relative path +
    size), the number of TRAIN mass variants and the merge-relevant config. A bucket built before a new external source
    (e.g. COCONUT) was standardized has a different signature and is REBUILT -- never silently kept TRAIN-only."""
    import hashlib
    std_root = Path(std_root)
    files = sorted((str(f.relative_to(std_root)).replace("\\", "/"), f.stat().st_size) for f in std_root.glob("*/chunk-*.parquet")) \
        if std_root.is_dir() else []
    blob = json.dumps({"files": files, "n_mass_variants": int(len(mass_variants)), "prefix_len": cfg.bucket_prefix_len,
                       "max_ids": cfg.max_source_ids_per_candidate}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def merge_bucket(bucket, std_root, mass_variants, out_root, train_structure_table=None, cfg=UniverseBuildConfig(), inputs_signature=None):
    """Stage B for one bucket. Resumes (skips) only when its done-marker exists AND was written for the same stage-A
    inputs (`stage_a_signature`); otherwise the bucket is rebuilt. Returns the bucket record."""
    out_root = Path(out_root)
    done = out_root / "_done" / f"bucket-{bucket}.json"
    sig = inputs_signature or stage_a_signature(std_root, mass_variants, cfg)
    if done.exists():
        rec = json.loads(done.read_text(encoding="utf-8"))
        if rec.get("inputs_signature") == sig:
            return rec
    ext = _read_bucket(std_root, bucket)
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
    u.to_parquet(out_root / "buckets" / f"bucket={bucket}.parquet", index=False)
    variants.to_parquet(out_root / "variants" / f"bucket={bucket}.parquet", index=False)
    rec = {"bucket": bucket, "n_connectivities": int(len(u)), "n_variants": int(len(variants)),
           "n_train": int(u["train_present"].sum()) if len(u) else 0, "n_external_records": int(len(ext)),
           "inputs_signature": sig, "finished_at": datetime.now(timezone.utc).isoformat()}
    _write_json_atomic(done, rec)
    return rec


def all_buckets(std_root, mass_variants, prefix_len):
    """Every bucket seen in the TRAIN variants or any standardized shard (sorted)."""
    import pyarrow.parquet as pq
    b = set(bucket_of(mass_variants["connectivity_key"], prefix_len))
    for f in sorted(Path(std_root).glob("*/chunk-*.parquet")):
        b |= set(pq.read_table(f, columns=["bucket"]).column("bucket").unique().to_pylist())
    return sorted(b)


# ---------------------------------------------------------------------------------------------
# stage C: finalize (global candidate ids + indexes)
# ---------------------------------------------------------------------------------------------

def finalize_universe(out_root, bucket_list, build_formula_index=True):
    """Global key-sorted `candidate_id` = bucket offset + row position (bucket files are key-sorted and
    buckets are key-prefix ranges, so the concatenation is globally key-sorted).

    Canonical layout (everything under `out_root`, the universe root; downstream loaders rely on it):
        candidate_keys.npy, bucket_offsets.json, universe_manifest.json, index/ (mass), formula_index/,
        buckets/, variants/
    The manifest embeds `source_summary` (TRAIN-only / external-only / shared counts), so the C2 protocol validity
    (`casmi.validation.c2_protocol`) can be decided from the manifest alone."""
    from collections import Counter
    from casmi.candidates.formula_index import CompactFormulaIndex
    from casmi.candidates.mass_index import CandidateMassIndex
    out_root = Path(out_root)
    offsets, keys, masses, ids, formulas, total, prev_last = {}, [], [], [], [], 0, ""
    combo, per_source = Counter(), Counter()
    for b in sorted(bucket_list):
        p = out_root / "buckets" / f"bucket={b}.parquet"
        t = read_parquet_safe(p, columns=["connectivity_key", "molecular_formula", "candidate_sources"])
        _count_sources(t["candidate_sources"], combo, per_source)
        k = t["connectivity_key"].astype(str).to_numpy()
        if len(k):
            assert (k[:-1] < k[1:]).all(), f"bucket {b} is not strictly key-sorted"
            assert k[0] > prev_last, f"bucket {b} overlaps the previous bucket (prefix ranges must be disjoint)"
            prev_last = k[-1]
        offsets[b] = total
        v = read_parquet_safe(out_root / "variants" / f"bucket={b}.parquet", columns=["connectivity_key", "exact_mass"])
        vk = v["connectivity_key"].astype(str).to_numpy()
        pos = np.searchsorted(k, vk)
        assert len(v) == 0 or ((pos < len(k)).all() and (k[np.clip(pos, 0, len(k) - 1)] == vk).all()), f"bucket {b}: variant without a structure row"
        keys.append(k.astype(f"S{KEY_LEN}"))
        masses.append(v["exact_mass"].to_numpy(np.float64))
        ids.append((pos + total).astype(np.int64))
        formulas.append(t["molecular_formula"].to_numpy(object))
        total += len(k)
    all_keys = np.concatenate(keys) if keys else np.zeros(0, f"S{KEY_LEN}")
    np.save(out_root / "candidate_keys.npy", all_keys)
    _write_json_atomic(out_root / "bucket_offsets.json", {"offsets": offsets, "n_candidates": total})
    idx = CandidateMassIndex(np.concatenate(masses) if masses else [], np.concatenate(ids) if ids else [], n_candidates=total)
    mmeta = idx.save(out_root / "index", extra={"universe_schema_version": UNIVERSE_SCHEMA_VERSION})
    fmeta = None
    if build_formula_index:
        fidx = CompactFormulaIndex.build(np.concatenate(formulas) if formulas else [], np.arange(total), total)
        fmeta = {**fidx.save(out_root / "formula_index"), "path": "formula_index"}
    manifest = {"universe_schema_version": UNIVERSE_SCHEMA_VERSION, "manifest_version": UNIVERSE_MANIFEST_VERSION,
                "n_candidates": total, "n_buckets": len(offsets), "mass_index": mmeta, "formula_index": fmeta,
                "source_summary": _summarize_sources(combo, per_source), "finalized_at": datetime.now(timezone.utc).isoformat()}
    _write_json_atomic(out_root / "universe_manifest.json", manifest)           # readers look for it next to the index
    return manifest


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
        t = read_parquet_safe(out_root / "buckets" / f"bucket={b}.parquet", columns=columns)
        parts.append(t.iloc[local].assign(candidate_id=local + starts[bi]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=(columns or V2_COLUMNS) + ["candidate_id"])


def _source_set(s):
    return set(s) if isinstance(s, (list, tuple, np.ndarray)) else set()


def _count_sources(candidate_sources, combo, per_source):
    for s in candidate_sources:
        ss = _source_set(s)
        combo["+".join(sorted(ss)) or "none"] += 1
        per_source.update(ss)


def _summarize_sources(combo, per_source):
    n = sum(combo.values())
    train_only = combo.get("TRAIN", 0)
    no_source = combo.get("none", 0)
    with_train = per_source.get("TRAIN", 0)
    external_any = n - train_only - no_source
    return {"n_candidates": int(n), "train_only": int(train_only), "external_only": int(external_any - (with_train - train_only)),
            "train_and_external": int(with_train - train_only), "external_any": int(external_any), "no_source": int(no_source),
            "per_source": {str(k): int(v) for k, v in sorted(per_source.items())},
            "overlap": {str(k): int(v) for k, v in combo.most_common()}}


def universe_source_summary(out_root):
    """Source composition of a finalized universe, bucket by bucket (never the whole universe in RAM).

    Returns a dict with `n_candidates`, `train_only`, `external_only`, `train_and_external`, `external_any`,
    `no_source`, `per_source` (connectivities containing each source) and `overlap` (exact source combination ->
    count). `external_any` (= external_only + train_and_external) is the number of structures a hidden Class-2
    truth could come from. Universes finalized by this module also carry it in `universe_manifest.json`."""
    from collections import Counter
    out_root = Path(out_root)
    off = json.loads((out_root / "bucket_offsets.json").read_text(encoding="utf-8"))["offsets"]
    combo, per_source = Counter(), Counter()
    for b in sorted(off, key=lambda x: off[x]):
        t = read_parquet_safe(out_root / "buckets" / f"bucket={b}.parquet", columns=["candidate_sources"])
        _count_sources(t["candidate_sources"], combo, per_source)
    return _summarize_sources(combo, per_source)


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
    """Counts of rejected (parse) and filtered-out rows by source and reason (reads the small side tables)."""
    out_dir = Path(out_dir)
    parts = []
    for kind, col in (("rejected", "failure_reason"), ("filtered_out", "filter_reason")):
        for f in sorted((out_dir / kind).glob("*/chunk-*.parquet")):
            t = pd.read_parquet(f, columns=["source", col])
            t[col] = t[col].astype(str).str.split(":").str[0]
            parts.append(t.groupby(["source", col]).size().rename("n").reset_index().rename(columns={col: "reason"}).assign(kind=kind))
    if not parts:
        return pd.DataFrame(columns=["kind", "source", "reason", "n"])
    return pd.concat(parts).groupby(["kind", "source", "reason"], as_index=False)["n"].sum()


__all__ = ["UniverseBuildConfig", "standardize_source", "stage_a_signature", "merge_bucket", "all_buckets", "finalize_universe", "read_candidates",
           "keys_to_ids", "load_candidate_keys", "build_universe_in_memory", "universe_source_summary", "provenance_view", "rejected_summary", "REJECT_COLUMNS"]
