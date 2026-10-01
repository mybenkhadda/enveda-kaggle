"""Memory-bounded, resumable, chunked QCR (query x candidate x reference) construction.

Queries are processed in chunks (default 50/chunk, see `chunk_query_ids`). For each chunk: if a
valid shard already exists (fingerprint + expected query-id set match, and the shard parquet is
readable -- see `shard_is_valid`), the chunk is skipped entirely: no peaks loaded, no reference
walk run, no new similarity computation. Otherwise the chunk is rebuilt from scratch and its
chunk-local state (identity/similarity peak dicts, ranked-reference lists, walk results) is
deleted and `gc.collect()`ed before the next chunk starts -- the caller never holds more than
ONE chunk's full per-reference walk state in memory at a time, regardless of total dataset size
(see spec section 12).
"""
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil
import pyarrow.parquet as pq

from casmi.io.parquet import load_rows_by_offset
from casmi.ranking.features import compute_single_pair_scores
from casmi.spectra.deduplication import classify_identity_tier
from casmi.spectra.preprocessing import remove_invalid_peaks, truncate_top_peaks
from casmi.spectra.reference_selection import MAX_REFS_PER_PROTOCOL, PROTOCOLS, compat_sort_key, walk_references
from casmi.spectra.similarity_cache import SimilarityCache
from casmi.qcr.fingerprint import code_version


@dataclass
class ChunkResult:
    chunk_id: int
    query_ids: list = field(default_factory=list)
    n_qcr_rows: int = 0
    n_pair_rows: int = 0
    loaded_from_shard: bool = False
    wall_seconds: float = 0.0
    peak_rss_gb: float = 0.0
    n_similarity_computed: int = 0
    n_similarity_hits: int = 0
    skip_reason: str = ""


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _current_rss_gb():
    try:
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return float("nan")


def estimate_max_workers(ram_budget_gb, est_worker_gb=0.5, current_rss_gb=None):
    """How many parallel chunk workers fit under `ram_budget_gb`, given the current process's
    already-resident memory (`current_rss_gb`, measured via psutil if not given) and a
    per-worker memory estimate (`est_worker_gb`). Always returns >= 1 -- never recommends zero
    workers even under a tight budget; the caller just runs serially in that case."""
    if current_rss_gb is None:
        current_rss_gb = _current_rss_gb()
    available = max(ram_budget_gb - current_rss_gb, 0.0)
    return max(int(available // est_worker_gb), 1)


def chunk_query_ids(query_ids, chunk_size=50):
    query_ids = list(query_ids)
    return [query_ids[i:i + chunk_size] for i in range(0, len(query_ids), chunk_size)]


def _shard_paths(output_dir, dataset_name, chunk_id):
    qcr_dir = Path(output_dir) / "qcr" / dataset_name
    pairs_dir = Path(output_dir) / "qcr_pairs" / dataset_name
    cache_dir = Path(output_dir) / "similarity_cache" / dataset_name
    for d in (qcr_dir, pairs_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    stem = f"part-{chunk_id:05d}"
    return {
        "qcr": qcr_dir / f"{stem}.parquet", "qcr_meta": qcr_dir / f"{stem}.meta.json",
        "pairs": pairs_dir / f"{stem}.parquet", "pairs_meta": pairs_dir / f"{stem}.meta.json",
        "cache": cache_dir / f"{stem}.parquet", "cache_meta": cache_dir / f"{stem}.meta.json",
    }


def shard_is_valid(paths, expected_fingerprint, expected_query_ids):
    """A chunk shard is safe to skip rebuilding only if: its metadata sidecar exists and is
    readable, its stored fingerprint matches `expected_fingerprint` EXACTLY, its stored
    query-id set matches `expected_query_ids` EXACTLY (as sets -- order-independent), and its
    data parquet file exists and its footer metadata can be read (a truncated/corrupt parquet
    is treated as invalid, never silently accepted). Returns `(is_valid, reason)`."""
    meta_path = paths["qcr_meta"]
    if not meta_path.exists():
        return False, "no metadata sidecar"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "unreadable metadata sidecar"
    if meta.get("fingerprint") != expected_fingerprint:
        return False, "fingerprint mismatch"
    if sorted(str(q) for q in meta.get("query_ids", [])) != sorted(str(q) for q in expected_query_ids):
        return False, "query id set mismatch"
    if not paths["qcr"].exists():
        return False, "missing qcr shard parquet"
    try:
        pq.ParquetFile(paths["qcr"]).metadata  # footer-only read: cheap, no row data loaded
    except Exception:
        return False, "unreadable qcr shard parquet"
    return True, "valid"


def _write_meta(path, fingerprint, chunk_id, query_ids, row_count, created_at, code_ver):
    meta = {
        "fingerprint": fingerprint, "chunk_id": chunk_id, "query_ids": [str(q) for q in query_ids],
        "row_count": row_count, "created_at": created_at, "code_version": code_ver,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_peaks(peak_store_path, offsets, similarity_config):
    raw = load_rows_by_offset(peak_store_path, ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"], list(offsets))
    raw = raw.rename(columns={"_row_offset": "row_offset"})
    identity_peaks, similarity_peaks = {}, {}
    max_peaks = similarity_config["max_peaks_similarity"]
    for r in raw.itertuples(index=False):
        mzs, ints = remove_invalid_peaks(r.ms2_mzs, r.ms2_normalized_intensities)
        identity_peaks[r.row_offset] = {"mzs": mzs, "intensities": ints, "precursor_mz": r.precursor_mz}
        sim_mzs, sim_ints = truncate_top_peaks(mzs, ints, max_peaks=max_peaks)
        similarity_peaks[r.row_offset] = {"mzs": sim_mzs, "intensities": sim_ints, "precursor_mz": r.precursor_mz}
    del raw
    gc.collect()
    return identity_peaks, similarity_peaks


def compute_reference_evidence(query_offset, reference_offset, identity_peaks, similarity_peaks, similarity_config):
    """THE single implementation of `(tier, cosine, modified_cosine, peak_overlap_frac,
    neutral_loss_cosine)` for one (query, reference) pair, keyed by pre-resolved row offsets --
    used by both the chunked builder's production `classify_fn` (wrapped in `SimilarityCache`)
    and any independent brute-force audit (called directly on independently-loaded peaks,
    bypassing the cache)."""
    tier = classify_identity_tier(
        identity_peaks[query_offset], identity_peaks[reference_offset],
        tolerant_mirror_kwargs={
            "precursor_diff_da": similarity_config["tolerant_mirror_precursor_diff_da"],
            "min_relative_intensity": similarity_config["tolerant_mirror_min_rel_intensity"],
            "ppm_tol": similarity_config["tolerant_mirror_ppm_tol"],
            "abs_tol_da": similarity_config["tolerant_mirror_abs_tol_da"],
            "match_fraction": similarity_config["tolerant_mirror_match_fraction"],
            "min_correlation": similarity_config["tolerant_mirror_min_correlation"],
        },
        near_dup_cosine_threshold=similarity_config["near_dup_cosine_threshold"],
        near_dup_precursor_diff_da=similarity_config["near_dup_precursor_diff_da"],
        bin_width=similarity_config["bin_width_da"],
    )
    c, m, o, n = compute_single_pair_scores(similarity_peaks[query_offset], similarity_peaks[reference_offset],
                                             bin_width=similarity_config["bin_width_da"], peak_tol_da=similarity_config["peak_tol_da"])
    return (tier, c, m, o, n)


def _rows_from_walk(dataset_name, chunk_query_rows, walk_results, row_query_meta, is_true_by_key, ref_meta_of, protocols):
    qcr_rows, pair_rows = [], []
    for row in chunk_query_rows.itertuples(index=False):
        key = (row.query_id, row.candidate_connectivity_key)
        wr = walk_results[key]
        qmeta = row_query_meta[row.query_id]
        for r in wr.rows:
            rmeta = ref_meta_of(r["reference_spectrum_id"])
            qcr_rows.append({
                "query_dataset": dataset_name, "query_id": row.query_id, "query_source": qmeta["source"],
                "query_instrument": qmeta["instrument"], "query_adduct": qmeta["adduct"], "query_ion_mode": qmeta["ion_mode"],
                "candidate_key": row.candidate_connectivity_key, "is_true": is_true_by_key[key],
                "ref_spectrum_id": r["reference_spectrum_id"], "compat_rank": r["compat_rank"],
                "ref_source": rmeta["source"], "ref_instrument": rmeta["instrument"], "ref_adduct": rmeta["adduct"], "ref_ion_mode": rmeta["ion_mode"],
                "ref_ce": rmeta["ce"], "ref_ce_unit": rmeta["ce_unit"],
                "tier": r["tier"], "mirror": r["tier"] in ("T1", "T2"), "near_dup_non_mirror": r["tier"] == "T3",
                "cosine": r["cosine"], "modified_cosine": r["modified_cosine"], "peak_overlap_frac": r["peak_overlap_frac"], "neutral_loss_cosine": r["neutral_loss_cosine"],
                **{f"eligible_{p}": r[f"eligible_{p}"] for p in protocols}, **{f"accepted_{p}": r[f"accepted_{p}"] for p in protocols},
            })
        pair_row = {"query_id": row.query_id, "candidate_key": row.candidate_connectivity_key, "is_true": is_true_by_key[key],
                    "n_refs_total": wr.n_references_total, "n_refs_walked": wr.n_references_walked, "exhausted": wr.exhausted}
        for p in protocols:
            pair_row[f"has_ge1_{p}"] = wr.has_at_least(p, 1)
            pair_row[f"has_ge5_{p}"] = wr.has_at_least(p, MAX_REFS_PER_PROTOCOL)
            pair_row[f"walk_depth_{p}"] = wr.walk_depth[p]
        pair_rows.append(pair_row)
    return qcr_rows, pair_rows


def build_qcr_chunk(dataset_name, chunk_query_rows, chunk_id, output_dir, *, reference_index, ref_meta_of, offset_of,
                     peak_store_path, similarity_config, config_hash, protocols=PROTOCOLS, max_refs=MAX_REFS_PER_PROTOCOL,
                     fingerprint=None, force_rebuild=False):
    """Build (or skip, if a valid shard already exists) ONE chunk of queries.

    `chunk_query_rows`: the pool rows (one per (query, candidate) pair) belonging to this
    chunk's query ids only -- never the full dataset's pool. `ref_meta_of`/`offset_of`: the
    same per-spectrum metadata lookups every v4a.1 notebook shares. `fingerprint`: the
    combined `casmi.qcr.fingerprint.artifact_fingerprint(...)` for this run; `None` disables
    resume (always rebuilds). Returns a `ChunkResult`. Chunk-local state (peak dicts,
    ranked-reference lists, walk results) is freed before returning, whether the chunk was
    built or skipped."""
    t0 = time.time()
    query_ids_this_chunk = sorted(chunk_query_rows["query_id"].unique().tolist())
    paths = _shard_paths(output_dir, dataset_name, chunk_id)

    if not force_rebuild and fingerprint is not None:
        valid, reason = shard_is_valid(paths, fingerprint, query_ids_this_chunk)
        if valid:
            qcr_meta = json.loads(paths["qcr_meta"].read_text(encoding="utf-8"))
            pairs_meta = json.loads(paths["pairs_meta"].read_text(encoding="utf-8")) if paths["pairs_meta"].exists() else {}
            return ChunkResult(chunk_id=chunk_id, query_ids=query_ids_this_chunk, n_qcr_rows=qcr_meta.get("row_count", 0),
                                n_pair_rows=pairs_meta.get("row_count", 0), loaded_from_shard=True,
                                wall_seconds=time.time() - t0, peak_rss_gb=_current_rss_gb(), skip_reason=reason)

    row_ranked_refs, row_query_meta = {}, {}
    for row in chunk_query_rows.itertuples(index=False):
        qmeta = row_query_meta.get(row.query_id)
        if qmeta is None:
            qmeta = ref_meta_of(row.query_id)
            row_query_meta[row.query_id] = qmeta
        full_refs = [rid for rid in reference_index.get(row.candidate_connectivity_key, ()) if rid != row.query_id]
        row_ranked_refs[(row.query_id, row.candidate_connectivity_key)] = sorted(
            full_refs, key=lambda rid: compat_sort_key(ref_meta_of(rid), qmeta))

    needed_ref_ids = set()
    for refs in row_ranked_refs.values():
        needed_ref_ids.update(refs)
    all_offsets = set(map(offset_of, chunk_query_rows["query_id"].unique())) | set(map(offset_of, needed_ref_ids))
    identity_peaks, similarity_peaks = _load_peaks(peak_store_path, all_offsets, similarity_config)

    cache = SimilarityCache(config_hash=config_hash)

    def make_classify_fn(query_id):
        q_off = offset_of(query_id)

        def classify(rid):
            def compute():
                return compute_reference_evidence(q_off, offset_of(rid), identity_peaks, similarity_peaks, similarity_config)
            return cache.get_or_compute(dataset_name, query_id, rid, compute)
        return classify

    walk_results, is_true_by_key = {}, {}
    for row in chunk_query_rows.itertuples(index=False):
        key = (row.query_id, row.candidate_connectivity_key)
        ranked = row_ranked_refs[key]
        lookup = {rid: ref_meta_of(rid) for rid in ranked}
        walk_results[key] = walk_references(ranked, row_query_meta[row.query_id], lookup, make_classify_fn(row.query_id),
                                             protocols=protocols, max_refs=max_refs)
        is_true_by_key[key] = bool(row.is_true_candidate)

    qcr_rows, pair_rows = _rows_from_walk(dataset_name, chunk_query_rows, walk_results, row_query_meta, is_true_by_key, ref_meta_of, protocols)
    qcr_df, pairs_df = pd.DataFrame(qcr_rows), pd.DataFrame(pair_rows)
    qcr_df.to_parquet(paths["qcr"], index=False)
    pairs_df.to_parquet(paths["pairs"], index=False)
    cache.save(paths["cache"])

    now, code_ver = _now_iso(), code_version()
    _write_meta(paths["qcr_meta"], fingerprint, chunk_id, query_ids_this_chunk, len(qcr_df), now, code_ver)
    _write_meta(paths["pairs_meta"], fingerprint, chunk_id, query_ids_this_chunk, len(pairs_df), now, code_ver)
    _write_meta(paths["cache_meta"], fingerprint, chunk_id, query_ids_this_chunk, len(cache), now, code_ver)

    result = ChunkResult(chunk_id=chunk_id, query_ids=query_ids_this_chunk, n_qcr_rows=len(qcr_df), n_pair_rows=len(pairs_df),
                          loaded_from_shard=False, wall_seconds=time.time() - t0, peak_rss_gb=_current_rss_gb(),
                          n_similarity_computed=cache.n_computed, n_similarity_hits=cache.n_hits)

    del row_ranked_refs, row_query_meta, identity_peaks, similarity_peaks, cache, walk_results, is_true_by_key, qcr_df, pairs_df
    gc.collect()
    return result


def build_qcr_resumable(dataset_name, pool_df, output_dir, *, reference_index, ref_meta_of, offset_of, peak_store_path,
                         similarity_config, config_hash, fingerprint, chunk_size=50, n_jobs=1,
                         protocols=PROTOCOLS, max_refs=MAX_REFS_PER_PROTOCOL, force_rebuild=False, timing_log_path=None):
    """Top-level orchestrator: splits `pool_df`'s query ids into chunks of `chunk_size`, builds
    (or resumes) each chunk via `build_qcr_chunk`, and appends one timing/RSS log entry per
    chunk to `timing_log_path` (default `<output_dir>/timing.json`). `n_jobs > 1` builds
    independent chunks concurrently via a thread pool (chunks write to disjoint shard files, so
    this is safe) -- use `estimate_max_workers` to pick a value that respects a RAM budget.
    Returns a list of `ChunkResult`, one per chunk, in chunk-id order."""
    query_ids = sorted(pool_df["query_id"].unique().tolist())
    chunks = chunk_query_ids(query_ids, chunk_size=chunk_size)
    pool_by_query = {qid: g for qid, g in pool_df.groupby("query_id")}

    def _run_one(chunk_id_and_ids):
        chunk_id, ids_in_chunk = chunk_id_and_ids
        chunk_rows = pd.concat([pool_by_query[qid] for qid in ids_in_chunk], ignore_index=True)
        return build_qcr_chunk(dataset_name, chunk_rows, chunk_id, output_dir, reference_index=reference_index,
                                ref_meta_of=ref_meta_of, offset_of=offset_of, peak_store_path=peak_store_path,
                                similarity_config=similarity_config, config_hash=config_hash, protocols=protocols,
                                max_refs=max_refs, fingerprint=fingerprint, force_rebuild=force_rebuild)

    indexed_chunks = list(enumerate(chunks))
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_run_one, indexed_chunks))
    else:
        results = [_run_one(ic) for ic in indexed_chunks]

    timing_log_path = Path(timing_log_path) if timing_log_path is not None else Path(output_dir) / "timing.json"
    _append_timing(timing_log_path, dataset_name, results)
    return results


def _append_timing(timing_log_path, dataset_name, chunk_results):
    log = {}
    if timing_log_path.exists():
        with open(timing_log_path, encoding="utf-8") as f:
            log = json.load(f)
    log.setdefault(dataset_name, [])
    for r in chunk_results:
        log[dataset_name].append({
            "chunk_id": r.chunk_id, "n_queries": len(r.query_ids), "n_qcr_rows": r.n_qcr_rows, "n_pair_rows": r.n_pair_rows,
            "similarity_hits": r.n_similarity_hits, "similarity_computations": r.n_similarity_computed,
            "wall_seconds": r.wall_seconds, "peak_rss_gb": r.peak_rss_gb, "loaded_from_shard": r.loaded_from_shard,
            "recorded_at": _now_iso(),
        })
    timing_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def load_qcr_shards(output_dir, dataset_name, kind="qcr"):
    """Warm-path loader: concatenates every existing shard parquet for `dataset_name` under
    `<output_dir>/<kind>/<dataset_name>/part-*.parquet` (`kind` in `{"qcr", "qcr_pairs"}`).
    Performs ZERO peak loading, ZERO reference walk, ZERO new similarity computation -- pure
    parquet reads, matching spec section 17's warm-path contract. Returns an empty DataFrame
    (not an error) if no shards exist yet."""
    shard_dir = Path(output_dir) / kind / dataset_name
    if not shard_dir.exists():
        return pd.DataFrame()
    parts = sorted(shard_dir.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
