"""Chunk resume test (spec section 64): build chunk 0, build chunk 1, simulate an interrupt
(nothing else done), then restart the full build -- the second run must reuse chunks 0/1
untouched (no peak reload, no new similarity computation) and produce a logically-equivalent
final concatenated table to a from-scratch build.
"""
import numpy as np
import pandas as pd
import pytest

from casmi.qcr.builder import build_qcr_chunk, build_qcr_resumable, chunk_query_ids, load_qcr_shards, shard_is_valid

SIMILARITY_CONFIG = {
    "bin_width_da": 0.1, "peak_tol_da": 0.02, "max_peaks_similarity": 100,
    "near_dup_cosine_threshold": 0.95, "near_dup_precursor_diff_da": 0.01,
    "tolerant_mirror_precursor_diff_da": 0.005, "tolerant_mirror_min_rel_intensity": 0.01,
    "tolerant_mirror_ppm_tol": 5.0, "tolerant_mirror_abs_tol_da": 0.002,
    "tolerant_mirror_match_fraction": 0.95, "tolerant_mirror_min_correlation": 0.99,
}


def _write_synthetic_peak_store(path, n_rows=16):
    rows = []
    for i in range(n_rows):
        drift = i * 0.05
        rows.append({
            "ms2_mzs": [100.0 + drift, 150.0, 200.0 - drift],
            "ms2_normalized_intensities": [1.0, 0.6, 0.3],
            "precursor_mz": 300.15,
        })
    pd.DataFrame(rows).to_parquet(path, index=False)


def _offset_of(spectrum_id):
    return int(spectrum_id.split("_")[1])


def _ref_meta_of(spectrum_id):
    return {"spectrum_id": spectrum_id, "adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV",
            "instrument": "Orbitrap", "source": "lib_b"}


def _pool_df(n_queries=4):
    return pd.DataFrame([
        {"query_id": f"q_{i}", "candidate_connectivity_key": "c0", "abs_mass_error_ppm": 1.0, "is_true_candidate": (i % 2 == 0)}
        for i in range(n_queries)
    ])


def _setup(tmp_path, n_queries=4):
    peak_store = tmp_path / "train.parquet"
    _write_synthetic_peak_store(peak_store, n_rows=16)
    reference_index = {"c0": ["r_10", "r_11", "r_12"]}
    return dict(
        dataset_name="host", pool_df=_pool_df(n_queries), output_dir=tmp_path / "outputs",
        reference_index=reference_index, ref_meta_of=_ref_meta_of, offset_of=_offset_of,
        peak_store_path=peak_store, similarity_config=SIMILARITY_CONFIG, config_hash="test-cfg",
        fingerprint="fp-v1", chunk_size=1,
    )


def test_second_run_reuses_completed_chunks_without_recomputation(tmp_path):
    kwargs = _setup(tmp_path, n_queries=4)

    # first run: chunks 0 and 1 "complete", simulate an interrupt before 2/3 ever start
    query_ids = sorted(kwargs["pool_df"]["query_id"].unique().tolist())
    chunks = chunk_query_ids(query_ids, chunk_size=1)
    pool_by_query = {qid: g for qid, g in kwargs["pool_df"].groupby("query_id")}

    completed = []
    for chunk_id, ids in list(enumerate(chunks))[:2]:
        rows = pd.concat([pool_by_query[q] for q in ids], ignore_index=True)
        result = build_qcr_chunk("host", rows, chunk_id, kwargs["output_dir"], reference_index=kwargs["reference_index"],
                                  ref_meta_of=kwargs["ref_meta_of"], offset_of=kwargs["offset_of"], peak_store_path=kwargs["peak_store_path"],
                                  similarity_config=kwargs["similarity_config"], config_hash=kwargs["config_hash"], fingerprint=kwargs["fingerprint"])
        completed.append(result)
        assert not result.loaded_from_shard  # first time building these two: must be a real build

    # restart: full resumable build over ALL 4 chunks
    results = build_qcr_resumable(**kwargs)
    assert len(results) == 4
    assert results[0].loaded_from_shard is True   # chunk 0 reused
    assert results[1].loaded_from_shard is True   # chunk 1 reused
    assert results[2].loaded_from_shard is False  # chunk 2 built for the first time now
    assert results[3].loaded_from_shard is False
    # reused chunks did zero new similarity computation
    assert results[0].n_similarity_computed == 0
    assert results[1].n_similarity_computed == 0


def test_resumed_build_produces_logically_equivalent_final_table_to_from_scratch_build(tmp_path):
    kwargs_resumed = _setup(tmp_path / "resumed", n_queries=4)
    kwargs_resumed["output_dir"].mkdir(parents=True, exist_ok=True)
    query_ids = sorted(kwargs_resumed["pool_df"]["query_id"].unique().tolist())
    chunks = chunk_query_ids(query_ids, chunk_size=1)
    pool_by_query = {qid: g for qid, g in kwargs_resumed["pool_df"].groupby("query_id")}
    for chunk_id, ids in list(enumerate(chunks))[:2]:
        rows = pd.concat([pool_by_query[q] for q in ids], ignore_index=True)
        build_qcr_chunk("host", rows, chunk_id, kwargs_resumed["output_dir"], reference_index=kwargs_resumed["reference_index"],
                         ref_meta_of=kwargs_resumed["ref_meta_of"], offset_of=kwargs_resumed["offset_of"],
                         peak_store_path=kwargs_resumed["peak_store_path"], similarity_config=kwargs_resumed["similarity_config"],
                         config_hash=kwargs_resumed["config_hash"], fingerprint=kwargs_resumed["fingerprint"])
    build_qcr_resumable(**kwargs_resumed)
    resumed_table = load_qcr_shards(kwargs_resumed["output_dir"], "host")

    kwargs_fresh = _setup(tmp_path / "fresh", n_queries=4)
    kwargs_fresh["output_dir"].mkdir(parents=True, exist_ok=True)
    build_qcr_resumable(**kwargs_fresh)
    fresh_table = load_qcr_shards(kwargs_fresh["output_dir"], "host")

    sort_cols = ["query_id", "candidate_key", "ref_spectrum_id"]
    resumed_sorted = resumed_table.sort_values(sort_cols).reset_index(drop=True)
    fresh_sorted = fresh_table.sort_values(sort_cols).reset_index(drop=True)
    assert len(resumed_sorted) == len(fresh_sorted)
    assert list(resumed_sorted["query_id"]) == list(fresh_sorted["query_id"])
    assert np.allclose(resumed_sorted["cosine"], fresh_sorted["cosine"], atol=1e-12)


def test_interrupted_run_leaves_no_partial_shard_for_unstarted_chunks(tmp_path):
    kwargs = _setup(tmp_path, n_queries=4)
    query_ids = sorted(kwargs["pool_df"]["query_id"].unique().tolist())
    chunks = chunk_query_ids(query_ids, chunk_size=1)
    pool_by_query = {qid: g for qid, g in kwargs["pool_df"].groupby("query_id")}
    rows0 = pool_by_query[chunks[0][0]]
    build_qcr_chunk("host", rows0, 0, kwargs["output_dir"], reference_index=kwargs["reference_index"],
                     ref_meta_of=kwargs["ref_meta_of"], offset_of=kwargs["offset_of"], peak_store_path=kwargs["peak_store_path"],
                     similarity_config=kwargs["similarity_config"], config_hash=kwargs["config_hash"], fingerprint=kwargs["fingerprint"])
    qcr_dir = kwargs["output_dir"] / "qcr" / "host"
    assert (qcr_dir / "part-00000.parquet").exists()
    assert not (qcr_dir / "part-00001.parquet").exists()


def test_fingerprint_mismatch_forces_rebuild_even_if_shard_exists(tmp_path):
    kwargs = _setup(tmp_path, n_queries=1)
    query_ids = sorted(kwargs["pool_df"]["query_id"].unique().tolist())
    chunks = chunk_query_ids(query_ids, chunk_size=1)
    pool_by_query = {qid: g for qid, g in kwargs["pool_df"].groupby("query_id")}
    rows0 = pool_by_query[chunks[0][0]]
    build_qcr_chunk("host", rows0, 0, kwargs["output_dir"], reference_index=kwargs["reference_index"],
                     ref_meta_of=kwargs["ref_meta_of"], offset_of=kwargs["offset_of"], peak_store_path=kwargs["peak_store_path"],
                     similarity_config=kwargs["similarity_config"], config_hash=kwargs["config_hash"], fingerprint="fp-v1")

    result = build_qcr_chunk("host", rows0, 0, kwargs["output_dir"], reference_index=kwargs["reference_index"],
                              ref_meta_of=kwargs["ref_meta_of"], offset_of=kwargs["offset_of"], peak_store_path=kwargs["peak_store_path"],
                              similarity_config=kwargs["similarity_config"], config_hash=kwargs["config_hash"], fingerprint="fp-v2-different")
    assert result.loaded_from_shard is False  # fingerprint changed -> must rebuild, not silently reuse stale shard
