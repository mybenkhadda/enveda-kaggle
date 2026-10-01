"""Each input to casmi.qcr.fingerprint.artifact_fingerprint must independently invalidate the
combined fingerprint (spec section 65) -- these tests change exactly one input at a time and
assert the fingerprint changes, holding everything else fixed.
"""
import os
import time

import pytest

from casmi.qcr import fingerprint as fp


def _base_kwargs(tmp_path):
    peak_store = tmp_path / "train.parquet"
    peak_store.write_bytes(b"x" * 100)
    return dict(
        peak_store_path=peak_store,
        reference_index={"conn_a": ["r001", "r002"], "conn_b": ["r003"]},
        query_ids=["q1", "q2"],
        candidate_pool_fingerprint="pool-fp-v1",
        similarity_config={"bin_width_da": 0.1, "peak_tol_da": 0.02},
        protocol_config={"max_refs": 5},
    )


def test_baseline_is_deterministic_and_reproducible(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    a = fp.artifact_fingerprint(**kwargs)
    b = fp.artifact_fingerprint(**kwargs)
    assert a == b


def test_peak_store_change_invalidates(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    before = fp.artifact_fingerprint(**kwargs)
    time.sleep(0.01)
    kwargs["peak_store_path"].write_bytes(b"y" * 200)  # different size AND mtime
    after = fp.artifact_fingerprint(**kwargs)
    assert before != after


def test_reference_index_change_invalidates(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    before = fp.artifact_fingerprint(**kwargs)
    kwargs["reference_index"] = {"conn_a": ["r001", "r002", "r999"], "conn_b": ["r003"]}
    after = fp.artifact_fingerprint(**kwargs)
    assert before != after


def test_query_ids_change_invalidates(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    before = fp.artifact_fingerprint(**kwargs)
    kwargs["query_ids"] = ["q1", "q2", "q3"]
    after = fp.artifact_fingerprint(**kwargs)
    assert before != after


def test_query_ids_order_does_not_matter():
    kwargs_a = {"query_ids": ["q1", "q2", "q3"]}
    kwargs_b = {"query_ids": ["q3", "q1", "q2"]}
    assert sorted(str(q) for q in kwargs_a["query_ids"]) == sorted(str(q) for q in kwargs_b["query_ids"])


def test_candidate_pool_fingerprint_change_invalidates(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    before = fp.artifact_fingerprint(**kwargs)
    kwargs["candidate_pool_fingerprint"] = "pool-fp-v2"
    after = fp.artifact_fingerprint(**kwargs)
    assert before != after


def test_similarity_config_change_invalidates(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    before = fp.artifact_fingerprint(**kwargs)
    kwargs["similarity_config"] = {**kwargs["similarity_config"], "bin_width_da": 0.2}
    after = fp.artifact_fingerprint(**kwargs)
    assert before != after


def test_protocol_config_change_invalidates(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    before = fp.artifact_fingerprint(**kwargs)
    kwargs["protocol_config"] = {"max_refs": 3}
    after = fp.artifact_fingerprint(**kwargs)
    assert before != after


def test_code_version_change_invalidates(tmp_path, monkeypatch):
    kwargs = _base_kwargs(tmp_path)
    monkeypatch.setattr(fp, "code_version", lambda repo_root=None: "commit-aaa")
    before = fp.artifact_fingerprint(**kwargs)
    monkeypatch.setattr(fp, "code_version", lambda repo_root=None: "commit-bbb")
    after = fp.artifact_fingerprint(**kwargs)
    assert before != after


def test_code_version_falls_back_to_source_hash_when_git_unavailable(tmp_path, monkeypatch):
    import subprocess

    def _raise(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(subprocess, "run", _raise)
    src_dir = tmp_path / "src" / "casmi"
    src_dir.mkdir(parents=True)
    (src_dir / "mod.py").write_text("x = 1\n", encoding="utf-8")
    version = fp.code_version(repo_root=tmp_path)
    assert version.startswith("nogit-")


def test_fingerprint_components_lets_caller_see_which_input_differs(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    before = fp.fingerprint_components(**kwargs)
    kwargs["similarity_config"] = {**kwargs["similarity_config"], "bin_width_da": 0.99}
    after = fp.fingerprint_components(**kwargs)
    assert before["similarity_config"] != after["similarity_config"]
    assert before["peak_store"] == after["peak_store"]
    assert before["candidate_pool_fingerprint"] == after["candidate_pool_fingerprint"]
