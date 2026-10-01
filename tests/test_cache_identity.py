"""Identity-namespaced caches: stale shards are never resumed after query / regime / hidden-set / config changes."""
import json

import pandas as pd
import pytest

from casmi.analog.pipeline import feature_identity, load_features, neighbor_identity
from casmi.workspace.cache_identity import (StaleCacheError, fingerprint_json, fingerprint_pairs, fingerprint_values, identity_hash, open_namespace,
                                            read_identity, reference_library_identity)

CFG = {"bin_width_da": 0.1, "peak_tol_da": 0.02, "candidate_ppm": 10.0, "top_k_analogs": 10}


def _q(ids, regimes=None):
    return pd.DataFrame({"query_id": ids, "regime": regimes or ["C1"] * len(ids), "fold": 0})


def _hs(hidden=(), removed=()):
    return {"hidden_reference_keys": set(hidden), "removed_structure_keys": set(removed)}


def test_fingerprints_are_order_free_and_sensitive():
    assert fingerprint_values(["b", "a"]) == fingerprint_values(["a", "b"]) != fingerprint_values(["a", "c"])
    assert fingerprint_pairs([("q1", "C1"), ("q2", "C2")]) == fingerprint_pairs([("q2", "C2"), ("q1", "C1")])
    assert fingerprint_pairs([("q1", "C1")]) != fingerprint_pairs([("q1", "C2")])
    assert fingerprint_json({"a": 1, "b": 2}) == fingerprint_json({"b": 2, "a": 1})


def test_same_identity_resumes_different_identity_gets_new_namespace(tmp_path):
    ident = {"fold": 0, "query_ids": fingerprint_values(["q1"])}
    d1 = open_namespace(tmp_path, ident, kind="analog_features")
    (d1 / "part-00000.parquet").write_bytes(b"x")
    assert open_namespace(tmp_path, ident, kind="analog_features") == d1                  # resume
    d2 = open_namespace(tmp_path, {**ident, "query_ids": fingerprint_values(["q1", "q2"])}, kind="analog_features")
    assert d2 != d1 and not any(d2.glob("part-*"))                                        # never resumes the old shards
    assert read_identity(d1)["fold"] == 0 and read_identity(d1)["kind"] == "analog_features"


def test_tampered_or_legacy_namespace_is_refused(tmp_path):
    ident = {"fold": 1}
    d = open_namespace(tmp_path, ident)
    (d / "identity.json").write_text(json.dumps({"fold": 999}))
    with pytest.raises(StaleCacheError, match="identity mismatch"):
        open_namespace(tmp_path, ident)
    legacy = tmp_path / f"ns-{identity_hash({'fold': 2})}"
    legacy.mkdir()
    (legacy / "part-00000.parquet").write_bytes(b"x")
    with pytest.raises(StaleCacheError, match="no identity.json"):
        open_namespace(tmp_path, {"fold": 2})


def test_regime_change_changes_both_identities_universe_change_only_features():
    q = _q(["q1", "q2"])
    base_nb = neighbor_identity(0, q, _hs({"K1"}), CFG, {"reference_library": {"bundle_version": "v2-A7"}})
    base_ft = feature_identity(0, q, _hs({"K1"}), CFG, 1000, {"reference_library": {"bundle_version": "v2-A7"}, "universe": {"n": 1}})
    # a truth becomes hidden after a regime rebuild -> different neighbor AND feature namespaces (no stale hiding masks)
    assert neighbor_identity(0, q, _hs({"K1", "K2"}), CFG, {"reference_library": {"bundle_version": "v2-A7"}}) != base_nb
    assert feature_identity(0, q, _hs({"K1", "K2"}), CFG, 1000, {"universe": {"n": 1}}) != base_ft
    # a new universe re-keys the features but not the (expensive) neighbor search
    assert neighbor_identity(0, q, _hs({"K1"}), CFG, {"reference_library": {"bundle_version": "v2-A7"}, "universe": {"n": 2}}) == base_nb
    assert feature_identity(0, q, _hs({"K1"}), CFG, 1000, {"reference_library": {"bundle_version": "v2-A7"}, "universe": {"n": 2}}) != base_ft
    # same queries, different regime labels -> different feature namespace; different chunk size -> different too
    assert feature_identity(0, _q(["q1", "q2"], ["C1", "C2"]), _hs({"K1"}), CFG, 1000) != feature_identity(0, q, _hs({"K1"}), CFG, 1000)
    assert feature_identity(0, q, _hs({"K1"}), CFG, 500) != feature_identity(0, q, _hs({"K1"}), CFG, 1000)


def test_load_features_reads_only_completed_shards_of_given_namespaces(tmp_path):
    d = open_namespace(tmp_path / "fold=0", {"fold": 0})
    pd.DataFrame({"query_id": ["q1"], "x": [1]}).to_parquet(d / "part-00000.parquet", index=False)
    (d / "part-00000.done").write_text("{}")
    pd.DataFrame({"query_id": ["q2"], "x": [2]}).to_parquet(d / "part-00001.parquet", index=False)   # no .done -> ignored
    stale = open_namespace(tmp_path / "fold=0", {"fold": 0, "old": True})
    pd.DataFrame({"query_id": ["STALE"], "x": [9]}).to_parquet(stale / "part-00000.parquet", index=False)
    (stale / "part-00000.done").write_text("{}")
    f = load_features([d])
    assert f["query_id"].tolist() == ["q1"]
    with pytest.raises(StaleCacheError):
        load_features([tmp_path / "fold=0"])                 # the base directory is not a namespace


def test_reference_library_identity_uses_labels_and_size_only(tmp_path):
    meta = tmp_path / "ref_meta.parquet"
    meta.write_bytes(b"12345")
    ident = reference_library_identity({"bundle_version": "v2-A7", "CONFIG_HASH": "60174e39a2a3c6b4"}, meta)
    assert ident == {"bundle_version": "v2-A7", "CONFIG_HASH": "60174e39a2a3c6b4", "ref_meta_bytes": 5}
