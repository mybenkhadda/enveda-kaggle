"""C1/C2/C3 regime builder: determinism, leakage, masking."""
import numpy as np
import pandas as pd
import pytest

from casmi.validation.regimes import (RegimeConfig, assert_regime_integrity, build_regime_table, hidden_sets, isin_sorted,
                                      reference_allowed_mask, removed_candidate_mask, stable_unit_hash, truth_eligibility)


def _toy(n_conn=60, n_folds=3, seed=0):
    rng = np.random.default_rng(seed)
    keys = [f"K{i:013d}" for i in range(n_conn)]
    fold_of = {k: i % n_folds for i, k in enumerate(keys)}
    n_spec = {k: int(rng.integers(1, 5)) for k in keys}
    ref_conn = np.array([k for k in keys for _ in range(n_spec[k])])
    queries = pd.DataFrame({"query_id": [f"q{i}" for i in range(len(ref_conn))], "true_connectivity_key": ref_conn,
                            "fold": [fold_of[k] for k in ref_conn]})
    folds = pd.DataFrame({"connectivity_key": keys, "fold": [fold_of[k] for k in keys]})
    # universe: every train key + some external-only keys; half the train keys also in COCONUT
    unified = pd.DataFrame({"connectivity_key": keys + [f"X{i:013d}" for i in range(20)],
                            "candidate_sources": [["COCONUT", "TRAIN"] if i % 2 == 0 else ["TRAIN"] for i in range(n_conn)] + [["COCONUT"]] * 20})
    return queries, ref_conn, folds, unified


def test_stable_hash_is_deterministic_and_seeded():
    a = stable_unit_hash(["x", "y"], 1)
    assert np.array_equal(a, stable_unit_hash(["x", "y"], 1))
    assert not np.array_equal(a, stable_unit_hash(["x", "y"], 2))
    assert ((a >= 0) & (a < 1)).all()


def test_regimes_are_reproducible_and_per_connectivity():
    q, ref, folds, uni = _toy()
    cfg = RegimeConfig(split_seed=7)
    r1 = build_regime_table(q, ref, uni, cfg)
    r2 = build_regime_table(q.sample(frac=1, random_state=1), ref, uni, cfg)
    pd.testing.assert_frame_equal(r1, r2)
    assert (r1.groupby("true_connectivity_key")["regime"].nunique(dropna=False) == 1).all()


def test_eligibility_rules():
    q, ref, folds, uni = _toy()
    e = truth_eligibility(q["true_connectivity_key"], ref, uni, RegimeConfig()).set_index("true_connectivity_key")
    counts = pd.Series(ref).value_counts()
    assert (e["eligible_C1"] == (counts.reindex(e.index) >= 2)).all()
    train_only = [k for k, s in zip(uni.connectivity_key, uni.candidate_sources) if s == ["TRAIN"]]
    assert not e.loc[e.index.isin(train_only), "eligible_C2"].any()      # C2 needs a non-TRAIN source
    e2 = truth_eligibility(q["true_connectivity_key"], ref, None, RegimeConfig())
    assert not e2["eligible_C2"].any() and (e2["c2_ineligible_reason"] == "no_external_universe").all()


def test_masks_and_integrity_pass():
    q, ref, folds, uni = _toy()
    cfg = RegimeConfig(split_seed=3)
    reg = build_regime_table(q, ref, uni, cfg)
    ukeys = np.sort(uni["connectivity_key"].to_numpy())
    for f in sorted(reg["fold"].unique()):
        t = assert_regime_integrity(reg, folds, ref, ukeys, f, cfg)
        assert t["passed"].all()
        hs = hidden_sets(reg, f)
        allowed = reference_allowed_mask(ref, hs["hidden_reference_keys"])
        c2 = set(reg[(reg.fold == f) & (reg.regime == "C2")].true_connectivity_key)
        c3 = set(reg[(reg.fold == f) & (reg.regime == "C3")].true_connectivity_key)
        assert not (set(ref[allowed]) & (c2 | c3))                       # C2 + C3 truth spectra removed
        removed = removed_candidate_mask(ukeys, hs["removed_structure_keys"])
        view = set(ukeys[~removed])
        assert c2 <= view                                                # C2 truth structure preserved
        assert not (c3 & view)                                           # C3 truth structure removed


def test_integrity_detects_fold_leakage():
    q, ref, folds, uni = _toy()
    cfg = RegimeConfig(split_seed=3)
    reg = build_regime_table(q, ref, uni, cfg)
    bad_folds = folds.copy()
    k = reg.loc[reg.fold == 0, "true_connectivity_key"].iloc[0]
    bad_folds = pd.concat([bad_folds, pd.DataFrame({"connectivity_key": [k], "fold": [1]})])   # same key in a training fold
    with pytest.raises(AssertionError):
        assert_regime_integrity(reg, bad_folds, ref, np.sort(uni.connectivity_key.to_numpy()), 0, cfg)


def test_queries_spanning_folds_are_rejected():
    q, ref, folds, uni = _toy()
    q.loc[0, "fold"] = (q.loc[0, "fold"] + 1) % 3
    q.loc[1, "true_connectivity_key"] = q.loc[0, "true_connectivity_key"]
    q.loc[1, "fold"] = (q.loc[0, "fold"] + 1) % 3
    with pytest.raises(AssertionError):
        build_regime_table(q, ref, uni, RegimeConfig())


def test_isin_sorted_and_removed_mask():
    keys = np.array(["A", "C", "E"])
    assert isin_sorted(keys, ["A", "B", "E", "Z"]).tolist() == [True, False, True, False]
    assert removed_candidate_mask(keys, {"C", "Q"}).tolist() == [False, True, False]


def test_only_fold_scope_supported():
    with pytest.raises(ValueError):
        RegimeConfig(hidden_scope="query")
