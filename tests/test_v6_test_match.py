"""v6 test-match diagnostic: the statistic is label-free and defined once, TEST and TL use the same
similarity implementation, strict / mirror differ only by eligibility, distributions / KS / reweighting
share one weighted definition and read only observable columns, the resemblance rule takes label-free
distances only, molecule aggregation is deterministic; realism metadata is additive."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi.validation import test_match as tm
from casmi.validation import realism

ROOT = Path(__file__).resolve().parents[1]


# ---- fixtures ------------------------------------------------------------------------------------

def _qcr():
    """q1 (source libA): candidate A refs a1 T1/libB .99, a2 T4/libA .90, a3 T4/libB .50;
    candidate B refs b1 T2/libB .97, b2 T3/libA .80. q2: no references at all."""
    rows = [("q1", "A", "a1", 1, "T1", "libB", .99), ("q1", "A", "a2", 2, "T4", "libA", .90), ("q1", "A", "a3", 3, "T4", "libB", .50),
            ("q1", "B", "b1", 1, "T2", "libB", .97), ("q1", "B", "b2", 2, "T3", "libA", .80)]
    q = pd.DataFrame(rows, columns=["query_id", "candidate_key", "ref_spectrum_id", "compat_rank", "tier", "ref_source", "cosine"])
    q["query_source"] = "libA"
    q["modified_cosine"] = q["cosine"] - 0.05
    q["peak_overlap_frac"] = q["cosine"] / 2
    q["neutral_loss_cosine"] = 0.1
    q["is_true"] = q["candidate_key"] == "A"                     # present in the shard, must never be read
    q["eligible_mirror_aware"] = q["tier"].isin(["T3", "T4"]) & (q["ref_source"] != q["query_source"])
    q["eligible_standard"] = True
    return q


def _pool():
    return pd.DataFrame({"query_id": ["q1", "q1", "q2"], "candidate_connectivity_key": ["A", "B", "C"], "abs_mass_error_ppm": [3.0, 1.0, 2.0]})


def _meta():
    return pd.DataFrame({"query_id": ["q1", "q2"], "adduct": "[M+H]+", "polarity": "positive", "precursor_mz": [301.0, 401.0],
                         "neutral_mass": [300.0, 400.0], "n_query_peaks": [10, 8]})


def _ref_meta():
    ids = ["a1", "a2", "a3", "b1", "b2"]
    return pd.DataFrame({"ref_spectrum_id": ids, "source": ["libB", "libA", "libB", "libB", "libA"], "instrument": "qtof", "adduct": "[M+H]+",
                         "collision_energy": 20.0, "precursor_mz": [301.0, 301.005, 350.0, 301.0, 301.2]})


def _write_shards(tmp_path):
    qp, pp = tmp_path / "qcr.parquet", tmp_path / "pairs.parquet"
    _qcr().to_parquet(qp, index=False)
    pd.DataFrame({"query_id": ["q1", "q1"], "candidate_key": ["A", "B"], "exhausted": [True, True]}).to_parquet(pp, index=False)
    return [qp], [pp]


def _stats(tmp_path, protocol):
    qp, pp = _write_shards(tmp_path)
    pairs = tm.qcr_protocol_pairs(qp, protocol, pool=_pool(), pair_paths=pp)
    return pairs, tm.finalize_query_stats(tm.query_stats_from_pairs(pairs, _pool(), _meta()), "TL_EVAL", protocol, _ref_meta())


# ---- label-freeness ------------------------------------------------------------------------------

def test_statistic_refuses_label_columns():
    pairs = _qcr()[tm.PAIR_COLS]
    with pytest.raises(tm.LabelLeakError):
        tm.query_stats_from_pairs(pairs.assign(is_true=True), _pool(), _meta())
    with pytest.raises(tm.LabelLeakError):
        tm.query_stats_from_pairs(pairs, _pool().assign(is_true_candidate=False), _meta())
    with pytest.raises(tm.LabelLeakError):
        tm.query_stats_from_pairs(pairs, _pool(), _meta().assign(true_connectivity_key="A"))


def test_label_free_read_lists_and_output(tmp_path):
    for cols in (tm.QCR_LABEL_FREE_COLS, tm.POOL_LABEL_FREE_COLS, tm.PAIR_COLS, tm.QUERY_STATS_COLUMNS):
        assert not (set(cols) & tm.LABEL_COLUMNS), cols
    pairs, s = _stats(tmp_path, tm.STRICT)
    assert "is_true" not in pairs.columns and list(s.columns) == tm.QUERY_STATS_COLUMNS
    tm.assert_label_free(s, "output")


def test_test_path_never_touches_truth():
    for fn in (tm.observed_test_query, tm.run_observed_test, tm._bundle_pairs, tm.library_query_meta, tm.qcr_protocol_pairs, tm.read_label_free_pool):
        src = inspect.getsource(fn)
        assert "is_true" not in src and "true_connectivity" not in src, fn.__name__
    nb = json.loads((ROOT / "src" / "10v6_00_test_match_diagnostic.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "is_true" not in code and "true_connectivity_key" not in code
    assert "mrr" not in code.lower() and "hit_at" not in code.lower()


# ---- the statistic ---------------------------------------------------------------------------------

def test_statistic_values_and_zero_reference_rule(tmp_path):
    _, s = _stats(tmp_path, tm.STRICT)
    s = s.set_index("query_id")
    # strict: a2 (.90), a3 (.50), b2 (.80) eligible -> best a2 on candidate A
    assert s.loc["q1", "max_cosine"] == pytest.approx(.90) and s.loc["q1", "best_reference_id"] == "a2"
    assert s.loc["q1", "best_candidate_connectivity"] == "A" and s.loc["q1", "best_candidate_mass_rank"] == 2      # B has the smaller ppm
    assert s.loc["q1", "matched_peaks_at_max_cosine"] == round(.45 * 10) and s.loc["q1", "eligible_reference_count"] == 3
    assert s.loc["q1", "near_duplicate_proxy"] == bool(False)                                                     # .90 < .95
    # q2: empty eligible set -> pre-registered 0.0, flagged
    assert s.loc["q2", "max_cosine"] == tm.NO_REFERENCE_VALUE and not s.loc["q2", "has_eligible_reference"] and s.loc["q2", "candidate_pool_size"] == 1


def test_strict_and_mirror_differ_only_by_eligibility(tmp_path):
    ps, s = _stats(tmp_path, tm.STRICT)
    pm, m = _stats(tmp_path, tm.MIRROR)
    pst, st = _stats(tmp_path, tm.STANDARD)
    assert set(pm["ref_spectrum_id"]) == {"a3"} and set(ps["ref_spectrum_id"]) == {"a2", "a3", "b2"}
    assert set(pst["ref_spectrum_id"]) == {"a1", "a2", "a3", "b1", "b2"}
    # identical values on shared pairs: same rows, same similarity implementation, only the filter differs
    j = ps.merge(pm, on=["query_id", "candidate_key", "ref_spectrum_id"], suffixes=("_s", "_m"))
    assert (j["cosine_s"] == j["cosine_m"]).all() and (j["modified_cosine_s"] == j["modified_cosine_m"]).all()
    assert m.set_index("query_id").loc["q1", "max_cosine"] == pytest.approx(.50)
    assert st.set_index("query_id").loc["q1", "max_cosine"] == pytest.approx(.99)
    assert (s["candidate_pool_size"].to_numpy() == m["candidate_pool_size"].to_numpy()).all()


def test_tie_rule_is_deterministic():
    pairs = pd.DataFrame({"query_id": "q", "candidate_key": ["B", "A", "A"], "ref_spectrum_id": ["r3", "r2", "r1"], "compat_rank": [1, 2, 1],
                          "cosine": [.7, .7, .7], "modified_cosine": .1, "peak_overlap_frac": .5})
    pool = pd.DataFrame({"query_id": "q", "candidate_key": ["A", "B"], "abs_mass_error_ppm": [1.0, 1.0]})
    meta = pd.DataFrame({"query_id": ["q"], "n_query_peaks": [4]})
    outs = [tm.query_stats_from_pairs(pairs.sample(frac=1, random_state=k), pool, meta).iloc[0] for k in range(5)]
    assert all(o["best_candidate_connectivity"] == "A" and o["best_reference_id"] == "r1" for o in outs)


def test_same_similarity_implementation_for_test_and_tl():
    src = inspect.getsource(tm._bundle_pairs)
    assert "casmi_infer.similarity import pair_scores" in src and "casmi_infer.compat import select_references" in src
    assert "use_backend(NUMPY)" in inspect.getsource(tm.run_observed_test)            # numpy == compute_single_pair_scores
    from casmi.qcr import builder
    assert "compute_single_pair_scores" in inspect.getsource(builder)                   # the QCR's kernel
    from casmi_infer import similarity
    assert "compute_single_pair_scores(query_sim_peaks, ref_sim_peaks" in inspect.getsource(similarity.pair_scores)
    obs = inspect.getsource(tm.observed_test_query)
    assert "frozenset()" not in obs or "exclude" not in obs.split("_bundle_pairs(")[1].split(")")[0]   # no exclusions passed for TEST


# ---- distributions / distances / reweighting -------------------------------------------------------

def test_weighted_definitions_reduce_to_raw():
    rng = np.random.default_rng(0)
    v = rng.random(101)
    for q in tm.QUANTILES:
        assert tm.weighted_quantile(v, q) == tm.weighted_quantile(v, q, np.full(101, 3.0))
        assert tm.weighted_quantile(v, q) in set(v)
    assert tm.weighted_share(v, ">=", .5) == pytest.approx((v >= .5).mean())
    assert tm.ks_distance(v, v) == 0.0
    a, b = rng.random(50), rng.random(70) + .3
    assert tm.ks_distance(a, b) == pytest.approx(tm.ks_distance(b, a))
    assert tm.ks_distance([0, 0, 0], [1, 1]) == 1.0
    assert tm.ks_distance(a, b, None, np.ones(70)) == pytest.approx(tm.ks_distance(a, b))


def _obs(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"query_id": [f"x{seed}_{i}" for i in range(n)], "adduct_group": rng.choice(["[M+H]+", "[M+Na]+", "OTHER"], n),
                         "polarity": rng.choice(["positive", "negative"], n), "pool_size_bin": rng.choice(["1-49", "50-149"], n),
                         "max_cosine": np.clip(rng.random(n) + shift, 0, 1)})


def test_reweighting_reads_only_observable_cells():
    tl, test = _obs(400, 1), _obs(300, 2)
    w1, info = tm.reweighting_weights_to_test(tl, test)
    w2, _ = tm.reweighting_weights_to_test(tl.assign(max_cosine=0.0), test.assign(max_cosine=1.0))
    assert np.allclose(w1.to_numpy(), w2.to_numpy()) and info["dims"]
    w3, _ = tm.reweighting_weights_to_test(tl[["query_id", "adduct_group", "polarity", "pool_size_bin"]], test[["adduct_group", "polarity", "pool_size_bin"]])
    assert np.allclose(w1.to_numpy(), w3.to_numpy())
    # reweighted TL cell shares match TEST cell shares on the chosen dims
    dims = info["dims"]
    cell = lambda d: d[dims].astype(str).agg("|".join, axis=1)
    tl_share = pd.Series(w1.to_numpy(), index=cell(tl).to_numpy()).groupby(level=0).sum() / w1.sum()
    t_share = cell(test).value_counts(normalize=True)
    common = t_share.index.intersection(tl_share.index)
    assert np.allclose(tl_share[common] / tl_share[common].sum(), t_share[common] / t_share[common].sum())


def test_distance_table_keeps_every_metric():
    t = _obs(200, 3).assign(max_modified_cosine=.5, matched_peaks_at_max_cosine=3, candidate_pool_size=10, max_peak_overlap=.2)
    s = _obs(200, 4).assign(max_modified_cosine=.5, matched_peaks_at_max_cosine=3, candidate_pool_size=10, max_peak_overlap=.2)
    m = _obs(200, 5, -.4).assign(max_modified_cosine=.1, matched_peaks_at_max_cosine=1, candidate_pool_size=10, max_peak_overlap=.1)
    d = tm.protocol_distances(t, {"strict": s, "mirror": m})
    assert set(d["metric"]) == {f"ks_{k}" for k in tm.KS_METRICS} | {"abs_diff_share_max_cosine_ge_0.95", "abs_diff_median_matched_peaks"}
    assert {"test_vs_strict_distance", "test_vs_mirror_distance", "closer"} <= set(d.columns)


# ---- pre-registered rule ---------------------------------------------------------------------------

def _ri(ks, mod, share, med):
    return {"ks_max_cosine": ks, "ks_max_modified_cosine": mod, "abs_diff_share_max_cosine_ge_0.95": share, "abs_diff_median_matched_peaks": med}


def test_rule_decisions():
    assert tm.decide_resemblance(_ri(.05, .05, .01, 0), _ri(.40, .30, .20, 3))["status"] == tm.STRICT_LIKE
    assert tm.decide_resemblance(_ri(.40, .30, .20, 3), _ri(.05, .05, .01, 0))["status"] == tm.MIRROR_LIKE
    assert tm.decide_resemblance(_ri(.10, .1, .1, 1), _ri(.11, .2, .2, 2))["status"] == tm.INCONCLUSIVE            # primary tie
    assert tm.decide_resemblance(_ri(.30, .1, .1, 1), _ri(.60, .2, .2, 2))["status"] == tm.INCONCLUSIVE            # far from both
    assert tm.decide_resemblance(_ri(.05, .3, .3, 3), _ri(.20, .1, .1, 1))["status"] == tm.INCONCLUSIVE            # contradicted 3/3
    assert tm.final_resemblance({"status": tm.STRICT_LIKE}, {"status": tm.MIRROR_LIKE}) == tm.INCONCLUSIVE
    assert tm.final_resemblance({"status": tm.STRICT_LIKE}, {"status": tm.STRICT_LIKE}) == tm.STRICT_LIKE


def test_rule_takes_no_model_performance_input():
    with pytest.raises((tm.LabelLeakError, ValueError)):
        tm.decide_resemblance({**_ri(.1, .1, .1, 1), "mrr_at_25": .9}, _ri(.2, .2, .2, 2))
    params = set(inspect.signature(tm.decide_resemblance).parameters)
    assert params == {"strict_inputs", "mirror_inputs", "rule"}
    assert set(tm.RULE_INPUTS) == {"ks_max_cosine", "ks_max_modified_cosine", "abs_diff_share_max_cosine_ge_0.95", "abs_diff_median_matched_peaks"}
    assert "MRR" in tm.RESEMBLANCE_RULE["excluded_inputs"]
    src = inspect.getsource(tm.rule_inputs)
    assert "score" not in src and "rank" not in src


def test_preregistration_precedes_statistics_in_notebook():
    nb = json.loads((ROOT / "src" / "10v6_00_test_match_diagnostic.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    first = lambda tok: next(i for i, s in enumerate(code) if tok in s)
    pre = first("write_preregistration(PREREG_PATH")
    for tok in ("run_observed_test(", "qcr_protocol_pairs(", "protocol_distances(", "decide_resemblance(", "distribution_summary("):
        assert pre < first(tok), tok
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


# ---- molecule matchability ---------------------------------------------------------------------------

def test_molecule_matchability_deterministic_and_proxy_wording():
    s = pd.DataFrame({"molecule_id": ["m2", "m1", "m1", "m2", "m3"], "max_cosine": [.3, .96, .6, .85, .45],
                      "has_eligible_reference": True, "near_duplicate_proxy": False, "n_t1_excluded_refs": [0, 1, 0, 0, 0]})
    a = tm.molecule_matchability(s)
    b = tm.molecule_matchability(s.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(a, b)
    a = a.set_index("molecule_id")
    assert list(a.index) == ["m1", "m2", "m3"] and a.loc["m1", "matchability_tier"] == "HIGH" and a.loc["m3", "matchability_tier"] == "LOW"
    assert a.loc["m1", "n_spectra_ge_0.95"] == 1 and a.loc["m2", "n_spectra_lt_0.50"] == 1
    p = tm.library_matchability_proxies(s)
    assert "proxy" in p["wording"] and not any("class 1 fraction" in k.lower() for k in p)


# ---- realism metadata -----------------------------------------------------------------------------------

def test_freeze_annotation_is_additive(tmp_path):
    p = tmp_path / "spectrum_model.json"
    p.write_text(json.dumps({"freeze_status": "FROZEN", "model_id": "M", "scaling_status": "PLATEAU"}))
    realism.annotate_freeze_realism(p, realism.PENDING)
    r = realism.annotate_freeze_realism(p, "MIRROR_LIKE")
    assert r["model_freeze_status"] == "FROZEN" and r["validation_realism_status"] == "MIRROR_LIKE"
    assert r["scaling_status"] == "PLATEAU" and r["freeze_status"] == "FROZEN" and len(r["validation_realism_history"]) == 2
    assert (tmp_path / "spectrum_model.pre_v6.json").exists()
    p.write_text(json.dumps({"freeze_status": "PROVISIONAL"}))
    with pytest.raises(RuntimeError):
        realism.annotate_freeze_realism(p, realism.PENDING)


def test_scaling_interpretation_and_protocol_mapping():
    assert realism.scaling_interpretation("MIRROR_LIKE", "PLATEAU")["scaling_label"] == "CEILING-LIMITED STRICT REGIME"
    assert realism.scaling_interpretation("STRICT_LIKE", "PLATEAU")["recorded_scaling_status_under_strict"] == "PLATEAU"
    assert realism.scaling_interpretation("INTERMEDIATE_OR_INCONCLUSIVE", "PLATEAU")["scaling_label"] == "PROTOCOL-SPECIFIC"
    assert realism.primary_eval_protocols("STRICT_LIKE") == ["test_simulated_strict"]
    assert realism.primary_eval_protocols("MIRROR_LIKE") == ["mirror_aware"]
    assert set(realism.primary_eval_protocols(realism.PENDING)) == {"test_simulated_strict", "mirror_aware"}


def test_repository_freeze_and_gap_metadata():
    f = ROOT / "outputs" / "v5" / "freeze" / "spectrum_model.json"
    g = ROOT / "outputs" / "v5" / "freeze" / "deployment_protocol_gap.json"
    if not f.exists() or not g.exists():
        pytest.skip("freeze artifacts not present")
    rec = json.loads(f.read_text(encoding="utf-8"))
    assert rec["freeze_status"] == "FROZEN" and rec["model_freeze_status"] == "FROZEN"
    assert rec["validation_realism_status"] in realism.REALISM_STATUSES and "scaling_status" in rec and "scale_curve" in rec
    gap = json.loads(g.read_text(encoding="utf-8"))
    assert gap["resolution_status"] == realism.GAP_WAIVED and gap["resolution_status_history"][0]["resolution_status"] == "PENDING_DEPLOYMENT_PARITY_TASK"
