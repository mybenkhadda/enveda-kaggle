"""v6.2 molecule aggregation: protocol-as-configuration notebook wiring (strict primary, mirror sensitivity,
no stale MOL_DEV_mirror_aware dependency), DEV-only calibration and selection, HOST after the lock,
manifest-based test-like resampling, molecule-level bootstrap, connectivity dedup before Top-25, label-free
matchability strata, and an exported config that reproduces the DEV ranking on the inference path."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi.ranking import molecule_eval as me
from casmi.ranking.selection import HostLeakError
from casmi_infer.aggregation import (FirstSpectrum, LogMeanExp, MaxProb, MaxScore, MeanProb, MostConfidentSpectrum, RRFAggregator, SumLogProb,
                                     make_aggregator)

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "src" / "11_02_molecule_aggregation.ipynb"
KEY = "candidate_connectivity_key"


def _code():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _first(cells, tok):
    return next(i for i, s in enumerate(cells) if tok in s)


# ---- notebook wiring -------------------------------------------------------------------------------------

def test_protocol_configuration_and_no_stale_mirror_dependency():
    cells = _code()
    src = "\n".join(cells)
    assert "PRIMARY_PROTOCOL = 'test_simulated_strict'" in cells[0] and "SENSITIVITY_PROTOCOLS = ['mirror_aware']" in cells[0]
    assert "read_parquet(D['features'] / 'MOL_DEV_mirror_aware.parquet')" not in src and "pd.read_parquet(legacy" not in src
    assert "pd.read_parquet(feature_path(D, 'MOL_DEV', p))" in src                  # one reader for every protocol
    assert "moldev_build_instructions(" in src and "MolDevNotReady(" in src           # stop cleanly, never substitute a population
    assert "primary_eval_protocols(REALISM)" in cells[0]


def test_frozen_model_untouched():
    src = "\n".join(_code())
    for tok in ("fit_fold_models", "LGBMRanker", "lgb.train", ".save_model(", "export_models("):
        assert tok not in src, tok
    assert "require_frozen_spectrum_model(" in _code()[0]


def test_calibration_uses_development_oof_only():
    cells = _code()
    fit = cells[_first(cells, "fit_temperature(")]
    assert "fit_temperature(oof," in fit and "guard_no_host(oof" in fit and "OOF overlaps MOL_DEV" in fit
    assert "PER_SPEC" not in fit and "host" not in fit.replace("guard_no_host", "").lower()


def test_selection_locked_before_any_host_access():
    cells = _code()
    lock = _first(cells, "lock_dev_selection(")
    for i, s in enumerate(cells[:lock + 1]):
        for tok in ("host_spec", "host_holdout_spectra", "build_host_protocol_artifacts(", "HOST_TAB", "feature_path(D, 'HOST'"):
            assert tok not in s, f"cell {i} touches HOST before the lock: {tok}"
    assert _first(cells, "require_dev_selection_lock(LOCK_PATH") > lock
    assert "select_aggregator_dev_only(" in cells[lock] and "for p in SELECTION_PROTOCOLS" in cells[lock]
    assert "selection changed after HOST" in "\n".join(cells[lock + 1:])


def test_outputs_and_bundle_swap_are_wired():
    src = "\n".join(_code())
    for f in ("calibration.json", "calibration_metrics.csv", "aggregator_metrics.csv", "aggregator_bootstrap.csv", "selected_aggregator.json",
              "moldev_predictions.parquet", "host_confirmation.csv", "aggregator_bundle_config.json"):
        assert f in src, f
    assert "verify_aggregator_roundtrip(" in src and "UPDATE_BUNDLE = False" in src and "apply_aggregator_config(" in src
    assert "NEXT: OPEN_CANDIDATES / CLASS2_VALIDATION" in src


# ---- selection is DEV-only ---------------------------------------------------------------------------------

def test_selection_refuses_host_inputs():
    t = pd.DataFrame({"aggregator": ["A1_max_score", "A6_rrf"], "dev_mrr_mean": [.5, .6], "delta_ci_low": [-.1, 0], "delta_ci_high": [.0, 0]})
    simp = {"A1_max_score": 1, "A6_rrf": 2}
    assert me.select_aggregator_dev_only(t, simp)[0] == "A1_max_score"          # within CI of best -> simplest
    with pytest.raises(HostLeakError):
        me.select_aggregator_dev_only(t.assign(host_mrr=.7), simp)
    with pytest.raises(HostLeakError):
        me.select_across_protocols({"HOST": "A6_rrf"}, simp)


# ---- resampling / evaluation ------------------------------------------------------------------------------

def _per_spec():
    """M1: s1 [A true, B], s2 [B, A]; M2: t1 [C true, D], t2 (no candidates)."""
    rows = [("s1", "M1", "A", 2.0, 1.0, True, .9), ("s1", "M1", "B", 1.0, 2.0, False, .5),
            ("s2", "M1", "B", 3.0, 1.0, False, .8), ("s2", "M1", "A", 0.5, 1.0, True, .7),
            ("t1", "M2", "C", 1.0, 1.0, True, .3), ("t1", "M2", "D", 0.9, 1.0, False, .2)]
    d = pd.DataFrame(rows, columns=["spectrum_id", "molecule_id", KEY, "score", "abs_mass_error_ppm", "is_true_candidate", "cosine_max"])
    d["rank"] = d.groupby("spectrum_id")["score"].rank(ascending=False, method="first").astype(int)
    from casmi_infer.aggregation import softmax_by_group
    d["prob"] = softmax_by_group(d["score"], d["spectrum_id"], 1.3)
    return d


ALL = pd.DataFrame({"molecule_id": ["M1", "M1", "M2", "M2"], "spectrum_id": ["s1", "s2", "t1", "t2"]})
TRUTH = {"M1": "A", "M2": "C"}
AGGS = [MaxScore(KEY), MaxProb(KEY), MeanProb(KEY), SumLogProb(eps=1e-6, cand_col=KEY), LogMeanExp(KEY), RRFAggregator(k=60, cand_col=KEY),
        MostConfidentSpectrum(KEY), FirstSpectrum(KEY)]
AGG_ID = lambda a: a.name


def test_resampling_draws_from_manifest_and_is_protocol_independent():
    per = _per_spec()
    other = per[per["spectrum_id"] != "s2"]                                       # a second "protocol" table missing s2's rows
    a = [set(s["spectrum_id"]) for _, s in me.multiplicity_resamples(per, [1], 30, 7, all_spectra=ALL)]
    b = [set(s["spectrum_id"]) for _, s in me.multiplicity_resamples(other, [1], 30, 7, all_spectra=ALL)]
    assert all(x - {"s2"} == y for x, y in zip(a, b))                             # same draws, only the available rows differ
    # t2 has no candidates but still consumes draws: M2 is sometimes empty
    assert any("t1" not in x for x in a) and any("t1" in x for x in a)
    legacy = [set(s["spectrum_id"]) for _, s in me.multiplicity_resamples(per, [1], 30, 7)]
    assert all("t1" in x for x in legacy)                                         # old behaviour: empty-pool spectra invisible


def test_evaluate_and_metrics_are_molecule_level_and_deterministic():
    rr1 = me.evaluate_aggregators(_per_spec(), TRUTH, AGGS, AGG_ID, [2], ALL, n_resamples=3, seed=1, cand_col=KEY)
    rr2 = me.evaluate_aggregators(_per_spec().sample(frac=1, random_state=4), TRUTH, AGGS, AGG_ID, [2], ALL, n_resamples=3, seed=1, cand_col=KEY)
    pd.testing.assert_frame_equal(rr1, rr2)
    assert set(rr1["molecule_id"]) == {"M1", "M2"} and len(rr1) == 3 * len(AGGS) * 2
    m, per_mol = me.aggregator_metrics(rr1, n_boot=50, seed=0)
    assert set(m["aggregator"]) == {a.name for a in AGGS} and (m["n_molecules"] == 2).all()
    assert list(per_mol.index) == ["M1", "M2"]
    d = me.paired_deltas(per_mol, "A3_mean_prob", me.PAIRED_BASELINES + (me.SPECTRUM_BASELINE,), n_boot=50, seed=0)
    assert set(d["other"]) == {"A6_rrf", "A1_max_score", "A7_most_confident_spectrum", "A8_first_spectrum"}


def test_union_dedup_before_top25():
    per = _per_spec()
    r = MeanProb(KEY).rank(per)
    m1 = r[r["molecule_id"] == "M1"]
    assert set(m1[KEY]) == {"A", "B"} and not r.duplicated(["molecule_id", KEY]).any()
    big = pd.concat([per.assign(**{KEY: per[KEY] + str(i)}) for i in range(20)], ignore_index=True)   # 40 candidates for M1
    top = me.molecule_top_k(MeanProb(KEY).rank(big))
    assert (top.groupby("molecule_id").size() <= me.MRR_K).all() and not top.duplicated(["molecule_id", KEY]).any()


def test_absent_candidate_policy_verified():
    out = me.verify_absent_candidate_policy(AGGS, cand_col=KEY)
    assert out["ok"].all()


# ---- scoring / calibration interface ---------------------------------------------------------------------

class _M:
    def __init__(self, b):
        self.b = b

    def predict(self, X):
        return X["f"].to_numpy(float) + self.b


def test_score_spectra_uses_inference_tie_rule_and_spectrum_softmax():
    f = pd.DataFrame({"query_id": [1, 1, 1, 2], KEY: ["B", "A", "C", "A"], "f": [1.0, 1.0, 0.5, 2.0], "abs_mass_error_ppm": [1.0, 1.0, 0.5, 3.0],
                      "true_connectivity_key": ["A", "A", "A", "A"], "is_true_candidate": [False, True, False, True]})
    per = me.score_spectra({0: _M(0.0), 1: _M(2.0)}, f, ["f"], 2.0, KEY)
    s1 = per[per["spectrum_id"] == "1"].sort_values("rank")
    assert list(s1[KEY]) == ["A", "B", "C"]                                        # score tie -> ppm tie -> key ASC
    assert np.allclose(per.groupby("spectrum_id")["prob"].sum(), 1.0) and set(per["molecule_id"]) == {"A"}
    assert per["spectrum_id"].map(type).eq(str).all()


def test_matchability_is_label_free():
    per = _per_spec()
    a = me.molecule_matchability(per, ALL)
    b = me.molecule_matchability(per.drop(columns=["is_true_candidate", "rank", "score", "prob"]), ALL)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc["M1", "matchability_tier"] == "HIGH" and a.loc["M2", "matchability_tier"] == "LOW"   # t2 (no pool) counts as 0


# ---- export / Kaggle ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("agg", AGGS, ids=lambda a: a.name)
def test_exported_config_reproduces_dev_ranking(agg):
    per = _per_spec()
    cfg = me.aggregator_export_config(agg, agg.name, 1.3)
    assert (cfg["temperature"] is None) == (not agg.needs_prob)
    assert type(make_aggregator(cfg, cand_col=KEY)) is type(agg)
    assert me.verify_aggregator_roundtrip(agg, cfg, per, KEY) > 0


def test_roundtrip_detects_a_wrong_export():
    per = _per_spec()
    with pytest.raises(AssertionError):
        me.verify_aggregator_roundtrip(MeanProb(KEY), me.aggregator_export_config(MaxScore(KEY), "A1_max_score", 1.3), per, KEY)


def test_bundle_swap_refuses_prob_aggregator_without_temperature(tmp_path):
    from casmi import bundle_export as bx
    with pytest.raises(ValueError):
        bx.apply_aggregator_config(tmp_path, {"name": "A3_mean_prob", "temperature": None}, {})


def test_mol_dev_requirements_refuse_training_overlap():
    man = pd.DataFrame({"connectivity_key": ["M1", "M1", "M2"], "source": "libX"})
    ok = me.mol_dev_requirements(man, ALL, ["Z"], ["Y"], ["H"], "hostlib")
    assert ok["disjoint_from_training"] and ok["multi_spectrum_molecules"]
    with pytest.raises(AssertionError):
        me.mol_dev_requirements(man, ALL, ["M2"], ["Y"], ["H"], "hostlib")
