"""v5 governance: held-out validity only counts a restriction that removed rows; TL_EVAL-only
headline selection refuses HOST; pre-registered scale decision; common evaluation population;
density reweighting."""
import numpy as np
import pandas as pd
import pytest

from casmi.ranking.scaling import density_reweighted_mrr, pool_size_stratification, scale_decision, score_population, select_scale_headline
from casmi.ranking.selection import (
    HostLeakError, apply_effective_validity, effective_held_out_validity, filter_audit, no_op_duplicate_annotations, select_from_dev_only,
)


def test_filter_audit_counts_and_rejects_additions():
    a = filter_audit("source", ["a", "b", "c"], ["a", "b"])
    assert a == {"filter": "source", "n_queries_before_filter": 3, "n_queries_after_filter": 2, "n_queries_removed": 1, "is_no_op": False}
    with pytest.raises(ValueError):
        filter_audit("source", ["a"], ["a", "z"])


def test_validity_only_increases_when_rows_removed():
    noop_source = {"source": filter_audit("source", "abc", "abc"), "structure": filter_audit("structure", "abc", "ab")}
    assert effective_held_out_validity("V1", noop_source) == 0
    assert effective_held_out_validity("V2", noop_source) == 0          # source filter removed nothing
    assert effective_held_out_validity("V3", noop_source) == 1          # only the structure filter counts
    both = {"source": filter_audit("source", "abcd", "abc"), "structure": filter_audit("structure", "abc", "ab")}
    assert effective_held_out_validity("V2", both) == 1 and effective_held_out_validity("V3", both) == 2
    assert effective_held_out_validity("V3", {}) == 0                  # no audit -> no claim


def test_no_op_duplicate_annotation():
    audits = {"source": filter_audit("source", "abc", "abc"), "structure": filter_audit("structure", "abc", "ab")}
    dups = no_op_duplicate_annotations(audits)
    assert "V2" in dups and "V3" not in dups


def test_selection_uses_effective_validity():
    t = pd.DataFrame([{"model_id": "V1", "dev_oof_mrr": 0.700, "dev_k1_mrr": 0.60, "held_out_validity": 0, "simplicity_rank": 0, "status": "OK"},
                      {"model_id": "V2", "dev_oof_mrr": 0.700, "dev_k1_mrr": 0.60, "held_out_validity": 1, "simplicity_rank": 1, "status": "OK"}])
    audits = {"source": filter_audit("source", "abc", "abc")}
    sel_nominal, _ = select_from_dev_only(t)
    sel_effective, ranked = select_from_dev_only(t, filter_audits=audits)
    assert sel_nominal == "V2" and sel_effective == "V1"
    _, corrections = apply_effective_validity(t, audits)
    assert corrections == [{"model_id": "V2", "nominal": 1, "effective": 0}]


def _tl(mrrs, n=(1000, 3000, 10000)):
    return pd.DataFrame({"model_id": ["V1_TL_1K", "V1_TL_3K", "V1_TL_10K"], "tl_eval_mrr": mrrs, "n_train_queries": list(n), "status": "OK"})


def test_headline_prefers_smaller_within_margin():
    assert select_scale_headline(_tl([0.600, 0.604, 0.603]))[0] == "V1_TL_1K"
    assert select_scale_headline(_tl([0.600, 0.610, 0.612]))[0] == "V1_TL_3K"
    assert select_scale_headline(_tl([0.600, 0.610, 0.620]))[0] == "V1_TL_10K"


def test_headline_refuses_host_and_unexpected_columns():
    with pytest.raises(HostLeakError):
        select_scale_headline(_tl([0.6, 0.6, 0.6]).assign(host_mrr=0.9))
    with pytest.raises(HostLeakError):
        select_scale_headline(_tl([0.6, 0.6, 0.6]).assign(anything_else=1))


def test_headline_ignores_rnd_and_not_run():
    t = pd.concat([_tl([0.5, 0.6, np.nan]).assign(status=["OK", "OK", "NOT_RUN"]),
                   pd.DataFrame([{"model_id": "V1_RND_1K", "tl_eval_mrr": 0.99, "n_train_queries": 1000, "status": "OK"}])])
    assert select_scale_headline(t)[0] == "V1_TL_3K"


def test_scale_decision_rule():
    ok = {"V1_TL_1K", "V1_TL_3K", "V1_TL_10K"}
    cont = {("V1_TL_10K", "V1_TL_3K"): {"delta": 0.01, "ci_low": 0.002, "ci_high": 0.02}}
    plat = {("V1_TL_10K", "V1_TL_3K"): {"delta": 0.01, "ci_low": -0.001, "ci_high": 0.02}}
    neg = {("V1_TL_10K", "V1_TL_3K"): {"delta": -0.01, "ci_low": -0.02, "ci_high": -0.001}}
    assert scale_decision(cont, ok)["scaling_status"] == "CONTINUE_SCALING"
    assert scale_decision(plat, ok)["scaling_status"] == "PLATEAU"
    assert scale_decision(neg, ok)["scaling_status"] == "PLATEAU"
    fallback = scale_decision({("V1_TL_3K", "V1_TL_1K"): {"delta": 0.02, "ci_low": 0.01, "ci_high": 0.03}}, {"V1_TL_1K", "V1_TL_3K"})
    assert fallback["pair"] == ["V1_TL_3K", "V1_TL_1K"] and fallback["scaling_status"] == "CONTINUE_SCALING"
    assert scale_decision({}, {"V1_TL_1K"})["scaling_status"] == "INSUFFICIENT_MODELS"


class _Const:
    def __init__(self, w):
        self.w = w

    def predict(self, X):
        return X.to_numpy() @ self.w


def test_common_population_scoring_uses_every_query_and_fold_mean():
    df = pd.DataFrame({"query_id": ["a", "a", "b", "b"], "candidate_connectivity_key": ["T", "D", "T", "D"], "f": [1.0, 0.0, 0.0, 1.0],
                       "abs_mass_error_ppm": [1.0, 1.0, 1.0, 1.0], "is_true_candidate": [True, False, True, False]})
    models = {0: _Const(np.array([1.0])), 1: _Const(np.array([3.0]))}
    pq, ranked = score_population(models, df, ["f"], ["a", "b", "c"])
    assert list(pq["query_id"]) == ["a", "b", "c"] and pq["rr"].tolist() == [1.0, 0.5, 0.0]
    assert np.allclose(ranked.sort_index()["score"], [2.0, 0.0, 0.0, 2.0])


def test_density_reweighting():
    host = pd.DataFrame({"query_id": list("abcd"), "n_candidates": [10, 20, 500, 600], "rr": [1.0, 1.0, 0.0, 0.5], "hit_at_1": [1, 1, 0, 0]})
    d = density_reweighted_mrr(host, test_pool_sizes=[10, 550, 560, 570])
    # bins: [0,50): host mrr 1.0, w 0.25 ; [400,800): host mrr 0.25, w 0.75
    assert d["estimate"] == pytest.approx(0.25 * 1.0 + 0.75 * 0.25)
    assert d["raw_host_mrr"] == pytest.approx(0.625) and d["label"] == "DENSITY-REWEIGHTED CLASS-1 ESTIMATE"
    assert d["table"]["low_n_host"].all()
    d2 = density_reweighted_mrr(host, test_pool_sizes=[10, 2000])       # >=800 bin has no HOST query
    assert d2["test_weight_uncovered"] == pytest.approx(0.5) and d2["estimate"] == pytest.approx(1.0)


def test_pool_size_stratification_shows_counts():
    pq = pd.DataFrame({"n_candidates": [10, 60, 60, 900], "rr": [1, 0.5, 0, 0], "hit_at_1": [1, 0, 0, 0]})
    t = pool_size_stratification(pq).set_index("pool_bin")
    assert t.loc["50-99", "n_queries"] == 2 and t.loc[">=800", "n_queries"] == 1 and t["low_n"].all()
