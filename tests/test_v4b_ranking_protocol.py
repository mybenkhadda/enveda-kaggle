"""v4b grouped ranking + HOST isolation: one group per query, no query split across a fold's
train/evaluation sides, fold-matched scoring, and selection/tuning functions that cannot consume
HOST information."""
import json

import numpy as np
import pandas as pd
import pytest

from casmi.ranking.baselines import select_alpha_dev
from casmi.ranking.lambdamart import (
    FORBIDDEN_MODEL_FEATURES, assert_feature_list_allowed, connectivity_disjointness, fit_fold_models, make_groups,
    predict_by_fold,
)
from casmi.ranking.selection import (
    HostLeakError, guard_no_host, lock_dev_selection, require_dev_selection_lock, select_from_dev_only,
    verify_preregistration, write_preregistration,
)


class RecordingRanker:
    """Stand-in for LGBMRanker: records what it was trained on; predicts a fixed feature."""

    def __init__(self, params=None):
        self.fitted_queries = None
        self.groups = None

    def fit(self, X, y, group):
        self.X, self.y, self.groups = X, y, np.asarray(group)
        return self

    def predict(self, X):
        return X.iloc[:, 0].to_numpy()


def _table(n_queries=10, n_cand=4, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_queries):
        for j in range(n_cand):
            rows.append({"query_id": f"q{i:02d}", "fold": i % n_folds, "f1": rng.random(), "is_true_candidate": j == 0,
                         "candidate_connectivity_key": f"k{j}"})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)  # scrambled row order


def test_make_groups_contiguous_one_group_per_query():
    d, groups = make_groups(_table())
    assert groups.sum() == len(d) and len(groups) == d["query_id"].nunique()
    runs = (d["query_id"] != d["query_id"].shift()).sum()
    assert runs == len(groups)


def test_fold_models_never_train_on_heldout_queries():
    t = _table()
    models, audit = fit_fold_models(t, ["f1"], model_factory=lambda p: RecordingRanker(),
                                    eval_query_ids_by_fold={f: set(t.loc[t["fold"] == f, "query_id"]) for f in range(5)})
    for f, m in models.items():
        heldout = set(t.loc[t["fold"] == f, "query_id"])
        assert m.groups.sum() == len(m.X)
        assert len(m.groups) == t.loc[t["fold"] != f, "query_id"].nunique()
    assert (audit["n_train_queries"] == 8).all()


def test_leak_assertion_fires():
    t = _table()
    with pytest.raises(AssertionError):
        fit_fold_models(t, ["f1"], model_factory=lambda p: RecordingRanker(), eval_query_ids_by_fold={0: {"q01"}})


def test_predict_by_fold_uses_matching_model():
    t = _table()

    class Const(RecordingRanker):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def predict(self, X):
            return np.full(len(X), float(self.v))

    models = {f: Const(f) for f in range(5)}
    s = predict_by_fold(models, t, ["f1"])
    assert (s == t["fold"].astype(float)).all()


def test_forbidden_features_rejected():
    for f in FORBIDDEN_MODEL_FEATURES:
        with pytest.raises(AssertionError):
            assert_feature_list_allowed(["abs_mass_error_ppm", f])


def test_connectivity_disjointness_report():
    q = pd.DataFrame({"query_id": ["a", "b", "c"], "fold": [0, 1, 1], "true_connectivity_key": ["X", "Y", "X"]})
    rep = connectivity_disjointness(q, q).set_index("fold")
    assert rep.loc[0, "n_overlap"] == 1  # X appears in fold 0 and fold 1 -> caught


# ---- HOST isolation ----------------------------------------------------------------------------

def _dev_table():
    return pd.DataFrame([
        {"model_id": "V1", "dev_oof_mrr": 0.700, "dev_k1_mrr": 0.60, "held_out_validity": 0, "simplicity_rank": 0, "status": "OK"},
        {"model_id": "V3", "dev_oof_mrr": 0.697, "dev_k1_mrr": 0.62, "held_out_validity": 2, "simplicity_rank": 2, "status": "OK"},
        {"model_id": "V4", "dev_oof_mrr": 0.690, "dev_k1_mrr": 0.70, "held_out_validity": 2, "simplicity_rank": 5, "status": "OK"},
        {"model_id": "V1_10k", "dev_oof_mrr": 0.99, "dev_k1_mrr": 0.99, "held_out_validity": 0, "simplicity_rank": 6, "status": "NOT_RUN"},
    ])


def test_selection_rule_near_tie_prefers_k1_then_validity():
    sel, ranked = select_from_dev_only(_dev_table())
    assert sel == "V3"          # within 0.005 of V1 and better k=1; V4 is outside the margin
    assert ranked["model_id"].tolist()[:3] == ["V3", "V1", "V4"]
    assert "V1_10k" not in ranked["model_id"].tolist()  # NOT_RUN never selectable


def test_selection_refuses_host_columns():
    t = _dev_table().assign(host_mrr=0.5)
    with pytest.raises(HostLeakError):
        select_from_dev_only(t)


def test_selection_refuses_unexpected_columns():
    with pytest.raises(HostLeakError):
        select_from_dev_only(_dev_table().assign(confirm_metric=0.1))


def test_guard_no_host_on_dicts_and_frames():
    with pytest.raises(HostLeakError):
        guard_no_host({"B4": 0.1, "HOST_B4": 0.2}, "x")
    guard_no_host(pd.DataFrame({"dev_mrr": [1.0]}), "x")


def test_alpha_selection_signature_has_no_host_input():
    import inspect
    params = inspect.signature(select_alpha_dev).parameters
    assert not any("host" in p.lower() for p in params)


def test_alpha_selection_is_dev_only_and_prefers_smaller_alpha_on_ties():
    df = pd.DataFrame({"query_id": ["a", "a", "b", "b"], "candidate_connectivity_key": ["T", "D", "T", "D"],
                       "peak_overlap_frac_max": [0.9, 0.1, 0.8, 0.2], "mass_loglik": [0.0, 0.0, 0.0, 0.0],
                       "abs_mass_error_ppm": [1.0, 1.0, 1.0, 1.0], "is_true_candidate": [True, False, True, False]})
    std = {"peak_overlap_frac_max": {"mean": 0.0, "std": 1.0}, "mass_loglik": {"mean": 0.0, "std": 1.0}}
    alpha, grid, cv = select_alpha_dev(df, ["a", "b"], {"a": 0, "b": 1}, "peak_overlap_frac_max", std, alpha_grid=(0.0, 0.5, 1.0))
    assert alpha == 0.0 and (grid["dev_mrr_at_25"] == 1.0).all() and len(cv) == 2


def test_preregistration_and_lock_order(tmp_path):
    prereg = tmp_path / "preregistration.json"
    lock = tmp_path / "dev_selection_lock.json"
    write_preregistration(prereg)
    with pytest.raises(HostLeakError):
        require_dev_selection_lock(lock, prereg)
    sel, ranked = select_from_dev_only(_dev_table())
    lock_dev_selection(lock, sel, ranked, {"B4": 0.1, "B5": 0.2}, prereg)
    token = require_dev_selection_lock(lock, prereg)
    assert token.selected_model_id == "V3"
    with pytest.raises(RuntimeError):
        lock_dev_selection(lock, "V1", ranked, {"B4": 0.1, "B5": 0.2}, prereg)  # cannot silently re-select


def test_preregistration_tamper_detected(tmp_path):
    prereg = tmp_path / "p.json"
    write_preregistration(prereg)
    rec = json.loads(prereg.read_text())
    rec["rule"]["near_tie_margin"] = 0.05
    prereg.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError):
        verify_preregistration(prereg)


def test_preregistration_cannot_change(tmp_path):
    prereg = tmp_path / "p.json"
    write_preregistration(prereg)
    write_preregistration(prereg)  # identical: no-op
    with pytest.raises(RuntimeError):
        write_preregistration(prereg, rule={"rule_id": "something-else"})
