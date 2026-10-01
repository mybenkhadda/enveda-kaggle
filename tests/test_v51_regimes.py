"""v5.1 regime classification, censored counts, Mode-A training filter (non-mutating, nested,
HOST / TL_EVAL-disjoint), Mode-A-only selection, routing rule, and the molecule freeze guard."""
import json

import numpy as np
import pandas as pd
import pytest

from casmi.ranking.freeze import (FreezeIntegrityError, NOT_FROZEN_MESSAGE, SpectrumModelNotFrozen, feature_order_hash,
                                  model_file_hashes, require_frozen_spectrum_model)
from casmi.ranking.scaling import MODE_A_NESTED_ORDER, population_ids, scale_decision, select_mode_a_headline
from casmi.ranking.selection import HostLeakError
from casmi.ranking.training_sets import derive_mode_a_manifests, filter_mode_a_training_queries
from casmi.validation.regime_audit import (
    INCONSISTENT, MODE_A_MIRROR, POOL_ABSENT, REF_ABSENT, ROUTE_CONTINUE, ROUTE_REF_ABSENT, STANDARD_ONLY, build_query_regimes,
    censored_median, classify_regime, metrics_by_regime, molecule_regime_mix, project_route, summary_row,
)


# ---- classification -------------------------------------------------------------------------

def test_classify_regime_all_cases():
    r = classify_regime([False, True, True, True, True, False], [9, 3, 2, 0, 0, 0], [9, 1, 0, 0, 2, 0])
    assert list(r) == [POOL_ABSENT, MODE_A_MIRROR, STANDARD_ONLY, REF_ABSENT, INCONSISTENT, POOL_ABSENT]


# ---- censored counts from a synthetic QCR ------------------------------------------------------

def _truth_rows(q, n_elig_std, n_elig_mir, exhausted):
    rows = []
    for i in range(max(n_elig_std, n_elig_mir)):
        std, mir = i < n_elig_std, i < n_elig_mir
        rows.append({"query_id": q, "candidate_key": "T", "is_true": True, "eligible_standard": std, "eligible_mirror_aware": mir,
                     "accepted_standard": std and sum(1 for r in rows if r["accepted_standard"]) < 5,
                     "accepted_mirror_aware": mir and sum(1 for r in rows if r["accepted_mirror_aware"]) < 5})
    return rows, {"query_id": q, "candidate_key": "T", "is_true": True, "exhausted": exhausted}


def _build(specs):
    qcr, pairs, pool, man = [], [], [], []
    for q, (in_pool, s, m, exh) in specs.items():
        man.append({"query_id": q, "connectivity_key": "T", "source": "libA", "instrument": "timsTOF", "adduct": "[M+H]+"})
        pool.append({"query_id": q, "candidate_connectivity_key": "T" if in_pool else "X", "is_true_candidate": in_pool})
        if in_pool and max(s, m):
            r, p = _truth_rows(q, s, m, exh)
            qcr += r
            pairs.append(p)
    qcr_df = pd.DataFrame(qcr) if qcr else pd.DataFrame(columns=["query_id", "candidate_key", "is_true", "eligible_standard", "eligible_mirror_aware",
                                                                 "accepted_standard", "accepted_mirror_aware"])
    pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["query_id", "candidate_key", "is_true", "exhausted"])
    return build_query_regimes(pd.DataFrame(man), pd.DataFrame(pool), qcr_df, pairs_df)


def test_censored_counts_exact_and_censored():
    r = _build({"q_exact3": (True, 3, 2, True), "q_exact5": (True, 5, 5, True), "q_cens": (True, 8, 6, False), "q_exact7": (True, 7, 7, True),
                "q_none": (True, 0, 0, True), "q_absent": (False, 0, 0, True)}).set_index("query_id")
    assert r.loc["q_exact3", "n_standard_eligible_exact_or_lower_bound"] == 3 and not r.loc["q_exact3", "standard_count_censored"]
    assert r.loc["q_exact5", "n_mirror_eligible_exact_or_lower_bound"] == 5 and not r.loc["q_exact5", "mirror_count_censored"]
    assert r.loc["q_cens", "n_mirror_eligible_exact_or_lower_bound"] == 5 and r.loc["q_cens", "mirror_count_censored"]
    assert r.loc["q_exact7", "n_standard_eligible_exact_or_lower_bound"] == 7 and not r.loc["q_exact7", "standard_count_censored"]
    assert r.loc["q_none", "regime"] == REF_ABSENT and r.loc["q_absent", "regime"] == POOL_ABSENT and r.loc["q_exact3", "regime"] == MODE_A_MIRROR


def test_standard_only_regime_from_evidence():
    r = _build({"q": (True, 4, 0, True)}).set_index("query_id")
    assert r.loc["q", "regime"] == STANDARD_ONLY and r.loc["q", "has_standard_ref"] and not r.loc["q", "has_mirror_ref"]


def test_censored_median_never_fabricates():
    assert censored_median([1, 2, 3], [False, False, False]) == 2.0
    assert censored_median([1, 5, 5], [False, True, True]) == ">=5"
    assert censored_median([0, 1, 5, 5], [False, False, True, False]) == 3.0
    assert censored_median([], []) is None


def test_summary_and_metrics_by_regime_keep_all_queries():
    r = pd.DataFrame({"query_id": list("abcd"), "regime": [MODE_A_MIRROR, REF_ABSENT, POOL_ABSENT, MODE_A_MIRROR], "truth_in_pool": [1, 1, 0, 1],
                      "n_standard_eligible_exact_or_lower_bound": [5, 0, 0, 2], "n_mirror_eligible_exact_or_lower_bound": [5, 0, 0, 1],
                      "mirror_count_censored": [True, False, False, False], "standard_count_censored": [True, False, False, False]})
    r["truth_in_pool"] = r["truth_in_pool"].astype(bool)
    s = summary_row(r)
    assert s["mode_a_share"] == 0.5 and s["pool_absent_share"] == 0.25 and s["mirror_ge5_share"] == 0.25
    pq = pd.DataFrame({"query_id": list("abcd"), "rr": [1.0, 0.0, 0.0, 0.5], "hit_at_1": [1, 0, 0, 0], "hit_at_5": [1, 0, 0, 1], "hit_at_25": [1, 0, 0, 1]})
    m = metrics_by_regime(pq, r).set_index("population")
    assert m.loc["ALL", "n_queries"] == 4 and m.loc["ALL", "mrr_at_25"] == pytest.approx(0.375)
    assert m.loc[MODE_A_MIRROR, "mrr_at_25"] == pytest.approx(0.75) and m.loc[POOL_ABSENT, "n_queries"] == 1


def test_molecule_regime_mix():
    r = pd.DataFrame({"connectivity_key": ["A", "A", "B", "B", "C", "D"], "regime": [MODE_A_MIRROR, MODE_A_MIRROR, MODE_A_MIRROR, REF_ABSENT, REF_ABSENT, STANDARD_ONLY]})
    mix = molecule_regime_mix(r).set_index("connectivity_key")["molecule_regime_mix"].to_dict()
    assert mix == {"A": "ALL_MODE_A", "B": "MIXED", "C": "ALL_REF_ABSENT", "D": "OTHER"}


# ---- Mode-A training filter -----------------------------------------------------------------

def _manifests():
    ids = [f"q{i:03d}" for i in range(60)]
    base = pd.DataFrame({"query_id": ids, "connectivity_key": [f"K{i}" for i in range(60)], "source": "libA", "selection_reason": "x"})
    regimes = pd.DataFrame({"query_id": ids, "regime": [[MODE_A_MIRROR, REF_ABSENT, STANDARD_ONLY, POOL_ABSENT][i % 4] for i in range(60)]})
    return {"TL_1K": base.iloc[:10], "TL_3K": base.iloc[:30], "TL_10K": base.iloc[:60]}, regimes


def test_filter_keeps_only_mode_a_and_does_not_mutate():
    ms, reg = _manifests()
    before = ms["TL_3K"].copy()
    out, audit = filter_mode_a_training_queries(ms["TL_3K"], reg)
    assert ms["TL_3K"].equals(before)
    assert set(out["query_id"].map(reg.set_index("query_id")["regime"])) == {MODE_A_MIRROR}
    assert audit["n_original"] == 30 and audit["n_mode_a_retained"] == len(out) and audit["retention_rate"] == pytest.approx(len(out) / 30)


def test_filtered_manifests_stay_nested():
    ms, reg = _manifests()
    out, audits = derive_mode_a_manifests(ms, {n: reg for n in ms})
    assert set(out["TL_1K_MODE_A"]["query_id"]) <= set(out["TL_3K_MODE_A"]["query_id"]) <= set(out["TL_10K_MODE_A"]["query_id"])
    assert set(audits) == {"TL_1K_MODE_A", "TL_3K_MODE_A", "TL_10K_MODE_A"}


def test_filter_requires_regime_for_every_query():
    ms, reg = _manifests()
    with pytest.raises(ValueError):
        filter_mode_a_training_queries(ms["TL_10K"], reg.iloc[:5])


def test_filtered_manifests_inherit_disjointness():
    ms, reg = _manifests()
    out, _ = derive_mode_a_manifests(ms, {n: reg for n in ms})
    host_keys, tl_eval_keys = {"H1", "H2"}, {"E1"}
    for m in out.values():
        assert not (set(m["connectivity_key"]) & host_keys) and not (set(m["connectivity_key"]) & tl_eval_keys)
        assert set(m["connectivity_key"]) <= set(ms["TL_10K"]["connectivity_key"])     # a filter can only remove


# ---- selection / routing ----------------------------------------------------------------------

def _mode_a_table(mrrs):
    return pd.DataFrame({"model_id": list(MODE_A_NESTED_ORDER), "tl_eval_mode_a_mrr": mrrs, "n_train_queries": [700, 2100, 7000], "status": "OK"})


def test_mode_a_winner_uses_tl_eval_mode_a_only():
    assert select_mode_a_headline(_mode_a_table([0.50, 0.503, 0.504]))[0] == "V1_TL_1K_MODE_A"
    assert select_mode_a_headline(_mode_a_table([0.50, 0.52, 0.53]))[0] == "V1_TL_10K_MODE_A"


@pytest.mark.parametrize("col", ["host_mrr", "tl_eval_all_mrr", "tl_eval_ref_absent_mrr", "tl_eval_standard_only_mrr"])
def test_other_populations_cannot_select(col):
    with pytest.raises(HostLeakError):
        select_mode_a_headline(_mode_a_table([0.5, 0.5, 0.5]).assign(**{col: 0.9}))


def test_mode_a_scale_decision_uses_mode_a_pair():
    ok = set(MODE_A_NESTED_ORDER)
    d = scale_decision({("V1_TL_10K_MODE_A", "V1_TL_3K_MODE_A"): {"delta": 0.01, "ci_low": 0.001, "ci_high": 0.02}}, ok, nested_order=MODE_A_NESTED_ORDER)
    assert d["scaling_status"] == "CONTINUE_SCALING" and d["pair"] == ["V1_TL_10K_MODE_A", "V1_TL_3K_MODE_A"]


def test_population_ids():
    r = pd.DataFrame({"query_id": list("abc"), "regime": [MODE_A_MIRROR, REF_ABSENT, STANDARD_ONLY]})
    assert population_ids(r, "TL_EVAL_MODE_A") == ["a"] and population_ids(r, "TL_EVAL_REF_ABSENT") == ["b"]
    assert population_ids(r, "TL_EVAL_STANDARD_ONLY") == ["c"] and population_ids(r, "TL_EVAL_ALL") == ["a", "b", "c"]


def test_routing_rule():
    assert project_route(0.49)["project_route"] == ROUTE_REF_ABSENT
    assert project_route(0.50)["project_route"] == ROUTE_CONTINUE
    assert project_route(0.9)["is_gate"] is False


# ---- molecule freeze guard ----------------------------------------------------------------------

def _freeze(tmp_path, status="FROZEN", tamper=None):
    mdir = tmp_path / "model"
    mdir.mkdir()
    for k in range(5):
        (mdir / f"fold_{k}.txt").write_text(f"booster {k}")
    feats = ["abs_mass_error_ppm", "cosine_max"]
    rec = {"freeze_status": status, "model_dir": str(mdir), "feature_names": feats, "feature_order_hash": feature_order_hash(feats),
           "model_hashes": model_file_hashes(mdir), "mirror_aware_protocol_hash": "P"}
    if tamper == "model":
        (mdir / "fold_0.txt").write_text("changed")
    if tamper == "features":
        rec["feature_names"] = list(reversed(feats))
    p = tmp_path / "spectrum_model.json"
    p.write_text(json.dumps(rec))
    return p


def test_guard_refuses_provisional(tmp_path):
    with pytest.raises(SpectrumModelNotFrozen, match="Mode-A spectrum model is not frozen"):
        require_frozen_spectrum_model(_freeze(tmp_path, status="PROVISIONAL (not every TL Mode-A scale model trained yet)"), protocol_hash_fn=lambda: "P")


def test_guard_allows_frozen(tmp_path):
    assert require_frozen_spectrum_model(_freeze(tmp_path), protocol_hash_fn=lambda: "P")["freeze_status"] == "FROZEN"


@pytest.mark.parametrize("tamper", ["model", "features"])
def test_guard_refuses_hash_mismatch(tmp_path, tamper):
    with pytest.raises(FreezeIntegrityError):
        require_frozen_spectrum_model(_freeze(tmp_path, tamper=tamper), protocol_hash_fn=lambda: "P")


def test_guard_refuses_protocol_change(tmp_path):
    with pytest.raises(FreezeIntegrityError):
        require_frozen_spectrum_model(_freeze(tmp_path), protocol_hash_fn=lambda: "CHANGED")


def test_guard_missing_file(tmp_path):
    with pytest.raises(SpectrumModelNotFrozen):
        require_frozen_spectrum_model(tmp_path / "nope.json")
    assert NOT_FROZEN_MESSAGE == "Mode-A spectrum model is not frozen. Complete v5 scaling first."
