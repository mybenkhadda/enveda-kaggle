"""v4b cache-settlement decision rule (spec section 51, "Cache settlement"): tiny floating
differences are accepted; a threshold crossing or a tier change forces REBUILD_ONCE; the
downstream gate refuses anything but an accepted/validated decision."""
import json

import numpy as np
import pandas as pd
import pytest

from casmi.qcr.context import V4A1_SIMILARITY_CONFIG
from casmi.qcr.settlement import (
    DECISION_ACCEPT, DECISION_REBUILD, DECISION_REBUILT_OK, DECISION_REBUILD_FAILED, CacheSettlementError,
    SIMILARITY_METRICS, annotate_differences, crosses_threshold, decide_after_rebuild, decide_cache,
    evidence_fingerprint, load_settlement_or_refuse, mismatch_table, recreate_c3_sample, summarize_settlement,
    write_settlement,
)


def _clean_summary(**over):
    s = {"n_pairs_checked": 1000, "n_tier_changes": 0, "n_threshold_crossings": 0, "max_abs_diff_cosine": 0.0,
         "max_abs_diff_modified_cosine": 0.0, "max_abs_diff_peak_overlap": 0.0, "max_abs_diff_neutral_loss": 0.0,
         "aggregated_feature_mismatch_count": 0}
    s.update(over)
    return s


def test_tiny_floating_differences_are_accepted():
    decision, reasons = decide_cache(_clean_summary(max_abs_diff_cosine=3e-16, max_abs_diff_modified_cosine=1e-10))
    assert decision == DECISION_ACCEPT and reasons == []


def test_difference_above_1e9_forces_rebuild():
    decision, reasons = decide_cache(_clean_summary(max_abs_diff_neutral_loss=2e-9))
    assert decision == DECISION_REBUILD and any("neutral_loss" in r for r in reasons)


def test_threshold_crossing_forces_rebuild_even_with_tiny_diff():
    decision, _ = decide_cache(_clean_summary(n_threshold_crossings=1, max_abs_diff_cosine=1e-12))
    assert decision == DECISION_REBUILD


def test_tier_change_forces_rebuild():
    decision, reasons = decide_cache(_clean_summary(n_tier_changes=1))
    assert decision == DECISION_REBUILD and any("tier" in r for r in reasons)


def test_aggregated_feature_mismatch_forces_rebuild():
    assert decide_cache(_clean_summary(aggregated_feature_mismatch_count=3))[0] == DECISION_REBUILD


def test_nan_or_missing_max_diff_is_never_a_pass():
    assert decide_cache(_clean_summary(max_abs_diff_cosine=float("nan")))[0] == DECISION_REBUILD
    s = _clean_summary()
    del s["max_abs_diff_peak_overlap"]
    assert decide_cache(s)[0] == DECISION_REBUILD
    assert decide_cache(_clean_summary(n_pairs_checked=0))[0] == DECISION_REBUILD


def test_decide_after_rebuild_never_loops():
    assert decide_after_rebuild(_clean_summary())[0] == DECISION_REBUILT_OK
    assert decide_after_rebuild(_clean_summary(n_tier_changes=2))[0] == DECISION_REBUILD_FAILED


@pytest.mark.parametrize("a,b,t,expected", [
    (0.9500000001, 0.9499999999, 0.95, True),
    (0.95, 0.9499999999, 0.95, True),       # exactly on threshold on one side counts (>= convention)
    (0.9500000001, 0.95, 0.95, True),       # (> convention)
    (0.96, 0.9600000001, 0.95, False),
    (0.10, 0.10000001, 0.95, False),
])
def test_crosses_threshold(a, b, t, expected):
    assert crosses_threshold(a, b, t) is expected


def _diag_frame(cached_cos, fresh_cos, cached_tier="T4", fresh_tier="T4"):
    rec = {"dataset": "host", "query_id": "train_1", "candidate_key": "K", "ref_spectrum_id": "train_2",
           "cached_tier": cached_tier, "fresh_tier": fresh_tier,
           "any_near_identity_threshold": False}
    for m in SIMILARITY_METRICS:
        rec[f"cached_{m}"] = 0.5
        rec[f"fresh_{m}"] = 0.5
        rec[f"legacy_{m}"] = 0.5
    rec["cached_cosine"], rec["fresh_cosine"], rec["legacy_cosine"] = cached_cos, fresh_cos, cached_cos
    for side in ("query", "ref"):
        rec[f"{side}_top_n_exact_tie"] = False
        rec[f"{side}_top_n_near_tie"] = False
        rec[f"{side}_top_n_selection_rule_sensitive"] = False
    return pd.DataFrame([rec])


def test_annotate_detects_cosine_crossing_and_legacy_explanation():
    d = annotate_differences(_diag_frame(0.9500001, 0.9499999), V4A1_SIMILARITY_CONFIG)
    s = summarize_settlement(d, aggregated_feature_mismatch_count=0)
    assert s["n_cosine_crossing_0_95"] == 1 and s["n_threshold_crossings"] == 1
    assert s["n_value_mismatches"] == 1 and s["n_mismatches_explained_by_legacy_topn"] == 1
    assert decide_cache(s)[0] == DECISION_REBUILD


def test_annotate_tier_change_counts_as_crossing_and_eligibility_change():
    d = annotate_differences(_diag_frame(0.5, 0.5, cached_tier="T3", fresh_tier="T4"), V4A1_SIMILARITY_CONFIG)
    s = summarize_settlement(d, 0)
    assert s["n_tier_changes"] == 1 and s["n_threshold_crossings"] == 1
    assert s["n_near_dup_eligibility_changes"] == 1 and s["n_mirror_eligibility_changes"] == 0


def test_identical_values_accept():
    d = annotate_differences(_diag_frame(0.7, 0.7), V4A1_SIMILARITY_CONFIG)
    s = summarize_settlement(d, 0)
    assert decide_cache(s)[0] == DECISION_ACCEPT
    t = mismatch_table(d)
    assert list(t["metric"]) == list(SIMILARITY_METRICS) and (t["n_mismatch"] == 0).all()


def test_nan_on_one_side_is_infinite_discrepancy():
    d = annotate_differences(_diag_frame(np.nan, 0.3), V4A1_SIMILARITY_CONFIG)
    assert np.isinf(d["absdiff_cosine"].iloc[0])


def test_recreate_c3_sample_matches_v4a1_sampling_code():
    host = pd.DataFrame({"query_id": [f"h{i}" for i in range(30)], "ref_spectrum_id": "r", "tier": "T4", "cosine": np.arange(30) / 30})
    dev = pd.DataFrame({"query_id": [f"d{i}" for i in range(50)], "ref_spectrum_id": "r", "tier": "T4", "cosine": np.arange(50) / 50})
    sample = recreate_c3_sample(host, dev, n_pairs=10, seed=42)
    combined = pd.concat([host.assign(dataset="host"), dev.assign(dataset="dev")], ignore_index=True)
    idx = np.random.RandomState(42).choice(len(combined), size=10, replace=False)
    assert sample["query_id"].tolist() == combined.iloc[idx]["query_id"].tolist()
    assert sample["dataset"].tolist() == combined.iloc[idx]["dataset"].tolist()


def _write_ok_settlement(tmp_path, decision=DECISION_ACCEPT):
    art = tmp_path / "a.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(art)
    fp, _ = evidence_fingerprint({"a": art})
    path = tmp_path / "cache_settlement.json"
    write_settlement(path, {"decision": decision, "evidence_artifacts": {"a": str(art)}, "evidence_fingerprint": fp})
    return path, art


def test_downstream_gate_accepts_valid_settlement(tmp_path):
    path, _ = _write_ok_settlement(tmp_path)
    assert load_settlement_or_refuse(path)["decision"] == DECISION_ACCEPT


@pytest.mark.parametrize("decision", [None, DECISION_REBUILD, DECISION_REBUILD_FAILED, "PASS"])
def test_downstream_gate_refuses_bad_decision(tmp_path, decision):
    path, _ = _write_ok_settlement(tmp_path, decision=decision)
    with pytest.raises(CacheSettlementError):
        load_settlement_or_refuse(path)


def test_downstream_gate_refuses_missing_file(tmp_path):
    with pytest.raises(CacheSettlementError):
        load_settlement_or_refuse(tmp_path / "nope.json")


def test_downstream_gate_refuses_changed_artifact(tmp_path):
    path, art = _write_ok_settlement(tmp_path)
    pd.DataFrame({"x": [1, 3]}).to_parquet(art)
    with pytest.raises(CacheSettlementError):
        load_settlement_or_refuse(path)


def test_v4a1_config_hash_is_the_historical_one():
    from casmi.qcr.context import config_hash
    assert config_hash(V4A1_SIMILARITY_CONFIG) == "93f7f9fa61d4"
