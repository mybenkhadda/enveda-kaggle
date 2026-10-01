import math

import pandas as pd

from casmi.validation.test_like import (
    detect_primary_holdout,
    failure_stage_table,
    mode_a_candidate_availability,
    mode_b_candidate_availability,
    mode_c_candidate_availability,
    summarize_modes,
)


def test_detect_primary_holdout_found():
    train_meta = pd.DataFrame({"ingest_lib": ["enveda-180", "enveda-np-examples", "gnps"]})
    found, subset = detect_primary_holdout(train_meta)
    assert found
    assert len(subset) == 1


def test_detect_primary_holdout_not_found():
    train_meta = pd.DataFrame({"ingest_lib": ["enveda-180", "gnps"]})
    found, subset = detect_primary_holdout(train_meta)
    assert not found
    assert subset is None


def test_detect_primary_holdout_missing_column():
    train_meta = pd.DataFrame({"other_col": [1, 2]})
    found, subset = detect_primary_holdout(train_meta)
    assert not found


def _pair_features(true_present_for):
    rows = []
    for q in ["q1", "q2", "q3"]:
        rows.append({"query_id": q, "is_true_candidate": q in true_present_for, "has_reference_spectrum": True})
    return pd.DataFrame(rows)


def test_mode_a_and_b_candidate_availability():
    known = _pair_features(true_present_for={"q1", "q2"})
    unseen = _pair_features(true_present_for={"q1"})
    assert math.isclose(mode_a_candidate_availability(known, ["q1", "q2", "q3"]), 2 / 3)
    assert math.isclose(mode_b_candidate_availability(unseen, ["q1", "q2", "q3"]), 1 / 3)


def test_mode_c_always_zero():
    assert mode_c_candidate_availability(["q1", "q2"]) == 0.0


def test_summarize_modes_reports_all_three_separately():
    known = _pair_features(true_present_for={"q1", "q2"})
    unseen = _pair_features(true_present_for={"q1"})
    table = summarize_modes(known, unseen, ["q1", "q2", "q3"])
    assert set(table["mode"]) == {"A_reference_available", "B_database_only", "C_structure_absent"}
    c_row = table[table["mode"] == "C_structure_absent"].iloc[0]
    assert c_row["candidate_availability"] == 0.0


def test_failure_stage_table_separates_missing_from_unscored():
    query_summary = pd.DataFrame({"query_id": ["q1", "q2", "q3"], "target_abs_mass_error_ppm": [1.0, 1.0, 100.0]})
    pair_features = pd.DataFrame({
        "query_id": ["q1"], "is_true_candidate": [True], "has_reference_spectrum": [True],
    })
    table = failure_stage_table(query_summary, pair_features, ["q1", "q2", "q3"], tolerance_ppm=5)
    stages = table.set_index("failure_stage")
    assert stages.loc["missing from candidate pool", "count"] == 1  # q3 (100ppm > 5ppm tolerance)
    assert stages.loc["candidate retrieved, direct reference unavailable", "count"] == 1  # q2: in pool, not scored
    assert stages.loc["candidate retrieved and scored", "count"] == 1  # q1
    assert table["count"].sum() == 3
