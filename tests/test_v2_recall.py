"""Phase 4 candidate-recall helpers."""
import numpy as np
import pandas as pd
import pytest

from casmi.candidates.mass_index import CandidateMassIndex
from casmi.candidates.recall import formula_element_class, gate_a_assessment, recall_by_ppm, recall_sweep, truth_reachable


def test_reachability_modes():
    rows = pd.DataFrame({"candidate_id": [0, 1, -1], "candidate_sources": [["TRAIN"], ["COCONUT", "TRAIN"], []]})
    assert truth_reachable(rows, "universe").tolist() == [True, True, False]
    assert truth_reachable(rows, "external_only").tolist() == [False, True, False]
    with pytest.raises(ValueError):
        truth_reachable(rows, "nope")


def test_recall_sweep_and_summary():
    idx = CandidateMassIndex([300.0, 300.0005, 300.003, 400.0], [0, 1, 2, 3])     # 0 / 1.67 / 10 ppm from 300 (no window-edge ties)
    masses = np.array([300.0, 300.0, 400.0])
    truths = np.array([1, 2, 3])
    has_ref = np.array([True, False, False, True])
    s = recall_sweep(idx, masses, truths, [2, 20], reachable=[True, True, False], query_ids=["a", "b", "c"], has_reference_mask=has_ref)
    two = s[s.ppm == 2].set_index("query_id")
    assert two.loc["a", "truth_rank"] == 2 and np.isnan(two.loc["b", "truth_rank"])          # 10 ppm away -> outside 2 ppm
    assert np.isnan(two.loc["c", "truth_rank"])                                               # not reachable
    assert two.loc["a", "pool_share_with_reference"] == pytest.approx(0.5)
    twenty = s[s.ppm == 20].set_index("query_id")
    assert twenty.loc["b", "truth_rank"] == 3
    summ = recall_by_ppm(s, k_values=(1, 25)).set_index("ppm")
    assert summ.loc[20, "recall_all"] == pytest.approx(2 / 3) and summ.loc[2, "recall_all"] == pytest.approx(1 / 3)


def test_gate_a_assessment_is_labelled_heuristic():
    t = pd.DataFrame({"ppm": [10], "recall_all": [0.5], "recall_at_25": [0.4]})
    a = gate_a_assessment(t, ppm_ref=10)
    assert a["decision"] == "CANDIDATE_COVERAGE" and "HEURISTIC" in a["label"]
    t2 = pd.DataFrame({"ppm": [10], "recall_all": [0.95], "recall_at_25": [0.5]})
    assert gate_a_assessment(t2)["decision"] == "RANKING"


def test_formula_element_class():
    assert formula_element_class("C10H12O2") == "CHO"
    assert formula_element_class("C10H12N2OCl") == "CHNO+X"
    assert formula_element_class(None) == "unknown"
