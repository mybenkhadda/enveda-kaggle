"""Local reproduction of the competition metric (MRR@25 on RDKit tautomer-canonical InChIKey14)."""
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from casmi.validation.official_metric import (KeyCache, dedupe_by_connectivity, reciprocal_rank, score_submission,  # noqa: E402
                                              split_predictions, validate_submission_frame)

PYRIDONE_ENOL, PYRIDONE_KETO = "Oc1ccccn1", "O=c1cccc[nH]1"     # tautomers -> same competition key
ALANINOL_R, ALANINOL_S = "C[C@H](N)CO", "C[C@@H](N)CO"            # stereo ignored by InChIKey14


def test_tautomers_and_stereo_share_the_competition_key():
    kc = KeyCache()
    assert kc(PYRIDONE_ENOL) == kc(PYRIDONE_KETO) is not None
    assert kc(ALANINOL_R) == kc(ALANINOL_S) is not None
    assert kc("CCO") != kc("CCCO")
    assert kc("not a smiles") is None and kc("") is None and kc(None) is None
    assert len(kc("CCO")) == 14


def test_positions_are_literal_and_capped_at_25():
    assert reciprocal_rank(["A", "B"], "B") == 0.5
    assert reciprocal_rank([None, "B"], "B") == 0.5            # invalid entries occupy their slot
    assert reciprocal_rank(["A", "A", "B"], "B") == pytest.approx(1 / 3)   # duplicates are NOT collapsed
    assert reciprocal_rank(["X"] * 25 + ["B"], "B") == 0.0       # beyond 25 ignored


def test_score_submission_end_to_end():
    truth = pd.DataFrame({"molecule_id": ["m1", "m2", "m3"], "smiles": [PYRIDONE_KETO, "CCO", "c1ccccc1"]})
    sub = pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": [f"CCC;{PYRIDONE_ENOL}", "xx;CCCO;OCC"]})
    mrr, per = score_submission(sub, truth)
    per = per.set_index("molecule_id")
    assert per.loc["m1", "rr"] == 0.5                             # tautomer of the truth at position 2
    assert per.loc["m2", "rr"] == pytest.approx(1 / 3) and per.loc["m2", "n_invalid"] == 1
    assert per.loc["m3", "rr"] == 0.0                             # missing row scores 0
    assert mrr == pytest.approx((0.5 + 1 / 3 + 0) / 3)
    with pytest.raises(ValueError):
        score_submission(sub, pd.DataFrame({"molecule_id": ["m1"], "smiles": ["(("]}))


def test_dedupe_by_connectivity_removes_wasted_slots():
    smiles, keys = dedupe_by_connectivity([PYRIDONE_ENOL, PYRIDONE_KETO, "bad(", "CCO", ALANINOL_R, ALANINOL_S, "OCC"])
    assert smiles == [PYRIDONE_ENOL, "CCO", ALANINOL_R] and len(set(keys)) == 3
    assert len(dedupe_by_connectivity([f"C{'C' * i}O" for i in range(40)])[0]) == 25


def test_validate_submission_frame():
    ok = pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["CCO;CCCO", "c1ccccc1"]})
    assert validate_submission_frame(ok, expected_ids=["m1", "m2"]) == []
    assert validate_submission_frame(ok.rename(columns={"smiles": "SMILES"}))
    bad = pd.DataFrame({"molecule_id": ["m1", "m1", "m3"], "smiles": ["CCO;OCC", "bad((", ";".join(["C" * (i + 1) for i in range(26)])]})
    probs = " | ".join(validate_submission_frame(bad, expected_ids=["m1", "m2"]))
    for needle in ("duplicate molecule_id", "expected molecule_id missing", "unexpected molecule_id", "unparseable", "repeated connectivity", "> 25"):
        assert needle in probs, needle
    assert split_predictions("A; B ;C") == ["A", "B", "C"] and split_predictions(None) == []
