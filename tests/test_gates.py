import pandas as pd
import pytest

from casmi.gates import evaluate, overall_status, summarize


def _spec(condition, results_key="k"):
    return [{"id": "G1", "description": "test requirement", "results_key": results_key, "condition": condition}]


def test_missing_results_key_is_not_proven_not_pass():
    df = evaluate(_spec("== true"), {})
    assert df.iloc[0]["status"] == "NOT_PROVEN"
    assert df.iloc[0]["value"] is None


def test_condition_eq_true():
    assert evaluate(_spec("== true"), {"k": True}).iloc[0]["status"] == "PASS"
    assert evaluate(_spec("== true"), {"k": False}).iloc[0]["status"] == "FAIL"


def test_condition_eq_false():
    assert evaluate(_spec("== false"), {"k": False}).iloc[0]["status"] == "PASS"
    assert evaluate(_spec("== false"), {"k": True}).iloc[0]["status"] == "FAIL"


def test_condition_eq_zero():
    assert evaluate(_spec("== 0"), {"k": 0}).iloc[0]["status"] == "PASS"
    assert evaluate(_spec("== 0"), {"k": 3}).iloc[0]["status"] == "FAIL"


def test_condition_le_and_ge():
    assert evaluate(_spec("<= 60"), {"k": 60}).iloc[0]["status"] == "PASS"
    assert evaluate(_spec("<= 60"), {"k": 60.5}).iloc[0]["status"] == "FAIL"
    assert evaluate(_spec(">= 5"), {"k": 5}).iloc[0]["status"] == "PASS"
    assert evaluate(_spec(">= 5"), {"k": 4}).iloc[0]["status"] == "FAIL"


def test_condition_exists():
    assert evaluate(_spec("exists"), {"k": None}).iloc[0]["status"] == "FAIL"
    assert evaluate(_spec("exists"), {"k": 0}).iloc[0]["status"] == "PASS"
    assert evaluate(_spec("exists"), {}).iloc[0]["status"] == "NOT_PROVEN"


def test_unsupported_condition_raises():
    with pytest.raises(ValueError):
        evaluate(_spec("~= 3"), {"k": 3})


def test_overall_status_pass_only_if_every_requirement_passes():
    all_pass = pd.DataFrame({"status": ["PASS", "PASS"]})
    assert overall_status(all_pass) == "PASS"

    one_fail = pd.DataFrame({"status": ["PASS", "FAIL"]})
    assert overall_status(one_fail) == "FAIL"

    one_not_proven = pd.DataFrame({"status": ["PASS", "NOT_PROVEN"]})
    assert overall_status(one_not_proven) == "FAIL"  # NOT_PROVEN must never count as passing


def test_summarize_counts_fail_and_not_proven_separately():
    df = pd.DataFrame({"status": ["PASS", "FAIL", "NOT_PROVEN", "NOT_PROVEN"]})
    status, n_fail, n_not_proven = summarize(df)
    assert status == "FAIL"
    assert n_fail == 1
    assert n_not_proven == 2


def test_evaluate_accepts_dict_with_requirements_key():
    spec = {"requirements": _spec("== true")}
    df = evaluate(spec, {"k": True})
    assert df.iloc[0]["status"] == "PASS"


def test_real_gate_yaml_loads_and_has_31_requirements():
    import yaml
    from pathlib import Path
    spec_path = Path(__file__).resolve().parents[1] / "configs" / "gates" / "v4a1_gate.yaml"
    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    ids = [r["id"] for r in spec["requirements"]]
    assert ids == [f"G{i}" for i in range(1, 32)]
