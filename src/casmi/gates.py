"""Evidence-derived gate evaluation for the v4a.1 validation system: reads a YAML gate spec
(id/description/results_key/condition per requirement, see `configs/gates/v4a1_gate.yaml`) and
a flat results dict (`casmi.results.load_all_results`), and decides PASS/FAIL/NOT_PROVEN per
requirement. A `results_key` missing from `results` is ALWAYS NOT_PROVEN, never PASS -- missing
evidence must never be silently treated as success.
"""
import re

import pandas as pd
import yaml

_COND_RE = re.compile(r"^(==|!=|<=|>=|<|>)\s*(.+)$")

GATE_STATUSES = ("PASS", "FAIL", "NOT_PROVEN")


def _parse_condition(condition):
    condition = condition.strip()
    if condition == "exists":
        return ("exists", None)
    m = _COND_RE.match(condition)
    if not m:
        raise ValueError(f"unsupported gate condition: {condition!r}")
    op, rhs = m.group(1), m.group(2).strip()
    if rhs == "true":
        rhs_val = True
    elif rhs == "false":
        rhs_val = False
    else:
        try:
            rhs_val = int(rhs)
        except ValueError:
            try:
                rhs_val = float(rhs)
            except ValueError:
                rhs_val = rhs.strip("\"'")
    return (op, rhs_val)


def _apply_condition(op, rhs, value):
    if op == "exists":
        return value is not None
    if value is None:
        return False
    try:
        if op == "==":
            return value == rhs
        if op == "!=":
            return value != rhs
        if op == "<=":
            return value <= rhs
        if op == ">=":
            return value >= rhs
        if op == "<":
            return value < rhs
        if op == ">":
            return value > rhs
    except TypeError:
        return False
    raise ValueError(f"unsupported operator: {op!r}")


def load_gate_spec(path):
    """Load a gate YAML file. Accepts either a bare top-level list of requirement dicts or a
    `{"requirements": [...]}` mapping (both are normalized to a list by `evaluate`)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(spec, results):
    """`spec`: a gate spec as returned by `load_gate_spec` (list of
    `{id, description, results_key, condition}` dicts, or `{"requirements": [...]}`).
    `results`: flat `{"a.b.c": value}` dict (see `casmi.results.load_all_results`).

    Returns `DataFrame[id, description, status, value, evidence]` with
    `status in {PASS, FAIL, NOT_PROVEN}`. A `results_key` absent from `results` is ALWAYS
    NOT_PROVEN -- this function never treats missing evidence as passing."""
    requirements = spec["requirements"] if isinstance(spec, dict) and "requirements" in spec else spec
    rows = []
    for req in requirements:
        req_id = req["id"]
        description = req.get("description", "")
        results_key = req["results_key"]
        condition = req["condition"]
        op, rhs = _parse_condition(condition)

        if results_key not in results:
            rows.append({"id": req_id, "description": description, "status": "NOT_PROVEN",
                         "value": None, "evidence": f"missing results key: {results_key!r}"})
            continue

        value = results[results_key]
        ok = _apply_condition(op, rhs, value)
        rows.append({
            "id": req_id, "description": description, "status": "PASS" if ok else "FAIL",
            "value": value, "evidence": f"{results_key} {condition} -> {value!r}",
        })
    return pd.DataFrame(rows, columns=["id", "description", "status", "value", "evidence"])


def overall_status(gate_df):
    """PASS only if every requirement is PASS; any FAIL or NOT_PROVEN makes the overall gate
    FAIL (see spec section 70 -- NOT_PROVEN is never treated as passing)."""
    if len(gate_df) and (gate_df["status"] == "PASS").all():
        return "PASS"
    return "FAIL"


def summarize(gate_df):
    """Returns `(overall_status, n_fail, n_not_proven)`."""
    n_fail = int((gate_df["status"] == "FAIL").sum())
    n_not_proven = int((gate_df["status"] == "NOT_PROVEN").sum())
    return overall_status(gate_df), n_fail, n_not_proven
