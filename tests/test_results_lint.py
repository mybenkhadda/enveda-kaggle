"""Record-literal lint (spec sections 11/60): none of the four v4a.1 stage notebooks may pass a
bare literal (True/False/int/float/str) as `record()`'s `value` argument -- every gate-evidence
value must come from a runtime-computed expression.
"""
import json
from pathlib import Path

import pytest

from casmi.validation.record_lint import find_literal_record_violations

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
STAGE_NOTEBOOKS = ["10v4a1_s1_host.ipynb", "10v4a1_s2_dev.ipynb", "10v4a1_s3_audits.ipynb", "10v4a1_s4_gate.ipynb"]


def _write_notebook(path, cell_sources):
    nb = {
        "cells": [{"cell_type": "code", "source": src, "metadata": {}, "outputs": [], "execution_count": None} for src in cell_sources],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f)


def test_literal_true_is_flagged(tmp_path):
    path = tmp_path / "nb.ipynb"
    _write_notebook(path, ['record("s1", "c1.pass", True)'])
    violations = find_literal_record_violations(path)
    assert len(violations) == 1


def test_literal_zero_is_flagged(tmp_path):
    path = tmp_path / "nb.ipynb"
    _write_notebook(path, ['record("s1", "c9.host.ref_mismatches", 0)'])
    assert len(find_literal_record_violations(path)) == 1


def test_literal_float_is_flagged(tmp_path):
    path = tmp_path / "nb.ipynb"
    _write_notebook(path, ['record("s1", "c6.rate", 51.42)'])
    assert len(find_literal_record_violations(path)) == 1


def test_computed_expression_is_allowed(tmp_path):
    path = tmp_path / "nb.ipynb"
    _write_notebook(path, ['record("s1", "c1.pass", bool(reproducibility_pass))'])
    assert find_literal_record_violations(path) == []


def test_computed_len_expression_is_allowed(tmp_path):
    path = tmp_path / "nb.ipynb"
    _write_notebook(path, ['record("s1", "c9.host.ref_mismatches", len(ref_mismatches))'])
    assert find_literal_record_violations(path) == []


def test_metadata_literal_is_allowed_only_value_is_checked(tmp_path):
    path = tmp_path / "nb.ipynb"
    _write_notebook(path, ['record("s1", "c8.n_sampled", int(n_sampled), metadata={"seed": 42})'])
    assert find_literal_record_violations(path) == []


def test_keyword_value_literal_is_flagged(tmp_path):
    path = tmp_path / "nb.ipynb"
    _write_notebook(path, ['record("s1", "c1.pass", value=True)'])
    assert len(find_literal_record_violations(path)) == 1


def test_syntax_error_cell_is_reported_not_silently_skipped(tmp_path):
    path = tmp_path / "nb.ipynb"
    _write_notebook(path, ["def broken(:\n    pass"])
    violations = find_literal_record_violations(path)
    assert len(violations) == 1
    assert "SYNTAX ERROR" in violations[0]["source_snippet"]


@pytest.mark.parametrize("notebook_name", STAGE_NOTEBOOKS)
def test_stage_notebook_has_no_literal_record_calls(notebook_name):
    path = NOTEBOOK_DIR / notebook_name
    if not path.exists():
        pytest.skip(f"{notebook_name} not yet generated")
    violations = find_literal_record_violations(path)
    assert violations == [], f"{notebook_name} has literal record() values: {violations}"
