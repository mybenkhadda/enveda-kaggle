"""Static checks on the v2 Colab notebooks (nothing is executed): valid nbformat JSON, every code cell
parses, no local Windows paths, no visible-test / submission files as validation input, Drive persistence,
logic lives in modules (cells stay short)."""
import ast
import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
V2 = ["10_colab_v2_setup.ipynb", "11_colab_hidden_like_validation.ipynb", "12_colab_candidate_universe.ipynb",
      "13_colab_candidate_recall.ipynb", "14_colab_analog_baseline.ipynb"]


def _cells(name):
    nb = json.loads((NB_DIR / name).read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return nb, ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("name", V2)
def test_code_cells_parse(name):
    _, cells = _cells(name)
    assert cells
    for src in cells:
        ast.parse(src)


@pytest.mark.parametrize("name", V2)
def test_no_local_paths_or_competition_test_inputs(name):
    nb, cells = _cells(name)
    code = "\n".join(cells)
    for bad in ("C:\\", "OneDrive", "test.parquet", "sample_submission", "casmi_runtime", "run_named_inference"):
        assert bad not in code, f"{name}: {bad!r}"
    assert "drive.mount" in code and "ENVEDA_DRIVE_ROOT" in code


@pytest.mark.parametrize("name", V2[1:])
def test_results_go_to_drive_paths(name):
    _, cells = _cells(name)
    code = "\n".join(cells)
    assert "/content/" not in code.replace("/content/drive", "")    # only Drive (+ config scratch) -- no unique results in /content
    assert "P." in code                                              # every artifact path comes from the v2 config


@pytest.mark.parametrize("name", V2)
def test_cells_stay_orchestration_sized(name):
    _, cells = _cells(name)
    assert max(len(c.splitlines()) for c in cells) <= 40
    assert not any(ast.parse(c).body and any(isinstance(n, ast.ClassDef) for n in ast.walk(ast.parse(c))) for c in cells)


def test_setup_notebook_trains_nothing():
    _, cells = _cells("10_colab_v2_setup.ipynb")
    code = "\n".join(cells)
    for bad in (".fit(", "backward(", "optimizer", "fit_fold_models"):
        assert bad not in code
