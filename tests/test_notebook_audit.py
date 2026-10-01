"""The notebook auditor must (1) pass every shipped v2 notebook and (2) actually catch each failure class it claims."""
import importlib.util
import json
from pathlib import Path

import pytest

from casmi.workspace.notebook_cells import BOOTSTRAP_CELL
from casmi.workspace.versions import NOTEBOOK_API_VERSION

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("audit_notebooks", REPO / "scripts" / "audit_notebooks.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

V2_NOTEBOOKS = sorted({p for g in audit.DEFAULT_GLOBS for p in REPO.glob(g)})


@pytest.mark.parametrize("nb", V2_NOTEBOOKS, ids=[p.stem for p in V2_NOTEBOOKS])
def test_shipped_notebooks_pass_audit(nb):
    errors = [f for f in audit.audit_notebook(nb) if f["severity"] == "ERROR"]
    assert not errors, "\n".join(f"cell {f['cell']} {f['code']} [{f['token']}]: {f['message']} -> {f['replacement']}" for f in errors)


@pytest.mark.parametrize("nb", V2_NOTEBOOKS, ids=[p.stem for p in V2_NOTEBOOKS])
def test_shipped_notebooks_have_no_outputs_and_cell_ids(nb):
    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    assert all(not c.get("outputs") for c in cells if c["cell_type"] == "code")
    assert len({c.get("id") for c in cells}) == len(cells) and all(c.get("id") for c in cells)


GOOD_SETUP = (f"from casmi.workspace.bootstrap import bootstrap\n"
              f"CTX = bootstrap('15_x', notebook_api='{NOTEBOOK_API_VERSION}', uses_paths=('reports_dir',), uses_artifacts=('universe_root',),\n"
              f"                requires=('validation_regimes',))\n"
              f"CFG, P, ARTIFACTS = CTX.cfg, CTX.paths, CTX.artifacts")


def _nb(tmp_path, *code, md="## LEAKAGE AUDIT\n", name="15_test.ipynb", setup=GOOD_SETUP, boot=BOOTSTRAP_CELL, pre=()):
    cells = [{"cell_type": "markdown", "id": "m0", "metadata": {}, "source": [md]}]
    cells += [{"cell_type": "code", "id": f"p{i}", "metadata": {}, "outputs": [], "execution_count": None, "source": [c]} for i, c in enumerate(pre)]
    cells += [{"cell_type": "code", "id": "b", "metadata": {}, "outputs": [], "execution_count": None, "source": [boot]},
              {"cell_type": "code", "id": "s", "metadata": {}, "outputs": [], "execution_count": None, "source": [setup]}]
    cells += [{"cell_type": "code", "id": f"c{i}", "metadata": {}, "outputs": [], "execution_count": None, "source": [c]} for i, c in enumerate(code)]
    p = tmp_path / name
    p.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    return p


LOG = "from casmi.workspace.experiments import log_experiment\nlog_experiment(P.reports_dir, 'e', 'd', ['m'])"


def _codes(path):
    return {f["code"] for f in audit.audit_notebook(path) if f["severity"] == "ERROR"}


def test_clean_synthetic_notebook(tmp_path):
    p = _nb(tmp_path, "x = P.reports_dir / 'a'\ny = ARTIFACTS.universe_root\nz = ARTIFACTS.validation_regimes", LOG)
    assert _codes(p) == set()


def test_catches_deprecated_alias_with_replacement(tmp_path):
    p = _nb(tmp_path, "x = P.universe\ny = P.reports_dir\nz = ARTIFACTS.universe_root", LOG)
    f = [x for x in audit.audit_notebook(p) if x["code"] == "deprecated-path-alias"]
    assert f and f[0]["token"] == "P.universe" and "ARTIFACTS.universe_root" in f[0]["replacement"]


def test_catches_unknown_and_undeclared_fields_and_artifacts(tmp_path):
    p = _nb(tmp_path, "a = P.made_up\nb = P.cache_dir\nc = ARTIFACTS.universe_root\nd = ARTIFACTS.regimes_dir\ne = ARTIFACTS['not_there']", LOG)
    assert {"unknown-path-field", "undeclared-path-field", "undeclared-artifact", "unknown-artifact"} <= _codes(p)


def test_catches_obsolete_signature_and_forbidden_api(tmp_path):
    p = _nb(tmp_path, "from casmi.candidates.universe import finalize_universe\n"
                      "finalize_universe(ARTIFACTS.universe_root, [], formula_dir=1, manifest_dir=2)\n"
                      "from casmi.workspace.contract import pipeline_status\n"
                      "from casmi.workspace.colab_paths import ColabPaths\n"
                      "from casmi.validation.regimes import no_such_thing\n"
                      "z = ARTIFACTS.universe_root", LOG)
    assert {"signature-mismatch", "forbidden-api", "obsolete-import"} <= _codes(p)


def test_catches_method_mismatch_on_instances(tmp_path):
    p = _nb(tmp_path, "from casmi.validation.regimes import RegimeConfig\nRC = RegimeConfig()\nRC.config_hash(1, 2)\nRC.nope()\n"
                      "z = ARTIFACTS.universe_root", LOG)
    msgs = [f["message"] for f in audit.audit_notebook(p) if f["code"] == "signature-mismatch"]
    assert any("config_hash" in m for m in msgs) and any("nope" in m for m in msgs)


def test_catches_undefined_used_before_defined_and_hard_coded_paths(tmp_path):
    p = _nb(tmp_path, "y = undefined_thing + later\nz = '/content/drive/MyDrive/x.parquet'\nw = 'C:/Users/x'", "later = 1\nq = ARTIFACTS.universe_root", LOG)
    assert {"undefined-name", "used-before-defined", "hard-coded-path"} <= _codes(p)


def test_catches_visible_test_leakage_and_ad_hoc_git(tmp_path):
    p = _nb(tmp_path, "t = 'data/raw/test.parquet'\nsubprocess.run(['git', '-C', '/x', 'pull'])\nz = ARTIFACTS.universe_root", LOG)
    assert {"visible-test-leakage", "ad-hoc-git"} <= _codes(p)


def test_catches_imports_before_bootstrap(tmp_path):
    p = _nb(tmp_path, "z = ARTIFACTS.universe_root", LOG, pre=("from casmi.workspace.config import load_v2_config",))
    f = [x for x in audit.audit_notebook(p) if x["code"] == "import-before-bootstrap"]
    assert f and f[0]["token"] == "casmi import"


def test_catches_stale_or_edited_bootstrap_and_api_mismatch(tmp_path):
    old = BOOTSTRAP_CELL.replace("casmi-v2-bootstrap-2", "casmi-v2-bootstrap-1")
    assert "stale-bootstrap" in _codes(_nb(tmp_path, "z = ARTIFACTS.universe_root", LOG, boot=old, name="15_a.ipynb"))
    edited = BOOTSTRAP_CELL.replace("--ff-only", "")
    assert "stale-bootstrap" in _codes(_nb(tmp_path, "z = ARTIFACTS.universe_root", LOG, boot=edited, name="15_b.ipynb"))
    bad_api = GOOD_SETUP.replace(NOTEBOOK_API_VERSION, "casmi-v2-notebooks-3")
    assert "notebook-api-mismatch" in _codes(_nb(tmp_path, "z = ARTIFACTS.universe_root", LOG, setup=bad_api, name="15_c.ipynb"))


def test_catches_missing_bootstrap_leakage_section_and_logging(tmp_path):
    p = tmp_path / "16_bad.ipynb"
    p.write_text(json.dumps({"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": ["x = 1"]}],
                             "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    assert {"missing-bootstrap", "missing-leakage-audit", "missing-experiment-log"} <= _codes(p)


def test_findings_report_cell_token_and_replacement(tmp_path):
    p = _nb(tmp_path, "x = P.regimes\nz = ARTIFACTS.universe_root", LOG)
    f = [x for x in audit.audit_notebook(p) if x["code"] == "deprecated-path-alias"][0]
    assert f["cell"] is not None and f["cell_id"] == "c0" and f["token"] == "P.regimes" and f["replacement"] == "ARTIFACTS.regimes_dir"
