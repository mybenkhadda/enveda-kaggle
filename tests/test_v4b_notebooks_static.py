"""Static checks on the two v4b notebooks (no execution): every code cell parses; notebook 01 never
calls a QCR / evidence builder; no HOST metric is produced before the DEV selection is locked;
notebook 00 never hard-codes the cache decision."""
import ast
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _find(name):
    """The v4b notebooks were moved from notebooks/ to src/ after execution -- accept either."""
    for d in (_ROOT / "notebooks", _ROOT / "src"):
        if (d / name).exists():
            return d / name
    return _ROOT / "notebooks" / name


NB00 = _find("10v4b_00_cache_settlement.ipynb")
NB01 = _find("10v4b_01_modeA_ranking_freeze.ipynb")


def _code_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("path", [NB00, NB01])
def test_notebook_exists_and_every_code_cell_parses(path):
    assert path.exists(), f"{path} missing"
    for i, src in enumerate(_code_cells(path)):
        try:
            ast.parse(src)
        except SyntaxError as e:  # pragma: no cover - failure path
            pytest.fail(f"{path.name} code cell {i}: {e}")


@pytest.mark.parametrize("path", [NB00, NB01])
def test_no_outputs_committed(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    if path.parent.name == "src":
        pytest.skip("executed copy under src/ legitimately carries outputs")
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "notebook must ship without outputs"


def test_nb01_never_rebuilds_evidence():
    forbidden = ("build_qcr_resumable", "build_qcr_chunk", "rebuild_dataset", "load_fresh_representations", "walk_references",
                 "compute_reference_evidence", "SimilarityCache")
    src = "\n".join(_code_cells(NB01))
    hits = [f for f in forbidden if re.search(rf"\b{f}\b", src)]
    assert not hits, f"10v4b_01 must only consume persisted evidence, found: {hits}"


def test_nb01_host_metrics_only_after_lock():
    cells = _code_cells(NB01)
    lock_idx = next(i for i, s in enumerate(cells) if "lock_dev_selection(" in s)
    require_idx = next(i for i, s in enumerate(cells) if "require_dev_selection_lock(" in s)
    assert require_idx > lock_idx
    host_metric_tokens = ("evaluate_host", "HOST_PQ", "HOSTP", "HOST_K", "host_stress", "host_ci", "host_boot")
    for i, s in enumerate(cells[:lock_idx + 1]):
        present = [t for t in host_metric_tokens if re.search(rf"\b{t}\b", s)]
        assert not present, f"cell {i} (before/at the DEV lock) references HOST metric objects: {present}"


def test_nb01_refuses_unsettled_cache_first():
    cells = _code_cells(NB01)
    first_gate = next(i for i, s in enumerate(cells) if "load_settlement_or_refuse(" in s)
    first_load = next(i for i, s in enumerate(cells) if "read_parquet(ARTS" in s)
    assert first_gate < first_load


def test_nb00_never_hardcodes_the_decision():
    src = "\n".join(_code_cells(NB00))
    assert not re.search(r"""\bdecision\s*=\s*['"]""", src), "cache decision must come from decide_cache(), never a literal"
    assert "decide_cache(" in src and "decide_after_rebuild(" in src
