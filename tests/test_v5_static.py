"""Static review of the v5 / v5.1 / 11 / Kaggle notebooks and the shipped inference code (no execution):
cells parse; selection happens before any HOST metric; the Mode-A selector is used; the molecule
notebook starts with the FROZEN guard; the Kaggle notebook self-tests before hidden-test inference,
writes only through the guarded writer, makes no network call and never submits; casmi_infer only
imports the shipped `casmi` subset; the bundle config carries hashes; no literal metric claims."""
import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("10v5_00_testlike_manifests", "10v5_01_scale_features", "10v5_02_scaling_and_freeze", "10v5_03_tl_regime_audit",
         "10v5_04_standard_only_provenance", "10v5_05_test_simulated_1k_pilot",
         "11_00_export_bundle", "11_01_inference_parity", "11_02_molecule_aggregation")


def _nb(name):
    """The v5 notebooks were moved to src/ -- accept src/ or notebooks/ (never the *.v5_executed archives)."""
    for d in (ROOT / "src", ROOT / "notebooks"):
        if (d / f"{name}.ipynb").exists():
            return d / f"{name}.ipynb"
    return ROOT / "src" / f"{name}.ipynb"


V5_NOTEBOOKS = [_nb(n) for n in NAMES]
KAGGLE = ROOT / "kaggle" / "kaggle_submit.ipynb"


def _cells(path, kind="code"):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == kind]


def _first(cells, token):
    return next(i for i, s in enumerate(cells) if token in s)


@pytest.mark.parametrize("path", V5_NOTEBOOKS + [KAGGLE], ids=lambda p: p.name)
def test_exists_and_parses(path):
    assert path.exists(), path
    for i, src in enumerate(_cells(path)):
        try:
            ast.parse(src)
        except SyntaxError as e:  # pragma: no cover
            pytest.fail(f"{path.name} cell {i}: {e}")


@pytest.mark.parametrize("name", ["10v5_03_tl_regime_audit", "10v5_02_scaling_and_freeze", "11_02_molecule_aggregation"])
def test_v51_notebooks_ship_without_outputs(name):
    nb = json.loads(_nb(name).read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


def test_regime_notebook_covers_both_protocols_sources_and_route():
    src = "\n".join(_cells(_nb("10v5_03_tl_regime_audit")))
    for tok in ("build_query_regimes(", "source_breakdown(", "accepted_standard", "accepted_mirror_aware", "project_route(",
                "derive_mode_a_manifests(", "NOT_BUILT", "metrics_by_regime(", "molecule_regime_mix("):
        assert tok in src, tok
    assert "fit_fold_models" not in src and "LGBMRanker" not in src          # no training in the audit
    md = "\n".join(_cells(_nb("10v5_03_tl_regime_audit"), "markdown"))
    assert "This is a question" in md                                          # the hypothesis is not asserted up front


def test_scaling_notebook_v53_testsim_strict_selects_on_tl_eval_before_host():
    cells = _cells(_nb("10v5_02_scaling_and_freeze"))
    src = "\n".join(cells)
    assert "PROTOCOL = V53_PROTOCOL" in src and "V53_RULE" in src
    for stale in ("_MODE_A", "MODE_A_MODEL_MANIFEST", "select_mode_a_headline", "TL_EVAL_mirror_aware", "preregistration_v5_1"):
        assert stale not in src, f"stale v5.1 mirror-aware path still present: {stale}"
    lock_idx = _first(cells, "lock_dev_selection(")
    assert "select_scale_headline(" in cells[lock_idx] and "candidates=V53_NESTED_ORDER" in cells[lock_idx]
    for i, s_ in enumerate(cells[:lock_idx + 1]):
        for tok in ("HOST_PQ", "host_eval", "RESULTS\\['host'\\]\\["):
            assert not re.search(tok, s_), f"cell {i} uses {tok} before the TL_EVAL lock"
    assert "Build missing scaling artifacts before model training." in src
    # every scale model scores the SAME fixed TL_EVAL table and query list
    assert "PQ = {mid: score_population(models, TL, BASE_FEATURES, TL_IDS)[0] for mid, models in MODELS.items()}" in src
    assert "freeze_status(FREEZE_EVIDENCE)" in src and "deployment_protocol_gap.json" in src


def test_molecule_notebook_guard_first_and_host_after_lock():
    cells = _cells(_nb("11_02_molecule_aggregation"))
    assert "require_frozen_spectrum_model(" in cells[0]
    assert cells[0].index("require_frozen_spectrum_model(") < cells[0].index("load_fold_models(")
    lock_idx = _first(cells, "lock_dev_selection(")
    for s in cells[:lock_idx + 1]:
        assert "host_spec" not in s and "host_holdout_spectra" not in s


def test_export_notebook_writes_selftest_fixture():
    src = "\n".join(_cells(_nb("11_00_export_bundle")))
    assert "build_fixture(" in src and "select_fixture_queries(" in src and "refresh_manifest(" in src.split("build_fixture(")[1]


def test_parity_notebook_separates_local_and_kaggle_p4():
    src = "\n".join(_cells(_nb("11_01_inference_parity")))
    assert "P4_LOCAL_PREFLIGHT" in src and "P4_KAGGLE_OFFLINE" in src and "p4_status(" in src
    assert "REPORT['P4'] =" not in src


def test_kaggle_selftest_before_inference_and_guarded_writer():
    cells = _cells(KAGGLE)
    st = _first(cells, "run_selftest_with_fallback(")
    first_inference = min(_first(cells, "candidate_features("), _first(cells, "bundle.search.search("))
    assert st < first_inference, "self-test must run before any hidden-test inference"
    src = "\n".join(cells)
    assert "write_submission(" in src and ".to_csv(WORK / 'submission.csv'" not in src
    for k in ("self_test_backend_initial", "self_test_backend_final", "self_test_feature_mismatches", "self_test_score_mismatches",
              "self_test_rank_mismatches", "numba_available", "numba_fallback_used", "offline_environment_proven"):
        assert k in src, k


def test_kaggle_notebook_offline_and_no_auto_submit():
    src = "\n".join(_cells(KAGGLE))
    forbidden = [r"\brequests\b", r"\burllib\b", r"\bsocket\b", r"\bhttp", r"kaggle\s+competitions\s+submit", r"KaggleApi", r"^\s*!",
                 r"\bsubprocess\b", r"pip install", r"network_is_disabled"]
    hits = [f for f in forbidden if re.search(f, src, flags=re.MULTILINE)]
    assert not hits, hits
    assert "/kaggle/input" in src and "run_report.json" in src and "EMERGENCY_CAP_ENABLED = False" in src


def test_inference_code_only_imports_shipped_subset():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from casmi_infer import SHARED_CASMI_MODULES
    shipped = {m.replace("/", ".")[:-3].replace(".__init__", "") for m in SHARED_CASMI_MODULES}
    files = list((ROOT / "src" / "casmi_infer").glob("*.py")) + [ROOT / "src" / m for m in SHARED_CASMI_MODULES]
    third_party_ok = {"numpy", "pandas", "lightgbm", "psutil", "pyarrow", "numba"}
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for m in mods:
                top = m.split(".")[0]
                if top == "casmi":
                    assert m in shipped, f"{f.name} imports {m}, which the bundle does not ship"
                elif top != "casmi_infer" and top not in third_party_ok:
                    assert top in sys.stdlib_module_names, f"{f.name} imports non-shipped package {m}"
                assert top != "rdkit", f"{f.name} imports RDKit"


def test_bundle_config_carries_hashes():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from casmi.bundle_export import RRF_BASELINE_AGGREGATOR, build_config
    from casmi.qcr.context import V4B_REBUILD_SIMILARITY_CONFIG
    from casmi_infer.validation import config_hash
    with pytest.raises(ValueError):                                  # v6.3: no implicit (RRF) aggregator default
        build_config(V4B_REBUILD_SIMILARITY_CONFIG, None)
    cfg = build_config(V4B_REBUILD_SIMILARITY_CONFIG, aggregator=RRF_BASELINE_AGGREGATOR)
    assert cfg["CONFIG_HASH"] == config_hash(cfg) and cfg["similarity_config_hash"] and cfg["reference_compatibility_version"]
    assert cfg["aggregator"]["name"] == "RRF" and cfg["aggregator"]["k"] == 60 and cfg["top_k_submission"] == 25
    assert config_hash(dict(cfg, top_k_submission=24)) != cfg["CONFIG_HASH"]


def test_route_threshold_fixed_in_source():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from casmi.validation.regime_audit import ROUTE_THRESHOLD
    assert ROUTE_THRESHOLD == 0.50


def test_no_literal_metric_claims_in_markdown():
    pat = re.compile(r"(MRR|Hit@\d+)[^\n|]{0,20}(=|≈|~)\s*0?\.\d{2,}")
    for p in V5_NOTEBOOKS + [KAGGLE]:
        for md in _cells(p, "markdown"):
            assert not pat.search(md), f"{p.name}: literal metric value in markdown: {pat.search(md).group(0)}"


# ---- v5.2 ---------------------------------------------------------------------------------------

def test_provenance_notebook_is_audit_only():
    src = "\n".join(_cells(_nb("10v5_04_standard_only_provenance")))
    for tok in ("fit_fold_models", "score_population", "LGBMRanker", "mrr_at_25"):
        assert tok not in src, tok
    for tok in ("build_truth_reference_pairs(", "query_level_summary(", "t2_policy(", "protocol_counts_from_shards(", "check_existing_flag_parity(",
                "assert_walk_complete(", "provenance_by_source(", "connectivity_source_summary(", "project_route_v2(", "protocol_semantics_table("):
        assert tok in src, tok
    md = "\n".join(_cells(_nb("10v5_04_standard_only_provenance"), "markdown"))
    assert "is not a leakage boundary" in md and "INDEPENDENT_REFERENCE_LIKE" in md


def test_pilot_is_1k_only_and_decides_before_host():
    cells = _cells(_nb("10v5_05_test_simulated_1k_pilot"))
    src = "\n".join(cells)
    assert "load_manifest('TL_3K'" not in src and "load_manifest('TL_10K'" not in src
    go = _first(cells, "scale_go_decision(")
    host_scoring = _first(cells, "FEATS[('HOST', p)], BASE_FEATURES")
    assert go < host_scoring, "SCALE_GO must be persisted before HOST is scored"
    assert "select_mode_a_headline" not in src and "STANDARD_SENS" in src


def test_scaling_notebook_is_gated_on_scale_go():
    cells = _cells(_nb("10v5_02_scaling_and_freeze"))
    gate = _first(cells, "scale_go_decision.json")
    train = _first(cells, "fit_fold_models(")
    assert gate < train and "RUN_3K_10K_SCALING" in cells[gate] and "raise RuntimeError" in cells[gate]


def test_production_inference_not_changed_to_test_simulated():
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "casmi_infer").glob("*.py"))
    assert "test_simulated" not in src
