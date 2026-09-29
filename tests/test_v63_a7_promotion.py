"""v6.3: promotion of the MOL_DEV-locked A7 aggregator into the bundle -- locked-artifact loading and refusal,
the aggregator + frozen-calibration contract at bundle load, A7 semantics (spectrum choice, deterministic ties,
connectivity dedup before Top-25), RRF only when explicit, the format-2 self-test (spectrum + molecule level)
blocking submission on any parity failure, stricter submission validation, run-report completeness, and a
submission logger that never submits."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from casmi import bundle_export as bx
from casmi import submissions
from casmi.ranking import molecule_eval as me
from casmi_infer import pipeline, selftest as st
from casmi_infer.aggregation import MostConfidentSpectrum, RRFAggregator, make_aggregator
from casmi_infer.features import SPECTRAL_FEATURES
from casmi_infer.ranker import rank_spectrum
from casmi_infer.submission import SubmissionBlocked, write_submission
from casmi_infer.validation import (RUN_REPORT_REQUIRED, AggregatorContractError, assert_run_report_complete, check_smiles_parse, sha256_file,
                                    validate_rankings, verify_aggregator_contract)

ROOT = Path(__file__).resolve().parents[1]
T = 1.1947
MODEL = "V1_TL_1K_TESTSIM_STRICT"
STRICT = "test_simulated_strict"


# ---- locked artifacts --------------------------------------------------------------------------------------

def _artifacts(tmp_path, **over):
    sel = {"selected_aggregator": "A7_most_confident_spectrum", "selection_protocols": [STRICT], "spectrum_model": MODEL, "temperature": T,
           "aggregator_config": {"name": "A7_most_confident_spectrum", "temperature": T}, "preregistration_sha256": "rule1"}
    lock = {"selected_model_id": "A7_most_confident_spectrum", "alphas": {"temperature": T}, "preregistration_sha256": "rule1"}
    cal = {"temperature_fit": {"temperature": T}, "host_used": False}
    for k, v in over.items():
        target, key = k.split("__")
        {"sel": sel, "lock": lock, "cal": cal}[target][key] = v
    paths = {}
    for name, obj in (("selected_aggregator.json", sel), ("aggregator_lock.json", lock), ("calibration.json", cal)):
        paths[name] = tmp_path / name
        paths[name].write_text(json.dumps(obj), encoding="utf-8")
    return paths


FREEZE = {"freeze_status": "FROZEN", "model_id": MODEL, "protocol": STRICT}


def _load(p, freeze=FREEZE):
    return bx.load_locked_aggregator(p["selected_aggregator.json"], p["aggregator_lock.json"], p["calibration.json"], freeze)


def test_locked_a7_loads_as_production_config(tmp_path):
    promo = _load(_artifacts(tmp_path))
    agg = promo["aggregator"]
    assert agg["name"] == "MOST_CONFIDENT_SPECTRUM" and agg["aggregator_id"] == "A7_most_confident_spectrum" and agg["temperature"] == T
    assert isinstance(make_aggregator(agg), MostConfidentSpectrum)
    assert set(promo["sha256"]) == {"selected_aggregator.json", "calibration.json", "aggregator_lock.json"}


@pytest.mark.parametrize("over, freeze", [
    ({"sel__selected_aggregator": "A6_rrf", "lock__selected_model_id": "A6_rrf"}, FREEZE),        # a different (RRF) selection is refused
    ({"sel__selection_protocols": [STRICT, "mirror_aware"]}, FREEZE),                             # sensitivity must not have selected
    ({"sel__spectrum_model": "V1_TL_1K"}, FREEZE),                                                # selection made for another model
    ({}, {**FREEZE, "model_id": "V1_TL_1K"}),                                                     # wrong frozen model
    ({}, {**FREEZE, "freeze_status": "PROVISIONAL"}),
    ({"cal__temperature_fit": {"temperature": T + 1e-9}}, FREEZE),                                # temperature drift
    ({"lock__alphas": {"temperature": 2.0}}, FREEZE),
    ({"cal__host_used": True}, FREEZE),
    ({"lock__preregistration_sha256": "other"}, FREEZE),
])
def test_inconsistent_locked_artifacts_are_rejected(tmp_path, over, freeze):
    with pytest.raises(bx.AggregatorPromotionError):
        _load(_artifacts(tmp_path, **over), freeze)


def test_missing_artifact_rejected(tmp_path):
    p = _artifacts(tmp_path)
    p["calibration.json"].unlink()
    with pytest.raises(bx.AggregatorPromotionError):
        _load(p)


def test_bundle_models_must_match_freeze(tmp_path):
    from casmi.ranking.freeze import feature_order_hash
    m = tmp_path / "models"
    m.mkdir()
    for k in range(2):
        (m / f"v1_fold{k}.txt").write_text(f"booster {k}", encoding="utf-8")
    names = ["a", "b"]
    (m / "feature_names.json").write_text(json.dumps(names), encoding="utf-8")
    (m / "model_info.json").write_text(json.dumps({"model_id": MODEL}), encoding="utf-8")
    good = {"model_id": MODEL, "feature_names": names, "feature_order_hash": feature_order_hash(names),
            "model_hashes": {f"fold_{k}.txt": sha256_file(m / f"v1_fold{k}.txt") for k in range(2)}}
    assert bx.verify_bundle_models_match_freeze(tmp_path, good)["n_folds"] == 2
    for bad in ({**good, "model_hashes": {**good["model_hashes"], "fold_1.txt": "0" * 64}}, {**good, "model_id": "V1_TL_1K"},
                {**good, "feature_names": ["b", "a"]}):
        with pytest.raises(bx.AggregatorPromotionError):
            bx.verify_bundle_models_match_freeze(tmp_path, bad)


# ---- aggregator + calibration contract at bundle load ----------------------------------------------------------

def _bundle_dir(tmp_path, t=T):
    d = tmp_path / "bundle"
    (d / "aggregation").mkdir(parents=True)
    (d / "aggregation" / "calibration.json").write_text(json.dumps({"temperature_fit": {"temperature": t}, "host_used": False}), encoding="utf-8")
    (d / "aggregation" / "selected_aggregator.json").write_text(json.dumps(
        {"selected_aggregator": "A7_most_confident_spectrum", "temperature": t, "aggregator_config": {"temperature": t}, "spectrum_model": MODEL}), encoding="utf-8")
    cfg = {"aggregator": {"name": "MOST_CONFIDENT_SPECTRUM", "temperature": t}, "aggregator_name": "MOST_CONFIDENT_SPECTRUM", "model_id": MODEL,
           "calibration_temperature": t, "calibration_path": "aggregation/calibration.json",
           "calibration_sha256": sha256_file(d / "aggregation" / "calibration.json"), "selection_artifact_path": "aggregation/selected_aggregator.json",
           "selection_artifact_sha256": sha256_file(d / "aggregation" / "selected_aggregator.json")}
    return d, cfg


def test_contract_accepts_consistent_a7_bundle(tmp_path):
    d, cfg = _bundle_dir(tmp_path)
    c = verify_aggregator_contract(cfg, d)
    assert c["aggregator_id"] == "A7_most_confident_spectrum" and c["temperature"] == T and c["needs_prob"]


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("calibration_path"),
    lambda c: c.pop("calibration_sha256"),
    lambda c: c.update(calibration_sha256="0" * 64),
    lambda c: c.update(calibration_temperature=T * 2),
    lambda c: c["aggregator"].pop("temperature"),
    lambda c: c.update(model_id="V1_TL_1K"),
    lambda c: c.update(aggregator_name="RRF"),
])
def test_missing_or_mismatched_calibration_rejected(tmp_path, mutate):
    d, cfg = _bundle_dir(tmp_path)
    mutate(cfg)
    with pytest.raises(AggregatorContractError):
        verify_aggregator_contract(cfg, d)


def test_calibration_file_tampering_rejected(tmp_path):
    d, cfg = _bundle_dir(tmp_path)
    (d / "aggregation" / "calibration.json").write_text(json.dumps({"temperature_fit": {"temperature": 2.0}}), encoding="utf-8")
    with pytest.raises(AggregatorContractError):
        verify_aggregator_contract(cfg, d)


# ---- RRF: explicit only --------------------------------------------------------------------------------------

def test_explicit_rrf_still_works(tmp_path):
    assert isinstance(make_aggregator({"name": "RRF", "k": 60}), RRFAggregator)
    c = verify_aggregator_contract({"aggregator": {"name": "RRF", "k": 60}}, tmp_path)
    assert c["aggregator_id"] == "A6_rrf" and not c["needs_prob"] and c["temperature"] is None
    from casmi.qcr.context import V4B_REBUILD_SIMILARITY_CONFIG
    cfg = bx.build_config(V4B_REBUILD_SIMILARITY_CONFIG, aggregator=bx.RRF_BASELINE_AGGREGATOR)
    assert cfg["aggregator_name"] == "RRF"


def test_rrf_never_substitutes_silently(tmp_path):
    for bad in ({}, None, {"name": ""}, {"k": 60}):
        with pytest.raises(ValueError):
            make_aggregator(bad)
    with pytest.raises(AggregatorContractError):
        verify_aggregator_contract({}, tmp_path)
    with pytest.raises(ValueError):
        bx.build_config({"bin_width_da": .1, "peak_tol_da": .02, "max_peaks_similarity": 100}, None)   # refused before hashing
    with pytest.raises(AggregatorContractError):                                       # A7 without T fails, it does not fall back
        verify_aggregator_contract({"aggregator": {"name": "MOST_CONFIDENT_SPECTRUM"}}, tmp_path)
    fake = SimpleNamespace(aggregator=make_aggregator({"name": "MOST_CONFIDENT_SPECTRUM"}), temperature=None)
    with pytest.raises(RuntimeError):
        pipeline.aggregate_molecules(fake, _per_spec())
    src = inspect.getsource(pipeline.aggregate_molecules) + inspect.getsource(pipeline.calibrated_probabilities)
    assert '.get("temperature", 1.0)' not in src


def test_config_only_swap_is_retired(tmp_path):
    with pytest.raises(bx.AggregatorPromotionError):
        bx.apply_aggregator_config(tmp_path, {"name": "MOST_CONFIDENT_SPECTRUM", "temperature": T}, {})


# ---- A7 semantics ----------------------------------------------------------------------------------------------

def _per_spec():
    """molecule M: s2 is the most confident spectrum; s1 and s3 tie with each other; candidate 7 appears in two spectra."""
    rows = [("M", "s1", 7, 2.0, 1.0), ("M", "s1", 8, 1.0, 1.0), ("M", "s2", 9, 6.0, 2.0), ("M", "s2", 7, 1.0, 1.0), ("M", "s2", 8, 0.5, 3.0),
            ("M", "s3", 5, 2.0, 1.0), ("M", "s3", 6, 1.0, 1.0), ("N", "t1", 1, 1.0, 1.0), ("N", "t1", 2, 1.0, 0.5)]
    d = pd.DataFrame(rows, columns=["molecule_id", "spectrum_id", "conn_idx", "score", "abs_mass_error_ppm"])
    d["rank"] = d.sort_values(["spectrum_id", "score", "abs_mass_error_ppm", "conn_idx"], ascending=[True, False, True, True]).groupby("spectrum_id").cumcount().reindex(d.index) + 1
    return d.assign(model_scored=True)


def _fake_bundle(t=T):
    return SimpleNamespace(aggregator=make_aggregator({"name": "MOST_CONFIDENT_SPECTRUM", "temperature": t}), temperature=t)


def test_a7_chooses_the_most_confident_spectrum():
    b = _fake_bundle()
    sel = pipeline.selected_spectra(b, _per_spec()).set_index("molecule_id")
    assert sel.loc["M", "spectrum_id"] == "s2"
    r = pipeline.aggregate_molecules(b, _per_spec())
    m = r[r["molecule_id"] == "M"].sort_values("mol_rank")
    assert m["conn_idx"].tolist() == [9, 7, 8]                                   # s2's own ranking; candidates absent from s2 are not ranked


def test_a7_confidence_ties_are_deterministic():
    d = _per_spec()
    d = d[d["spectrum_id"] != "s2"]                                             # s1 and s3 have identical score profiles -> tie
    b = _fake_bundle()
    outs = [pipeline.selected_spectra(b, d.sample(frac=1, random_state=k)).set_index("molecule_id").loc["M", "spectrum_id"] for k in range(6)]
    assert outs == ["s1"] * 6                                                   # confidence DESC, spectrum_id ASC
    n = pipeline.selected_spectra(b, d).set_index("molecule_id").loc["N"]
    assert n["spectrum_id"] == "t1"


def test_fallback_spectra_do_not_win_a7():
    d = pd.concat([_per_spec(), pd.DataFrame({"molecule_id": "M", "spectrum_id": "s0", "conn_idx": [99, 98], "score": [0.0, -50.0],
                                              "abs_mass_error_ppm": [0.0, 50.0], "rank": [1, 2], "model_scored": False})], ignore_index=True)
    counters = {}
    sel = pipeline.selected_spectra(_fake_bundle(), d, counters).set_index("molecule_id")
    assert sel.loc["M", "spectrum_id"] == "s2" and counters["fallback_spectra_excluded_from_prob_aggregation"] == 1
    only_fb = d[d["spectrum_id"] == "s0"]                                      # a molecule with ONLY fallback spectra keeps the fallback
    assert pipeline.selected_spectra(_fake_bundle(), only_fb).set_index("molecule_id").loc["M", "spectrum_id"] == "s0"


def test_dedup_before_top25():
    n = 40
    d = pd.DataFrame({"molecule_id": "M", "spectrum_id": "s1", "conn_idx": np.arange(n), "score": np.linspace(5, 0, n),
                      "abs_mass_error_ppm": 1.0, "rank": np.arange(1, n + 1), "model_scored": True})
    r = pipeline.aggregate_molecules(_fake_bundle(), d)
    assert not r.duplicated(["molecule_id", "conn_idx"]).any() and len(r) == n       # one row per connectivity BEFORE any cut
    from casmi_infer.submission import molecule_top_k
    top = molecule_top_k(r, [f"C{c}" for c in range(n)], top_k=25)
    assert top["M"] == [f"C{c}" for c in range(25)]                                  # Top-25 = the first 25 unique, in rank order
    assert validate_rankings({"M": [1, 2, 2]}, ["M"]) and not validate_rankings({"M": list(range(25))}, ["M"])
    assert validate_rankings({"M": list(range(26))}, ["M"])


# ---- format-2 self-test: spectrum AND molecule level ------------------------------------------------------------

def _infer(bundle, row):
    k = int(row["spectrum_id"][1:])
    rng = np.random.default_rng(k)
    n = 30
    f = pd.DataFrame({"conn_idx": rng.choice(60, n, replace=False), "abs_mass_error_ppm": rng.uniform(0, 5, n), **{c: rng.random(n) for c in SPECTRAL_FEATURES}})
    folds = np.vstack([rng.normal(size=n) * (1 + k % 3), rng.normal(size=n)])
    f = f.assign(score_fold0=folds[0], score_fold1=folds[1])
    return rank_spectrum(f, folds.mean(axis=0))


def _fixture_bundle(tmp_path, t=T):
    return SimpleNamespace(dir=tmp_path, config={"CONFIG_HASH": "h1", "top_k_submission": 25, "aggregator": {"name": "MOST_CONFIDENT_SPECTRUM", "temperature": t}},
                           manifest={"files": {"models/v1_fold0.txt": "a", "models/v1_fold1.txt": "b"}},
                           ranker=SimpleNamespace(feature_names=list(SPECTRAL_FEATURES)), temperature=t,
                           aggregator=make_aggregator({"name": "MOST_CONFIDENT_SPECTRUM", "temperature": t}))


def _queries():
    return pd.DataFrame({"spectrum_id": [f"s{i}" for i in range(1, 11)]})


def test_selftest_fixture_roundtrip_passes(tmp_path):
    b = _fixture_bundle(tmp_path)
    man, _ = st.build_fixture(b, _queries(), tmp_path, infer_fn=_infer)
    assert man["format"] == st.FIXTURE_FORMAT and man["n_molecules"] >= 3 and man["temperature"] == T
    res = st.run_selftest(b, "numpy", infer_fn=_infer)
    assert res["passed"] and all(res[k] == 0 for k in st.MISMATCH_KEYS) and res["n_fixture_molecules"] == man["n_molecules"]
    exp_mol = pd.read_parquet(tmp_path / "selftest" / "expected_molecules.parquet")
    assert exp_mol["selected_spectrum_id"].notna().all()


def test_selftest_detects_molecule_level_drift_and_blocks_submission(tmp_path):
    b = _fixture_bundle(tmp_path)
    st.build_fixture(b, _queries(), tmp_path, infer_fn=_infer)

    def drifted(bundle, row):
        r = _infer(bundle, row)
        if row["spectrum_id"] == "s1":
            r = r.assign(score_fold0=r["score_fold0"] * 3)
            r = rank_spectrum(r.drop(columns=["score", "rank"]), r[["score_fold0", "score_fold1"]].mean(axis=1).to_numpy())
        return r
    res = st.run_selftest(b, "numpy", infer_fn=drifted)
    assert not res["passed"] and res["fold_score_mismatches"] > 0 and res["prob_mismatches"] > 0
    with pytest.raises(SubmissionBlocked):
        write_submission(pd.DataFrame({"molecule_id": ["m"], "smiles": ["C"]}), tmp_path / "submission.csv", res)
    assert not (tmp_path / "submission.csv").exists()


def test_stale_fixture_refused(tmp_path):
    b = _fixture_bundle(tmp_path)
    st.build_fixture(b, _queries(), tmp_path, infer_fn=_infer)
    for stale in (_fixture_bundle(tmp_path, t=T * 1.01), SimpleNamespace(**{**vars(_fixture_bundle(tmp_path)), "config": {**b.config, "CONFIG_HASH": "h2"}})):
        with pytest.raises(st.SelfTestFailed):
            st.run_selftest(stale, "numpy", infer_fn=_infer)
    m = json.loads((tmp_path / "selftest" / "fixture_manifest.json").read_text())
    (tmp_path / "selftest" / "fixture_manifest.json").write_text(json.dumps({**m, "format": "casmi-selftest-1"}))
    with pytest.raises(st.SelfTestFailed):
        st.load_fixture(tmp_path)


def test_compare_molecules_counts():
    em = pd.DataFrame({"molecule_id": ["A", "B"], "selected_spectrum_id": ["s1", "s3"], "confidence": [.9, .5]})
    er = pd.DataFrame({"molecule_id": ["A", "A", "B"], "mol_rank": [1, 2, 1], "conn_idx": [1, 2, 3]})
    assert st.compare_molecules(em, em, er, er)["molecule_ranking_mismatches"] == 0
    am = em.assign(selected_spectrum_id=["s1", "s4"], confidence=[.9, .5 + 1e-6])
    ar = er.assign(conn_idx=[2, 1, 3])
    c = st.compare_molecules(am, em, ar, er)
    assert c["selected_spectrum_mismatches"] == 1 and c["confidence_mismatches"] == 1 and c["molecule_ranking_mismatches"] == 1


def test_fixture_molecules_deterministic():
    a = st.assign_fixture_molecules(["s3", "s1", "s2", "s10", "s4"])
    assert a == st.assign_fixture_molecules(["s10", "s4", "s3", "s2", "s1"])
    assert pd.Series(a).value_counts().sort_index().tolist() == [3, 1, 1]


# ---- submission validation / run report / logger -------------------------------------------------------------------

def test_invalid_submission_blocked_by_validators():
    assert validate_rankings({"A": [1], "B": []}, ["A", "B"]) and validate_rankings({"A": [1]}, ["A", "A"])
    assert validate_rankings({"A": [1]}, ["A", None]) and validate_rankings({"A": [1], "Z": [2]}, ["A"])
    assert check_smiles_parse(["C", "X("], parse_fn=lambda s: s == "C")[0] == "FAIL"
    assert check_smiles_parse(["C"], parse_fn=lambda s: True)[0] == "PARSER_PASS"
    assert check_smiles_parse(["C"], None, None)[0] == "UNVERIFIED"
    assert check_smiles_parse(["C", "Q"], {"n_checked": 2, "failed_smiles": ["Q"]})[0] == "FAIL"
    assert check_smiles_parse(["C"], {"n_checked": 2, "failed_smiles": ["Q"]})[0] == "EXPORT_VALIDATED"


def test_run_report_fields_required():
    full = {k: 1 for k in RUN_REPORT_REQUIRED} | {"aggregator": "MOST_CONFIDENT_SPECTRUM"}
    assert assert_run_report_complete(full)
    for k in ("calibration_hash", "self_test_status", "candidate_pair_count"):
        with pytest.raises(ValueError):
            assert_run_report_complete({**full, k: None})
    with pytest.raises(ValueError):
        assert_run_report_complete({k: v for k, v in full.items() if k != "submission_validation_status"})
    assert assert_run_report_complete({**full, "aggregator": "RRF", "calibration_hash": None, "aggregator_artifact_hash": None})


def test_submission_logger_never_submits(tmp_path):
    import ast
    src = inspect.getsource(submissions)
    imported = {a.name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"requests", "urllib", "subprocess", "socket", "http", "kaggle"}, imported
    assert "KaggleApi" not in src and "competitions submit" not in src
    rep = {"bundle_version": "v2-A7-h-M", "CONFIG_HASH": "abc", "model_id": MODEL, "aggregator": "MOST_CONFIDENT_SPECTRUM",
           "aggregator_config": {"aggregator_id": "A7_most_confident_spectrum"}, "candidate_universe_version": "closed_world_train_library:x:y",
           "self_test_status": "PASS", "backend": "numba", "numpy_fallback_used": False, "runtime_seconds": 1.0, "peak_ram": 2.0}
    (tmp_path / "run_report.json").write_text(json.dumps(rep))
    f = submissions.fields_from_run_report(tmp_path / "run_report.json")
    e = submissions.log_submission(tmp_path / "log.jsonl", submission_name="anchor-A7", notes="anchor", **f)
    assert e["CONFIG_HASH"] == "abc" and e["aggregator"] == "MOST_CONFIDENT_SPECTRUM" and e["aggregator_id"] == "A7_most_confident_spectrum"
    assert e["public_score"] is None and e["candidate_universe_version"] == "closed_world_train_library:x:y"
    assert set(submissions.FIELDS) <= set(submissions.load_submission_log(tmp_path / "log.jsonl")[0])


def test_stale_selection_wording_fixed():
    assert me.select_across_protocols({STRICT: "A7_most_confident_spectrum"}, {"A7_most_confident_spectrum": 1})[1] == \
        "selected on primary protocol; sensitivity protocol did not influence selection"
    assert "all evaluation protocols agree" not in inspect.getsource(me)


# ---- notebooks (static) -------------------------------------------------------------------------------------------

def _code(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_kaggle_notebook_a7_wiring():
    cells = _code(ROOT / "kaggle" / "kaggle_submit.ipynb")
    src = "\n".join(cells)
    first = lambda tok: next(i for i, s in enumerate(cells) if tok in s)
    assert "EXPECTED_AGGREGATOR = 'MOST_CONFIDENT_SPECTRUM'" in src
    assert first("run_selftest_with_fallback(") < first("candidate_features(") < first("aggregate_molecules(") < first("write_submission(")
    w = cells[first("write_submission(")]
    assert w.index("validate_rankings(") < w.index("write_submission(") and "check_smiles_parse(" in w and "raise RuntimeError" in w
    assert "assert_run_report_complete(report)" in src
    for k in RUN_REPORT_REQUIRED:
        assert f"'{k}'" in src, k
    assert "model_scored=model_scored" in src


def test_export_notebook_promotes_locked_aggregator_before_writing():
    nb = ROOT / "src" / "11_00_export_bundle.ipynb"
    if not nb.exists():
        pytest.skip("local-only export notebook (not shipped to the Colab runtime repo)")
    cells = _code(nb)
    first = lambda tok: next(i for i, s in enumerate(cells) if tok in s)
    assert first("load_locked_aggregator(") < first("export_structures(")                  # refuse BEFORE anything is written
    assert first("archive_bundle_metadata(") < first("export_structures(")                 # never overwrite silently
    c = cells[first("export_models(")]
    assert c.index("export_models(") < c.index("verify_bundle_models_match_freeze(") < c.index("build_config(")
    assert "aggregator=AGGREGATOR_CFG" in c and "calibration=CALIBRATION" in c and "candidate_universe_version=" in c
    assert "PROMOTE_LOCKED_AGGREGATOR = True" in cells[0]
