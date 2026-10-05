"""Validation protocol v2: legacy universe adoption + legacy Stage-A discovery, eligibility-based regimes with an
identity-bound manifest, panels (TIMSTOF_DEV never decides, HOST file-backed, NP_PANEL structure-only), protocol validity
cases A-D, A8 threshold block, round-trip diagnosis / deployment gate, bound experiment rows, public-baseline interface,
fingerprint exclusion contract, PubChem build-script source checks, notebooks 11a / 11b / 13 / 14 static structure.
Synthetic fixtures only; nothing here reads project data. RDKit is never required."""
import ast
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi.candidates.legacy_adoption import (ADOPTED_MANIFEST_NAME, LEGACY_EXTERNAL_VERIFIED, LEGACY_TRAIN_ONLY, VERIFICATION_FAILED,
                                              LegacyAdoptionRefused, adopt_legacy_universe, discover_legacy_stage_a,
                                              effective_universe_identity, select_stage_a_namespaces, verify_legacy_universe)
from casmi.ranking.fingerprint_contract import (ExclusionContractError, assert_dataset_respects_exclusions,
                                                build_fingerprint_dataset_manifest)
from casmi.validation.external_coverage import DecisionPolicy
from casmi.validation.panels import (ALL_DEV, HOST, HOST_PANEL_SOURCE_MISSING, NOT_AVAILABLE, NP_PANEL, NP_THRESHOLD_NOT_SET, TIMSTOF_DEV,
                                     PanelConfig, build_panels, canonical_panel_name, panel_manifest, select_decision_panel)
from casmi.validation.protocol_v2 import (BLOCKED_A8_THRESHOLDS, INVALID, ROUNDTRIP_DEPLOYMENT_BLOCKED, ROUNDTRIP_DEPLOYMENT_OK, VALID,
                                          VALID_READ_ONLY, ProtocolV2Error, a8_threshold_status, final_decision_v2, policy_identity_v2,
                                          protocol_validity_v2, require_protocol_v2_ready, roundtrip_deployment_check)
from casmi.validation.public_baseline import PublicBaselineSpec, load_public_baseline
from casmi.validation.regimes_v2 import (REGIME_CURRENT, REGIME_MANIFEST_MISSING, REGIME_REBUILD_REQUIRED, UNRESOLVED,
                                         RegimeRebuildRequired, archive_legacy_regimes, assert_regime_v2_consistency, assign_regimes_v2,
                                         build_regime_identity, check_regime_manifest, external_reachability, molecule_records,
                                         regime_counts_by_panel, regime_signature, require_current_regimes, write_exclusion_manifests,
                                         write_regime_manifest)
from casmi.validation.roundtrip_diagnostics import diagnose_roundtrip_failures, diagnosis_summary
from casmi.workspace.experiments import (ExperimentBindingError, comparable_experiments, load_experiments, log_bound_experiment,
                                         log_experiment)

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def _k(i):
    return f"{chr(65 + i // 100)}{i % 100:02d}" + "X" * 11          # 14-char, sorts by bucket letter


# ---------------------------------------------------------------------------------------------
# synthetic legacy universe (pre-v3 manifest layout of commit 85d43d9)
# ---------------------------------------------------------------------------------------------

def _legacy_universe(root, rows, mass_n=None, formula_n=None):
    """rows: list of (key, sources). Writes buckets, candidate_keys.npy, bucket_offsets.json, index / formula meta, legacy manifest."""
    root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["connectivity_key", "candidate_sources"]).sort_values("connectivity_key")
    df["in_train"] = df["candidate_sources"].map(lambda s: "TRAIN" in s)
    df["bucket"] = df["connectivity_key"].str[0]
    offsets, start = {}, 0
    (root / "buckets").mkdir(exist_ok=True)
    for b, g in df.groupby("bucket", sort=True):
        g.drop(columns="bucket").to_parquet(root / "buckets" / f"bucket={b}.parquet", index=False)
        offsets[b] = start
        start += len(g)
    np.save(root / "candidate_keys.npy", df["connectivity_key"].to_numpy().astype("S14"))
    (root / "bucket_offsets.json").write_text(json.dumps({"offsets": offsets, "n_candidates": start}))
    for d, name, n in (("index", "mass_index.meta.json", mass_n), ("formula_index", "formula_index.meta.json", formula_n)):
        (root / d).mkdir(exist_ok=True)
        (root / d / name).write_text(json.dumps({"n_candidates": start if n is None else n, "n_rows": start, "n_indexed": start}))
    (root / "universe_manifest.json").write_text(json.dumps({"universe_schema_version": "v2", "n_candidates": start, "n_buckets": len(offsets)}))
    return root


ROWS = [(_k(0), ["TRAIN"]), (_k(1), ["TRAIN", "COCONUT"]), (_k(2), ["COCONUT"]), (_k(101), ["COCONUT"]), (_k(102), ["TRAIN"])]


def test_legacy_universe_verified_and_adopted_without_rewriting(tmp_path):
    u = _legacy_universe(tmp_path / "universe", ROWS)
    legacy_bytes = (u / "universe_manifest.json").read_bytes()
    assert effective_universe_identity(u)["kind"] == "legacy_unadopted"
    v = verify_legacy_universe(u, expected_counts={"n_candidates": 5, "train": 3, "coconut": 3, "train_and_coconut": 1})
    assert v["status"] == LEGACY_EXTERNAL_VERIFIED and v["contents_verified"]
    assert v["counts"]["external_only"] == 2 and v["per_source"] == {"COCONUT": 3, "TRAIN": 3}
    res = adopt_legacy_universe(u, expected_counts={"coconut": 3})
    side = json.loads((u / ADOPTED_MANIFEST_NAME).read_text())
    assert res["written"] and side["adopted_from_legacy"] and side["contents_verified"] and not side["reproducible_build_id_available"]
    assert side["sources"]["COCONUT"]["n_candidates"] == 3 and side["limitations"]
    assert (u / "universe_manifest.json").read_bytes() == legacy_bytes            # legacy manifest never rewritten
    uid = effective_universe_identity(u)
    assert uid["kind"] == "legacy_adopted" and uid["external_sources"] == ["COCONUT"] and uid["build_id"].startswith("adopted-")
    assert not adopt_legacy_universe(u)["written"]                                # idempotent


def test_adoption_refused_on_invariant_failure_or_train_only(tmp_path):
    u = _legacy_universe(tmp_path / "u1", ROWS)
    with pytest.raises(LegacyAdoptionRefused):
        adopt_legacy_universe(u, expected_counts={"coconut": 4})                   # expected count mismatch
    assert not (u / ADOPTED_MANIFEST_NAME).exists()
    bad = _legacy_universe(tmp_path / "u2", ROWS, mass_n=4)
    assert verify_legacy_universe(bad)["status"] == VERIFICATION_FAILED
    train_only = _legacy_universe(tmp_path / "u3", [(_k(0), ["TRAIN"]), (_k(1), ["TRAIN"])])
    assert verify_legacy_universe(train_only)["status"] == LEGACY_TRAIN_ONLY
    with pytest.raises(LegacyAdoptionRefused):
        adopt_legacy_universe(train_only)


def test_adopted_universe_detects_later_changes(tmp_path):
    u = _legacy_universe(tmp_path / "universe", ROWS)
    adopt_legacy_universe(u)
    keys = np.load(u / "candidate_keys.npy")
    np.save(u / "candidate_keys.npy", keys[::-1])
    uid = effective_universe_identity(u)
    assert uid["kind"] == "contradiction" and uid["problems"]
    assert protocol_validity_v2(uid)["protocol_case"] == "D"


def test_legacy_stage_a_markers_are_discovered(tmp_path):
    w = tmp_path / "universe_work"
    for i in (0, 1):
        (w / "_done").mkdir(parents=True, exist_ok=True)
        (w / "_done" / f"COCONUT-chunk-{i:05d}.json").write_text(json.dumps({"n_input": 10, "n_prefiltered": 1, "n_rejected": 1,
                                                                             "n_filtered": 2, "n_kept": 6}))
        for d in ("standardized", "rejected", "filtered_out"):
            (w / d / "COCONUT").mkdir(parents=True, exist_ok=True)
            (w / d / "COCONUT" / f"chunk-{i:05d}.parquet").write_bytes(b"")
    rec = discover_legacy_stage_a(w)["COCONUT"]
    assert rec["legacy"] and rec["status"] == "complete_contiguous" and rec["chunk_ids"] == [0, 1]
    assert rec["n_input"] == 20 and rec["n_kept"] == 12 and rec["inferred_identity"]["kind"] == "INFERRED_FROM_MARKERS"
    (w / "rejected" / "COCONUT" / "chunk-00001.parquet").unlink()
    assert discover_legacy_stage_a(w)["COCONUT"]["status"] == "outputs_missing"
    disc = {"COCONUT": [{**rec, "status": "complete_contiguous"}]}
    ns, rows = select_stage_a_namespaces(disc, {"kind": "legacy_adopted"})
    assert ns == [str(w)] and rows[0]["selected"]
    assert select_stage_a_namespaces(disc, {"kind": "modern"})[0] == []


# ---------------------------------------------------------------------------------------------
# protocol validity + A8 + final decision + round trip
# ---------------------------------------------------------------------------------------------

def test_protocol_validity_cases():
    assert protocol_validity_v2({"kind": "modern", "external_sources": [], "build_id": "b"})["protocol_case"] == "A"
    ver = {"status": LEGACY_EXTERNAL_VERIFIED, "per_source": {"TRAIN": 3, "COCONUT": 3}}
    b = protocol_validity_v2({"kind": "legacy_unadopted", "external_sources": []}, ver)
    assert b["protocol_case"] == "B" and b["protocol_validity"] == VALID_READ_ONLY and b["protocol_warnings"]
    c = protocol_validity_v2({"kind": "legacy_adopted", "external_sources": ["COCONUT"], "build_id": "adopted-1"})
    assert c["protocol_case"] == "C" and c["protocol_validity"] == VALID and c["protocol_valid"]
    d = protocol_validity_v2({"kind": "legacy_adopted", "external_sources": ["COCONUT"], "build_id": "adopted-1"}, regime_universe_build_id="other")
    assert d["protocol_case"] == "D" and d["protocol_validity"] == INVALID
    assert protocol_validity_v2({"kind": "legacy_unadopted", "external_sources": []})["protocol_validity"] == INVALID


def test_a8_thresholds_block_the_final_decision():
    p = DecisionPolicy()
    a8 = a8_threshold_status(p)
    assert a8["status"] == BLOCKED_A8_THRESHOLDS and p.rule_a_min_incremental_pp == 10.0
    proto = {"protocol_validity": VALID}
    rules = {"decision": "PROCEED_TO_ANALOG_BASELINE"}
    ok_rt = {"status": ROUNDTRIP_DEPLOYMENT_OK}
    f = final_decision_v2(rules, proto, a8, {"decision_panel": "HOST"}, ok_rt)
    assert f["decision"] == BLOCKED_A8_THRESHOLDS and f["final_decision_refused"] and not f["permits_ranking"]
    set_p = DecisionPolicy(rule_b_min_reachability=0.5, rule_b_large_pool_threshold=100, rule_c_min_rescued_fraction=0.05,
                           rule_d_min_coconut_reachability=0.2, rule_d_max_lost_fraction=0.05)
    a8s = a8_threshold_status(set_p)
    assert final_decision_v2(rules, proto, a8s, {"decision_panel": "HOST"}, ok_rt)["permits_ranking"]
    assert not final_decision_v2(rules, proto, a8s, {"decision_panel": "HOST"}, {"status": ROUNDTRIP_DEPLOYMENT_BLOCKED})["permits_ranking"]
    assert final_decision_v2(rules, proto, a8s, {"decision_panel": None}, ok_rt)["decision"] == "DECISION_PANEL_UNAVAILABLE"
    assert policy_identity_v2(p, a8, ["HOST", "NP_PANEL"]) != policy_identity_v2(set_p, a8s, ["HOST", "NP_PANEL"])


def test_roundtrip_deployment_gate():
    fails = pd.DataFrame({"candidate_id": [3, 7]})
    assert roundtrip_deployment_check(fails)["status"] == ROUNDTRIP_DEPLOYMENT_BLOCKED
    assert roundtrip_deployment_check(fails, emitted_candidate_ids=[1, 2])["status"] == ROUNDTRIP_DEPLOYMENT_OK
    assert roundtrip_deployment_check(fails, blacklist_ids=[3, 7])["status"] == ROUNDTRIP_DEPLOYMENT_OK
    assert roundtrip_deployment_check(fails.iloc[0:0], exhaustive=False)["status"] == ROUNDTRIP_DEPLOYMENT_BLOCKED


def test_roundtrip_diagnosis_outcomes_with_a_fake_canonicalizer():
    keymap = {"rawA": "KA", "repA": "RA", "goodA": "KA", "rawB": "OTHER", "rawC": "KC", "rawC2": "RC", "rawD": "KD", "repD": "RD"}
    keyfn = lambda smiles: [(keymap.get(s), None) for s in smiles]   # noqa: E731
    fails = pd.DataFrame({"candidate_id": [1, 2, 3, 4, 5], "stored_connectivity_key": ["KA", "KB", "KC", "KD", "KE"],
                          "representative_smiles": ["repA", "repB", "repC", "repD", "repE"],
                          "recomputed_connectivity_key": ["RA", "RB", "RC", "RD", "RE"], "stratum": "COCONUT only"})
    recs = pd.DataFrame({"source": "COCONUT", "source_id": ["a1", "a2", "b1", "c1", "c2", "d1"],
                         "raw_smiles": ["rawA", "rawA", "rawB", "rawC", "rawC2", "rawD"],
                         "normalized_smiles": ["repA", "goodA", "x", "repC", "repC", "repD"],
                         "representative_smiles": ["repA", "goodA", "x", "repC", "repC", "repD"],
                         "connectivity_key": ["KA", "KA", "KB", "KC", "KC", "KD"]})
    d = diagnose_roundtrip_failures(fails, recs, keyfn_batch=keyfn).set_index("stored_connectivity_key")["outcome"]
    assert d.to_dict() == {"KA": "MERGE_REPRESENTATIVE_SELECTION_FIX", "KB": "SOURCE_CANONICALIZATION_FIX",
                           "KC": "EXPECTED_CANONICALIZER_VARIATION", "KD": "REPRESENTATIVE_SMILES_FIX_ONLY", "KE": "UNRESOLVED"}
    s = diagnosis_summary(diagnose_roundtrip_failures(fails, recs, keyfn_batch=keyfn))
    assert s["n_failures"] == 5 and sum(s["by_outcome"].values()) == 5


# ---------------------------------------------------------------------------------------------
# regimes v2
# ---------------------------------------------------------------------------------------------

def _regime_inputs():
    q = pd.DataFrame({"query_id": ["q1", "q2", "q3", "q4", "q5", "q6"], "true_connectivity_key": ["K1", "K1", "K2", "K3", "K4", "K5"],
                      "fold": [0, 0, 1, 1, 2, 2], "population": "ALL_DEV"})
    mirror = pd.DataFrame({"query_id": ["q1", "q2", "q3", "q4", "q6"], "flag": [False, True, False, False, False]})
    strict = pd.DataFrame({"query_id": ["q3"], "flag": [True]})
    memb = pd.DataFrame({"connectivity_key": ["K1", "K2", "K3", "K4", "K5"], "in_coconut": [True, True, False, True, False],
                         "in_pubchem": False, "in_other_external": False, "pubchem_available": False})
    return q, mirror, strict, external_reachability(memb, ["COCONUT"])


def test_regimes_are_eligibility_based_with_natural_counts():
    q, mirror, strict, memb = _regime_inputs()
    reg, mol = assign_regimes_v2(q, mirror, memb, ["COCONUT"], strict_evidence=strict)
    r = mol.set_index("connectivity_key")["regime"].to_dict()
    assert r == {"K1": "C1", "K2": "C2", "K3": "C3", "K4": UNRESOLVED, "K5": "C3"}       # K1: OR over its queries
    assert mol.set_index("connectivity_key").loc["K2", "has_test_simulated_strict_reference"]  # sensitivity never decides C1
    assert assert_regime_v2_consistency(mol)["passed"].all()
    assert set(reg["regime"]) <= {"C1", "C2", "C3", UNRESOLVED} and len(reg) == 6
    panels = pd.DataFrame({"panel": ["ALL_DEV"] * 5 + ["HOST"], "connectivity_key": ["K1", "K2", "K3", "K4", "K5", "K9"]})
    rec = molecule_records(panels, mol)
    assert rec.loc[rec["connectivity_key"] == "K9", "regime"].item() == UNRESOLVED     # explicit, never 'UNASSIGNED'
    c = regime_counts_by_panel(rec)
    assert c.loc[c["panel"] == "ALL_DEV", "n_molecules"].sum() == 5


def test_pubchem_source_requires_an_index():
    memb = pd.DataFrame({"connectivity_key": ["K1"], "in_coconut": [False], "in_pubchem": [False], "in_other_external": [False],
                         "pubchem_available": [False]})
    with pytest.raises(ValueError):
        external_reachability(memb, ["COCONUT", "PUBCHEM"])


def _identity(**over):
    base = dict(git_commit="abc", dev_identity={"n": 1}, split_identity={"n": 2}, universe_identity={"kind": "legacy_adopted", "build_id": "adopted-1"},
                external_sources=["COCONUT"], membership_identity=None, mirror_identity={"p": 1}, panel_manifest_identity="pm1", seed=1,
                config={"x": 1})
    base.update(over)
    return build_regime_identity(**base)


def test_regime_manifest_is_identity_bound(tmp_path):
    man = tmp_path / "validation_regimes_v2.manifest.json"
    assert check_regime_manifest(man, _identity())["status"] == REGIME_MANIFEST_MISSING
    write_regime_manifest(man, _identity(), counts={})
    assert check_regime_manifest(man, _identity())["status"] == REGIME_CURRENT
    for change in ({"external_sources": ["COCONUT", "PUBCHEM"]}, {"universe_identity": {"kind": "modern", "build_id": "s1"}},
                   {"panel_manifest_identity": "pm2"}, {"mirror_identity": {"p": 2}}):
        chk = check_regime_manifest(man, _identity(**change))
        assert chk["status"] == REGIME_REBUILD_REQUIRED and chk["mismatches"]
    git = check_regime_manifest(man, _identity(git_commit="def"))
    assert git["status"] == REGIME_CURRENT and git["warnings"]
    assert check_regime_manifest(man, _identity(git_commit="def"), strict_git_commit=True)["status"] == REGIME_REBUILD_REQUIRED
    assert regime_signature(_identity()) == regime_signature(_identity(git_commit="zzz"))
    with pytest.raises(RegimeRebuildRequired):
        require_current_regimes(man, _identity(external_sources=["PUBCHEM"]))


def test_old_regimes_archived_not_deleted(tmp_path):
    pdir = tmp_path / "processed"
    pdir.mkdir()
    pd.DataFrame({"query_id": ["1"], "regime": ["C2"]}).to_parquet(pdir / "validation_regimes.parquet")
    (pdir / "validation_regimes.metadata.json").write_text("{}")
    rep = archive_legacy_regimes(pdir)
    assert (pdir / "legacy" / "validation_regimes_pre_external_universe.parquet").is_file()
    assert (pdir / "validation_regimes.parquet").is_file() and (pdir / "validation_regimes.STALE.json").is_file()
    assert len(rep["archived"]) == 2 and not archive_legacy_regimes(pdir)["archived"]          # idempotent
    pd.DataFrame({"query_id": ["2"], "regime": ["C1"]}).to_parquet(pdir / "validation_regimes.parquet")
    assert archive_legacy_regimes(pdir)["refused"]                                           # never overwrites a different archive


def test_exclusion_manifests_and_fingerprint_contract(tmp_path):
    rec = pd.DataFrame({"connectivity_key": ["K1", "K2", "K3", "K4"], "panel": ["HOST", "ALL_DEV", "ALL_DEV", "NP_PANEL"],
                        "regime": ["C1", "C2", "C3", "C1"]})
    meta = write_exclusion_manifests(rec, tmp_path / "ex", "sig1", "pm1")
    assert set(meta["files"]) == {f"{n}_connectivities.parquet" for n in ("host", "np_panel", "c2", "c3")}
    kept, man = build_fingerprint_dataset_manifest(["K1", "K2", "K3", "K4", "K5"], tmp_path / "ex")
    assert kept == ["K5"] and man["regime_signature"] == "sig1" and not man["training_performed"]
    assert assert_dataset_respects_exclusions(kept, man, tmp_path / "ex")
    with pytest.raises(ExclusionContractError):
        assert_dataset_respects_exclusions(["K2", "K5"], man, tmp_path / "ex")


# ---------------------------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------------------------

DEV = pd.DataFrame({"query_id": ["1", "2", "3"], "true_connectivity_key": ["K1", "K2", "K3"], "fold": [0, 1, 2],
                    "neutral_mass": [100.0, 200.0, 300.0], "instrument_type": ["timsTOF", "QTOF", "timsTOF"]})


def test_timstof_is_never_the_decision_panel_and_host_is_never_fabricated():
    cfg = PanelConfig()
    q, st = build_panels(DEV, cfg)
    assert st[HOST]["status"] == NOT_AVAILABLE and st[HOST]["reason"] == HOST_PANEL_SOURCE_MISSING
    assert st[NP_PANEL]["reason"] == NP_THRESHOLD_NOT_SET and st[TIMSTOF_DEV]["n_queries"] == 2 and st[ALL_DEV]["n_queries"] == 3
    assert select_decision_panel(st)["decision_panel"] is None                  # TIMSTOF_DEV never substitutes
    with pytest.raises(ValueError):
        select_decision_panel(st, order=[TIMSTOF_DEV])
    with pytest.raises(ValueError):
        PanelConfig(decision_panel_order=["HOST", "TIMSTOF_DEV"]).validate()
    host = pd.DataFrame({"query_id": ["h1", "h2"], "true_connectivity_key": ["H1", "H1"], "neutral_mass": [1.0, 1.0], "fold": [0, 0]})
    q2, st2 = build_panels(DEV, cfg, host_frame=host)
    assert st2[HOST]["n_molecules"] == 1 and select_decision_panel(st2)["decision_panel"] == HOST and st2[HOST]["warning"]
    assert panel_manifest(st, cfg, select_decision_panel(st))["panel_manifest_identity"] != \
        panel_manifest(st2, cfg, select_decision_panel(st2))["panel_manifest_identity"]


def test_np_panel_is_structure_score_based_with_a_user_threshold():
    cfg = PanelConfig(np_threshold=0.5, np_threshold_provenance="user")
    scores = pd.DataFrame({"connectivity_key": ["K1", "K2", "K3"], "np_score": [1.2, 0.1, np.nan]})
    q, st = build_panels(DEV, cfg, np_scores=scores, np_info={"method": "file"})
    assert st[NP_PANEL]["n_molecules"] == 1 and st[NP_PANEL]["n_score_missing"] == 1
    assert select_decision_panel(st)["decision_panel"] == NP_PANEL


def test_host_like_alias_is_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert canonical_panel_name("host_like") == TIMSTOF_DEV
        cfg = PanelConfig.from_dict({"host_like": {"column": "instrument_type", "values": ["timsTOF"]}})
    assert cfg.timstof_values == ["timsTOF"] and any(issubclass(x.category, DeprecationWarning) for x in w)


# ---------------------------------------------------------------------------------------------
# experiments, public baseline, notebook-14 preflight
# ---------------------------------------------------------------------------------------------

BIND = dict(notebook="13_candidate_coverage", git_commit="abc1234", universe_build_id="adopted-1", regime_signature="sig",
            panel_manifest_identity="pm", policy_identity="pol", model_name="none", status="BLOCKED_A8_THRESHOLDS")


def test_bound_experiments(tmp_path):
    with pytest.raises(ExperimentBindingError):
        log_bound_experiment(tmp_path, "e0", **{**BIND, "git_commit": ""})
    with pytest.raises(ExperimentBindingError):
        log_bound_experiment(tmp_path, "e1", kind="ranking", **BIND)                # ranking needs feature contract + checkpoint
    log_experiment(tmp_path, "old", "legacy row", ["mass"])
    log_bound_experiment(tmp_path, "e2", **BIND)
    log_bound_experiment(tmp_path, "e3", kind="ranking", feature_contract="fc", checkpoint_identity="ck", **BIND)
    df = load_experiments(tmp_path)
    assert df.set_index("experiment_id").loc["old", "legacy_unbound"] and not df.set_index("experiment_id").loc["e2", "legacy_unbound"]
    comp = comparable_experiments(df, df.set_index("experiment_id").loc["e2"].to_dict())
    assert set(comp["experiment_id"]) == {"e2", "e3"}


def test_public_baseline_interface(tmp_path):
    st, df = load_public_baseline(PublicBaselineSpec(), tmp_path)
    assert st["status"] == "NOT_AVAILABLE" and df is None
    pd.DataFrame({"query_id": [1, 2], "connectivity_key": ["K1", "K2"]}).to_parquet(tmp_path / "pub.parquet")
    st, df = load_public_baseline(PublicBaselineSpec(path="pub.parquet", provenance="test fixture"), tmp_path)
    assert st["status"] == "AVAILABLE" and list(df["query_id"]) == ["1", "2"]


def test_notebook14_preflight(tmp_path):
    with pytest.raises(ProtocolV2Error):
        require_protocol_v2_ready(tmp_path, tmp_path / "missing.json")
    man = tmp_path / "m.json"
    body = write_regime_manifest(man, _identity(), counts={})
    gate = {"completed": True, "protocol_validity": VALID, "permits_ranking": True, "decision": "PROCEED_TO_ANALOG_BASELINE",
            "roundtrip_deployment": {"status": ROUNDTRIP_DEPLOYMENT_OK}, "decision_panel": "HOST",
            "regime_signature": body["regime_signature"], "universe_build_id": "adopted-1"}
    (tmp_path / "coverage_gate_summary.json").write_text(json.dumps(gate))
    assert require_protocol_v2_ready(tmp_path, man, _identity())["preflight_problems"] == []
    (tmp_path / "coverage_gate_summary.json").write_text(json.dumps({**gate, "regime_signature": "other"}))
    with pytest.raises(ProtocolV2Error):
        require_protocol_v2_ready(tmp_path, man)
    assert require_protocol_v2_ready(tmp_path, man, override=True)["override_used"]


# ---------------------------------------------------------------------------------------------
# PubChem build script (source checks only; no build) + config + notebooks
# ---------------------------------------------------------------------------------------------

def _script():
    spec = importlib.util.spec_from_file_location("pubchem_script", ROOT / "scripts" / "build_pubchem_membership_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pubchem_script_reports_missing_source(tmp_path):
    s = _script()
    ok, info = s.validate_source(tmp_path / "CID-InChI-Key.gz")
    assert not ok and info["status"] == s.SOURCE_MISSING
    assert s.main(["--input", str(tmp_path / "CID-InChI-Key.gz"), "--out-dir", str(tmp_path / "o")]) == 2
    assert not s.validate_source("https://example.org/x.gz")[0]
    import gzip
    p = tmp_path / "ok.gz"
    with gzip.open(p, "wt") as f:
        f.write("1\tInChI=1S/CH4/h1H4\tVNWKTOKETHGBQD-UHFFFAOYSA-N\n")
    assert s.validate_source(p)[0]


def test_config_panels_and_decision():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "casmi_v2_colab.yaml").read_text(encoding="utf-8"))
    assert "host_like" not in json.dumps(cfg["coverage"].get("panels") or {}) and cfg["coverage"]["decision"]["natural_product_panel"] != "host_like"
    assert cfg["panels"]["decision_panel_order"] == ["HOST", "NP_PANEL"] and cfg["panels"]["NP_PANEL"]["threshold"] is None
    assert cfg["validation_v2"]["external_sources"] == ["COCONUT"]
    assert cfg["coverage"]["decision"]["rule_a_min_incremental_pp"] == 10.0
    assert all(cfg["coverage"]["decision"][k] is None for k in ("rule_b_min_reachability", "rule_b_large_pool_threshold",
                                                              "rule_c_min_rescued_fraction", "rule_d_min_coconut_reachability",
                                                              "rule_d_max_lost_fraction"))


def _nb(name):
    return json.loads((NB / name).read_text(encoding="utf-8"))


def _src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


@pytest.mark.parametrize("name", ["11a_colab_legacy_universe_adoption.ipynb", "11b_colab_validation_regimes_v2.ipynb", "13_candidate_coverage.ipynb"])
def test_new_notebooks_define_names_before_use(name):
    import builtins
    defined = set(dir(builtins)) | {"display", "__file__"}
    for c in [c for c in _nb(name)["cells"] if c["cell_type"] == "code"]:
        assert c["execution_count"] is None and c["outputs"] == []
        tree = ast.parse(_src(c))
        local = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                local.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                local.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, (ast.FunctionDef, ast.Lambda)):
                local.update(a.arg for a in n.args.args)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
        missing = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - defined - local
        assert not missing, f"{name}:{c['id']}: used before definition {sorted(missing)}"
        defined |= local


def test_notebook13_requires_current_regimes_before_any_coverage_number():
    cells = _nb("13_candidate_coverage.ipynb")["cells"]
    ids = [c["id"] for c in cells]
    reg = _src(cells[ids.index("c-regimes")])
    assert "REGIME_REBUILD_REQUIRED" in reg and "check_regime_manifest" in reg and "regime_identity_from_run" in reg
    assert ids.index("c-regimes") < ids.index("c-membership") < ids.index("c-decision")
    code = "\n".join(_src(c) for c in cells if c["cell_type"] == "code")
    assert "fillna('UNASSIGNED')" not in code and "host_like" not in code and "dev_grouped" not in code
    assert "protocol_validity_v2" in code and "final_decision_v2" in code and "select_stage_a_namespaces" in code
    assert "log_bound_experiment" in code and "roundtrip_deployment_check" in code


def test_notebook11b_builds_identity_bound_regimes():
    code = "\n".join(_src(c) for c in _nb("11b_colab_validation_regimes_v2.ipynb")["cells"] if c["cell_type"] == "code")
    for needle in ("assign_regimes_v2", "regime_identity_from_run", "write_regime_manifest", "archive_legacy_regimes",
                   "write_exclusion_manifests", "assert_regime_integrity", "panels_for_run", "load_reference_evidence"):
        assert needle in code, needle
    assert "regime_fractions" not in code and "RegimeConfig(split_seed" not in code


def test_notebook10_default_branch():
    code = "\n".join(_src(c) for c in _nb("10_colab_v2_setup.ipynb")["cells"])
    assert "'feature/validation-protocol-v2'" in code
