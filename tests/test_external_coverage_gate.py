"""External candidate COVERAGE gate (notebook 13_candidate_coverage) + the engineering fixes it depends on:
external-only reachability contract (TRAIN never counts), PubChem key membership, mass-window coverage / pool density,
filter-loss attribution, charge / salt rescue provenance, representative-SMILES round trip, official-metric / public
baseline NOT_AVAILABLE behaviour, decision rules, Stage-B / Stage-C invalidation, pandas bool handling, scratch root,
notebook-10 c6-verify and notebook-13 static structure. RDKit-dependent tests are skipped without RDKit."""
import ast
import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi.candidates.filters import UniverseFilterConfig
from casmi.candidates.mass_index import CandidateMassIndex
from casmi.candidates.merge import unify
from casmi.candidates.pubchem_membership import (MEMBERSHIP_LABEL, MembershipConfigError, PubChemMembershipConfig, PubChemMembershipIndex,
                                                 build_membership_index, connectivity_from_inchikey, membership_summary)
from casmi.candidates.pubchem_tier1 import PubChemTier1Config, projected_pool_counts, tier1_prefilter
from casmi.candidates.rescue import (CLEANUP_COLUMNS, apply_cleanup_to_records, effective_filters, get_policy, is_default_policy,
                                     policy_identity)
from casmi.candidates.universe import (STAGE_B_VERSION, StageBIncompleteError, UniverseBuildConfig, _normalize_dtypes, _write_bucketed,
                                       finalize_universe, merge_bucket, run_stage_b, stage_b_id_of, stage_b_identity, universe_manifest_status)
from casmi.utils.booleans import as_bool
from casmi.validation.external_coverage import (DecisionPolicy, CoverageGateError, adduct_charge_audit, audit_filter_losses,
                                                build_external_membership, build_population_table, c2_external_protocol,
                                                classify_external_reachability, decision_metrics, estimate_runtime,
                                                evaluate_decision_rules, evaluate_mass_window_coverage, filter_loss_summary, loss_categories,
                                                mass_bin_labels, molecule_table, official_metric_parity, public_baseline_coverage,
                                                reachability_summary, remove_train_provenance_for_truth, require_coverage_gate,
                                                stratify_by_mass_bin, summarize_pool_density, write_json)
from casmi.validation.smiles_roundtrip import roundtrip_frame, stratum_of
from casmi.workspace.config import CANONICAL_COLAB_SCRATCH_ROOT, load_v2_config, missing_required_inputs, resolve_paths

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"
CFG_PATH = REPO / "configs" / "casmi_v2_colab.yaml"
KA, KB, KC, KD = "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD"


def _nb(name):
    return json.loads((NB / name).read_text(encoding="utf-8"))


def _src(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


# ---------------------------------------------------------------------------------------------------------------
# external-coverage regression contract (A: TRAIN, B: TRAIN+COCONUT, C: COCONUT, D: PUBCHEM)
# ---------------------------------------------------------------------------------------------------------------

def _universe_rows():
    return pd.DataFrame({"connectivity_key": [KA, KB, KC], "candidate_id": [0, 1, 2],
                         "candidate_sources": [["TRAIN"], ["COCONUT", "TRAIN"], ["COCONUT"]]})


def test_train_only_truth_is_never_externally_reachable():
    m = classify_external_reachability(build_external_membership([KA, KB, KC, KD], _universe_rows(), pubchem={KD}))
    r = m.set_index("connectivity_key")
    # A: present in the full TRAIN-union universe ...
    assert r.at[KA, "membership_in_full_train_union"] and r.at[KA, "in_train"]
    # ... but that is NOT external reachability (the old leakage)
    assert not r.at[KA, "external_reachable"] and r.at[KA, "reachability_status"] == "EXTERNAL_UNREACHABLE"
    # B: TRAIN removed, COCONUT remains
    assert r.at[KB, "candidate_sources_without_train"] == ["COCONUT"] and r.at[KB, "external_reachable"]
    # C / D reachable through an external source (D only through PubChem membership, not in the finalized universe)
    assert r.at[KC, "external_reachable"] and r.at[KC, "coconut_only"]
    assert r.at[KD, "external_reachable"] and r.at[KD, "pubchem_only"] and not r.at[KD, "membership_in_full_train_union"]
    assert r.at[KA, "reachability_labels"] == "TRAIN_REACHABLE" and r.at[KB, "reachability_labels"] == "TRAIN_REACHABLE;COCONUT_REACHABLE"


def test_remove_train_provenance_only_for_truths():
    out = remove_train_provenance_for_truth([["TRAIN", "COCONUT"], ["TRAIN"]], [True, False])
    assert out == [["COCONUT"], ["TRAIN"]]


def test_pubchem_union_and_unavailable_semantics():
    no_pc = classify_external_reachability(build_external_membership([KA, KD], _universe_rows(), pubchem=None))
    assert not no_pc["pubchem_available"].any() and not no_pc["in_pubchem"].any()
    with_pc = classify_external_reachability(build_external_membership([KA, KB, KD], _universe_rows(), pubchem={KB, KD}))
    r = with_pc.set_index("connectivity_key")
    assert r.at[KB, "coconut_and_pubchem"] and r.at[KD, "pubchem_only"] and r.at[KA, "neither"]
    mol = with_pc.assign(panel="p")
    s = reachability_summary(mol).iloc[0]
    assert s["union_external_reachability"] == pytest.approx(2 / 3) and s["coconut_reachability"] == pytest.approx(1 / 3)
    assert s["incremental_pubchem_over_coconut_pp"] == pytest.approx(100 / 3) and s["pubchem_label"] == MEMBERSHIP_LABEL
    s0 = reachability_summary(no_pc.assign(panel="p")).iloc[0]
    assert np.isnan(s0["incremental_pubchem_over_coconut_pp"]) and s0["pubchem_label"] == "NOT_AVAILABLE"


def test_c2_external_protocol_flags_legacy_train_only_c2():
    m = classify_external_reachability(build_external_membership([KA, KB], _universe_rows())).assign(regime=["C2", "C2"])
    m = c2_external_protocol(m).set_index("connectivity_key")
    assert not m.at[KA, "c2_external_eligible"] and m.at[KA, "legacy_c2_train_only"]
    assert m.at[KB, "c2_external_eligible"] and not m.at[KB, "legacy_c2_train_only"]


def test_population_panels_are_kept_separate_and_missing_file_panel_raises():
    from casmi.validation.external_coverage import PanelSpec
    dev = pd.DataFrame({"query_id": [1, 2, 3], "true_connectivity_key": [KA, KB, KB], "fold": [0, 1, 1], "neutral_mass": [300.0, 400.0, 400.0],
                        "instrument_type": ["timsTOF", "QTOF", "timsTOF"]})
    reg = pd.DataFrame({"query_id": ["1", "2", "3"], "regime": ["C1", "C2", "C2"]})
    q = build_population_table(dev, reg, [PanelSpec("TIMSTOF_DEV", column="instrument_type", values=["timsTOF"])])
    assert q.groupby("panel").size().to_dict() == {"dev_grouped": 3, "TIMSTOF_DEV": 2}
    assert set(q.columns) >= {"panel", "query_id", "connectivity_key", "regime"}
    mol = molecule_table(q)
    assert len(mol) == 4 and mol["truth_exact_mass"].isna().all()            # missing exact mass -> NaN, never a failure
    with pytest.raises(FileNotFoundError):
        build_population_table(dev, None, [PanelSpec("np_file", kind="file", path="x.parquet")], {})


def test_mass_bins_follow_project_edges():
    bins = [0, 200, 300, 1000, 5000]
    assert mass_bin_labels(bins) == ["<200", "200-300", "300-1000", "1000-5000", ">=5000"]
    assert list(stratify_by_mass_bin([150, 250, 999.9, 1000, 6000, None], bins)) == ["<200", "200-300", "300-1000", "1000-5000", ">=5000", "unknown"]


# ---------------------------------------------------------------------------------------------------------------
# mass-window coverage + pool density
# ---------------------------------------------------------------------------------------------------------------

def _index():
    # candidate 0: TRAIN only (300.0000), 1: TRAIN+COCONUT (300.0010), 2: COCONUT (300.0500), 3: far (400.0)
    return CandidateMassIndex([300.0, 300.001, 300.05, 400.0], [0, 1, 2, 3], n_candidates=4), np.array([False, True, True, True])


def test_mass_window_coverage_separates_database_and_mass_failures():
    idx, ext_mask = _index()
    q = pd.DataFrame({"query_id": ["q0", "q1", "q2"], "panel": "p", "neutral_mass": [300.0, 300.001, 300.001],
                      "truth_candidate_id": [0, 1, 2], "external_reachable": [False, True, True]})
    w = evaluate_mass_window_coverage(idx, q, ppm_values=(5, 200), external_mask=ext_mask).set_index(["query_id", "ppm"])
    assert w.loc[("q0", 5), "coverage_outcome"] == "database_coverage_failure" and not w.loc[("q0", 5), "truth_in_mass_window"]
    assert w.loc[("q1", 5), "coverage_outcome"] == "retrieved" and w.loc[("q1", 5), "truth_in_mass_window"]
    assert w.loc[("q2", 5), "coverage_outcome"] == "mass_retrieval_failure"          # reachable, ~163 ppm away
    assert w.loc[("q2", 200), "coverage_outcome"] == "retrieved"
    assert w.loc[("q1", 5), "full_current_universe_pool_size"] == 2 and w.loc[("q1", 5), "external_only_pool_size"] == 1
    assert np.isnan(w.loc[("q1", 5), "projected_pubchem_tier1_pool_size"])


def test_pool_density_summary_keeps_projected_counts_separate():
    w = pd.DataFrame({"panel": "p", "ppm": 5, "query_id": list("abcd"), "full_current_universe_pool_size": [1, 2, 3, 100],
                      "external_only_pool_size": [0, 1, 1, 50], "projected_pubchem_tier1_pool_size": [np.nan] * 4})
    s = summarize_pool_density(w).set_index("pool")
    assert s.at["full_current_universe_pool_size", "median"] == 2.5 and s.at["full_current_universe_pool_size", "max"] == 100
    assert s.at["projected_pubchem_tier1_pool_size", "n"] == 0
    assert s.at["projected_pubchem_tier1_pool_size", "count_kind"] == "pubchem_projected_raw_count"
    assert list(projected_pool_counts(np.array([100.0, 100.0001, 200.0]), [100.0, np.nan, -1], 5)) == [2, 0, 0]


# ---------------------------------------------------------------------------------------------------------------
# filter-loss attribution
# ---------------------------------------------------------------------------------------------------------------

def test_loss_categories():
    assert loss_categories("mass_out_of_range;charged") == ["charge", "mass"]
    assert loss_categories("parse_failed: parse_failed") == ["parse"] and loss_categories("prefilter_not_organic") == ["organic"]


def test_filter_loss_attribution_and_rescue():
    mol = pd.DataFrame({"panel": "p", "connectivity_key": [KA, KB, KC, KD], "truth_exact_mass": [300.0, 310.0, 320.0, 330.0],
                        "in_coconut": [True, False, False, False]})
    lost = pd.DataFrame({"source": "COCONUT", "source_id": ["1", "2", "3"], "raw_smiles": ["a", "b", "c"],
                         "connectivity_key": [KB, KC, None], "loss_reason": ["charged", "multi_component", "parse_failed: x"],
                         "loss_stage": ["filter", "filter", "parse"]})
    lost["loss_categories"] = lost["loss_reason"].map(lambda r: ";".join(loss_categories(r)))
    rescued = pd.DataFrame({"source": "COCONUT", "rescued_connectivity_key": [KB, KD], "passes_after_rescue": [True, True],
                            "rescue_kind": ["uncharger", "fragment_parent"]})
    t = audit_filter_losses(mol, lost, rescued, external_sources=("COCONUT",), mass_bins=[0, 200, 1000]).set_index("connectivity_key")
    assert t.at[KA, "kept_current_policy"] and not t.at[KA, "lost_by_current_filter"]
    assert t.at[KB, "lost_by_current_filter"] and t.at[KB, "loss_reason"] == "charge" and t.at[KB, "rescued_by_uncharger"]
    assert t.at[KC, "lost_by_current_filter"] and t.at[KC, "loss_reason"] == "component" and not t.at[KC, "reachable_after_rescue"]
    assert t.at[KD, "newly_reachable_by_rescue"] and t.at[KD, "rescued_by_fragment_parent"] and not t.at[KD, "reachable_raw"]
    s = filter_loss_summary(t.reset_index()).iloc[0]
    assert s["reachable_before_filters"] == 3 and s["lost_count"] == 2 and s["lost_by_charge"] == 1 and s["lost_by_component"] == 1
    assert s["rescued_count"] == 2 and s["newly_reachable_by_rescue"] == 1      # KB (uncharger) + KD (fragment parent)
    assert filter_loss_summary(pd.DataFrame()).empty


# ---------------------------------------------------------------------------------------------------------------
# cleanup policies (charge / salt rescue)
# ---------------------------------------------------------------------------------------------------------------

def test_current_policy_is_a_no_op_and_not_part_of_identity():
    rec = pd.DataFrame({"source": "COCONUT", "source_id": ["1"], "raw_smiles": ["CC(=O)[O-].[Na+]"]})
    assert apply_cleanup_to_records(rec, "current_policy") is rec
    assert is_default_policy("current_policy") and not is_default_policy("rescue_policy")
    f = UniverseFilterConfig()
    assert effective_filters(f, "current_policy").neutral_only and not effective_filters(f, "rescue_policy").neutral_only
    assert policy_identity("rescue_policy")["fragment_parent"] is True
    with pytest.raises(ValueError):
        get_policy("aggressive_policy")


def test_stage_a_identity_unchanged_for_current_policy(tmp_path):
    from casmi.candidates.sources import load_source_config
    from casmi.candidates.stage_a import StageAPerformance, build_id_of, stage_a_identity
    src = tmp_path / "coconut.csv"
    src.write_text("identifier,canonical_smiles,name,molecular_formula\nC1,CCO,a,C2H6O\n", encoding="utf-8")
    cfg = load_source_config(REPO / "configs" / "v6" / "coconut_source.json")
    base = stage_a_identity(cfg, src, UniverseBuildConfig(), StageAPerformance())
    assert "cleanup_policy" not in base                                          # existing builds keep their build id
    resc = stage_a_identity(cfg, src, UniverseBuildConfig(cleanup_policy="rescue_policy"), StageAPerformance())
    assert build_id_of(base) != build_id_of(resc) and resc["cleanup_policy"]["name"] == "rescue_policy"


def test_uncharger_and_fragment_parent_preserve_provenance():
    pytest.importorskip("rdkit")
    from casmi.candidates.rescue import cleanup_structure
    r = cleanup_structure("CC(=O)[O-].[Na+]", "rescue_policy")
    assert r["original_source_smiles"] == "CC(=O)[O-].[Na+]" and r["original_component_count"] == 2 and r["salt_stripped"]
    assert r["formal_charge_after_cleanup"] == 0 and r["charge_cleanup_status"] in ("neutralized", "neutral")
    assert "." not in r["cleaned_smiles"]
    rec = pd.DataFrame({"source": "COCONUT", "source_id": ["CNP1", "CNP2"], "raw_smiles": ["CC(=O)[O-].[Na+]", "not_a_smiles"]})
    out = apply_cleanup_to_records(rec, "rescue_policy")
    assert set(CLEANUP_COLUMNS) <= set(out.columns) and out["source_id"].tolist() == ["CNP1", "CNP2"]
    assert out.loc[0, "original_source_smiles"] == "CC(=O)[O-].[Na+]" and out.loc[0, "raw_smiles"] != "CC(=O)[O-].[Na+]"
    assert out.loc[1, "raw_smiles"] == "not_a_smiles" and out.loc[1, "charge_cleanup_status"] == "parse_failed"


def test_permanent_charge_is_retained_not_discarded():
    pytest.importorskip("rdkit")
    from casmi.candidates.rescue import cleanup_structure
    r = cleanup_structure("C[N+](C)(C)C", "rescue_policy")                      # quaternary ammonium: no proton to remove
    assert r["permanently_charged"] and r["charge_cleanup_status"] == "permanent_charge_retained"
    assert r["formal_charge_after_cleanup"] == 1 and "+" in r["cleaned_smiles"]
    assert cleanup_structure("C[N+](C)(C)C", "current_policy")["charge_cleanup_status"] == "not_applied"


# ---------------------------------------------------------------------------------------------------------------
# representative SMILES: overlap precedence determinism + round trip
# ---------------------------------------------------------------------------------------------------------------

def _mv():
    return pd.DataFrame({"mass_variant_id": ["mv1"], "connectivity_key": [KA], "exact_mass": [300.1], "representative_smiles": ["TRAIN_SMILES"],
                         "molecular_formula": ["C10H12O2"], "n_train_spectra": [3]})


def _ext(order):
    rows = [("COCONUT", "c2", KA, "coco_A_2"), ("COCONUT", "c1", KA, "coco_A_1"), ("COCONUT", "c9", KB, "coco_B_9"), ("COCONUT", "c3", KB, "coco_B_3")]
    rows = [rows[i] for i in order]
    d = pd.DataFrame(rows, columns=["source", "source_id", "connectivity_key", "representative_smiles"])
    return d.assign(molecular_formula="C10H12O2", neutral_monoisotopic_mass=300.1, name=None)


def test_overlap_representative_smiles_precedence_is_deterministic():
    a, _ = unify(_mv(), _ext([0, 1, 2, 3]))
    b, _ = unify(_mv(), _ext([3, 2, 1, 0]))
    pd.testing.assert_frame_equal(a, b)
    r = a.set_index("connectivity_key")
    assert r.at[KA, "representative_smiles"] == "TRAIN_SMILES"             # TRAIN representative kept for shared connectivities
    assert r.at[KA, "open_representative_smiles"] == "coco_A_1"            # first external record by (source order, source_id)
    assert r.at[KB, "representative_smiles"] == "coco_B_3"


def test_roundtrip_frame_and_strata():
    rows = pd.DataFrame({"candidate_id": [0, 1, 2], "connectivity_key": [KA, KB, KC], "representative_smiles": ["a", "b", "c"],
                         "candidate_sources": [["TRAIN"], ["COCONUT"], ["COCONUT", "TRAIN"]], "source_ids": [[], [], []]})
    fr = roundtrip_frame(rows, [(KA, None), ("ZZZZZZZZZZZZZZ", None), (None, "parse_failed")])
    assert fr["roundtrip_ok"].tolist() == [True, False, False]
    assert fr["stratum"].tolist() == ["TRAIN only", "COCONUT only", "TRAIN+COCONUT"]
    assert stratum_of(["PUBCHEM", "COCONUT"]) == "COCONUT+PUBCHEM" and stratum_of(None) == "none"


def test_roundtrip_uses_the_project_canonicalizer():
    pytest.importorskip("rdkit")
    from casmi.chemistry.connectivity import competition_connectivity_key
    from casmi.validation.smiles_roundtrip import project_keys
    keys = project_keys(["Oc1ccccn1", "O=c1cccc[nH]1", "not_a_smiles"])
    assert keys[0][0] == keys[1][0] == competition_connectivity_key("Oc1ccccn1")
    assert keys[2][0] is None and keys[2][1] == "parse_failed"


# ---------------------------------------------------------------------------------------------------------------
# official metric / public baseline / adducts / estimates
# ---------------------------------------------------------------------------------------------------------------

def test_official_metric_unavailable_without_a_competition_file(tmp_path):
    assert official_metric_parity(None, None, ["CCO"])["OFFICIAL_METRIC_PARITY"] == "NOT_AVAILABLE"
    assert official_metric_parity(tmp_path / "missing.py", "key", ["CCO"])["OFFICIAL_METRIC_PARITY"] == "NOT_AVAILABLE"
    f = tmp_path / "official.py"
    f.write_text("def key(s):\n    return s.upper()[:14]\n", encoding="utf-8")
    out = official_metric_parity(f, "key", ["cco", "ccn"], project_keyfn=lambda s: s.upper()[:14])
    assert out["OFFICIAL_METRIC_PARITY"] == "AVAILABLE" and out["n_equal"] == 2


def test_public_baseline_unavailable_and_overlap():
    status, cmp_ = public_baseline_coverage(None, None)
    assert status["PUBLIC_BASELINE_COVERAGE"] == "NOT_AVAILABLE" and cmp_.empty
    q = pd.DataFrame({"panel": "p", "query_id": ["1", "2", "3"], "connectivity_key": [KA, KB, KC]})
    win = pd.DataFrame({"panel": "p", "query_id": ["1", "2", "3"], "ppm": 5, "truth_in_mass_window": [True, False, True]})
    pub = pd.DataFrame({"query_id": ["1", "2"], "connectivity_key": [KA, KB]})
    status, cmp_ = public_baseline_coverage(pub, q, win, ppm=5)
    row = cmp_.set_index("panel").iloc[0]
    assert status["PUBLIC_BASELINE_COVERAGE"] == "AVAILABLE" and row["both"] == 1 and row["public_only"] == 1 and row["current_only"] == 1


def test_adduct_audit_surfaces_intrinsic_ions_only_when_present():
    t = adduct_charge_audit(["[M+H]+", "[M+H]+", "[M]+", "[M-H]-", "bogus"]).set_index("adduct")
    assert t.at["[M]+", "permanently_charged_compatible"] and not t.at["[M+H]+", "permanently_charged_compatible"]
    assert t.at["[M+H]+", "n_occurrences"] == 2 and not t.at["bogus", "supported"] and t.at["[M-H]-", "polarity"] == "negative"


def test_runtime_estimate_is_labelled():
    e = estimate_runtime(1_000_000, 100.0)
    assert e["label"] == "ESTIMATE_ONLY" and e["estimated_seconds"] == pytest.approx(10_000.0)
    assert estimate_runtime(1000, 10.0, observed_workers=2, target_workers=4, parallel_efficiency=0.5)["assumed_records_per_second"] == 10.0
    with pytest.raises(ValueError):
        estimate_runtime(10, 0)


def test_tier1_prefilter_skips_absent_columns_and_invents_no_threshold():
    chunk = pd.DataFrame({"cid": ["1", "2", "3"], "exact_mass": ["100", "300", "2000"], "molecular_formula": ["C6H6", "C10H12O2", "NaCl"]})
    keep, reasons, applied, skipped = tier1_prefilter(chunk, PubChemTier1Config(), UniverseFilterConfig())
    assert keep.tolist() == [False, True, False] and "mass_out_of_range" in applied and "not_organic" in applied
    cfg = PubChemTier1Config(min_literature_count=5)
    _, _, _, skipped = tier1_prefilter(chunk, cfg, UniverseFilterConfig())
    assert any("literature_count" in s for s in skipped)                        # absent column -> rule skipped, never guessed
    assert PubChemTier1Config().min_literature_count is None and PubChemTier1Config().min_patent_count is None


# ---------------------------------------------------------------------------------------------------------------
# PubChem key membership (local file only)
# ---------------------------------------------------------------------------------------------------------------

def test_pubchem_membership_build_and_lookup(tmp_path):
    f = tmp_path / "CID-InChI-Key.gz"
    lines = ["1\tInChI=1S/x\tAAAAAAAAAAAAAA-BBBBBBBBBB-N", "2\tInChI=1S/y\tAAAAAAAAAAAAAA-CCCCCCCCCC-N", "3\tInChI=1S/z\tCCCCCCCCCCCCCC-DDDDDDDDDD-N",
             "4\tInChI=1S/w\tnot-a-key"]
    with gzip.open(f, "wt", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")
    cfg = PubChemMembershipConfig(path=str(f), chunksize=2)
    man = build_membership_index(cfg, tmp_path / "idx", tmp_path / "spool", log=None)
    assert man["n_input_rows"] == 4 and man["n_invalid_inchikeys"] == 1 and man["n_unique_connectivity_keys"] == 2 and man["label"] == MEMBERSHIP_LABEL
    idx = PubChemMembershipIndex(tmp_path / "idx")
    look = idx.lookup([KA, KB, KC]).set_index("connectivity_key")
    assert look.at[KA, "in_pubchem"] and look.at[KA, "pubchem_n_records"] == 2 and int(look.at[KA, "pubchem_min_cid"]) == 1
    assert not look.at[KB, "in_pubchem"] and look.at[KC, "in_pubchem"]
    assert membership_summary(None)["status"] == "NOT_AVAILABLE"
    assert membership_summary(idx, [KA, KB])["truth_in_pubchem_rate"] == 0.5
    assert build_membership_index(cfg, tmp_path / "idx", tmp_path / "spool", log=None)["n_input_rows"] == 4    # resumed from markers


def test_pubchem_membership_refuses_urls_and_bad_keys():
    with pytest.raises(MembershipConfigError):
        PubChemMembershipConfig(path="https://ftp.example.org/CID-InChI-Key.gz").validate()
    keys = connectivity_from_inchikey(["AAAAAAAAAAAAAA-BBBBBBBBBB-N", "aaaa", None, "AAAAAAAAAAAAAA-BBBBBBBBBB"])
    assert keys.iloc[0] == KA and keys.iloc[1:].isna().all()


# ---------------------------------------------------------------------------------------------------------------
# decision rules + notebook-14 precondition
# ---------------------------------------------------------------------------------------------------------------

def _metrics(**kw):
    m = {"np_panel_present": True, "pubchem_incremental_pp": None, "np_reachability_union": 0.8, "np_coconut_reachability": 0.6,
         "pool_statistic": 50.0, "lost_fraction": 0.01, "rescued_fraction": 0.05}
    m.update(kw)
    return m


def test_unset_thresholds_are_not_evaluated_and_never_guessed():
    d = evaluate_decision_rules(_metrics(), DecisionPolicy())
    assert d["rules"]["A"]["status"] == "NOT_EVALUATED"                 # PubChem membership unavailable
    assert all(d["rules"][r]["status"] == "NOT_EVALUATED" for r in ("B", "C", "D"))
    assert d["decision"] == "UNDETERMINED_THRESHOLDS_OR_INPUTS_MISSING"


def test_rule_a_and_priority():
    d = evaluate_decision_rules(_metrics(pubchem_incremental_pp=10.0), DecisionPolicy())
    assert d["rules"]["A"]["fired"] and d["decision"] == "BUILD_PUBCHEM_TIER1_NEXT"
    assert not evaluate_decision_rules(_metrics(pubchem_incremental_pp=9.99), DecisionPolicy())["rules"]["A"]["fired"]
    p = DecisionPolicy(rule_c_min_rescued_fraction=0.02, rule_d_min_coconut_reachability=0.5, rule_d_max_lost_fraction=0.05,
                       rule_b_min_reachability=0.5, rule_b_large_pool_threshold=10.0)
    d = evaluate_decision_rules(_metrics(pubchem_incremental_pp=1.0), p)
    assert d["fired_rules"] == ["C", "B", "D"] and d["decision"] == "REBUILD_COCONUT_WITH_RESCUE_POLICY"
    d = evaluate_decision_rules(_metrics(pubchem_incremental_pp=1.0, rescued_fraction=0.0, pool_statistic=1.0), p)
    assert d["decision"] == "PROCEED_TO_ANALOG_BASELINE"
    assert evaluate_decision_rules(_metrics(pubchem_incremental_pp=50.0), p, protocol_valid=False)["decision"] == "PROTOCOL_INVALID"
    assert evaluate_decision_rules(_metrics(), DecisionPolicy(stage_b_invalidation_fixed=False,
                                                              rule_c_min_rescued_fraction=0.0))["rules"]["C"]["status"] == "NOT_EVALUATED"


def test_decision_metrics_from_tables():
    reach = pd.DataFrame({"panel": ["HOST"], "n_molecules": [10], "incremental_pubchem_over_coconut_pp": [np.nan],
                          "union_external_reachability": [0.7], "coconut_reachability": [0.7]})
    pool = pd.DataFrame({"panel": ["HOST"], "ppm": [5], "pool": ["full_current_universe_pool_size"], "median": [42.0]})
    loss = pd.DataFrame({"panel": ["HOST"], "reachable_before_filters": [8], "lost_count": [2], "rescued_count": [1],
                         "newly_reachable_by_rescue": [1]})
    m = decision_metrics(reach, pool, loss, DecisionPolicy())
    assert m["np_panel_present"] and m["pubchem_incremental_pp"] is None and m["pool_statistic"] == 42.0
    assert m["lost_fraction"] == 0.25 and m["rescued_fraction"] == 0.25
    assert not decision_metrics(reach.assign(panel="other"), pool, loss, DecisionPolicy())["np_panel_present"]


def test_notebook14_precondition(tmp_path):
    with pytest.raises(CoverageGateError):
        require_coverage_gate(tmp_path)
    write_json(tmp_path / "coverage_gate_summary.json", {"completed": True, "protocol_valid": True, "decision": "BUILD_PUBCHEM_TIER1_NEXT"})
    with pytest.raises(CoverageGateError):
        require_coverage_gate(tmp_path)
    assert require_coverage_gate(tmp_path, override=True)["override_used"]
    write_json(tmp_path / "coverage_gate_summary.json", {"completed": True, "protocol_valid": True, "decision": "PROCEED_TO_ANALOG_BASELINE"})
    assert not require_coverage_gate(tmp_path)["override_used"]


# ---------------------------------------------------------------------------------------------------------------
# Stage-B / Stage-C invalidation
# ---------------------------------------------------------------------------------------------------------------

def _inputs(size=1000, filters=None, cleanup=None, sources=("COCONUT",)):
    srcs = {}
    for s in sources:
        ident = {"source_file_name": f"{s.lower()}.csv", "source_size_bytes": size, "source_columns": {"id_col": "identifier"},
                 "canonicalizer": {"profile": "universe_minimal", "rdkit_version": "x"}, "filters": filters or {"neutral_only": True},
                 "prefilter": {"by_source_formula": True}, "chunk_records": 100000}
        if cleanup:
            ident["cleanup_policy"] = cleanup
        srcs[s] = {"status": "complete", "build_id": f"b-{s}-{size}", "n_chunks": 1, "n_kept": 1, "identity": ident,
                   "std_files": {f"/w/{s}/chunk-00000.parquet": 10}}
    return {"sources": srcs, "std_files": []}


def test_stage_b_identity_tracks_sources_filters_cleanup_and_sizes():
    mv, cfg = _mv(), UniverseBuildConfig()
    base = stage_b_id_of(stage_b_identity(_inputs(), mv, None, cfg))
    assert base == stage_b_id_of(stage_b_identity(_inputs(), mv, None, cfg))
    assert base != stage_b_id_of(stage_b_identity(_inputs(size=1001), mv, None, cfg))                      # source byte size
    assert base != stage_b_id_of(stage_b_identity(_inputs(filters={"neutral_only": False}), mv, None, cfg))  # per-source filters
    assert base != stage_b_id_of(stage_b_identity(_inputs(cleanup={"name": "rescue_policy"}), mv, None, cfg))
    assert base != stage_b_id_of(stage_b_identity(_inputs(sources=("COCONUT", "PUBCHEM")), mv, None, cfg))  # external source added
    assert base != stage_b_id_of(stage_b_identity(_inputs(), mv, None, UniverseBuildConfig(cleanup_policy="rescue_policy")))
    assert base != stage_b_id_of(stage_b_identity(_inputs(), mv, None, UniverseBuildConfig(filters=UniverseFilterConfig(min_exact_mass=100))))
    assert base != stage_b_id_of(stage_b_identity(_inputs(), mv, None, UniverseBuildConfig(prefilter_mass_margin_da=2.0)))
    ident = stage_b_identity(_inputs(), mv, None, cfg)
    assert ident["stage_b_version"] == STAGE_B_VERSION and ident["sources"]["COCONUT"]["source_size_bytes"] == 1000


def test_stale_or_legacy_bucket_marker_is_not_reused(tmp_path):
    out, std = tmp_path / "u", tmp_path / "std"
    r1 = merge_bucket("A", std, _mv(), out, None, UniverseBuildConfig())
    assert r1["build_id"] == r1["stage_b_id"] and "created_at" in r1 and r1["bucket"] == "A"
    marker = out / "_done" / "stage_b" / r1["stage_b_id"] / "bucket-A.json"
    legacy = {k: v for k, v in r1.items() if k != "build_id"}                   # pre-v3 marker format
    marker.write_text(json.dumps({**legacy, "created_at": "LEGACY"}), encoding="utf-8")
    r2 = merge_bucket("A", std, _mv(), out, None, UniverseBuildConfig())
    assert r2["created_at"] != "LEGACY" and r2["build_id"] == r1["stage_b_id"]   # rebuilt, not silently reused


def test_stage_c_depends_on_its_parent_stage_b(tmp_path):
    out = tmp_path / "u"
    sb = run_stage_b({"sources": {}, "std_files": []}, _mv(), None, out, UniverseBuildConfig(), log=None)
    man = finalize_universe(out, sb["buckets"], stage_b_id=sb["stage_b_id"])
    assert man["parent_stage_b_build_id"] == sb["stage_b_id"] and man["manifest_version"] == 3
    assert man["mass_index_count"] == 1 and man["formula_index_count"] >= 1 and man["overlap_counts"] == {"TRAIN": 1}
    assert man["cleanup_policy"]["name"] == "current_policy" and man["filter_policy"]["neutral_only"] is True
    st = universe_manifest_status(out, expected_stage_b_id="some-other-stage-b")
    assert st["stale"] and not st["legacy_manifest_detected"]
    assert universe_manifest_status(out, expected_stage_b_id=sb["stage_b_id"])["stale"] is False
    marker = out / "_done" / "stage_b" / sb["stage_b_id"] / "bucket-A.json"
    marker.write_text(json.dumps({"stage_b_id": "foreign", "bucket": "A"}), encoding="utf-8")
    with pytest.raises(StageBIncompleteError):
        finalize_universe(out, sb["buckets"], stage_b_id=sb["stage_b_id"])


def test_legacy_manifest_is_detected_but_readable(tmp_path):
    (tmp_path / "universe_manifest.json").write_text(json.dumps({"manifest_version": 2, "build_id": "sb-old", "n_candidates": 696447,
                                                                  "external_source_present": True}), encoding="utf-8")
    st = universe_manifest_status(tmp_path, expected_stage_b_id="sb-new")
    assert st["legacy_manifest_detected"] and st["parent_stage_b_build_id"] == "sb-old" and st["stale"] and st["n_candidates"] == 696447


# ---------------------------------------------------------------------------------------------------------------
# pandas bool handling (no deprecated object fillna downcast)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("values,expected", [
    (pd.Series([True, None, False], dtype=object), [True, False, False]),
    (pd.Series([True, False]), [True, False]),
    (pd.Series(pd.array([True, None, False], dtype="boolean")), [True, False, False]),
    (pd.Series([1.0, np.nan, 0.0]), [True, False, False]),
    (pd.Series([], dtype=object), []),
])
def test_as_bool_dtypes(values, expected):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)          # as_bool itself must be warning-free
        out = as_bool(values)
    assert out.dtype == bool and out.tolist() == expected


def test_as_bool_missing_true_and_index_kept():
    s = pd.Series([None, False], index=[10, 11], dtype=object)
    out = as_bool(s, missing=True)
    assert out.tolist() == [True, False] and out.index.tolist() == [10, 11]


def test_universe_and_merge_bool_paths_raise_no_futurewarning():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*[Dd]owncasting.*", category=FutureWarning)   # the bool fill / cast warnings
        d = _normalize_dtypes(pd.DataFrame({"is_charged": pd.Series([True, None], dtype=object), "tautomer_hit_cap": [None, None]}))
        u, _ = unify(_mv(), _ext([0, 1, 2, 3]))
    assert d["is_charged"].tolist() == [True, False] and d["tautomer_hit_cap"].tolist() == [False, False]
    assert u["has_reference_spectrum"].dtype == bool and u.set_index("connectivity_key").at[KB, "has_reference_spectrum"] == False  # noqa: E712


def test_bucket_writer_still_works_with_object_bools(tmp_path):
    df = pd.DataFrame({"source": "COCONUT", "source_id": ["1"], "connectivity_key": [KA], "is_charged": pd.Series([None], dtype=object)})
    _write_bucketed(df, tmp_path / "x.parquet", 1)
    assert pd.read_parquet(tmp_path / "x.parquet")["is_charged"].tolist() == [False]


# ---------------------------------------------------------------------------------------------------------------
# scratch root + notebook static checks
# ---------------------------------------------------------------------------------------------------------------

def test_canonical_scratch_root_everywhere():
    cfg, p = load_v2_config(CFG_PATH, environ={})
    assert str(p.scratch_dir).replace("\\", "/") == CANONICAL_COLAB_SCRATCH_ROOT == "/content/enveda_work"
    assert resolve_paths(cfg, environ={"ENVEDA_SCRATCH_ROOT": "/tmp/x"}).scratch_dir == Path("/tmp/x")
    c2 = _src(next(c for c in _nb("10_colab_v2_setup.ipynb")["cells"] if c.get("id") == "c2-roots"))
    assert "ENVEDA_SCRATCH_ROOT', '/content/enveda_work'" in c2
    for name in ("10_colab_v2_setup.ipynb", "12_colab_candidate_universe.ipynb", "13_candidate_coverage.ipynb", "14_colab_analog_baseline.ipynb"):
        assert "/content/scratch" not in json.dumps(_nb(name)), name


def test_missing_required_inputs_reports_without_creating(tmp_path):
    cfg, p = load_v2_config(CFG_PATH, environ={"ENVEDA_DRIVE_ROOT": str(tmp_path)})
    miss = missing_required_inputs(cfg, p)
    assert {n for n, _ in miss} == set(cfg["inputs"]) and not (tmp_path / "data").exists()


def test_notebook10_c6_verify_is_code():
    cells = {c.get("id"): c for c in _nb("10_colab_v2_setup.ipynb")["cells"]}
    c6 = cells["c6-verify"]
    assert c6["cell_type"] == "code" and c6["execution_count"] is None and c6["outputs"] == []
    src = _src(c6)
    ast.parse(src)
    assert "missing_required_inputs" in src and "IDENTITY" in src and "colab_paths" not in src
    assert all(c["cell_type"] == "markdown" or (c["execution_count"] is None and c["outputs"] == [])
               for c in _nb("10_colab_v2_setup.ipynb")["cells"])


NB13_REQUIRED_CELLS = ["md-title", "md-leakage", "md-decision-rules", "md-protocol-v2", "c-boot", "c-decision-policy", "c-universe",
                       "c-populations", "c-regimes", "c-membership", "c-reachability", "c-pubchem", "c-mass-window", "c-pool-density",
                       "c-filter-loss-load", "c-rescue", "c-mass-range", "c-adduct-charge", "c-roundtrip", "c-roundtrip-diagnose",
                       "c-official-metric", "c-public-baseline", "c-estimate", "c-decision", "c-fig-reachability", "c-fig-pools",
                       "c-fig-loss-roundtrip", "c-save", "md-final-summary"]
NB13_ARTIFACTS = ["external_reachability_by_molecule.parquet", "external_reachability_summary.csv", "reachability_by_mass_bin.csv",
                  "mass_pool_density.parquet", "mass_pool_density_summary.csv", "filter_loss_by_molecule.parquet", "filter_loss_summary.csv",
                  "smiles_roundtrip_failures.parquet", "smiles_roundtrip_summary.json", "official_metric_parity.json",
                  "pubchem_membership_summary.json", "pubchem_incremental_coverage.csv", "decision_rules.json", "coverage_gate_summary.json",
                  "public_baseline_coverage.json", "public_baseline_coverage_comparison.csv", "regime_counts_by_panel.csv",
                  "smiles_roundtrip_diagnosis.parquet", "smiles_roundtrip_diagnosis.json", "panel_status.json"]


def test_notebook13_static_structure():
    nb = _nb("13_candidate_coverage.ipynb")
    assert nb["nbformat"] == 4 and "kernelspec" in nb["metadata"]
    ids = [c.get("id") for c in nb["cells"]]
    assert ids == NB13_REQUIRED_CELLS
    assert _src(nb["cells"][0]).startswith("# 13 — External Candidate Coverage Gate")
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["execution_count"] is None and c["outputs"] == [] for c in code)
    # decision rules (markdown + pre-registration cell) precede every computation
    assert ids.index("md-decision-rules") < ids.index("c-decision-policy") < ids.index("c-universe") < ids.index("c-membership")
    all_code = "\n".join(_src(c) for c in code)
    for a in NB13_ARTIFACTS:
        assert a in all_code, a
    for forbidden in ("fit_fold_models", "lambdarank", ".fit(", "0.335", "test.parquet"):
        assert forbidden not in all_code, forbidden
    for c in code:
        tree = ast.parse(_src(c))
        assert len(_src(c).splitlines()) <= 40
        assert not any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))


def test_notebook13_names_are_defined_before_use():
    """Every module-level name a code cell LOADS was assigned / imported by that cell or an earlier one (static)."""
    import builtins
    defined = set(dir(builtins)) | {"display", "__file__"}
    for c in [c for c in _nb("13_candidate_coverage.ipynb")["cells"] if c["cell_type"] == "code"]:
        tree = ast.parse(_src(c))
        local = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                local.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                local.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, (ast.FunctionDef, ast.Lambda)):
                local.update(a.arg for a in n.args.args)
            elif isinstance(n, ast.comprehension):
                local.update(x.id for x in ast.walk(n.target) if isinstance(x, ast.Name))
            elif isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
        loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = loads - defined - local
        assert not missing, f"{c['id']}: used before definition {sorted(missing)}"
        defined |= local


def test_notebook14_requires_the_coverage_gate():
    cells = _nb("14_colab_analog_baseline.ipynb")["cells"]
    ids = [c.get("id") for c in cells]
    assert ids.index("c-coverage-gate") == ids.index("c-boot") + 1
    src = _src(cells[ids.index("c-coverage-gate")])
    assert "require_coverage_gate" in src and "PROCEED_TO_ANALOG_BASELINE" in src and "ALLOW_ANALOG_WITHOUT_GATE = False" in src
    assert "require_protocol_v2_ready" in src and "validation_v2" in src


def test_legacy_recall_notebook_is_labelled():
    nb = _nb("13_colab_candidate_recall.ipynb")
    assert "LEGACY_DIAGNOSTIC_ONLY" in _src(nb["cells"][0])
    assert "gateA_c2_recall_10ppm'" not in json.dumps(nb)
