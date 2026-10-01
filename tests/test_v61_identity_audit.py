"""v6.1: exact-library-identity audit (TRAIN-only universe, hash vs array identity, honest wording,
legacy-T1 parity) and label-free multi-spectrum connectivity consensus."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi.spectra.deduplication import compute_peak_hash
from casmi.spectra.preprocessing import remove_invalid_peaks
from casmi.validation import library_identity_audit as lia
from casmi.validation import test_match as tm

ROOT = Path(__file__).resolve().parents[1]

# ---- fixtures ------------------------------------------------------------------------------------
# TRAIN rows:
#   train_0  byte-identical copy of TEST q1                                  (conn CA)
#   train_1  q2 with one m/z moved by 1e-5 -> same 4-decimal hash, arrays differ (conn CB)
#   train_2  q2 with one intensity moved by 1e-7 -> same hash, arrays differ    (conn CC: one hash, two connectivities)
#   train_3  unrelated                                                         (conn CD)
#   train_4  empty peak list, precursor 700 -> degenerate hash                 (conn CE)
TRAIN = [
    ([100.0, 150.5, 200.25], [1.0, 0.5, 0.25], 250.0, "CA", "libX"),
    ([110.00001, 160.0], [1.0, 0.3], 300.0, "CB", "libY"),
    ([110.0, 160.0], [1.0, 0.3000001], 300.0, "CC", "libZ"),
    ([300.0, 301.0], [1.0, 1.0], 400.0, "CD", "libX"),
    ([], [], 700.0, "CE", "libY"),
]
TEST = [
    ("q1", "m1", [100.0, 150.5, 200.25], [1.0, 0.5, 0.25], 250.0),
    ("q2", "m1", [110.0, 160.0], [1.0, 0.3], 300.0),
    ("q3", "m2", [500.0, 510.0], [1.0, 0.2], 600.0),
    ("q4", "m2", [], [], 700.0),
]


class _Lib:
    def __init__(self, ids=None, extra_rows=0):
        n = len(TRAIN)
        self.spectrum_id = np.array(ids if ids is not None else [f"train_{i}" for i in range(n)])
        self.precursor_mz = np.array([t[2] for t in TRAIN], dtype=float)
        self.peak_hash = np.array([compute_peak_hash(*remove_invalid_peaks(t[0], t[1]), t[2]) for t in TRAIN], dtype=object)
        self.meta = pd.DataFrame({"ref_row": np.arange(n), "ref_spectrum_id": self.spectrum_id, "connectivity_key": [t[3] for t in TRAIN],
                                  "source": [t[4] for t in TRAIN], "instrument": "qtof", "adduct": "[M+H]+", "collision_energy": 20.0,
                                  "precursor_mz": self.precursor_mz})
        self.index_ids = np.arange(n + extra_rows)


def _train_parquet(tmp_path):
    p = tmp_path / "train.parquet"
    pd.DataFrame({"ms2_mzs": [np.asarray(t[0], float) for t in TRAIN], "ms2_normalized_intensities": [np.asarray(t[1], float) for t in TRAIN],
                  "precursor_mz": [t[2] for t in TRAIN]}).to_parquet(p, index=False)
    return p


def _test_df():
    return pd.DataFrame({"spectrum_id": [t[0] for t in TEST], "molecule_id": [t[1] for t in TEST], "ms2_mzs": [np.asarray(t[2], float) for t in TEST],
                         "ms2_normalized_intensities": [np.asarray(t[3], float) for t in TEST], "precursor_mz": [t[4] for t in TEST],
                         "adduct": "[M+H]+", "ionization_mode": "positive", "instrument_type": "qtof", "collision_energy_ev": 20.0})


POOL = pd.DataFrame({"query_id": ["q1", "q2", "q2", "q3"], "candidate_key": ["CA", "CB", "CX", "CD"], "abs_mass_error_ppm": [1.0, 1.0, 2.0, 3.0]})
LEGACY = pd.Series({"q1": 1, "q2": 1, "q3": 0, "q4": 0})


def _audit(tmp_path, legacy=LEGACY, lib=None):
    return lia.audit_test_identity(lib or _Lib(), _test_df(), _train_parquet(tmp_path), 100, n_train_rows=len(TRAIN), test_pool=POOL,
                                   legacy_t1=legacy, best_reference=pd.Series({"q3": "train_3"}), progress=False)


# ---- TRAIN-only reference universe -----------------------------------------------------------------

def test_reference_index_refuses_test_ids_foreign_ids_and_misalignment():
    lib = _Lib()
    assert lia.assert_train_only_reference_index(lib, ["q1"], n_train_rows=len(TRAIN))["reference_universe"] == "TRAIN_ONLY"
    with pytest.raises(lia.ReferenceIndexContaminationError):          # a TEST spectrum inserted into the reference index
        lia.assert_train_only_reference_index(_Lib(ids=["train_0", "train_1", "q1", "train_3", "train_4"]), ["q1"])
    with pytest.raises(lia.ReferenceIndexContaminationError):          # TRAIN-looking id that is also a TEST id
        lia.assert_train_only_reference_index(lib, ["train_2"])
    with pytest.raises(lia.ReferenceIndexContaminationError):          # rows not at their train.parquet offsets
        lia.assert_train_only_reference_index(_Lib(ids=["train_1", "train_0", "train_2", "train_3", "train_4"]), [])
    with pytest.raises(lia.ReferenceIndexContaminationError):          # rows added (e.g. TEST+TRAIN library)
        lia.assert_train_only_reference_index(lib, [], n_train_rows=len(TRAIN) - 1)
    with pytest.raises(lia.ReferenceIndexContaminationError):          # CSR pointing outside the library
        lia.assert_train_only_reference_index(_Lib(extra_rows=1), [])


def test_hash_index_is_built_from_train_only():
    params = set(inspect.signature(lia.TrainHashIndex.__init__).parameters)
    assert params == {"self", "lib", "test_ids", "n_train_rows"}        # no channel for TEST hashes
    idx = lia.TrainHashIndex(_Lib(), ["q1", "q2"], len(TRAIN))
    assert idx.provenance == "TRAIN_ONLY" and set(idx.ref_ids) == {f"train_{i}" for i in range(len(TRAIN))}
    _, h3 = lia.query_identity(_test_df().iloc[2].to_dict(), 100)
    assert len(idx.lookup(h3)) == 0 and len(idx.lookup(None)) == 0
    _, h2 = lia.query_identity(_test_df().iloc[1].to_dict(), 100)
    assert idx.lookup(h2).tolist() == [1, 2]
    no_hash = _Lib()
    no_hash.peak_hash = np.full(len(TRAIN), None, dtype=object)
    with pytest.raises(ValueError):                                      # compute_hash=False library: T1 was a no-op
        lia.TrainHashIndex(no_hash, [], None)


def test_query_hash_uses_the_production_function():
    src = inspect.getsource(lia.query_identity)
    assert "casmi_infer.spectrum import clean_query, query_peak_hash" in src


# ---- hash vs array identity ------------------------------------------------------------------------

def test_audit_separates_hash_and_array_identity(tmp_path):
    a, m, _ = _audit(tmp_path)
    a = a.set_index("spectrum_id")
    assert list(a.index) == ["q1", "q2", "q3", "q4"]
    # q1: byte-identical measurement in TRAIN
    assert a.loc["q1", "identity_relation"] == lia.FULL_IDENTICAL and a.loc["q1", "possible_contamination_flag"]
    assert a.loc["q1", "hash_equal"] and a.loc["q1", "full_peak_array_equal"] and a.loc["q1", "best_train_reference_id"] == "train_0"
    # q2: hash equal to two TRAIN spectra of two connectivities, arrays differ -> NOT a duplicate
    assert a.loc["q2", "hash_equal"] and not a.loc["q2", "full_peak_array_equal"] and not a.loc["q2", "possible_contamination_flag"]
    assert a.loc["q2", "identity_relation"] == lia.HASH_ONLY and a.loc["q2", "n_train_hash_matches"] == 2
    assert a.loc["q2", "n_distinct_train_connectivities_in_hash_matches"] == 2
    # q3: no hash match -> compared with the max-cosine reference, flags stay informative
    assert a.loc["q3", "match_basis"] == lia.BASIS_BEST_COSINE and not a.loc["q3", "hash_equal"] and a.loc["q3", "n_train_hash_matches"] == 0
    # q4: peak-free spectrum -> degenerate hash flagged
    assert a.loc["q4", "query_hash_degenerate"] and not a.loc["q1", "query_hash_degenerate"]
    assert not a["query_in_reference_index"].any()
    assert (a["ref_hash_recomputed_equal"].dropna().astype(bool)).all()
    assert set(m["ref_spectrum_id"]) <= {f"train_{i}" for i in range(len(TRAIN))}


def test_legacy_t1_definition_is_reproduced(tmp_path):
    a, _, idx = _audit(tmp_path)
    a = a.set_index("spectrum_id")
    # in-pool = hash matches whose TRAIN connectivity is a mass-window candidate (q2: CB in pool, CC not)
    assert a.loc["q1", "n_train_hash_matches_in_candidate_pool"] == 1 and a.loc["q2", "n_train_hash_matches_in_candidate_pool"] == 1
    assert a["legacy_parity_ok"].astype(bool).all()
    s = lia.identity_audit_summary(a.reset_index(), idx)
    assert s["legacy_parity"] == {"n_checked": 4, "n_mismatch": 0}
    v = lia.interpret_identity_audit(s)
    assert v["verdict"] in lia.VERDICTS and v["protocol_work_status"] == "CLOSED" and not v["correctness_failure"]


def test_legacy_parity_mismatch_reopens_protocol_work(tmp_path):
    a, _, idx = _audit(tmp_path, legacy=pd.Series({"q1": 1, "q2": 5, "q3": 0, "q4": 0}))
    s = lia.identity_audit_summary(a, idx)
    v = lia.interpret_identity_audit(s)
    assert s["legacy_parity"]["n_mismatch"] == 1
    assert v["verdict"] == lia.INTEGRITY_FAILURE and v["protocol_work_status"] == "REOPEN_REQUIRED"


def test_train_row_misalignment_is_detected(tmp_path):
    p = _train_parquet(tmp_path)
    exp = np.array([t[2] for t in TRAIN], float)
    assert set(lia.load_train_identity_peaks(p, [0, 2], exp)) == {0, 2}
    with pytest.raises(RuntimeError):
        lia.load_train_identity_peaks(p, [0, 2], exp[::-1])


def test_audit_refuses_truth_columns(tmp_path):
    with pytest.raises(tm.LabelLeakError):
        lia.audit_test_identity(_Lib(), _test_df().assign(inchikey="X"), _train_parquet(tmp_path), 100, progress=False)


# ---- wording / identity claims --------------------------------------------------------------------------

def test_exact_duplicate_wording_needs_full_arrays():
    with pytest.raises(lia.IdentityWordingError):
        lia.assert_identity_wording(["exact_library_duplicate_share"])
    with pytest.raises(lia.IdentityWordingError):                         # never a molecule-identity claim
        lia.assert_identity_wording(["same_molecule_share"], full_array_verified={"same_molecule_share"})
    assert lia.assert_identity_wording(["share_exact_library_duplicate"], full_array_verified=lia.FULL_ARRAY_BACKED_KEYS)
    s = pd.DataFrame({"max_cosine": [.9, .4], "has_eligible_reference": True, "near_duplicate_proxy": False, "n_t1_excluded_refs": [1, 0]})
    keys = tm.library_matchability_proxies(s)
    assert not any("exact" in k.lower() and "duplicate" in k.lower() for k in keys)
    assert any(k.startswith("t1_peak_hash_hit") and "hash-level" in k for k in keys)


def test_hash_equality_is_never_proof_of_identity(tmp_path):
    a, _, idx = _audit(tmp_path)
    assert not any(t in c.lower() for c in a.columns for t in lia.FORBIDDEN_IDENTITY_CLAIMS)
    src = inspect.getsource(lia.audit_test_identity)
    assert 'rec["full_peak_array_equal"] and rec["precursor_mz_equal"]' in src   # contamination = arrays + precursor
    bad = a.copy()
    bad.loc[bad["spectrum_id"] == "q2", "possible_contamination_flag"] = True   # hash-only row marked as a duplicate
    with pytest.raises(lia.IdentityWordingError):
        lia.assert_audit_integrity(bad, idx, ["q1", "q2", "q3", "q4"])
    assert lia.assert_audit_integrity(a, idx, ["q1", "q2", "q3", "q4"])
    s = lia.identity_audit_summary(a, idx)
    assert s["share_train_hash_match_anywhere"] > s["share_exact_library_duplicate"]


def test_deterministic_example_sample(tmp_path):
    a, _, _ = _audit(tmp_path)
    s1 = lia.deterministic_sample(a, n=3)
    s2 = lia.deterministic_sample(a.sample(frac=1, random_state=7), n=3)
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))
    assert len(s1) == 3 and s1["identity_relation"].nunique() == 3
    txt = lia.format_identity_example(s1.iloc[0])
    for tok in ("TEST spectrum", "matched TRAIN", "source", "instrument", "adduct", "CE", "precursor m/z", "peak counts", "hash equal?", "full arrays equal?"):
        assert tok in txt


# ---- connectivity consensus (label-free) -----------------------------------------------------------------

def _cons_stats():
    rows = [("a1", "M1", "X", .97), ("a2", "M1", "X", .91), ("a3", "M1", "Y", .45),        # majority X; low-cos dissenter
            ("b1", "M2", "P", .95), ("b2", "M2", "Q", .90),                               # confident conflict (split)
            ("c1", "M3", "Z", .80),                                                        # single spectrum
            ("d1", "M4", None, 0.0), ("d2", "M4", "W", .60)]                               # one non-voting spectrum
    s = pd.DataFrame(rows, columns=["query_id", "molecule_id", "best_candidate_connectivity", "max_cosine"])
    return s.assign(has_eligible_reference=s["best_candidate_connectivity"].notna(), adduct="[M+H]+", polarity="positive", neutral_mass=300.0)


def test_connectivity_consensus_values():
    mol, spec = tm.connectivity_consensus(_cons_stats())
    m = mol.set_index("molecule_id")
    assert m.loc["M1", "consensus_class"] == "MAJORITY" and m.loc["M1", "modal_connectivity"] == "X" and m.loc["M1", "modal_share"] == pytest.approx(2 / 3)
    assert m.loc["M2", "consensus_class"] == "SPLIT" and m.loc["M2", "confident_conflict"] and m.loc["M2", "modal_connectivity"] == "P"  # tie -> summed cosine
    assert m.loc["M3", "consensus_class"] == "SINGLE_SPECTRUM" and m.loc["M4", "consensus_class"] == "SINGLE_SPECTRUM"
    sp = spec.set_index("query_id")
    assert sp.loc["a1", "loo_agrees"] and not sp.loc["a3", "loo_agrees"] and "d1" not in sp.index
    assert sp.loc["a3", "cosine_bin"] == "<0.50" and sp.loc["a1", "cosine_bin"] == ">=0.95"
    summ = tm.connectivity_consensus_summary(mol, spec)
    assert summ["n_multi_spectrum_molecules"] == 2 and "NOT accuracy" in summ["wording"]


def test_connectivity_consensus_deterministic_and_label_free():
    s = _cons_stats()
    a = tm.connectivity_consensus(s)
    b = tm.connectivity_consensus(s.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    c = tm.connectivity_consensus(s.assign(is_true=False, score=1.0, rank=1))      # extra columns are never read
    pd.testing.assert_frame_equal(a[0], c[0])
    assert not (set(tm.CONSENSUS_INPUT_COLS) & tm.LABEL_COLUMNS)
    src = inspect.getsource(tm.connectivity_consensus) + inspect.getsource(tm.connectivity_consensus_summary)
    assert "is_true" not in src and "true_connectivity" not in src and "mrr" not in src.lower()
    assert set(inspect.signature(tm.connectivity_consensus).parameters) == {"stats", "pool", "molecule_col"}


# ---- notebook wiring ---------------------------------------------------------------------------------------

def test_notebook_audit_and_consensus_sections():
    nb = json.loads((ROOT / "src" / "10v6_00_test_match_diagnostic.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    first = lambda tok: next(i for i, s in enumerate(code) if tok in s)
    assert first("run_observed_test(") < first("audit_test_identity(") < first("connectivity_consensus(") < first("protocol_closure.json")
    audit_cell = code[first("audit_test_identity(")]
    assert "assert_audit_integrity(" in audit_cell and "legacy_t1=" in audit_cell and "exact_identity_audit.parquet" in audit_cell
    assert "deterministic_sample(AUDIT, n=20)" in "\n".join(code)
    closure = code[first("protocol_closure.json")]
    assert "raise RuntimeError" in closure and "'status'" not in closure.split("_res[")[-1]   # resemblance status is not re-decided
