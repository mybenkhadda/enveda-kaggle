"""v2 candidate universe: filters, connectivity-key dedup across external databases, bucketed resumable
build, id-based mass index (sorted search, parity with brute force), compact formula index, and the
canonicalizer reuse (RDKit tests are skipped when RDKit is absent)."""
import json

import numpy as np
import pandas as pd
import pytest

from casmi.candidates.filters import UniverseFilterConfig, apply_filters, is_organic_formula
from casmi.candidates.formula_index import CompactFormulaIndex
from casmi.candidates.mass_index import CandidateMassIndex, brute_force_open_search
from casmi.candidates.universe import (_write_bucketed, all_buckets, build_universe_in_memory, finalize_universe, keys_to_ids,
                                       load_candidate_keys, merge_bucket, read_candidates, UniverseBuildConfig)

KA, KB, KC, KD, KE = "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD", "EEEEEEEEEEEEEE"


def _mv():
    return pd.DataFrame({"mass_variant_id": ["mv1", "mv2"], "connectivity_key": [KA, KB], "exact_mass": [300.1, 400.2],
                         "representative_smiles": ["CCO_train_A", "CCO_train_B"], "molecular_formula": ["C10H12O2", "C20H24O4"],
                         "n_train_spectra": [3, 0]})


def _ext():
    rows = [  # source, id, smiles, key, formula, mass, charge, n_frag
        ("COCONUT", "c1", "smi_A_coconut", KA, "C10H12O2", 300.1, 0, 1),
        ("COCONUT", "c2", "smi_C_coconut", KC, "C15H20O3", 350.3, 0, 1),
        ("PUBCHEM", "p1", "smi_C_pubchem_stereo", KC, "C15H20O3", 350.3, 0, 1),   # same connectivity, other SMILES / id / full InChIKey
        ("PUBCHEM", "p2", "smi_D_small", KD, "C3H6O", 58.04, 0, 1),               # out of mass range
        ("PUBCHEM", "p3", "smi_E_charged", KE, "C12H18N+", 176.1, 1, 1),         # charged
    ]
    d = pd.DataFrame(rows, columns=["source", "source_id", "raw_smiles", "connectivity_key", "molecular_formula", "neutral_monoisotopic_mass",
                                    "formal_charge", "n_fragments"])
    d["normalized_smiles"] = d["raw_smiles"]
    d["representative_smiles"] = d["raw_smiles"]
    d["molecular_weight"] = d["neutral_monoisotopic_mass"] + 0.1
    d["name"] = None
    return d


def test_filters_keep_reasons_and_never_touch_train():
    ext = _ext()
    kept, filt = apply_filters(ext, UniverseFilterConfig())
    assert set(filt["connectivity_key"]) == {KD, KE}
    assert filt.set_index("connectivity_key").loc[KD, "filter_reason"] == "mass_out_of_range"
    assert "charged" in filt.set_index("connectivity_key").loc[KE, "filter_reason"]
    train = ext.assign(source="TRAIN")
    k2, f2 = apply_filters(train, UniverseFilterConfig(apply_to_train=False))
    assert len(f2) == 0 and len(k2) == len(train)


def test_organic_formula():
    assert is_organic_formula("C10H12N2O")
    assert not is_organic_formula("NaCl") and not is_organic_formula("C2H3NaO2") and not is_organic_formula(None)


def test_in_memory_universe_dedups_by_connectivity():
    u, variants = build_universe_in_memory(_mv(), _ext())
    assert u["connectivity_key"].tolist() == [KA, KB, KC]                    # key-sorted, D and E filtered
    assert u["candidate_id"].tolist() == [0, 1, 2]
    a, c = u.set_index("connectivity_key").loc[KA], u.set_index("connectivity_key").loc[KC]
    assert a["representative_smiles"] == "CCO_train_A" and a["train_present"] and a["coconut_present"]
    assert sorted(c["candidate_sources"]) == ["COCONUT", "PUBCHEM"] and c["source_count"] == 2
    assert c["representative_smiles"] == "smi_C_coconut"                      # source order: COCONUT before PUBCHEM
    assert sorted(c["source_ids"]) == ["COCONUT:c2", "PUBCHEM:p1"]
    assert not c["train_present"] and not c["has_reference_spectrum"]
    assert set(variants["candidate_id"]) == {0, 1, 2}


def test_bucketed_build_matches_in_memory(tmp_path):
    ext = _ext()
    cfg = UniverseBuildConfig(bucket_prefix_len=1)
    kept, _ = apply_filters(ext, cfg.filters)
    for src, g in kept.groupby("source"):
        _write_bucketed(g, tmp_path / "work" / "standardized" / src / "chunk-00000.parquet", 1)
    std_root = tmp_path / "work" / "standardized"
    buckets = all_buckets(std_root, _mv(), 1)
    assert buckets == ["A", "B", "C"]
    for b in buckets:
        merge_bucket(b, std_root, _mv(), tmp_path / "u", None, cfg)
    rec = merge_bucket("A", std_root, _mv(), tmp_path / "u", None, cfg)      # resumable: done-marker -> skipped
    assert rec["bucket"] == "A"
    man = finalize_universe(tmp_path / "u", buckets)
    assert man["n_candidates"] == 3
    src = man["source_summary"]                                    # embedded: C2 validity is decidable from the manifest alone
    assert src["n_candidates"] == 3 and src["external_any"] >= 1 and sum(src["overlap"].values()) == 3
    assert (tmp_path / "u" / "universe_manifest.json").exists() and (tmp_path / "u" / "formula_index").is_dir()
    keys = load_candidate_keys(tmp_path / "u")
    assert [k.decode() for k in keys] == [KA, KB, KC]
    assert keys_to_ids(keys, [KC, "ZZZZZZZZZZZZZZ"]).tolist() == [2, -1]
    rows = read_candidates(tmp_path / "u", [2, 0], columns=["connectivity_key", "candidate_sources"])
    assert dict(zip(rows.candidate_id, rows.connectivity_key)) == {0: KA, 2: KC}
    idx = CandidateMassIndex.load(tmp_path / "u" / "index")
    ids, ap = idx.search_ppm(350.3, 5)
    assert ids.tolist() == [2] and ap[0] == pytest.approx(0, abs=1e-9)
    fidx = CompactFormulaIndex.load(tmp_path / "u" / "formula_index")
    assert fidx.lookup("C15H20O3").tolist() == [2]


def test_bucket_built_before_a_new_source_is_rebuilt(tmp_path):
    """A TRAIN-only bucket must not be silently kept after an external source (COCONUT) is standardized."""
    cfg = UniverseBuildConfig(bucket_prefix_len=1)
    std_root = tmp_path / "work" / "standardized"
    rec = merge_bucket("A", std_root, _mv(), tmp_path / "u", None, cfg)            # no external source yet
    assert rec["n_external_records"] == 0
    kept, _ = apply_filters(_ext(), cfg.filters)
    _write_bucketed(kept[kept.source == "COCONUT"], std_root / "COCONUT" / "chunk-00000.parquet", 1)
    rec2 = merge_bucket("A", std_root, _mv(), tmp_path / "u", None, cfg)
    assert rec2["inputs_signature"] != rec["inputs_signature"] and rec2["n_external_records"] >= 1
    assert merge_bucket("A", std_root, _mv(), tmp_path / "u", None, cfg)["finished_at"] == rec2["finished_at"]   # now resumed


def _random_index(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, n // 2, size=n)                   # several mass variants per candidate
    masses = np.round(rng.uniform(150, 1200, size=n), 3)
    masses[:10] = masses[10]                                 # duplicate masses
    return ids, masses


def test_candidate_mass_index_parity_with_brute_force():
    ids, masses = _random_index()
    idx = CandidateMassIndex(masses, ids)
    keys = np.array([f"{i:08d}" for i in ids])               # key order == id order
    for q in list(masses[:30]) + [149.0, 5000.0, float(masses[10])]:
        for ppm in (2, 5, 20, 200):
            a_ids, a_ap = idx.search_ppm(q, ppm)
            b_keys, b_ap = brute_force_open_search(keys, masses, q, ppm)
            assert [f"{i:08d}" for i in a_ids] == b_keys.tolist()
            assert np.allclose(a_ap, b_ap)


def test_batch_equals_single_and_exclusion():
    ids, masses = _random_index(seed=1)
    idx = CandidateMassIndex(masses, ids)
    qs = masses[:25]
    off, bids, bap = idx.search_ppm_batch(qs, 50)
    for i, q in enumerate(qs):
        s_ids, _ = idx.search_ppm(q, 50)
        assert bids[off[i]:off[i + 1]].tolist() == s_ids.tolist()
    excl = np.zeros(idx.n_candidates, bool)
    excl[bids[:3]] = True
    off2, bids2, _ = idx.search_ppm_batch(qs, 50, exclude_mask=excl)
    assert not np.isin(bids2, np.flatnonzero(excl)).any()
    assert (idx.variant_counts(qs, 50) >= np.diff(off)).all()


def test_truth_rank_and_truth_ppm(tmp_path):
    idx = CandidateMassIndex([300.0, 300.0003, 300.001, 500.0], [5, 2, 7, 1], n_candidates=8)
    rank, size = idx.truth_rank([300.0, 300.0], [2, 9], ppm=5)
    assert rank[0] == 2 and np.isnan(rank[1]) and size.tolist() == [3, 3]
    tp = idx.truth_abs_ppm([300.0, 300.0], [7, -1])
    assert tp[0] == pytest.approx(0.001 / 300 * 1e6) and np.isnan(tp[1])
    idx.save(tmp_path / "ix")
    re = CandidateMassIndex.load(tmp_path / "ix")
    assert re.search_ppm(300.0, 5)[0].tolist() == idx.search_ppm(300.0, 5)[0].tolist()


def test_formula_index_roundtrip(tmp_path):
    f = CompactFormulaIndex.build(["C2H6O", None, "C2H6O", "CH4"], [0, 1, 2, 3], n_candidates=4)
    assert f.lookup("C2H6O").tolist() == [0, 2] and f.lookup("CH4").tolist() == [3] and len(f.lookup("XX")) == 0
    assert f.candidate_formula_id.tolist()[1] == -1
    assert f.formula_frequency([0, 1, 3]).tolist() == [2, 0, 1]
    f.save(tmp_path / "f")
    g = CompactFormulaIndex.load(tmp_path / "f")
    assert g.lookup("C2H6O").tolist() == [0, 2]
    assert json.loads((tmp_path / "f" / "formula_index.meta.json").read_text())["hard_filter"] is False


def test_external_canonicalization_reuses_training_contract():
    pytest.importorskip("rdkit")
    from casmi.candidates.standardize import standardize_records
    from casmi.chemistry.connectivity import competition_connectivity_key
    rec = pd.DataFrame({"source": "COCONUT", "source_id": ["t1", "t2", "s1", "s2", "bad"],
                        "raw_smiles": ["Oc1ccccn1", "O=c1cccc[nH]1", "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "not_a_smiles"]})
    std, rej = standardize_records(rec)
    assert rej["source_id"].tolist() == ["bad"] and rej["failure_reason"].iloc[0].startswith("parse_failed")
    k = std.set_index("source_id")["connectivity_key"]
    assert k["t1"] == k["t2"]                         # tautomers -> one connectivity
    assert k["s1"] == k["s2"]                         # enantiomers -> one connectivity (stereo-free first block)
    assert k["t1"] == competition_connectivity_key("Oc1ccccn1")
    empty_mv = pd.DataFrame(columns=["mass_variant_id", "connectivity_key", "exact_mass", "representative_smiles", "molecular_formula", "n_train_spectra"])
    u, _ = build_universe_in_memory(empty_mv, std, cfg=UniverseBuildConfig(filters=UniverseFilterConfig(min_exact_mass=0)))
    assert len(u) == 2 and u["source_count"].tolist() == [1, 1]
