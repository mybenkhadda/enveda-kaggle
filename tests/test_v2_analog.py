"""Analog channel: binning parity with the existing representation, packed Tanimoto, propagation features,
feature-schema stability / shortcut guard, pool truncation. No RDKit or GPU needed."""
import numpy as np
import pandas as pd
import pytest

from casmi.analog.features import ANALOG_FEATURES, MASS_FEATURES, candidate_pools, feature_schema
from casmi.analog.propagation import propagate
from casmi.analog.retrieval import AnalogConfig, binned_csr
from casmi.candidates.mass_index import CandidateMassIndex
from casmi.candidates.provenance import FORBIDDEN_SHORTCUT_FEATURES
from casmi.chemistry.fingerprint_store import FingerprintCache, popcount_u8, tanimoto_matrix
from casmi.spectra.binning import bin_spectrum
from casmi.spectra.neutral_loss import neutral_loss_cosine_similarity
from casmi.spectra.similarity import binned_cosine_similarity

SPECTRA = [
    (np.array([50.01, 50.04, 77.0, 105.03, 180.9]), np.array([0.2, 0.3, 1.0, 0.5, 0.1]), 200.0),
    (np.array([51.0, 77.02, 120.0]), np.array([0.4, 1.0, 0.2]), 190.0),
    (np.array([]), np.array([]), 150.0),
]


def _csr(spectra):
    counts = [len(m) for m, _, _ in spectra]
    off = np.concatenate([[0], np.cumsum(counts)])
    mz = np.concatenate([m for m, _, _ in spectra])
    it = np.concatenate([i for _, i, _ in spectra])
    return off, mz, it, np.array([p for _, _, p in spectra])


def test_binned_csr_matches_bin_spectrum_rows():
    cfg = AnalogConfig()
    off, mz, it, prec = _csr(SPECTRA)
    M = binned_csr(off, mz, it, prec, cfg).toarray()
    for r, (m, i, _) in enumerate(SPECTRA):
        b, v = bin_spectrum(m, i, bin_width=cfg.bin_width_da)
        dense = np.zeros(cfg.n_bins)
        dense[b] = v
        assert np.allclose(M[r], dense, atol=1e-6)
    assert np.allclose(M[2], 0)                                    # empty spectrum -> zero row


def test_prefilter_cosines_equal_existing_similarities():
    cfg = AnalogConfig()
    off, mz, it, prec = _csr(SPECTRA[:2])
    F = binned_csr(off, mz, it, prec, cfg)
    N = binned_csr(off, mz, it, prec, cfg, neutral_loss=True)
    (m0, i0, p0), (m1, i1, p1) = SPECTRA[0], SPECTRA[1]
    b0, v0 = bin_spectrum(m0, i0); b1, v1 = bin_spectrum(m1, i1)
    assert float((F[0] @ F[1].T).toarray()[0, 0]) == pytest.approx(binned_cosine_similarity(b0, v0, b1, v1), abs=1e-6)
    assert float((N[0] @ N[1].T).toarray()[0, 0]) == pytest.approx(neutral_loss_cosine_similarity(m0, i0, p0, m1, i1, p1), abs=1e-6)


def test_binned_csr_rebases_offsets_slice():
    cfg = AnalogConfig()
    off, mz, it, prec = _csr(SPECTRA)
    full = binned_csr(off, mz, it, prec, cfg).toarray()
    part = binned_csr(off[1:3], mz, it, prec[1:2], cfg).toarray()
    assert np.allclose(part[0], full[1])


def _bits(sets, n_bits=64):
    out = np.zeros((len(sets), n_bits), np.uint8)
    for r, s in enumerate(sets):
        out[r, list(s)] = 1
    return np.packbits(out, axis=1)


def test_tanimoto_matrix_manual():
    a = _bits([{1, 2, 3}, set()])
    b = _bits([{2, 3, 4, 5}, {1, 2, 3}])
    T = tanimoto_matrix(a, b)
    assert T[0, 0] == pytest.approx(2 / 5) and T[0, 1] == pytest.approx(1.0)
    assert np.isnan(T[1, 0]) or T[1, 0] == 0                       # empty fingerprint: union 0 -> NaN
    assert popcount_u8(np.array([255, 1, 0], np.uint8)).tolist() == [8, 1, 0]


def test_propagation_features():
    cand = _bits([{1, 2, 3}, {10, 11}, {1, 2, 3, 9}])
    ana = _bits([{1, 2, 3}, {10, 11, 12}])
    f = propagate(cand, np.ones(3, bool), np.array(["C0", "C1", "C2"]), np.array(["F1", "F2", "F1"]), ana, np.ones(2, bool),
                  analog_sims=[0.9, 0.3], analog_ranks=[1, 2], analog_keys=["C0", "Z9"], analog_formulas=["F1", "F9"],
                  analog_masses=[300.0, 310.0], analog_precursor_deltas=[0.0, -10.0], analog_same_adduct=[True, False],
                  query_neutral_mass=300.5, support_threshold=0.5, weight_power=2.0, top_k=10)
    assert f["analog_structural_similarity"][0] == pytest.approx(1.0)
    assert f["analog_exact_structure_support"].tolist() == pytest.approx([0.9, 0.0, 0.0])
    assert f["analog_rank"].tolist() == [1, 2, 1]
    assert f["analog_support_count"].tolist() == [1, 1, 1]
    assert f["analog_same_formula_support"][0] == pytest.approx(0.81 / (0.81 + 0.09))
    assert f["analog_mass_delta"][0] == pytest.approx(0.5)
    assert f["analog_best_score"][0] == pytest.approx(0.9) and f["analog_confidence"][0] == pytest.approx(0.6)
    assert f["analog_same_adduct_support"][0] == pytest.approx(0.5) and f["analog_n_analogs"][0] == 2
    assert f["analog_propagation_score"][0] > f["analog_propagation_score"][1]


def test_propagation_without_analogs():
    f = propagate(_bits([{1}]), np.ones(1, bool), ["C0"], ["F"], np.zeros((0, 8), np.uint8), np.zeros(0, bool), [], [], [], [], [], [], [],
                  300.0, top_k=10)
    assert f["analog_rank"][0] == 11 and f["analog_n_analogs"][0] == 0 and np.isnan(f["analog_mass_delta"][0])


def test_feature_schema_is_stable_and_shortcut_free():
    s1, s2 = feature_schema(), feature_schema()
    assert s1 == s2 and s1["features"] == MASS_FEATURES + ANALOG_FEATURES
    assert len(set(ANALOG_FEATURES)) == len(ANALOG_FEATURES) and all(c.startswith("analog_") for c in ANALOG_FEATURES)
    assert not set(s1["features"]) & set(FORBIDDEN_SHORTCUT_FEATURES)


def test_candidate_pool_truncation_is_reported():
    idx = CandidateMassIndex(np.full(10, 300.0) + np.arange(10) * 1e-5, np.arange(10))
    off, ids, ap, n_trunc = candidate_pools(idx, [300.0, 999.0], 100, max_per_query=4)
    assert off.tolist() == [0, 4, 4] and n_trunc == 1 and ids.tolist() == [0, 1, 2, 3]


def test_fingerprint_cache_does_not_persist_unknown_smiles(tmp_path):
    pytest.importorskip("rdkit")
    c = FingerprintCache(tmp_path, radius=2, n_bits=256)
    bits, valid = c.get(["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"], lambda miss: ["CCO" if m.startswith("A") else None for m in miss])
    assert valid.tolist() == [True, False] and bits[1].sum() == 0
    assert len(c) == 1                                              # B was not cached as "invalid forever"
    c.flush()                                                       # writes are buffered until flush()
    c2 = FingerprintCache(tmp_path, radius=2, n_bits=256)           # reload from shards
    b2, v2 = c2.get(["AAAAAAAAAAAAAA"], lambda miss: pytest.fail("must not recompute a cached key"))
    assert v2.all() and np.array_equal(b2[0], bits[0])


def test_fingerprint_cache_buffers_writes_and_compacts(tmp_path):
    pytest.importorskip("rdkit")
    smiles = {f"K{i:013d}": "C" * (i % 7 + 1) + "O" for i in range(200)}
    calls = []

    def smiles_of(keys):
        calls.append(len(keys))
        return [smiles[k] for k in keys]

    c = FingerprintCache(tmp_path, radius=2, n_bits=256)
    keys = list(smiles)
    for s in range(0, 200, 10):                                      # 20 small batches -> no shard yet
        b, v = c.get(keys[s:s + 10], smiles_of)
        assert v.all()
    assert len(c) == 200 and not list(c.dir.glob("shard-*.npz"))
    b2, _ = c.get(keys[:10], smiles_of)                              # served from the pending buffer
    assert sum(calls) == 200 and (b2 == c.get(keys[:10], smiles_of)[0]).all()
    assert c.flush() == 200 and len(list(c.dir.glob("shard-*.npz"))) == 1
    c2 = FingerprintCache(tmp_path, radius=2, n_bits=256)
    assert len(c2) == 200 and (c2.get(keys[:10], smiles_of)[0] == b2).all() and sum(calls) == 200
    for i in range(40):                                              # many shards -> compacted on load
        c2.get([f"X{i:013d}"], lambda ks: ["CCN"])
        c2.flush()
    assert len(list(c2.dir.glob("shard-*.npz"))) == 41
    c3 = FingerprintCache(tmp_path, radius=2, n_bits=256)
    assert len(c3) == 240 and len(list(c3.dir.glob("shard-*.npz"))) == 1
