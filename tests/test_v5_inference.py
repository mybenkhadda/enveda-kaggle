"""casmi_infer parity on synthetic fixtures: adduct export vs training parser, neutral-mass
reconstruction, searchsorted window + mass-variant dedupe vs the training generator, fallback
sequence, compat ordering vs `compat_sort_key` (incl. missing values and ties), eligibility / T1,
stable top-N, similarity + aggregation vs the training functions, fold-mean scoring."""
import json

import numpy as np
import pandas as pd
import pytest

from casmi.candidates.generator import CandidateGenerator, dedupe_pool_to_connectivity
from casmi.candidates.mass_index import MassIndex
from casmi.chemistry.adducts import neutral_mass_from_precursor
from casmi.data.metadata import summarize_collision_energy
from casmi.ranking.features import aggregate_pair_scores_from_values, compute_single_pair_scores
from casmi.spectra.deduplication import compute_peak_hash
from casmi.spectra.preprocessing import remove_invalid_peaks, truncate_top_peaks
from casmi.spectra.reference_selection import compat_sort_key, is_eligible
from casmi_infer.adducts import AdductRules, export_adduct_rules
from casmi_infer.compat import compat_order, select_references
from casmi_infer.features import SPECTRAL_FEATURES, candidate_features
from casmi_infer.mass_search import MassSearch
from casmi_infer.ranker import FoldMeanRanker, rank_spectrum
from casmi_infer.reference_index import ReferenceLibrary
from casmi_infer.spectrum import ce_mean, clean_query

ADDUCTS = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+CH2O2-H]-", "[2M+H]+", "[M+2H]2+", "[M+Cl]-", "[M+K]+", "[M+Xx]+"]


# ---- adducts ---------------------------------------------------------------------------------

def test_adduct_export_parity_with_training_parser():
    rules = AdductRules(export_adduct_rules(ADDUCTS, observed_in_test=["[M+H]+"]))
    for a in ADDUCTS:
        for mz in (101.0, 333.3, 999.99):
            exp = neutral_mass_from_precursor(mz, a)
            got = rules.neutral_mass(mz, a)
            assert (exp is None and got is None) or got == exp
    assert not rules.supported("[M+Xx]+") and rules.neutral_mass(200.0, "[M+Xx]+") is None
    assert rules.neutral_mass(200.0, "never-seen") is None
    assert rules.payload["rules"]["[M+H]+"]["observed_in_test"] and len(rules.payload["parser_version"]) == 16


def test_neutral_mass_reconstruction_known_values():
    rules = AdductRules(export_adduct_rules(["[M+H]+", "[2M+H]+"]))
    m = 180.0633881
    for a, mult in (("[M+H]+", 1), ("[2M+H]+", 2)):
        mz = (mult * m + rules.rules[a]["mass_shift"]) / abs(rules.rules[a]["charge"])
        assert rules.neutral_mass(mz, a) == pytest.approx(m, abs=1e-9)


def test_polarity_default():
    assert AdductRules.polarity_default("negative") == "[M-H]-" and AdductRules.polarity_default("positive") == "[M+H]+"


# ---- mass search -----------------------------------------------------------------------------

def _library(seed=0, n_conn=400):
    rng = np.random.default_rng(seed)
    keys = sorted(f"K{i:04d}" for i in range(n_conn))
    rows = []
    for k in keys:
        base = rng.uniform(150, 900)
        rows.append((f"{k}__v0", k, base))
        if rng.random() < 0.1:
            rows.append((f"{k}__v1", k, base + rng.uniform(-0.003, 0.003)))   # second mass variant, very close
    mv = pd.DataFrame(rows, columns=["mass_variant_id", "connectivity_key", "exact_mass"])
    conn = pd.DataFrame({"connectivity_key": keys, "conn_idx": np.arange(len(keys))})
    st = mv.assign(conn_idx=mv["connectivity_key"].map(dict(zip(keys, range(len(keys)))))).sort_values(["exact_mass", "connectivity_key"])
    return mv, conn, MassSearch(st["exact_mass"].to_numpy(), st["conn_idx"].to_numpy(), conn["connectivity_key"].to_numpy())


@pytest.mark.parametrize("ppm", [5.0, 50.0])
def test_window_and_dedupe_match_training_generator(ppm):
    mv, conn, ms = _library()
    gen = CandidateGenerator(MassIndex.from_dataframe(mv, mass_col="exact_mass", key_col="mass_variant_id"))
    v2c = mv.set_index("mass_variant_id")["connectivity_key"]
    rng = np.random.default_rng(1)
    for nm in list(mv["exact_mass"].sample(40, random_state=2)) + list(rng.uniform(150, 900, 40)):
        pool = dedupe_pool_to_connectivity(gen.generate_dataframe(pd.DataFrame({"query_id": ["q"], "neutral_mass": [nm]}), tolerance_ppm=ppm), v2c)
        exp = dict(zip(pool["candidate_connectivity_key"], pool["abs_mass_error_ppm"]))
        ci, ap, level = ms.search(nm, ppm, (), 0)
        got = dict(zip(conn["connectivity_key"].to_numpy()[ci], ap))
        if exp:
            assert level == "primary" and set(got) == set(exp)
            assert max(abs(got[k] - exp[k]) for k in exp) <= 1e-12
            assert list(ap) == sorted(ap)


def test_fallback_sequence():
    ms = MassSearch(np.array([100.0, 200.0, 300.0]), np.array([0, 1, 2]), np.array(["A", "B", "C"]))
    assert ms.search(100.0 * (1 + 30e-6), 50.0, (100.0, 200.0), 2)[2] == "primary"
    assert ms.search(100.0 * (1 + 80e-6), 50.0, (100.0, 200.0), 2)[2] == "ppm_100"
    ci, _, level = ms.search(100.0 * (1 + 150e-6), 50.0, (100.0, 200.0), 2)
    assert level == "ppm_200" and ci.tolist() == [0]
    ci, ap, level = ms.search(240.0, 50.0, (100.0, 200.0), 2)
    assert level == "nearest_2" and ci.tolist() == [1, 2] and list(ap) == sorted(ap)
    assert ms.search(None, 50.0)[2] == "none"
    mv, conn, big = _library()
    ci, ap, level = big.search(5000.0, 50.0, (100.0, 200.0), 25)     # far above the library
    assert level == "nearest_25" and len(ci) == 25 and len(set(ci)) == 25 and list(ap) == sorted(ap)


# ---- synthetic reference library -----------------------------------------------------------

def _write_library(tmp_path, seed=3):
    rng = np.random.default_rng(seed)
    n = 40
    rows, mz_all, it_all, offs = [], [], [], [0]
    raw_peaks = {}
    for i in range(n):
        mz = np.sort(rng.uniform(50, 500, rng.integers(5, 140)))
        it = rng.choice([1.0, 2.0, 3.0, 5.0], len(mz))      # ties everywhere -> exercises the stable top-N
        prec = 510.0 + i * 0.01
        imz, iit = remove_invalid_peaks(mz, it)
        smz, sit = truncate_top_peaks(imz, iit, 100)
        raw_peaks[i] = (imz, iit, prec, smz, sit)
        mz_all.append(smz); it_all.append(sit); offs.append(offs[-1] + len(smz))
        rows.append({"ref_row": i, "ref_spectrum_id": f"train_{1000 - i}", "connectivity_key": f"K{i % 4}",
                     "source": ["libA", "libB", None][i % 3], "instrument": ["timsTOF", "QTOF", None, ""][i % 4],
                     "adduct": ["[M+H]+", "[M+Na]+", None][i % 3], "polarity": ["positive", "negative"][i % 2],
                     "collision_energy": [20.0, 35.0, np.nan, 20.0][i % 4], "precursor_mz": prec,
                     "peak_hash": compute_peak_hash(imz, iit, prec)})
    meta = pd.DataFrame(rows)
    meta["sid_rank"] = meta["ref_spectrum_id"].rank(method="first").astype(np.int64) - 1
    meta.to_parquet(tmp_path / "ref_meta.parquet", index=False)
    conn_idx = meta["connectivity_key"].str[1:].astype(int).to_numpy()
    order = np.lexsort((meta["ref_row"].to_numpy(), conn_idx))
    np.save(tmp_path / "ref_index_ids.npy", meta["ref_row"].to_numpy()[order])
    np.save(tmp_path / "ref_index_offsets.npy", np.concatenate([[0], np.cumsum(np.bincount(conn_idx, minlength=4))]))
    np.save(tmp_path / "ref_peaks_offsets.npy", np.array(offs, dtype=np.int64))
    np.save(tmp_path / "ref_peaks_mz.npy", np.concatenate(mz_all))
    np.save(tmp_path / "ref_peaks_int.npy", np.concatenate(it_all))
    return ReferenceLibrary(tmp_path, mmap=False), meta, raw_peaks


def _train_meta(row):
    """How training's SpectrumLookups.ref_meta_of presents a reference (missing -> NaN / None)."""
    ce = row["collision_energy"]
    ce = None if pd.isna(ce) else float(ce)
    nanify = lambda v: float("nan") if v is None else v
    return {"spectrum_id": row["ref_spectrum_id"], "adduct": nanify(row["adduct"]), "ion_mode": nanify(row["polarity"]), "ce": ce,
            "ce_unit": "eV" if ce is not None else None, "instrument": nanify(row["instrument"]), "source": row["source"]}


@pytest.mark.parametrize("q", [
    {"adduct": "[M+H]+", "ion_mode": "positive", "instrument": "timsTOF", "ce": 20.0},
    {"adduct": "[M+Na]+", "ion_mode": "negative", "instrument": "", "ce": 30.0},
    {"adduct": "[M-H]-", "ion_mode": "negative", "instrument": "QTOF", "ce": float("nan")},
    {"adduct": None, "ion_mode": None, "instrument": None, "ce": None},
])
def test_compat_order_matches_training_sort_key(tmp_path, q):
    lib, meta, _ = _write_library(tmp_path)
    rows = np.arange(len(meta))
    qm_train = {"adduct": float("nan") if q["adduct"] is None else q["adduct"], "ion_mode": float("nan") if q["ion_mode"] is None else q["ion_mode"],
                "instrument": float("nan") if q["instrument"] is None else q["instrument"],
                "ce": None if q["ce"] is None or np.isnan(q["ce"]) else q["ce"]}
    qm_train["ce_unit"] = "eV" if qm_train["ce"] is not None else None
    expected = sorted(meta.to_dict("records"), key=lambda r: compat_sort_key(_train_meta(r), qm_train))
    got = rows[compat_order(lib, rows, q)]
    assert [meta.loc[i, "ref_spectrum_id"] for i in got] == [r["ref_spectrum_id"] for r in expected]


def test_compat_order_deterministic(tmp_path):
    lib, meta, _ = _write_library(tmp_path)
    q = {"adduct": "[M+H]+", "ion_mode": "positive", "instrument": "timsTOF", "ce": 20.0}
    rows = np.arange(len(meta))
    a = rows[compat_order(lib, rows, q)]
    perm = np.random.default_rng(0).permutation(rows)
    b = perm[compat_order(lib, perm, q)]
    assert np.array_equal(a, b)


def test_eligibility_matches_mirror_aware_and_t1(tmp_path):
    lib, meta, raw = _write_library(tmp_path)
    q = {"adduct": "[M+H]+", "ion_mode": "positive", "instrument": "timsTOF", "ce": 20.0, "source": "libA"}
    imz, iit, prec, _, _ = raw[4]                          # query IS an exact mirror of reference row 4 (conn K0)
    qhash = compute_peak_hash(imz, iit, prec)
    sel, n_total, n_t1 = select_references(lib, 0, q, qhash, exclude_sources={"libA"}, max_refs=5)
    assert n_t1 == 1 and 4 not in set(sel)
    refs = lib.refs_of(0)
    ordered = refs[compat_order(lib, refs, q)]
    tier = lambda r: "T1" if r == 4 else "T4"
    q_train = {"source": "libA"}
    expected = [r for r in ordered if is_eligible("mirror_aware", {"source": meta.loc[r, "source"]}, q_train, tier(r))][:5]
    assert list(sel) == expected
    sel_test, _, _ = select_references(lib, 0, q, None, exclude_sources=frozenset(), max_refs=5)   # hidden-test mode
    assert list(sel_test) == list(ordered[:5])
    sel_self, _, _ = select_references(lib, 0, q, None, exclude_ids={meta.loc[ordered[0], "ref_spectrum_id"]})
    assert ordered[0] not in set(sel_self)


def test_features_match_training_aggregation(tmp_path):
    lib, meta, raw = _write_library(tmp_path)
    cfg = {"bin_width_da": 0.1, "peak_tol_da": 0.02, "max_peaks_similarity": 100, "top_reference_count": 5}
    rng = np.random.default_rng(9)
    qmz = np.sort(rng.uniform(50, 500, 180)); qit = rng.choice([1.0, 2.0, 4.0], 180)
    identity, sim = clean_query(qmz, qit, 520.0, 100)
    q = {"adduct": "[M+H]+", "ion_mode": "positive", "instrument": "timsTOF", "ce": 20.0}
    from casmi_infer.backend import use_backend
    with use_backend("numpy"):     # exact equality is the numpy-backend contract; numba is proven by the bundle self-test
        f = candidate_features(lib, np.array([0, 1, 2, 3]), np.array([1.0, 2.0, 3.0, 4.0]), sim, q, None, cfg)
    for ci in range(4):
        sel, _, _ = select_references(lib, ci, q, None, max_refs=5)
        vals = [compute_single_pair_scores(sim, {"mzs": raw[r][3], "intensities": raw[r][4], "precursor_mz": raw[r][2]}, 0.1, 0.02) for r in sel]
        exp = aggregate_pair_scores_from_values(*zip(*vals))
        row = f[f["conn_idx"] == ci].iloc[0]
        for c in SPECTRAL_FEATURES:
            assert row[c] == exp[c]
    assert list(f.columns[:10]) == ["conn_idx", "abs_mass_error_ppm", *SPECTRAL_FEATURES]


def test_clean_query_uses_stable_topn():
    mz = np.arange(150, dtype=float) + 50.0
    it = np.ones(150)
    _, sim = clean_query(mz, it, 300.0, 100)
    assert np.array_equal(sim["mzs"], mz[:100])            # all tied -> lowest m/z win, deterministically


def test_ce_mean_parity_with_training():
    vals = [20.0, [10.0, 30.0], None, [np.nan, 40.0], np.nan, []]
    df = pd.DataFrame({"collision_energy_ev": vals})
    ref = summarize_collision_energy(df)["ce_mean"].to_numpy()
    got = np.array([ce_mean(v) for v in vals])
    assert np.array_equal(np.isnan(ref), np.isnan(got)) and np.allclose(ref[~np.isnan(ref)], got[~np.isnan(got)])


# ---- ranker ------------------------------------------------------------------------------------

class _Booster:
    def __init__(self, names, w):
        self.names, self.w = names, np.asarray(w, float)

    def feature_name(self):
        return self.names

    def predict(self, X):
        return X.to_numpy() @ self.w


def test_fold_mean_scoring_and_ties():
    names = ["abs_mass_error_ppm", *SPECTRAL_FEATURES]
    boosters = [_Booster(names, [0] * 9), _Booster(names, [0] * 9)]
    r = FoldMeanRanker(boosters, names)
    df = pd.DataFrame({"conn_idx": [7, 3, 5], "abs_mass_error_ppm": [2.0, 2.0, 1.0], **{c: [0.1, 0.2, 0.3] for c in SPECTRAL_FEATURES}})
    s = r.predict(df)
    assert np.allclose(s, 0.0)
    ranked = rank_spectrum(df, s)
    assert ranked["conn_idx"].tolist() == [5, 3, 7]          # all tied: abs ppm ASC, then key (conn_idx) ASC
    b2 = [_Booster(names, [1] + [0] * 8), _Booster(names, [3] + [0] * 8)]
    assert np.allclose(FoldMeanRanker(b2, names).predict(df), [4.0, 4.0, 2.0])


def test_ranker_refuses_feature_order_mismatch():
    names = ["abs_mass_error_ppm", *SPECTRAL_FEATURES]
    with pytest.raises(ValueError):
        FoldMeanRanker([_Booster(list(reversed(names)), [0] * 9)], names)
