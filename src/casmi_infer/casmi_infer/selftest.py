"""Bundle self-test: frozen expected outputs for ~50 HOST queries, re-derived at runtime by the
NORMAL inference path (numba if active) and compared before any hidden-test inference.

    bundle/selftest/queries.parquet            fixture spectra + metadata + per-query exclusion sets
    bundle/selftest/expected_features.parquet  (spectrum_id, conn_idx, abs_mass_error_ppm, 8 features)
    bundle/selftest/expected_scores.parquet    (spectrum_id, conn_idx, fold-mean score)
    bundle/selftest/expected_ranks.parquet     (spectrum_id, conn_idx, rank)
    bundle/selftest/fixture_manifest.json      file sha256s + the CONFIG_HASH / model files it was made for

Expected values are produced LOCALLY with the numpy backend (the path validated against training
by 11_01 P1/P2). The fixture references the normal exported reference library -- no peaks are
duplicated -- and needs nothing outside the bundle at runtime (no HOST dataset).

Comparison: candidate sets identical; features atol 1e-9, rtol 0 (NaN == NaN); scores atol 1e-9
(LightGBM on identical features is deterministic -- any larger gap means a feature moved across a
split); ranks identical. Fallback policy (`run_selftest_with_fallback`): numba failing -> rerun on
numpy -> NUMPY_FALLBACK if it passes; any numpy failure raises `SelfTestFailed` (no submission).

v6.3 (fixture format 2) adds, on the same fixture spectra:
    expected_scores.parquet            + score_fold<k> (every V1 fold, atol 1e-9)
    expected_probs.parquet             (spectrum_id, conn_idx, prob) = softmax(score / T_frozen), atol 1e-9
    expected_molecules.parquet         (molecule_id, selected_spectrum_id, confidence) -- A7 choice, exact id / atol 1e-9
    expected_molecule_ranking.parquet  (molecule_id, mol_rank, conn_idx) -- configured aggregator, Top-25, EXACT order
Fixture spectra are grouped into deterministic pseudo-molecules (`assign_fixture_molecules`, sizes 3,1,2,4
cycling over sorted spectrum ids) so the molecule-level path -- calibration, confidence ties, A7 spectrum
choice, connectivity dedup, Top-25 -- is exercised exactly as on the hidden test.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_infer.backend import NUMBA, NUMPY, active_backend, numba_available, set_backend, use_backend
from casmi_infer.features import SPECTRAL_FEATURES, candidate_features
from casmi_infer.ranker import rank_spectrum
from casmi_infer.spectrum import clean_query, query_meta, query_peak_hash
from casmi_infer.validation import sha256_file

FEATURE_ATOL = 1e-9
SCORE_ATOL = 1e-9
PROB_ATOL = 1e-9
CONFIDENCE_ATOL = 1e-9
FIXTURE_DIR = "selftest"
FIXTURE_FORMAT = "casmi-selftest-2"
FIXTURE_MOLECULE_SIZES = (3, 1, 2, 4)
FILES = ("queries.parquet", "expected_features.parquet", "expected_scores.parquet", "expected_ranks.parquet",
         "expected_probs.parquet", "expected_molecules.parquet", "expected_molecule_ranking.parquet")
MISMATCH_KEYS = ("candidate_set_mismatches", "feature_mismatches", "score_mismatches", "fold_score_mismatches", "rank_mismatches",
                 "prob_mismatches", "selected_spectrum_mismatches", "confidence_mismatches", "molecule_ranking_mismatches")


class SelfTestFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# runtime: infer one fixture query exactly like the submission path
# ---------------------------------------------------------------------------------------------

def infer_fixture_query(bundle, row):
    identity, sim = clean_query(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"], bundle.sim_cfg["max_peaks_similarity"])
    qm = query_meta(row["adduct"], row.get("ionization_mode"), row.get("instrument_type"), row.get("collision_energy_ev"), source=row.get("exclude_source"))
    nm = bundle.adducts.neutral_mass(row["precursor_mz"], row["adduct"])
    ci, ap, _ = bundle.search.search(nm, bundle.primary_ppm(row["adduct"]), tuple(bundle.ppm["fallback_ppm"]), bundle.ppm["nearest_n"])
    ex_src = frozenset([row["exclude_source"]]) if isinstance(row.get("exclude_source"), str) else frozenset()
    ex_ids = frozenset([row["exclude_id"]]) if isinstance(row.get("exclude_id"), str) else frozenset()
    feats = candidate_features(bundle.lib, ci, ap, sim, qm, query_peak_hash(identity), bundle.sim_cfg, ex_src, ex_ids)
    folds = bundle.ranker.predict_folds(feats)                    # fold-mean below == ranker.predict (same arithmetic)
    feats = feats.assign(**{f"score_fold{k}": folds[k] for k in range(folds.shape[0])})
    return rank_spectrum(feats, folds.mean(axis=0))


def assign_fixture_molecules(spectrum_ids, sizes=FIXTURE_MOLECULE_SIZES):
    """{spectrum_id: pseudo molecule id}: sorted ids chunked by `sizes` cycling (deterministic)."""
    ids, out, i, k = sorted(map(str, spectrum_ids)), {}, 0, 0
    while i < len(ids):
        for s in ids[i:i + sizes[k % len(sizes)]]:
            out[s] = f"FXM{k:03d}"
        i += sizes[k % len(sizes)]
        k += 1
    return out


def molecule_outputs(bundle, actual, queries, top_k=None):
    """(probs, molecules, molecule_ranking) through the PRODUCTION molecule path (`pipeline.aggregate_molecules`
    with the configured aggregator and the frozen temperature)."""
    from casmi_infer.pipeline import aggregate_molecules, calibrated_probabilities, selected_spectra
    top_k = int(top_k or bundle.config["top_k_submission"])
    mol_of = dict(zip(queries["spectrum_id"].astype(str), queries["molecule_id"].astype(str)))
    df = actual.assign(spectrum_id=actual["spectrum_id"].astype(str), model_scored=True)
    df["molecule_id"] = df["spectrum_id"].map(mol_of)
    probs = (calibrated_probabilities(bundle, df) if bundle.temperature is not None else df.assign(prob=np.nan))[["spectrum_id", "conn_idx", "prob"]]
    ranking = aggregate_molecules(bundle, df)
    ranking = ranking[ranking["mol_rank"] <= top_k][["molecule_id", "mol_rank", "conn_idx"]].reset_index(drop=True)
    sel = selected_spectra(bundle, df)
    if sel is None:
        mols = pd.DataFrame({"molecule_id": sorted(df["molecule_id"].unique()), "selected_spectrum_id": None, "confidence": np.nan})
    else:
        mols = sel.rename(columns={"spectrum_id": "selected_spectrum_id"})[["molecule_id", "selected_spectrum_id", "confidence"]]
    return probs.reset_index(drop=True), mols.reset_index(drop=True), ranking


def _close(a, b, atol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= atol)


def compare_fold_scores(actual, exp_scores, atol=SCORE_ATOL):
    """Rows whose per-fold V1 scores differ (missing rows / columns count as mismatches)."""
    key = ["spectrum_id", "conn_idx"]
    cols = [c for c in exp_scores.columns if c.startswith("score_fold")]
    if not cols:
        return 0
    if any(c not in actual.columns for c in cols):
        return int(len(exp_scores))
    m = exp_scores[key + cols].merge(actual[key + cols], on=key, how="outer", suffixes=("_exp", "_act"), indicator=True)
    bad = (m["_merge"] != "both").to_numpy().copy()
    for c in cols:
        bad |= (m["_merge"] == "both").to_numpy() & ~_close(m[f"{c}_exp"], m[f"{c}_act"], atol)
    return int(bad.sum())


def compare_probs(actual_probs, exp_probs, atol=PROB_ATOL):
    key = ["spectrum_id", "conn_idx"]
    a = actual_probs.assign(spectrum_id=actual_probs["spectrum_id"].astype(str))
    e = exp_probs.assign(spectrum_id=exp_probs["spectrum_id"].astype(str))
    m = e.merge(a, on=key, how="outer", suffixes=("_exp", "_act"), indicator=True)
    both = (m["_merge"] == "both").to_numpy()
    return int((~both | (both & ~_close(m["prob_exp"], m["prob_act"], atol))).sum())


def compare_molecules(act_mol, exp_mol, act_rank, exp_rank, atol=CONFIDENCE_ATOL):
    """Selected spectrum (exact), its confidence (atol), and the Top-k connectivity ORDER per molecule (exact)."""
    m = exp_mol.merge(act_mol, on="molecule_id", how="outer", suffixes=("_exp", "_act"), indicator=True)
    both = (m["_merge"] == "both").to_numpy()
    se, sa = m["selected_spectrum_id_exp"], m["selected_spectrum_id_act"]
    same_sel = (se.isna() & sa.isna()) | (se.astype(str) == sa.astype(str))
    lists = lambda r: {mid: g.sort_values("mol_rank")["conn_idx"].astype(int).tolist() for mid, g in r.groupby("molecule_id")}
    le, la = lists(exp_rank), lists(act_rank)
    return {"n_fixture_molecules": int(len(exp_mol)),
            "selected_spectrum_mismatches": int((~both | (both & ~same_sel.to_numpy())).sum()),
            "confidence_mismatches": int((~both | (both & ~_close(m["confidence_exp"], m["confidence_act"], atol))).sum()),
            "molecule_ranking_mismatches": int(sum(le.get(k) != la.get(k) for k in set(le) | set(la)))}


def compare_to_expected(actual, exp_features, exp_scores, exp_ranks):
    """Counts of mismatching (spectrum_id, conn_idx) rows per check; missing/extra rows count as
    mismatches in every check."""
    key = ["spectrum_id", "conn_idx"]
    exp = exp_features.merge(exp_scores, on=key).merge(exp_ranks, on=key)
    m = exp.merge(actual[key + ["abs_mass_error_ppm", *SPECTRAL_FEATURES, "score", "rank"]], on=key, how="outer",
                  suffixes=("_exp", "_act"), indicator=True)
    both = m["_merge"] == "both"
    n_set = int((~both).sum())
    feat_bad = ~both.to_numpy().copy()
    for c in ("abs_mass_error_ppm", *SPECTRAL_FEATURES):
        feat_bad |= both.to_numpy() & ~_close(m[f"{c}_exp"], m[f"{c}_act"], FEATURE_ATOL)
    score_bad = ~both.to_numpy() | (both.to_numpy() & ~_close(m["score_exp"], m["score_act"], SCORE_ATOL))
    rank_bad = ~both.to_numpy() | (both.to_numpy() & (m["rank_exp"].to_numpy() != m["rank_act"].to_numpy()))
    return {"n_candidate_rows": int(len(exp)), "candidate_set_mismatches": n_set, "feature_mismatches": int(feat_bad.sum()),
            "score_mismatches": int(score_bad.sum()), "rank_mismatches": int(rank_bad.sum())}


def load_fixture(bundle_dir):
    d = Path(bundle_dir) / FIXTURE_DIR
    if not (d / "fixture_manifest.json").exists():
        raise SelfTestFailed(f"no self-test fixture in {d}")
    man = json.loads((d / "fixture_manifest.json").read_text(encoding="utf-8"))
    if man.get("format") != FIXTURE_FORMAT:
        raise SelfTestFailed(f"fixture format {man.get('format')!r} != {FIXTURE_FORMAT!r} (no molecule-level parity) -- re-export it (11_00)")
    for f in FILES:
        if sha256_file(d / f) != man["files"][f]:
            raise SelfTestFailed(f"fixture file {f} does not match its recorded sha256")
    return man, {f: pd.read_parquet(d / f) for f in FILES}


def check_fixture_matches_bundle(bundle, man):
    """Expected scores depend on the model and config: a fixture made for other files is STALE."""
    if man["CONFIG_HASH"] != bundle.config["CONFIG_HASH"]:
        raise SelfTestFailed("self-test fixture was built for a different CONFIG_HASH -- re-export it (11_00)")
    current = {k: v for k, v in bundle.manifest["files"].items() if k.startswith("models/")}
    if man["model_files"] != current:
        raise SelfTestFailed("self-test fixture was built for different model files -- re-export it (11_00)")
    if list(man.get("feature_names") or []) != list(bundle.ranker.feature_names):
        raise SelfTestFailed("self-test fixture feature order != the shipped models' feature order")
    if man.get("temperature") != bundle.temperature:
        raise SelfTestFailed(f"self-test fixture temperature {man.get('temperature')!r} != bundle calibration {bundle.temperature!r}")


def run_selftest(bundle, backend, fixture=None, infer_fn=infer_fixture_query):
    """Spectrum level (candidates, features, fold scores, fold-mean score, ranks) AND molecule level (calibrated
    probabilities, A7 selected spectrum + confidence, final Top-k connectivity order) on one backend."""
    man, data = fixture if fixture is not None else load_fixture(bundle.dir)
    check_fixture_matches_bundle(bundle, man)
    queries = data["queries.parquet"]
    with use_backend(backend):
        parts = []
        for row in queries.to_dict("records"):
            parts.append(infer_fn(bundle, row).assign(spectrum_id=row["spectrum_id"]))
        actual = pd.concat(parts, ignore_index=True)
        probs, mols, mrank = molecule_outputs(bundle, actual, queries)
    res = compare_to_expected(actual, data["expected_features.parquet"], data["expected_scores.parquet"], data["expected_ranks.parquet"])
    res["fold_score_mismatches"] = compare_fold_scores(actual, data["expected_scores.parquet"])
    res["prob_mismatches"] = compare_probs(probs, data["expected_probs.parquet"])
    res.update(compare_molecules(mols, data["expected_molecules.parquet"], mrank, data["expected_molecule_ranking.parquet"]))
    res.update(backend=backend, n_fixture_queries=int(len(queries)), aggregator=bundle.config["aggregator"].get("name"))
    res["passed"] = all(res[k] == 0 for k in MISMATCH_KEYS)
    return res


def run_selftest_with_fallback(bundle, runner=run_selftest, log=print):
    """Initial backend = the normal one (numba if available). numba fails -> numpy; numpy pass ->
    NUMPY_FALLBACK (and the process-wide backend is set to numpy for the real inference); numpy fail
    -> SelfTestFailed. Returns the report dict."""
    initial = active_backend()
    first = runner(bundle, initial)
    report = {"self_test_backend_initial": initial, "numba_available": numba_available(), "attempts": [first], "numba_fallback_used": False}
    if first["passed"]:
        set_backend(initial)
        report.update(self_test_backend_final=initial, passed=True, final=first)
        return report
    if initial == NUMBA:
        log("NUMBA_PARITY_FAILED -- rerunning the self-test with the numpy backend")
        second = runner(bundle, NUMPY)
        report["attempts"].append(second)
        if second["passed"]:
            set_backend(NUMPY)
            log("USING_NUMPY_FALLBACK")
            report.update(self_test_backend_final="NUMPY_FALLBACK", numba_fallback_used=True, passed=True, final=second)
            return report
    report.update(self_test_backend_final=None, passed=False, final=report["attempts"][-1])
    raise SelfTestFailed(f"bundle self-test failed on every backend -- no submission will be written: {report['attempts']}")


# ---------------------------------------------------------------------------------------------
# export side (local): choose ~50 representative HOST queries and freeze their expected outputs
# ---------------------------------------------------------------------------------------------

def select_fixture_queries(host_meta, pool_size, truth_refs, n=50, n_hard=10, seed=42):
    """Deterministic stratified pick over (adduct, pool-size quartile, truth-ref bin, CE known) plus
    `n_hard` large-pool multi-reference queries. `host_meta`: query_id, adduct, ce_known."""
    d = host_meta.copy().sort_values("query_id").reset_index(drop=True)
    d["pool_size"] = d["query_id"].map(pool_size).fillna(0)
    d["truth_refs"] = d["query_id"].map(truth_refs).fillna(0)
    d["pool_q"] = pd.qcut(d["pool_size"].rank(method="first"), 4, labels=False)
    d["ref_bin"] = np.select([d["truth_refs"] <= 0, d["truth_refs"] < 5], ["0", "1-4"], default=">=5")
    rng = np.random.default_rng(seed)
    hard = d[(d["pool_size"] >= d["pool_size"].quantile(0.9)) & (d["truth_refs"] >= 5)]
    picked = list(hard["query_id"].iloc[rng.permutation(len(hard))[:n_hard]]) if len(hard) else []
    strata = {k: list(g["query_id"].iloc[rng.permutation(len(g))]) for k, g in d.groupby(["adduct", "pool_q", "ref_bin", "ce_known"], sort=True)}
    while len(picked) < min(n, len(d)) and any(strata.values()):
        for k in sorted(strata):
            if strata[k] and len(picked) < n:
                q = strata[k].pop(0)
                if q not in picked:
                    picked.append(q)
    return sorted(picked)


def build_fixture(bundle, query_rows, out_bundle_dir, infer_fn=infer_fixture_query):
    """`query_rows`: DataFrame with spectrum_id, peaks, precursor_mz, adduct, ionization_mode,
    instrument_type, collision_energy_ev, exclude_source, exclude_id. Expected outputs are computed
    with the NUMPY backend. Writes bundle/selftest/* (the caller refreshes the bundle manifest)."""
    d = Path(out_bundle_dir) / FIXTURE_DIR
    d.mkdir(parents=True, exist_ok=True)
    query_rows = query_rows.assign(spectrum_id=query_rows["spectrum_id"].astype(str))
    if "molecule_id" not in query_rows.columns:
        query_rows = query_rows.assign(molecule_id=query_rows["spectrum_id"].map(assign_fixture_molecules(query_rows["spectrum_id"])))
    parts = []
    with use_backend(NUMPY):
        for row in query_rows.to_dict("records"):
            parts.append(infer_fn(bundle, row).assign(spectrum_id=row["spectrum_id"]))
        exp = pd.concat(parts, ignore_index=True)
        probs, mols, mrank = molecule_outputs(bundle, exp, query_rows)
    key = ["spectrum_id", "conn_idx"]
    folds = [c for c in exp.columns if c.startswith("score_fold")]
    query_rows.to_parquet(d / "queries.parquet", index=False)
    exp[key + ["abs_mass_error_ppm", *SPECTRAL_FEATURES]].to_parquet(d / "expected_features.parquet", index=False)
    exp[key + ["score", *folds]].to_parquet(d / "expected_scores.parquet", index=False)
    exp[key + ["rank"]].to_parquet(d / "expected_ranks.parquet", index=False)
    probs.to_parquet(d / "expected_probs.parquet", index=False)
    mols.to_parquet(d / "expected_molecules.parquet", index=False)
    mrank.to_parquet(d / "expected_molecule_ranking.parquet", index=False)
    man = {"format": FIXTURE_FORMAT, "backend_used_for_expected": NUMPY, "n_queries": int(len(query_rows)), "n_candidate_rows": int(len(exp)),
           "n_molecules": int(query_rows["molecule_id"].nunique()), "molecule_sizes": list(FIXTURE_MOLECULE_SIZES),
           "CONFIG_HASH": bundle.config["CONFIG_HASH"], "aggregator": bundle.config["aggregator"], "temperature": bundle.temperature,
           "feature_names": list(bundle.ranker.feature_names), "n_folds": len(folds),
           "model_files": {k: v for k, v in bundle.manifest["files"].items() if k.startswith("models/")},
           "tolerances": {"feature_atol": FEATURE_ATOL, "score_atol": SCORE_ATOL, "prob_atol": PROB_ATOL, "confidence_atol": CONFIDENCE_ATOL,
                          "ranks": "identical", "selected_spectrum": "identical", "molecule_ranking": "identical order"},
           "files": {f: sha256_file(d / f) for f in FILES}}
    (d / "fixture_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man, exp
