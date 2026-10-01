"""v6 test-match diagnostic: LABEL-FREE query <-> TRAINING-library similarity statistics, computed
IDENTICALLY for the visible competition TEST and for TL_EVAL (optionally HOST) under a reference-
eligibility protocol, plus the pre-registered protocol-resemblance rule.

Question (not an assumption): does `test_simulated_strict` resemble the hidden test's relationship to
the training spectral library, or does it admit unusually close sibling spectra that hidden test
molecules will not have? The query -> library similarity is observable for TEST without labels, so
the same statistic can be compared across populations.

ONE statistic definition, TWO pair producers
--------------------------------------------
`query_stats_from_pairs` is the ONLY place a statistic is defined. It consumes a long frame of the
RANKER-VISIBLE (query, candidate, reference) pairs -- the first `top_reference_count` (5) ELIGIBLE
references of every mass-window candidate in compat order, i.e. exactly the references whose
similarities feed the frozen ranker's features -- and the query's mass-window pool. Two producers
differ ONLY in the eligibility filter:

  TEST  (`test_observed`)  `casmi_infer` production path, unchanged: `clean_query` (remove_invalid_peaks
        + deterministic top-N) -> bundle adduct rules -> `MassSearch` (primary window, logged
        fallbacks) -> `compat.select_references` with exclude_sources = {} and exclude_ids = {}
        (T1 via peak hash; T2 not applied -- the documented deployment behaviour) -> `pair_scores`
        forced to the NUMPY backend (== `casmi.ranking.features.compute_single_pair_scores`).
  TL_EVAL / HOST           the persisted QCR (same `compute_single_pair_scores`, same peak cleaning,
        same compat walk) filtered by `casmi.qcr.protocols.select_protocol_refs(PROTOCOL_DEFS[p])`.

`bundle_library_query_pairs` re-derives TL queries through the TEST producer (with the benchmark
exclusions) so the notebook can PROVE the two producers agree (parity cell), instead of trusting it.

Statistic (per query):
  max_cosine / max_modified_cosine / max_peak_overlap   max over the ranker-visible pairs (each metric
        maximised independently); 0.0 when the query has NO eligible pair (`has_eligible_reference`
        False) -- pre-registered, identical for every population.
  matched_peaks_at_max_cosine   round(peak_overlap_frac * n_query_similarity_peaks) of the best-cosine
        pair (`peak_overlap` divides the greedy match count by the QUERY's similarity-peak count).
  best pair tie rule            cosine DESC, abs_mass_error_ppm ASC, candidate_key ASC, compat_rank
        ASC, ref_spectrum_id ASC.
  best_candidate_mass_rank      rank of the best-match candidate in the pool ordered by abs ppm ASC,
        candidate_key ASC.

Label-freeness: inputs are refused if they carry any truth column (`LABEL_COLUMNS`); QCR shards and
pools are READ with label-free column lists only (`is_true` / `is_true_candidate` never loaded);
TL/HOST `molecule_id` is left empty (a development molecule would be its truth connectivity).
No function here accepts a model score, rank or MRR.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.qcr.protocols import PROTOCOL_DEFS, TOP_K, assert_walk_complete, protocol_counts_from_shards, select_protocol_refs

TEST_OBSERVED = "test_observed"
STRICT, MIRROR, STANDARD = "test_simulated_strict", "mirror_aware", "standard"
POP_TEST = f"TEST:{TEST_OBSERVED}"


def population(dataset, protocol):
    return f"{dataset}:{protocol}"


POP_STRICT, POP_MIRROR, POP_STANDARD = population("TL_EVAL", STRICT), population("TL_EVAL", MIRROR), population("TL_EVAL", STANDARD)

# truth-bearing columns that must never enter the label-free statistic
LABEL_COLUMNS = frozenset({"is_true", "is_true_candidate", "true_connectivity_key", "connectivity_key", "truth_abs_ppm", "inchikey",
                           "inchikey14", "normalized_smiles", "smiles", "molecular_formula", "rr", "mrr", "truth_rank", "score", "rank"})
QCR_LABEL_FREE_COLS = ["query_id", "candidate_key", "ref_spectrum_id", "compat_rank", "tier", "ref_source", "query_source",
                       "cosine", "modified_cosine", "peak_overlap_frac"]
POOL_LABEL_FREE_COLS = ["query_id", "candidate_connectivity_key", "abs_mass_error_ppm"]
PAIR_COLS = ["query_id", "candidate_key", "ref_spectrum_id", "compat_rank", "cosine", "modified_cosine", "peak_overlap_frac"]
META_COLS = ["query_id", "molecule_id", "instrument", "adduct", "polarity", "precursor_mz", "neutral_mass", "retrieval_level",
             "n_query_peaks", "n_t1_excluded_refs"]
QUERY_STATS_COLUMNS = [
    "dataset", "protocol", "population", "query_id", "molecule_id", "instrument", "adduct", "polarity", "precursor_mz", "neutral_mass",
    "retrieval_level", "n_query_peaks", "candidate_pool_size", "eligible_reference_count", "n_candidates_with_reference",
    "has_eligible_reference", "max_cosine", "max_modified_cosine", "max_peak_overlap", "matched_peaks_at_max_cosine",
    "best_candidate_connectivity", "best_candidate_mass_rank", "best_candidate_abs_ppm", "best_reference_id", "best_reference_source",
    "best_reference_instrument", "best_reference_adduct", "best_reference_collision_energy", "best_reference_precursor_mz",
    "best_ref_precursor_diff_da", "near_duplicate_proxy", "n_t1_excluded_refs"]
NO_REFERENCE_VALUE = 0.0
# label-free translation of the existing T3 near-duplicate thresholds (V4B similarity config) -- a PROXY only
NEAR_DUP_COSINE, NEAR_DUP_PRECURSOR_DA = 0.95, 0.01


class LabelLeakError(RuntimeError):
    """A truth-bearing column reached the label-free test-match statistic."""


def assert_label_free(df, where):
    bad = sorted(set(map(str, df.columns)) & LABEL_COLUMNS)
    if bad:
        raise LabelLeakError(f"{where}: truth-bearing columns are not allowed in the label-free diagnostic: {bad}")
    return True


def normalize_polarity(v):
    s = str(v).strip().lower()
    return "positive" if s.startswith("pos") or s == "+" else "negative" if s.startswith("neg") or s == "-" else "unknown"


# ---------------------------------------------------------------------------------------------
# THE statistic
# ---------------------------------------------------------------------------------------------

def query_stats_from_pairs(pairs, pool, query_meta):
    """One row per query of `query_meta` (every query of the population, including ones with an empty
    pool or no eligible reference). `pairs`: PAIR_COLS of the ranker-visible eligible pairs;
    `pool`: query_id, candidate_key (or candidate_connectivity_key), abs_mass_error_ppm;
    `query_meta`: META_COLS (missing optional columns are filled with NA)."""
    for name, df in (("pairs", pairs), ("pool", pool), ("query_meta", query_meta)):
        assert_label_free(df, f"query_stats_from_pairs({name})")
    pool = pool.rename(columns={"candidate_connectivity_key": "candidate_key"})[["query_id", "candidate_key", "abs_mass_error_ppm"]]
    meta = query_meta.copy()
    for c in META_COLS:
        if c not in meta.columns:
            meta[c] = pd.NA
    if meta["query_id"].duplicated().any():
        raise ValueError("query_meta must hold one row per query")
    p = pool.sort_values(["query_id", "abs_mass_error_ppm", "candidate_key"], kind="mergesort").reset_index(drop=True)
    if p.duplicated(["query_id", "candidate_key"]).any():
        raise ValueError("pool must hold one row per (query, candidate connectivity)")
    p["mass_rank"] = p.groupby("query_id", sort=False).cumcount() + 1
    pool_size = p.groupby("query_id").size()

    pr = pairs[PAIR_COLS].merge(p[["query_id", "candidate_key", "abs_mass_error_ppm", "mass_rank"]], on=["query_id", "candidate_key"],
                                how="left", validate="many_to_one")
    if pr["mass_rank"].isna().any():
        raise ValueError(f"{int(pr['mass_rank'].isna().sum())} pairs reference a candidate outside the query's pool")
    g = pr.groupby("query_id")
    agg = pd.DataFrame({"eligible_reference_count": g.size(), "n_candidates_with_reference": g["candidate_key"].nunique(),
                        "max_cosine": g["cosine"].max(), "max_modified_cosine": g["modified_cosine"].max(),
                        "max_peak_overlap": g["peak_overlap_frac"].max()})
    best = pr.sort_values(["query_id", "cosine", "abs_mass_error_ppm", "candidate_key", "compat_rank", "ref_spectrum_id"],
                          ascending=[True, False, True, True, True, True], kind="mergesort").drop_duplicates("query_id").set_index("query_id")

    out = meta[META_COLS].set_index("query_id")
    out["candidate_pool_size"] = pool_size.reindex(out.index).fillna(0).astype(int)
    out = out.join(agg, how="left")
    out["has_eligible_reference"] = out["eligible_reference_count"].notna()
    out["eligible_reference_count"] = out["eligible_reference_count"].fillna(0).astype(int)
    out["n_candidates_with_reference"] = out["n_candidates_with_reference"].fillna(0).astype(int)
    for c in ("max_cosine", "max_modified_cosine", "max_peak_overlap"):
        out[c] = out[c].astype(float).fillna(NO_REFERENCE_VALUE)
    b = best.reindex(out.index)
    nqp = pd.to_numeric(out["n_query_peaks"], errors="coerce")
    out["matched_peaks_at_max_cosine"] = np.where(out["has_eligible_reference"], np.rint(b["peak_overlap_frac"].to_numpy(float) * nqp.to_numpy(float)), 0.0)
    out["best_candidate_connectivity"] = b["candidate_key"]
    out["best_candidate_mass_rank"] = b["mass_rank"]
    out["best_candidate_abs_ppm"] = b["abs_mass_error_ppm"]
    out["best_reference_id"] = b["ref_spectrum_id"]
    return out.reset_index()


def attach_reference_metadata(stats, ref_meta):
    """Best-reference metadata from the exported reference table (bundle `ref_meta.parquet`, the same
    table for every population) + the label-free near-duplicate proxy."""
    rm = ref_meta.rename(columns={"ref_spectrum_id": "best_reference_id", "source": "best_reference_source", "instrument": "best_reference_instrument",
                                  "adduct": "best_reference_adduct", "collision_energy": "best_reference_collision_energy",
                                  "precursor_mz": "best_reference_precursor_mz"})
    cols = ["best_reference_id", "best_reference_source", "best_reference_instrument", "best_reference_adduct", "best_reference_collision_energy",
            "best_reference_precursor_mz"]
    s = stats.drop(columns=[c for c in cols[1:] if c in stats.columns]).merge(rm[cols].drop_duplicates("best_reference_id"),
                                                                            on="best_reference_id", how="left", validate="many_to_one")
    s["best_ref_precursor_diff_da"] = (pd.to_numeric(s["precursor_mz"], errors="coerce") - s["best_reference_precursor_mz"]).abs()
    s["near_duplicate_proxy"] = s["has_eligible_reference"] & (s["max_cosine"] > NEAR_DUP_COSINE) & (s["best_ref_precursor_diff_da"] < NEAR_DUP_PRECURSOR_DA)
    return s


def finalize_query_stats(stats, dataset, protocol, ref_meta):
    s = attach_reference_metadata(stats, ref_meta)
    s.insert(0, "population", population(dataset, protocol))
    s.insert(0, "protocol", protocol)
    s.insert(0, "dataset", dataset)
    s["polarity"] = s["polarity"].map(normalize_polarity)
    s["adduct"] = s["adduct"].astype(str)
    s = s[QUERY_STATS_COLUMNS].sort_values("query_id", kind="mergesort").reset_index(drop=True)
    assert_label_free(s, "finalize_query_stats(output)")
    return s


# ---------------------------------------------------------------------------------------------
# producer 1: TEST (and TL parity) through the production casmi_infer path
# ---------------------------------------------------------------------------------------------

def _bundle_pairs(bundle, query_id, sim_peaks, qmeta, qhash, candidates, exclude_sources=frozenset(), exclude_ids=frozenset()):
    """Ranker-visible pairs of one query: EXACTLY `casmi_infer.features.candidate_features`' reference
    selection + `pair_scores`, but keeping the per-pair values (candidate_features only keeps aggregates)."""
    from casmi_infer.compat import select_references
    from casmi_infer.similarity import pair_scores
    ci, ap = candidates
    keys = bundle.conn_table["connectivity_key"].to_numpy()
    rows, n_t1 = [], 0
    for c, _ in zip(ci, ap):
        sel, _, t1 = select_references(bundle.lib, int(c), qmeta, qhash, exclude_sources, exclude_ids, bundle.sim_cfg["top_reference_count"])
        n_t1 += int(t1)
        for pos, r in enumerate(sel, start=1):
            cos, mod, ov, _nl = pair_scores(sim_peaks, bundle.lib.peaks(int(r)), bundle.sim_cfg)
            rows.append((query_id, keys[int(c)], bundle.lib.spectrum_id[int(r)], pos, float(cos), float(mod), float(ov)))
    pairs = pd.DataFrame(rows, columns=PAIR_COLS)
    pool = pd.DataFrame({"query_id": query_id, "candidate_key": keys[np.asarray(ci, dtype=np.int64)], "abs_mass_error_ppm": np.asarray(ap, float)})
    return pairs, pool, n_t1


def observed_test_query(bundle, row, log):
    """One visible-TEST spectrum under production semantics (unpublished query: exclude_sources = {},
    exclude_ids = {}; T1 by peak hash). `row` is a test.parquet record; only observable fields are read."""
    from casmi_infer.pipeline import resolve_neutral_mass
    from casmi_infer.spectrum import clean_query, query_meta, query_peak_hash
    identity, sim = clean_query(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"], bundle.sim_cfg["max_peaks_similarity"])
    qm = query_meta(row["adduct"], row.get("ionization_mode"), row.get("instrument_type"), row.get("collision_energy_ev"), source=None)
    nm, used = resolve_neutral_mass(bundle, row, log)
    ci, ap, level = bundle.search.search(nm, bundle.primary_ppm(used), tuple(bundle.ppm["fallback_ppm"]), bundle.ppm["nearest_n"])
    qid = str(row["spectrum_id"])
    pairs, pool, n_t1 = _bundle_pairs(bundle, qid, sim, qm, query_peak_hash(identity), (ci, ap))
    meta = {"query_id": qid, "molecule_id": row.get("molecule_id"), "instrument": row.get("instrument_type"), "adduct": str(row["adduct"]),
            "polarity": row.get("ionization_mode"), "precursor_mz": float(row["precursor_mz"]), "neutral_mass": nm, "retrieval_level": level,
            "n_query_peaks": int(len(sim["mzs"])), "n_t1_excluded_refs": n_t1}
    return pairs, pool, meta


def run_observed_test(bundle, test_df, cache_dir, chunk_size=250, limit=None, progress=True):
    """All visible-TEST spectra (sorted by spectrum_id), NUMPY backend, chunk-cached under `cache_dir`
    (a chunk is reused only if its query-id list and the bundle's reference-library sha match).
    Returns (pairs, pool, meta)."""
    import time

    from casmi_infer.backend import NUMPY, use_backend
    from casmi_infer.pipeline import RunLog
    assert_label_free(test_df, "run_observed_test(test_df)")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lib_sha = (bundle.manifest.get("files") or {}).get("ref_meta.parquet", "") + "|" + (bundle.manifest.get("files") or {}).get("ref_peaks_mz.npy", "")
    df = test_df.sort_values("spectrum_id", kind="mergesort").reset_index(drop=True)
    if limit:
        df = df.iloc[:int(limit)]
    log, parts = RunLog(), {"pairs": [], "pool": [], "meta": []}
    t0 = time.time()
    with use_backend(NUMPY):
        for k, start in enumerate(range(0, len(df), chunk_size)):
            chunk = df.iloc[start:start + chunk_size]
            ids = chunk["spectrum_id"].astype(str).tolist()
            f = {n: cache_dir / f"{n}-{k:05d}.parquet" for n in parts}
            if all(p.exists() for p in f.values()):
                m = pd.read_parquet(f["meta"])
                if m["query_id"].astype(str).tolist() == ids and (m["_lib_sha"] == lib_sha).all():
                    for n in parts:
                        parts[n].append(pd.read_parquet(f[n]).drop(columns=["_lib_sha"], errors="ignore"))
                    continue
            cp, cpool, cm = [], [], []
            for row in chunk.to_dict("records"):
                pr, po, me = observed_test_query(bundle, row, log)
                cp.append(pr)
                cpool.append(po)
                cm.append(me)
            out = {"pairs": pd.concat(cp, ignore_index=True) if cp else pd.DataFrame(columns=PAIR_COLS),
                   "pool": pd.concat(cpool, ignore_index=True) if cpool else pd.DataFrame(columns=["query_id", "candidate_key", "abs_mass_error_ppm"]),
                   "meta": pd.DataFrame(cm, columns=META_COLS)}
            out["meta"]["_lib_sha"] = lib_sha
            for n in parts:
                out[n].to_parquet(f[n], index=False)
                parts[n].append(out[n].drop(columns=["_lib_sha"], errors="ignore"))
            if progress:
                print(f"[test_observed] {min(start + chunk_size, len(df))}/{len(df)} spectra  {time.time() - t0:.0f}s")
    res = tuple(pd.concat(parts[n], ignore_index=True) for n in ("pairs", "pool", "meta"))
    return (*res, log)


def library_query_meta(bundle, query_ids):
    """Observable metadata of LIBRARY queries (TL_EVAL / HOST spectra are training spectra): read from
    the exported reference table at ref_row == train row offset, neutral mass by the bundle adduct
    rules (serialized from the training parser), similarity-peak count from the exported peak store."""
    lib = bundle.lib
    rows = []
    for q in query_ids:
        off = int(str(q).split("_")[1])
        m = lib.meta.iloc[off]
        if str(m["ref_spectrum_id"]) != str(q):
            raise ValueError(f"ref_meta row {off} is {m['ref_spectrum_id']!r}, expected {q!r}")
        prec = float(lib.precursor_mz[off])
        rows.append({"query_id": str(q), "molecule_id": None, "instrument": m["instrument"], "adduct": str(m["adduct"]), "polarity": m["polarity"],
                     "precursor_mz": prec, "neutral_mass": bundle.adducts.neutral_mass(prec, str(m["adduct"])), "retrieval_level": "primary",
                     "n_query_peaks": int(len(lib.peaks(off)["mzs"])), "n_t1_excluded_refs": np.nan})
    return pd.DataFrame(rows, columns=META_COLS)


def bundle_library_query_pairs(bundle, query_id, protocol):
    """PARITY PRODUCER: a library query re-derived through the TEST producer with the benchmark
    exclusions of `protocol` (mirror_aware: exclude the query's source + its own id; strict/standard:
    own id only). T2 is not applicable on this path (not shipped), so T2-affected queries are the only
    admissible differences vs the QCR producer."""
    lib = bundle.lib
    off = int(str(query_id).split("_")[1])
    m = lib.meta.iloc[off]
    qm = {"adduct": m["adduct"], "ion_mode": m["polarity"], "instrument": m["instrument"], "ce": float(lib.ce[off]), "source": lib.source[off]}
    prec = float(lib.precursor_mz[off])
    nm = bundle.adducts.neutral_mass(prec, str(m["adduct"]))
    ci, ap = bundle.search._dedupe(np.arange(*bundle.search.window(nm, bundle.primary_ppm(str(m["adduct"])))), nm)
    pdef = PROTOCOL_DEFS[protocol]
    ex_src = frozenset([lib.source[off]]) if pdef.source_exclusion else frozenset()
    qhash = lib.peak_hash[off] if "T1" in pdef.excluded_tiers else None
    return _bundle_pairs(bundle, str(query_id), lib.peaks(off), qm, qhash, (ci, ap), ex_src, frozenset([str(query_id)]))


# ---------------------------------------------------------------------------------------------
# producer 2: TL_EVAL / HOST from the persisted QCR (protocol filter only)
# ---------------------------------------------------------------------------------------------

def qcr_protocol_pairs(qcr_paths, protocol, pool=None, pair_paths=None, check_walk=True):
    """Ranker-visible pairs under `protocol` from QCR shards, read with LABEL-FREE columns only.
    For a protocol other than mirror_aware, the walk-completeness argument is re-checked on the data
    (`assert_walk_complete`); the counts need a pool, to which a constant placeholder
    `is_true_candidate=False` is attached (carried, never read)."""
    pdef = PROTOCOL_DEFS[protocol]
    if check_walk and protocol != MIRROR:
        if pool is None or pair_paths is None:
            raise ValueError("walk-completeness check needs the pool and the qcr_pairs shards")
        placeholder = pool[POOL_LABEL_FREE_COLS].assign(is_true_candidate=False)
        assert_walk_complete(protocol_counts_from_shards(list(qcr_paths), list(pair_paths), placeholder, [protocol])[protocol])
    parts = []
    for p in qcr_paths:
        q = pd.read_parquet(p, columns=QCR_LABEL_FREE_COLS)
        if len(q):
            parts.append(select_protocol_refs(q, pdef, TOP_K)[PAIR_COLS])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PAIR_COLS)


def read_label_free_pool(path_or_paths, query_ids=None):
    paths = [path_or_paths] if isinstance(path_or_paths, (str, Path)) else list(path_or_paths)
    pool = pd.concat([pd.read_parquet(p, columns=POOL_LABEL_FREE_COLS) for p in paths], ignore_index=True)
    if query_ids is not None:
        pool = pool[pool["query_id"].isin(set(map(str, query_ids)))]
    return pool.drop_duplicates(["query_id", "candidate_connectivity_key"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# distributions, distances, strata, reweighting (one weighted implementation; raw = unit weights)
# ---------------------------------------------------------------------------------------------

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
COSINE_SHARES = ((">=", 0.95), (">=", 0.90), (">=", 0.80), (">=", 0.70), ("<", 0.50))
QUANTILE_METRICS = ("max_cosine", "matched_peaks_at_max_cosine", "max_modified_cosine", "max_peak_overlap", "candidate_pool_size",
                    "eligible_reference_count")
KS_METRICS = ("max_cosine", "max_modified_cosine", "matched_peaks_at_max_cosine", "candidate_pool_size", "max_peak_overlap")
POOL_SIZE_BINS = (0, 1, 50, 150, 300, 600, np.inf)
POOL_SIZE_LABELS = ("0", "1-49", "50-149", "150-299", "300-599", "600+")
NEUTRAL_MASS_BINS = (0, 200, 300, 400, 500, 700, np.inf)
NEUTRAL_MASS_LABELS = ("<200", "200-299", "300-399", "400-499", "500-699", "700+")
TOP_ADDUCTS = 6
MIN_STRATUM_N = 30
REWEIGHT_DIMS = (("adduct_group", "polarity", "pool_size_bin"), ("adduct_group", "polarity"), ("polarity", "pool_size_bin"), ("polarity",))
REWEIGHT_MIN_CELL, REWEIGHT_MIN_COVERAGE = 5, 0.95


def _w(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & (w > 0)
    return v[ok], w[ok]


def weighted_quantile(values, q, weights=None):
    """Inverted-CDF quantile (a value that occurs in the sample); unit weights = the raw quantile."""
    v, w = _w(values, weights)
    if not len(v):
        return float("nan")
    o = np.argsort(v, kind="mergesort")
    v, cw = v[o], np.cumsum(w[o]) / w.sum()
    return float(v[min(int(np.searchsorted(cw, q - 1e-12, side="left")), len(v) - 1)])


def weighted_share(values, op, threshold, weights=None):
    v, w = _w(values, weights)
    if not len(v):
        return float("nan")
    hit = v >= threshold if op == ">=" else v < threshold
    return float((w * hit).sum() / w.sum())


def ks_distance(a, b, wa=None, wb=None):
    """Two-sample (weighted) Kolmogorov-Smirnov statistic: sup_x |F_a(x) - F_b(x)| over the pooled support."""
    va, wa = _w(a, wa)
    vb, wb = _w(b, wb)
    if not len(va) or not len(vb):
        return float("nan")
    oa, ob = np.argsort(va, kind="mergesort"), np.argsort(vb, kind="mergesort")
    va, ca = va[oa], np.cumsum(wa[oa]) / wa.sum()
    vb, cb = vb[ob], np.cumsum(wb[ob]) / wb.sum()
    x = np.union1d(va, vb)
    fa = np.concatenate([[0.0], ca])[np.searchsorted(va, x, side="right")]
    fb = np.concatenate([[0.0], cb])[np.searchsorted(vb, x, side="right")]
    return float(np.max(np.abs(fa - fb)))


def add_strata(stats, test_stats):
    """Observable stratum columns. The adduct groups are the TOP_ADDUCTS most frequent TEST adducts
    (everything else -> OTHER); fixed pool-size and neutral-mass bins."""
    top = test_stats["adduct"].astype(str).value_counts().sort_index().sort_values(ascending=False, kind="mergesort").index[:TOP_ADDUCTS]
    s = stats.copy()
    s["adduct_group"] = np.where(s["adduct"].astype(str).isin(set(top)), s["adduct"].astype(str), "OTHER")
    s["pool_size_bin"] = pd.cut(s["candidate_pool_size"], POOL_SIZE_BINS, right=False, labels=POOL_SIZE_LABELS).astype(str)
    s["neutral_mass_bin"] = pd.cut(pd.to_numeric(s["neutral_mass"], errors="coerce"), NEUTRAL_MASS_BINS, right=False,
                                   labels=NEUTRAL_MASS_LABELS).astype(str)
    return s


def distribution_row(stats, weights=None):
    row = {"n_queries": int(len(stats)), "n_effective": float((np.sum(weights) ** 2 / np.sum(np.square(weights)))) if weights is not None and len(stats) else float(len(stats)),
           "share_no_eligible_reference": weighted_share((~stats["has_eligible_reference"].astype(bool)).astype(float), ">=", 0.5, weights)}
    for m in QUANTILE_METRICS:
        for q in QUANTILES:
            row[f"{m}_P{int(round(q * 100))}"] = weighted_quantile(stats[m], q, weights)
    for op, t in COSINE_SHARES:
        row[f"share_max_cosine_{'ge' if op == '>=' else 'lt'}_{t:.2f}"] = weighted_share(stats["max_cosine"], op, t, weights)
    row["share_near_duplicate_proxy"] = weighted_share(stats["near_duplicate_proxy"].astype(float), ">=", 0.5, weights)
    return row


def distribution_summary(query_stats, weights_by_population=None):
    """One row per (population, weighting); RAW always, TEST_REWEIGHTED where weights are given."""
    rows = []
    for pop, g in query_stats.groupby("population", sort=True):
        rows.append({"population": pop, "weighting": "RAW", **distribution_row(g)})
        w = (weights_by_population or {}).get(pop)
        if w is not None:
            rows.append({"population": pop, "weighting": "TEST_REWEIGHTED", **distribution_row(g, w.reindex(g["query_id"]).to_numpy(float))})
    return pd.DataFrame(rows)


def rule_inputs(test, other, w_other=None):
    """The four pre-registered label-free comparisons of TEST vs one protocol population."""
    return {"ks_max_cosine": ks_distance(test["max_cosine"], other["max_cosine"], None, w_other),
            "ks_max_modified_cosine": ks_distance(test["max_modified_cosine"], other["max_modified_cosine"], None, w_other),
            "abs_diff_share_max_cosine_ge_0.95": abs(weighted_share(test["max_cosine"], ">=", 0.95) - weighted_share(other["max_cosine"], ">=", 0.95, w_other)),
            "abs_diff_median_matched_peaks": abs(weighted_quantile(test["matched_peaks_at_max_cosine"], 0.5)
                                                 - weighted_quantile(other["matched_peaks_at_max_cosine"], 0.5, w_other))}


def protocol_distances(test, others, weights=None):
    """Wide table: metric x (test_vs_<name>_distance). `others`: {"strict": df, "mirror": df, ["standard": df]}.
    KS rows for every KS_METRICS entry plus the two share/median rule rows; nothing is dropped when
    metrics disagree."""
    weights = weights or {}
    rows = []
    for m in KS_METRICS:
        rows.append({"metric": f"ks_{m}", **{f"test_vs_{k}_distance": ks_distance(test[m], o[m], None, weights.get(k)) for k, o in others.items()}})
    for key in ("abs_diff_share_max_cosine_ge_0.95", "abs_diff_median_matched_peaks"):
        rows.append({"metric": key, **{f"test_vs_{k}_distance": rule_inputs(test, o, weights.get(k))[key] for k, o in others.items()}})
    t = pd.DataFrame(rows)
    if "test_vs_strict_distance" in t and "test_vs_mirror_distance" in t:
        t["closer"] = np.where(np.isclose(t["test_vs_strict_distance"], t["test_vs_mirror_distance"], atol=1e-12, rtol=0), "tie",
                               np.where(t["test_vs_strict_distance"] < t["test_vs_mirror_distance"], "strict", "mirror"))
    return t


def stratified_distances(test, others, strata=("adduct_group", "polarity", "pool_size_bin", "neutral_mass_bin"), min_n=MIN_STRATUM_N):
    """KS(max_cosine) of TEST vs each protocol inside every observable stratum with >= min_n queries on
    both sides (sparse strata are listed with status TOO_SMALL, not silently dropped)."""
    rows = []
    for dim in strata:
        for level in sorted(set(test[dim].astype(str))):
            t = test[test[dim].astype(str) == level]
            r = {"stratum_dim": dim, "stratum": level, "n_test": int(len(t))}
            ok = len(t) >= min_n
            for k, o in others.items():
                sub = o[o[dim].astype(str) == level]
                r[f"n_{k}"] = int(len(sub))
                ok &= len(sub) >= min_n
                r[f"ks_test_vs_{k}"] = ks_distance(t["max_cosine"], sub["max_cosine"]) if len(sub) and len(t) else float("nan")
                r[f"median_max_cosine_{k}"] = weighted_quantile(sub["max_cosine"], 0.5)
            r["median_max_cosine_test"] = weighted_quantile(t["max_cosine"], 0.5)
            r["status"] = "OK" if ok else "TOO_SMALL"
            if ok and "strict" in others and "mirror" in others:
                d = r["ks_test_vs_strict"] - r["ks_test_vs_mirror"]
                r["closer"] = "tie" if abs(d) <= 1e-12 else ("strict" if d < 0 else "mirror")
            rows.append(r)
    return pd.DataFrame(rows)


def stratified_summary(strat):
    ok = strat[strat["status"] == "OK"]
    if not len(ok) or "closer" not in ok:
        return {"n_strata_ok": 0}
    w = ok["n_test"].astype(float)
    return {"n_strata_ok": int(len(ok)), "n_strata_strict_closer": int((ok["closer"] == "strict").sum()),
            "n_strata_mirror_closer": int((ok["closer"] == "mirror").sum()),
            "test_weighted_share_strict_closer": float((w * (ok["closer"] == "strict")).sum() / w.sum())}


def _cell(df, dims):
    return df[list(dims)].astype(str).agg("|".join, axis=1)


def reweighting_weights_to_test(tl, test, dims_candidates=REWEIGHT_DIMS, min_cell=REWEIGHT_MIN_CELL, min_coverage=REWEIGHT_MIN_COVERAGE):
    """Weights (indexed by TL query_id, mean 1) that make TL's distribution over OBSERVABLE cells
    match visible TEST's. The first dims tuple whose TL-supported cells (>= min_cell TL queries) cover
    >= min_coverage of TEST is used (coarser fallbacks avoid sparse cross-products). Only the columns
    in `dims` are read -- no statistic, score or label."""
    for dims in dims_candidates:
        tc = _cell(test, dims).value_counts(normalize=True)
        lc = _cell(tl, dims).value_counts()
        supported = lc[lc >= min_cell].index
        coverage = float(tc[tc.index.isin(supported)].sum())
        if coverage >= min_coverage or dims == dims_candidates[-1]:
            break
    tc_s = tc[tc.index.isin(supported)] / max(coverage, 1e-12)
    cell = _cell(tl, dims)
    w = cell.map(lambda c: tc_s.get(c, 0.0) / lc[c] if c in tc_s.index else 0.0).to_numpy(float)
    w = w * (len(w) / w.sum()) if w.sum() > 0 else w
    info = {"dims": list(dims), "test_coverage": coverage, "n_cells_used": int(len(tc_s)), "uncovered_test_share": 1 - coverage,
            "min_cell": min_cell, "min_coverage": min_coverage, "n_tl_zero_weight": int((w == 0).sum())}
    return pd.Series(w, index=tl["query_id"].to_numpy()), info


# ---------------------------------------------------------------------------------------------
# pre-registered protocol-resemblance rule (label-free distances ONLY)
# ---------------------------------------------------------------------------------------------

STRICT_LIKE, MIRROR_LIKE, INCONCLUSIVE = "STRICT_LIKE", "MIRROR_LIKE", "INTERMEDIATE_OR_INCONCLUSIVE"
RESEMBLANCE_STATUSES = (STRICT_LIKE, MIRROR_LIKE, INCONCLUSIVE)
PRIMARY_INPUT = "ks_max_cosine"
SUPPORTING_INPUTS = ("ks_max_modified_cosine", "abs_diff_share_max_cosine_ge_0.95", "abs_diff_median_matched_peaks")
RULE_INPUTS = (PRIMARY_INPUT,) + SUPPORTING_INPUTS
RESEMBLANCE_RULE = {
    "rule_id": "v6-test-match-1",
    "question": "which TL_EVAL reference-eligibility protocol does the visible TEST's label-free query->training-library similarity resemble?",
    "statistic": "max cosine over the ranker-visible (first 5 eligible references per mass-window candidate) pairs; 0.0 when no eligible pair; "
                 "identical preprocessing / kernel / retrieval for every population",
    "populations": {"TEST": "visible competition test, production casmi_infer eligibility (exclude_sources={}, T1 by peak hash, T2 not applied)",
                    "strict": "TL_EVAL under test_simulated_strict (QCR filter)", "mirror": "TL_EVAL under mirror_aware (QCR filter)",
                    "standard": "TL_EVAL under standard (optimistic sensitivity; never decisive)"},
    "primary": "KS(max_cosine), TEST vs strict and TEST vs mirror",
    "supporting": ["KS(max_modified_cosine)", "|share(max_cosine>=0.95) difference|", "|median matched_peaks_at_max_cosine difference|"],
    "primary_tie_tolerance": 0.02,
    "max_primary_distance_for_resemblance": 0.25,
    "decision": [
        "if |KS_strict - KS_mirror| <= primary_tie_tolerance -> INTERMEDIATE_OR_INCONCLUSIVE",
        "closer = protocol with the smaller primary KS; if that KS > max_primary_distance_for_resemblance -> INTERMEDIATE_OR_INCONCLUSIVE (TEST far from both)",
        "each supporting comparison votes for the protocol with the smaller distance (exact ties abstain)",
        "if >= 2 of the 3 supporting votes go to the OTHER protocol (majority contradiction) -> INTERMEDIATE_OR_INCONCLUSIVE",
        "else closer == strict -> STRICT_LIKE; closer == mirror -> MIRROR_LIKE",
    ],
    "raw_vs_reweighted": "the rule is applied to RAW and to TEST-REWEIGHTED TL distances; final status = the RAW decision when the "
                         "reweighted decision agrees, otherwise INTERMEDIATE_OR_INCONCLUSIVE",
    "reweighting": {"dims_fallback_order": [list(d) for d in REWEIGHT_DIMS], "min_cell": REWEIGHT_MIN_CELL, "min_test_coverage": REWEIGHT_MIN_COVERAGE},
    "strata": {"adduct_group": f"top {TOP_ADDUCTS} TEST adducts + OTHER", "pool_size_bins": list(POOL_SIZE_LABELS),
               "neutral_mass_bins": list(NEUTRAL_MASS_LABELS), "min_n": MIN_STRATUM_N, "role": "reported, not decisive"},
    "excluded_inputs": "model scores, ranks, MRR / Hit@k, truth identity, reference availability of the truth -- none can reach the rule",
    "no_reference_value": NO_REFERENCE_VALUE,
}


def _check_rule_inputs(d, where):
    keys = set(d)
    if keys != set(RULE_INPUTS):
        raise ValueError(f"{where}: rule inputs must be exactly {RULE_INPUTS}, got {sorted(keys)}")
    leaked = [k for k in keys if any(t in k.lower() for t in ("mrr", "hit", "score", "rank", "is_true"))]
    if leaked:
        raise LabelLeakError(f"{where}: model-performance inputs are not allowed: {leaked}")


def decide_resemblance(strict_inputs, mirror_inputs, rule=RESEMBLANCE_RULE):
    """`*_inputs`: `rule_inputs(TEST, TL_protocol)` dicts. Returns {status, reason, votes, ...}."""
    _check_rule_inputs(strict_inputs, "decide_resemblance(strict)")
    _check_rule_inputs(mirror_inputs, "decide_resemblance(mirror)")
    ps, pm = float(strict_inputs[PRIMARY_INPUT]), float(mirror_inputs[PRIMARY_INPUT])
    out = {"primary_ks_strict": ps, "primary_ks_mirror": pm, "votes": {}}
    if not (np.isfinite(ps) and np.isfinite(pm)):
        return {**out, "status": INCONCLUSIVE, "reason": "primary distance not computable"}
    for k in SUPPORTING_INPUTS:
        s, m = float(strict_inputs[k]), float(mirror_inputs[k])
        out["votes"][k] = "abstain" if not (np.isfinite(s) and np.isfinite(m)) or abs(s - m) <= 1e-12 else ("strict" if s < m else "mirror")
    if abs(ps - pm) <= rule["primary_tie_tolerance"]:
        return {**out, "status": INCONCLUSIVE, "reason": f"primary KS tie (|{ps:.4f} - {pm:.4f}| <= {rule['primary_tie_tolerance']})"}
    closer = "strict" if ps < pm else "mirror"
    if min(ps, pm) > rule["max_primary_distance_for_resemblance"]:
        return {**out, "status": INCONCLUSIVE, "closer_on_primary": closer,
                "reason": f"TEST is far from both protocols (closer KS {min(ps, pm):.4f} > {rule['max_primary_distance_for_resemblance']})"}
    against = sum(v not in (closer, "abstain") for v in out["votes"].values())
    if against >= 2:
        return {**out, "status": INCONCLUSIVE, "closer_on_primary": closer, "reason": f"primary favours {closer} but {against}/3 supporting comparisons contradict"}
    return {**out, "status": STRICT_LIKE if closer == "strict" else MIRROR_LIKE, "closer_on_primary": closer,
            "reason": f"primary favours {closer}; {against}/3 supporting comparisons contradict"}


def final_resemblance(raw_decision, reweighted_decision):
    if reweighted_decision is None or raw_decision["status"] == reweighted_decision["status"]:
        return raw_decision["status"]
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------------------
# matchability proxies (NOT class labels)
# ---------------------------------------------------------------------------------------------

MATCH_HIGH, MATCH_LOW = 0.80, 0.50


def matchability_tier(max_cos):
    v = np.asarray(max_cos, dtype=float)
    return np.where(v >= MATCH_HIGH, "HIGH", np.where(v >= MATCH_LOW, "MEDIUM", "LOW"))


def library_matchability_proxies(stats):
    """Spectrum-level PROXIES for direct-library-matchable spectra (never a 'Class 1 fraction')."""
    s = stats
    out = {"wording": "proxy for direct-library-matchable spectra (label-free; not a true Class-1 fraction)", "n_spectra": int(len(s)),
           "very_high_match_share (max_cosine>=0.95)": weighted_share(s["max_cosine"], ">=", 0.95),
           "high_match_share (max_cosine>=0.80)": weighted_share(s["max_cosine"], ">=", 0.80),
           "low_match_share (max_cosine<0.50)": weighted_share(s["max_cosine"], "<", 0.50),
           "no_eligible_reference_share": float((~s["has_eligible_reference"].astype(bool)).mean()) if len(s) else float("nan"),
           "near_duplicate_proxy_share (T3-like: cos>0.95 & |dprecursor|<0.01 Da)": float(s["near_duplicate_proxy"].astype(bool).mean()) if len(s) else float("nan")}
    t1 = pd.to_numeric(s["n_t1_excluded_refs"], errors="coerce")
    if t1.notna().any():
        # v6.1: formerly (mis)named 'exact_library_duplicate_share'. This is HASH equality (4/3/3-decimal rounding) over every
        # reference of every mass-window candidate, not array identity and not molecule identity; see library_identity_audit.
        out["t1_peak_hash_hit_in_candidate_refs_share (hash-level; excluded at inference)"] = float((t1.fillna(0) > 0).mean())
    from casmi.validation.library_identity_audit import assert_identity_wording
    assert_identity_wording(out)
    return out


def molecule_matchability(stats, molecule_col="molecule_id"):
    """One row per molecule (deterministic: molecule ASC) from its spectra's max-library cosines."""
    s = stats[stats[molecule_col].notna()]
    g = s.groupby(molecule_col, sort=True)["max_cosine"]
    m = pd.DataFrame({"n_spectra": g.size(), "max_spectrum_max_cosine": g.max(), "mean_max_cosine": g.mean(), "median_max_cosine": g.median(),
                      "n_spectra_ge_0.95": g.apply(lambda v: int((v >= 0.95).sum())), "n_spectra_ge_0.80": g.apply(lambda v: int((v >= 0.80).sum())),
                      "n_spectra_lt_0.50": g.apply(lambda v: int((v < 0.50).sum()))})
    m["matchability_tier"] = matchability_tier(m["max_spectrum_max_cosine"])
    return m.reset_index().rename(columns={molecule_col: "molecule_id"})


def molecule_matchability_shares(mol):
    n = len(mol)
    return {"n_molecules": int(n), "wording": "proxy (label-free): best-spectrum max library cosine per molecule",
            **{f"share_{t}": float((mol["matchability_tier"] == t).mean()) if n else float("nan") for t in ("HIGH", "MEDIUM", "LOW")},
            "share_best_spectrum_ge_0.95": float((mol["max_spectrum_max_cosine"] >= 0.95).mean()) if n else float("nan"),
            "share_with_any_spectrum_lt_0.50": float((mol["n_spectra_lt_0.50"] > 0).mean()) if n else float("nan")}


# ---------------------------------------------------------------------------------------------
# v6.1 multi-spectrum connectivity consensus (LABEL-FREE: consistency, NOT accuracy)
# ---------------------------------------------------------------------------------------------
# Per spectrum, the label-free "best library match" is `best_candidate_connectivity`: the candidate
# owning the max-cosine ranker-visible pair (tie rule of `query_stats_from_pairs`). A molecule's spectra
# should point to one connectivity if the library evidence is coherent. Agreement is measured without
# any truth: it says whether the spectra AGREE, not whether they are RIGHT. Leave-one-out agreement
# (a spectrum vs the modal connectivity of its sibling spectra) shows at which cosine a single spectrum
# stops being trustworthy, which is the question multi-spectrum aggregation has to answer.

CONSENSUS_CONFIDENT_COSINE = MATCH_HIGH
CONSENSUS_COSINE_BINS = (0.0, 0.50, 0.70, 0.80, 0.90, 0.95, np.inf)
CONSENSUS_COSINE_LABELS = ("<0.50", "0.50-0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", ">=0.95")
CONSENSUS_INPUT_COLS = ["query_id", "molecule_id", "best_candidate_connectivity", "max_cosine", "has_eligible_reference", "adduct", "polarity",
                        "neutral_mass"]
CONSENSUS_CLASSES = ("NO_REFERENCE", "SINGLE_SPECTRUM", "UNANIMOUS", "MAJORITY", "SPLIT")


def _modal(conns, cos):
    """Modal connectivity: count DESC, summed max-cosine DESC, connectivity ASC. (None, 0) if empty."""
    agg = {}
    for c, w in zip(conns, cos):
        n, s = agg.get(c, (0, 0.0))
        agg[c] = (n + 1, s + float(w))
    if not agg:
        return None, 0
    c = min(agg, key=lambda k: (-agg[k][0], -agg[k][1], str(k)))
    return c, agg[c][0]


def _consensus_class(n_ref, n_distinct, modal_share):
    if n_ref == 0:
        return "NO_REFERENCE"
    if n_ref == 1:
        return "SINGLE_SPECTRUM"
    if n_distinct == 1:
        return "UNANIMOUS"
    return "MAJORITY" if modal_share > 0.5 else "SPLIT"


def connectivity_consensus(stats, pool=None, molecule_col="molecule_id"):
    """(molecules, spectra), deterministic (molecule ASC, query ASC). Reads ONLY `CONSENSUS_INPUT_COLS`
    of label-free query stats (+ optionally the label-free pool: query_id, candidate_key) -- no score,
    rank or truth. A spectrum without an eligible reference has no best connectivity and does not vote."""
    s = stats.rename(columns={molecule_col: "molecule_id"})
    miss = [c for c in CONSENSUS_INPUT_COLS if c not in s.columns]
    if miss:
        raise ValueError(f"connectivity_consensus needs {miss}")
    s = s[CONSENSUS_INPUT_COLS].copy()
    assert_label_free(s, "connectivity_consensus(stats)")
    s = s[s["molecule_id"].notna()].sort_values(["molecule_id", "query_id"], kind="mergesort")
    s["votes"] = s["has_eligible_reference"].astype(bool) & s["best_candidate_connectivity"].notna()
    pool_sets = None
    if pool is not None:
        p = pool.rename(columns={"candidate_connectivity_key": "candidate_key"})
        assert_label_free(p[["query_id", "candidate_key"]], "connectivity_consensus(pool)")
        pool_sets = {str(q): set(g) for q, g in p.groupby("query_id")["candidate_key"]}
    mol_rows, spec_rows = [], []
    for mid, g in s.groupby("molecule_id", sort=True):
        v = g[g["votes"]]
        qids, conns, cos = v["query_id"].astype(str).tolist(), v["best_candidate_connectivity"].tolist(), v["max_cosine"].astype(float).tolist()
        modal, modal_n = _modal(conns, cos)
        n_ref, n_distinct = len(conns), len(set(conns))
        modal_share = modal_n / n_ref if n_ref else float("nan")
        conf = [c for c, w in zip(conns, cos) if w >= CONSENSUS_CONFIDENT_COSINE]
        best_i = min(range(n_ref), key=lambda i: (-cos[i], qids[i])) if n_ref else None
        best_conn = conns[best_i] if best_i is not None else None
        nm = pd.to_numeric(g["neutral_mass"], errors="coerce")
        r = {"molecule_id": mid, "n_spectra": int(len(g)), "n_with_reference": n_ref, "n_distinct_best_connectivity": n_distinct,
             "modal_connectivity": modal, "modal_count": modal_n, "modal_share": modal_share,
             "n_confident": len(conf), "n_distinct_confident_connectivity": len(set(conf)), "confident_conflict": len(set(conf)) >= 2,
             "best_spectrum_connectivity": best_conn, "best_spectrum_max_cosine": cos[best_i] if best_i is not None else NO_REFERENCE_VALUE,
             "best_spectrum_agrees_with_modal": bool(best_conn is not None and best_conn == modal),
             "n_adducts": int(g["adduct"].astype(str).nunique()), "n_polarities": int(g["polarity"].astype(str).nunique()),
             "neutral_mass_range_da": float(nm.max() - nm.min()) if nm.notna().any() else float("nan"),
             "consensus_class": _consensus_class(n_ref, n_distinct, modal_share)}
        if pool_sets is not None:
            ps = [pool_sets.get(str(q), set()) for q in g["query_id"]]
            r["pool_intersection_size"] = len(set.intersection(*ps)) if ps else 0
            r["modal_in_all_pools"] = bool(modal is not None and all(modal in x for x in ps))
        mol_rows.append(r)
        for i, (q, c, w) in enumerate(zip(qids, conns, cos)):
            oc, ow = conns[:i] + conns[i + 1:], cos[:i] + cos[i + 1:]
            lm, _ = _modal(oc, ow)
            spec_rows.append({"molecule_id": mid, "query_id": q, "best_candidate_connectivity": c, "max_cosine": w, "n_other_voting_spectra": len(oc),
                              "loo_modal_connectivity": lm, "loo_agrees": bool(lm is not None and c == lm), "supported_by_any_other": c in oc})
    mol = pd.DataFrame(mol_rows)
    if len(mol):
        mol["matchability_tier"] = matchability_tier(mol["best_spectrum_max_cosine"])
    spec = pd.DataFrame(spec_rows, columns=["molecule_id", "query_id", "best_candidate_connectivity", "max_cosine", "n_other_voting_spectra",
                                            "loo_modal_connectivity", "loo_agrees", "supported_by_any_other"])
    spec["cosine_bin"] = pd.cut(spec["max_cosine"], CONSENSUS_COSINE_BINS, right=False, labels=CONSENSUS_COSINE_LABELS).astype(str)
    return mol, spec


def connectivity_consensus_summary(mol, spec):
    """Shares over molecules with >= 2 voting spectra; LOO agreement over spectra with >= 1 voting sibling."""
    multi = mol[mol["n_with_reference"] >= 2]
    loo = spec[spec["n_other_voting_spectra"] >= 1]

    def sh(x):
        x = pd.Series(x).astype(bool)
        return float(x.mean()) if len(x) else float("nan")

    def bucket(n):
        return str(int(n)) if n < 5 else "5+"

    out = {"wording": "label-free CONSISTENCY of the best-library-match connectivity across a molecule's spectra; agreement is NOT accuracy",
           "confident_cosine": CONSENSUS_CONFIDENT_COSINE, "n_molecules": int(len(mol)), "n_multi_spectrum_molecules": int(len(multi)),
           "consensus_class_counts": {k: int((mol["consensus_class"] == k).sum()) for k in CONSENSUS_CLASSES},
           "share_unanimous_multi": sh(multi["consensus_class"] == "UNANIMOUS"),
           "share_majority_or_unanimous_multi": sh(multi["consensus_class"].isin(["UNANIMOUS", "MAJORITY"])),
           "share_split_multi": sh(multi["consensus_class"] == "SPLIT"),
           "share_confident_conflict_multi": sh(multi["confident_conflict"]),
           "share_best_spectrum_agrees_with_modal_multi": sh(multi["best_spectrum_agrees_with_modal"]),
           "modal_share_quantiles_multi": {f"P{q}": weighted_quantile(multi["modal_share"], q / 100) for q in (10, 25, 50, 75)},
           "by_n_voting_spectra": {}, "by_matchability_tier": {}, "loo_agreement_by_cosine_bin": {}}
    for b, g in multi.groupby(multi["n_with_reference"].map(bucket), sort=True):
        out["by_n_voting_spectra"][b] = {"n": int(len(g)), "share_unanimous": sh(g["consensus_class"] == "UNANIMOUS"),
                                         "share_split": sh(g["consensus_class"] == "SPLIT"), "share_confident_conflict": sh(g["confident_conflict"])}
    for t, g in multi.groupby("matchability_tier", sort=True):
        out["by_matchability_tier"][t] = {"n": int(len(g)), "share_unanimous": sh(g["consensus_class"] == "UNANIMOUS"),
                                          "share_split": sh(g["consensus_class"] == "SPLIT")}
    for b in CONSENSUS_COSINE_LABELS:
        g = loo[loo["cosine_bin"] == b]
        out["loo_agreement_by_cosine_bin"][b] = {"n": int(len(g)), "share_loo_agrees": sh(g["loo_agrees"]),
                                                 "share_supported_by_any_other": sh(g["supported_by_any_other"])}
    if "modal_in_all_pools" in multi.columns:
        out["share_modal_in_all_pools_multi"] = sh(multi["modal_in_all_pools"])
        out["share_empty_pool_intersection_multi"] = sh(multi["pool_intersection_size"] == 0)
    nmr = pd.to_numeric(multi["neutral_mass_range_da"], errors="coerce")
    out["neutral_mass_range_da_P99_multi"] = float(nmr.quantile(0.99)) if nmr.notna().any() else float("nan")
    return out


# ---------------------------------------------------------------------------------------------
# runtime report text (values come from the run; nothing is hard-coded)
# ---------------------------------------------------------------------------------------------

def format_diagnostic_report(summary_raw, distances, status, proxies, interpretation, populations=(POP_TEST, POP_STRICT, POP_MIRROR)):
    def line(pop, label):
        r = summary_raw[summary_raw["population"] == pop]
        if not len(r):
            return [f"{label}:", "  (not computed)"]
        r = r.iloc[0]
        return [f"{label}:", f"  n = {int(r['n_queries'])}", f"  median max cosine = {r['max_cosine_P50']:.4f}", f"  P90 = {r['max_cosine_P90']:.4f}",
                f"  share >= 0.95 = {r['share_max_cosine_ge_0.95']:.4f}", f"  share >= 0.80 = {r['share_max_cosine_ge_0.80']:.4f}",
                f"  share < 0.50 = {r['share_max_cosine_lt_0.50']:.4f}"]
    labels = {POP_TEST: "TEST", POP_STRICT: "TL STRICT", POP_MIRROR: "TL MIRROR", POP_STANDARD: "TL STANDARD (sensitivity)"}
    out = ["TEST MATCH DIAGNOSTIC", ""]
    for p in populations:
        out += line(p, labels.get(p, p)) + [""]
    ks = distances.set_index("metric")
    out += ["DISTANCES (KS max cosine):", f"  TEST<->STRICT = {ks.loc['ks_max_cosine', 'test_vs_strict_distance']:.4f}",
            f"  TEST<->MIRROR = {ks.loc['ks_max_cosine', 'test_vs_mirror_distance']:.4f}", "",
            "VALIDATION RESEMBLANCE:", f"  {status}", "", "LIBRARY-MATCHABILITY PROXY (not a Class-1 fraction):"]
    out += [f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in proxies.items()]
    out += ["", "INTERPRETATION:", *[f"  {t}" for t in interpretation]]
    return "\n".join(out)
