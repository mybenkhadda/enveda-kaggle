"""v5 query manifests: the fixed test-like evaluation set (TL_EVAL), the molecule-level
aggregation development set (MOL_DEV), the nested test-like training sets (TL_1K ⊂ TL_3K ⊂ TL_10K)
and the random-composition size control (RND_1K).

Construction order (each later set is connectivity-disjoint from every earlier one):

    TL_EVAL   <- universe minus HOST source, HOST structures, DEV-1k structures, and every spectrum
                 the frozen v4b mass-likelihood model was fitted on (so frozen B5 scores TL_EVAL
                 uncontaminated); timsTOF-type instrument, observed test adduct, test neutral-mass
                 range, supported adduct, non-empty candidate window; one spectrum per connectivity
    MOL_DEV   <- minus TL_EVAL structures; test-like connectivities with >= 3 eligible spectra
    TL_*      <- minus TL_EVAL + MOL_DEV structures; test-like importance order, nested prefixes
    RND_1K    <- same pool as TL_*, uniform order

Only OBSERVABLE, label-free test metadata is used (instrument, adduct, polarity, neutral mass).
HOST is never used to choose anything here -- its source and structures are only EXCLUDED.
All randomness: `numpy.random.default_rng(seed)` over a canonical (query_id-sorted) order, via
Efraimidis-Spirakis weighted keys, so a whole weighted permutation is drawn once and every nested
set is a prefix of it.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.chemistry.adducts import neutral_mass_from_precursor, parse_adduct
from casmi.ranking.training_sets import test_like_weights

MANIFEST_COLUMNS = ["query_id", "connectivity_key", "source", "instrument", "adduct", "ionization_mode", "neutral_mass",
                    "precursor_mz", "fold", "selection_reason", "selection_order"]
TRAINING_MANIFESTS = ("TL_1K", "TL_3K", "TL_10K", "RND_1K")
ALL_MANIFESTS = ("TL_EVAL", "MOL_DEV") + TRAINING_MANIFESTS


# ---------------------------------------------------------------------------------------------
# universe + observable test profile
# ---------------------------------------------------------------------------------------------

def build_query_universe(train_meta, connectivity_folds):
    """One row per train spectrum with a reconstructable neutral mass and a connectivity fold."""
    tm = train_meta
    adducts = pd.Series(tm["adduct"].astype(str).unique())
    supported = {a: parse_adduct(a)["supported"] for a in adducts}
    u = pd.DataFrame({
        "query_id": tm["train_spectrum_id"].astype(str).to_numpy(), "connectivity_key": tm["connectivity_key"].to_numpy(),
        "source": tm["ingest_lib"].astype(str).to_numpy(), "instrument": tm["instrument_type"].astype(str).to_numpy(),
        "adduct": tm["adduct"].astype(str).to_numpy(), "ionization_mode": tm["ionization_mode"].astype(str).to_numpy(),
        "precursor_mz": tm["precursor_mz"].to_numpy(float), "exact_mass": tm["exact_mass"].to_numpy(float),
    })
    u["adduct_supported"] = u["adduct"].map(supported).fillna(False).astype(bool)
    nm = np.full(len(u), np.nan)
    ok = u["adduct_supported"].to_numpy() & np.isfinite(u["precursor_mz"].to_numpy())
    nm[ok] = [neutral_mass_from_precursor(p, a) for p, a in zip(u.loc[ok, "precursor_mz"], u.loc[ok, "adduct"])]
    u["neutral_mass"] = nm
    u["truth_abs_ppm"] = np.abs(u["exact_mass"] - u["neutral_mass"]) / u["neutral_mass"] * 1e6
    u = u.merge(connectivity_folds[["connectivity_key", "fold"]], on="connectivity_key", how="left")
    u = u[u["connectivity_key"].notna()].sort_values("query_id", kind="mergesort").reset_index(drop=True)
    return u


def test_observable_profile(test_meta):
    """Label-free test-side profile: instrument names, observed adducts, polarity shares, neutral
    mass range (from supported adducts only)."""
    t = test_meta.copy()
    t["neutral_mass"] = [neutral_mass_from_precursor(p, a) if parse_adduct(a)["supported"] else np.nan
                         for p, a in zip(t["precursor_mz"], t["adduct"].astype(str))]
    nm = t["neutral_mass"].dropna()
    return {
        "instruments": sorted(t["instrument_type"].astype(str).unique().tolist()),
        "adducts": sorted(t["adduct"].astype(str).unique().tolist()),
        "adducts_supported": {a: bool(parse_adduct(a)["supported"]) for a in sorted(t["adduct"].astype(str).unique())},
        "polarity_share": t["ionization_mode"].astype(str).value_counts(normalize=True).to_dict(),
        "neutral_mass_min": float(nm.min()), "neutral_mass_max": float(nm.max()),
        "n_test_spectra": int(len(t)), "n_test_molecules": int(t["molecule_id"].nunique()) if "molecule_id" in t else None,
    }


def test_like_frame(test_meta):
    """Test metadata in the column names `test_like_weights` expects."""
    t = test_meta.copy()
    t["neutral_mass"] = [neutral_mass_from_precursor(p, a) if parse_adduct(a)["supported"] else np.nan
                         for p, a in zip(t["precursor_mz"], t["adduct"].astype(str))]
    return t[["instrument_type", "adduct", "ionization_mode", "neutral_mass"]].astype(
        {"instrument_type": str, "adduct": str, "ionization_mode": str})


def candidate_counts(universe, mass_index, tolerance_ppm):
    """Candidate-window size per query (mass-variant rows, before connectivity dedupe) -- a cheap
    searchsorted check that retrieval is non-empty."""
    nm = universe["neutral_mass"].to_numpy(float)
    ok = np.isfinite(nm)
    tol = np.abs(nm[ok]) * tolerance_ppm * 1e-6  # identical window arithmetic to MassIndex._window
    out = np.zeros(len(nm), dtype=int)
    out[ok] = (np.searchsorted(mass_index.masses, nm[ok] + tol, side="right")
               - np.searchsorted(mass_index.masses, nm[ok] - tol, side="left"))
    return out


# ---------------------------------------------------------------------------------------------
# deterministic weighted ordering
# ---------------------------------------------------------------------------------------------

def weighted_order(query_ids, weights, seed):
    """Efraimidis-Spirakis: key = log(u) / w (w > 0), sort DESC. Returns positions of the w>0
    items in draw order. Deterministic for a given (query_ids order, weights, seed); the caller
    passes query_id-sorted inputs."""
    w = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    u = rng.random(len(w))
    keys = np.full(len(w), -np.inf)
    pos = w > 0
    keys[pos] = np.log(u[pos]) / w[pos]
    if len(query_ids) != len(w):
        raise ValueError("query_ids and weights must be aligned")
    order = np.lexsort((np.arange(len(w)), -keys))  # keys DESC; position (== query_id order) breaks exact ties
    return order[pos[order]]


def _take_one_per_connectivity(df, order, n):
    seen, picked = set(), []
    keys = df["connectivity_key"].to_numpy()
    for i in order:
        k = keys[i]
        if k in seen:
            continue
        seen.add(k)
        picked.append(i)
        if len(picked) >= n:
            break
    return picked


# ---------------------------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------------------------

def tl_eval_eligibility(universe, profile, host_source, host_keys, excluded_keys=(), excluded_query_ids=(),
                        min_candidates=1, candidate_count_col="n_candidates_window"):
    """Adds one boolean column per criterion plus `eligible` and a human-readable `failed_criteria`."""
    u = universe.copy()
    crit = {
        "timsTOF_like_instrument": u["instrument"].isin(profile["instruments"]),
        "test_adduct": u["adduct"].isin(profile["adducts"]),
        "in_test_mass_range": u["neutral_mass"].between(profile["neutral_mass_min"], profile["neutral_mass_max"]),
        "neutral_mass_reconstructed": u["neutral_mass"].notna(),
        "candidate_window_nonempty": u[candidate_count_col] >= min_candidates,
        "not_host_source": u["source"] != host_source,
        "not_host_structure": ~u["connectivity_key"].isin(set(host_keys)),
        "not_excluded_structure": ~u["connectivity_key"].isin(set(excluded_keys)),
        "not_excluded_query": ~u["query_id"].isin(set(excluded_query_ids)),
        "has_fold": u["fold"].notna(),
    }
    for k, v in crit.items():
        u[f"crit_{k}"] = v.to_numpy(bool)
    u["eligible"] = np.logical_and.reduce([u[f"crit_{k}"].to_numpy() for k in crit])
    return u, list(crit)


def select_tl_eval(eligible_universe, test_frame, n=1000, seed=42):
    """~n queries, one per connectivity, drawn in test-like weighted order from the eligible set
    (weights rebalance adduct/polarity/mass toward the test marginals). If fewer eligible
    connectivities exist, all are used and the actual count is recorded by the caller."""
    e = eligible_universe[eligible_universe["eligible"]].sort_values("query_id", kind="mergesort").reset_index(drop=True)
    w, _ = test_like_weights(e.rename(columns={"instrument": "instrument_type"}), test_frame)
    order = weighted_order(e["query_id"], w, seed)
    picked = _take_one_per_connectivity(e, order, n)
    out = e.iloc[picked].copy()
    out["selection_order"] = np.arange(1, len(out) + 1)
    out["selection_reason"] = "TL_EVAL: eligible (timsTOF-like, test adduct, test mass range, retrievable, non-HOST, non-DEV1k, not in v4b mass-fit set); test-like weighted draw; 1 spectrum/connectivity"
    return out


def select_mol_dev(pool, test_frame, n_connectivities=400, min_spectra=3, max_spectra_per_connectivity=10, seed=42):
    """Molecule-level development set: connectivities with >= `min_spectra` eligible test-like
    spectra, drawn in test-like weighted order (connectivity weight = mean spectrum weight); up to
    `max_spectra_per_connectivity` spectra kept per connectivity (lowest query_id first)."""
    p = pool[pool["eligible"]].sort_values("query_id", kind="mergesort").reset_index(drop=True)
    counts = p.groupby("connectivity_key").size()
    keep_keys = counts[counts >= min_spectra].index
    p = p[p["connectivity_key"].isin(keep_keys)].reset_index(drop=True)
    w, _ = test_like_weights(p.rename(columns={"instrument": "instrument_type"}), test_frame)
    conn = pd.DataFrame({"connectivity_key": p["connectivity_key"], "w": w}).groupby("connectivity_key", sort=True)["w"].mean().reset_index()
    order = weighted_order(conn["connectivity_key"], conn["w"].to_numpy(), seed)
    chosen = conn.iloc[order[:n_connectivities]]["connectivity_key"].tolist()
    rank_of = {k: i + 1 for i, k in enumerate(chosen)}
    out = p[p["connectivity_key"].isin(set(chosen))].copy()
    out = out.groupby("connectivity_key", group_keys=False).head(max_spectra_per_connectivity)
    out["selection_order"] = out["connectivity_key"].map(rank_of)
    out["selection_reason"] = f"MOL_DEV: test-like connectivity with >={min_spectra} eligible spectra; <= {max_spectra_per_connectivity} spectra kept"
    return out.sort_values(["selection_order", "query_id"]).reset_index(drop=True)


def training_pool(universe, host_source, host_keys, excluded_keys, min_candidates=1, candidate_count_col="n_candidates_window"):
    """Valid training queries: reconstructable neutral mass, non-empty candidate window, a fold,
    non-HOST source, and a connectivity outside HOST / TL_EVAL / MOL_DEV."""
    u = universe
    mask = (u["neutral_mass"].notna() & (u[candidate_count_col] >= min_candidates) & u["fold"].notna()
            & (u["source"] != host_source) & ~u["connectivity_key"].isin(set(host_keys)) & ~u["connectivity_key"].isin(set(excluded_keys)))
    return u[mask].sort_values("query_id", kind="mergesort").reset_index(drop=True)


def select_training_manifests(pool, test_frame, sizes=(("TL_1K", 1000), ("TL_3K", 3000), ("TL_10K", 10000)),
                              rnd_name="RND_1K", rnd_size=1000, seed=42):
    """Nested TL sets = prefixes of ONE test-like weighted permutation (queries with test weight 0 --
    non-test adducts -- are never drawn); RND_1K = prefix of a uniform permutation of the same pool.
    A set larger than the eligible count becomes 'all eligible' and records its actual size."""
    w, wreport = test_like_weights(pool.rename(columns={"instrument": "instrument_type"}), test_frame)
    tl_order = weighted_order(pool["query_id"], w, seed)
    out, counts = {}, {}
    for name, size in sizes:
        idx = tl_order[:size]
        m = pool.iloc[idx].copy()
        m["selection_order"] = np.arange(1, len(m) + 1)
        m["test_like_weight"] = w[idx]
        m["selection_reason"] = (f"{name}: test-like weighted draw (seed {seed}), prefix {len(m)} of the TL permutation"
                                 + ("" if len(m) == size else f" -- ALL {len(m)} eligible (requested {size})"))
        out[name], counts[name] = m, {"requested": size, "actual": len(m)}
    rnd_order = weighted_order(pool["query_id"], np.ones(len(pool)), seed + 1)
    idx = rnd_order[:rnd_size]
    r = pool.iloc[idx].copy()
    r["selection_order"] = np.arange(1, len(r) + 1)
    r["test_like_weight"] = w[idx]
    r["selection_reason"] = f"{rnd_name}: uniform draw (seed {seed + 1}) from the same valid pool as TL_* (composition control)"
    out[rnd_name], counts[rnd_name] = r, {"requested": rnd_size, "actual": len(r)}
    return out, counts, wreport


# ---------------------------------------------------------------------------------------------
# fingerprints, persistence, validation
# ---------------------------------------------------------------------------------------------

def manifest_fingerprint(manifest):
    """sha256 over the sorted (query_id, connectivity_key, fold) triples -- content, not file bytes."""
    m = manifest[["query_id", "connectivity_key", "fold"]].astype(str).sort_values("query_id")
    payload = "\n".join("|".join(r) for r in m.itertuples(index=False, name=None))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_manifest(manifest, name, out_dir, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in MANIFEST_COLUMNS if c in manifest.columns] + [c for c in manifest.columns if c not in MANIFEST_COLUMNS]
    manifest[cols].to_parquet(out_dir / f"{name}.parquet", index=False)
    meta = {"name": name, "n_queries": int(len(manifest)), "n_connectivities": int(manifest["connectivity_key"].nunique()),
            "manifest_sha256": manifest_fingerprint(manifest), "written_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def load_manifest(name, out_dir, verify=True):
    out_dir = Path(out_dir)
    m = pd.read_parquet(out_dir / f"{name}.parquet")
    meta = json.loads((out_dir / f"{name}.meta.json").read_text(encoding="utf-8"))
    if verify and manifest_fingerprint(m) != meta["manifest_sha256"]:
        raise RuntimeError(f"manifest {name} content does not match its recorded sha256")
    return m, meta


def manifest_checks(manifests, host_keys, host_source):
    """Every structural guarantee as a list of `{check, passed, detail}` rows (the caller asserts)."""
    keys = {n: set(m["connectivity_key"]) for n, m in manifests.items()}
    ids = {n: set(m["query_id"]) for n, m in manifests.items()}
    host_keys = set(host_keys)
    rows = []

    def add(check, passed, detail=""):
        rows.append({"check": check, "passed": bool(passed), "detail": detail})

    for a, b in (("TL_1K", "TL_3K"), ("TL_3K", "TL_10K")):
        if a in ids and b in ids:
            add(f"{a} ⊂ {b}", ids[a] <= ids[b], f"{len(ids[a] - ids[b])} of {a} missing from {b}")
    for n in TRAINING_MANIFESTS:
        if n not in keys:
            continue
        for other in ("TL_EVAL", "MOL_DEV"):
            if other in keys:
                add(f"{n} ∩ {other} connectivity = ∅", not (keys[n] & keys[other]), f"{len(keys[n] & keys[other])} shared")
        add(f"{n} ∩ HOST connectivity = ∅", not (keys[n] & host_keys), f"{len(keys[n] & host_keys)} shared")
        add(f"{n} excludes HOST source", host_source not in set(manifests[n]["source"]))
    for n in ("TL_EVAL", "MOL_DEV"):
        if n in keys:
            add(f"{n} ∩ HOST connectivity = ∅", not (keys[n] & host_keys), f"{len(keys[n] & host_keys)} shared")
            add(f"{n} excludes HOST source", host_source not in set(manifests[n]["source"]))
    if "TL_EVAL" in keys and "MOL_DEV" in keys:
        add("TL_EVAL ∩ MOL_DEV connectivity = ∅", not (keys["TL_EVAL"] & keys["MOL_DEV"]))
    for n, m in manifests.items():
        add(f"{n} has no duplicate query_id", not m["query_id"].duplicated().any())
    return pd.DataFrame(rows)
