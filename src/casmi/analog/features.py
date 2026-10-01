"""Candidate-level analog feature table (v2 Phase 5).

One row per (query spectrum, candidate connectivity):
    query_id, fold, regime, candidate_id, connectivity_key, abs_mass_error_ppm, is_true_candidate,
    + ANALOG_FEATURES (fixed order; `feature_schema()` records it with a hash).

Candidates come from `CandidateMassIndex.search_ppm_batch` on the FOLD VIEW (C3 truths excluded via
`exclude_mask`); analog neighbors come from `casmi.analog.retrieval.build_neighbors` with the fold's hidden
references removed. Labels are connectivity-level (`connectivity_key == true_connectivity_key`).
Reference-availability shortcuts (`casmi.candidates.provenance.FORBIDDEN_SHORTCUT_FEATURES`) are never
features here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.analog.propagation import CANDIDATE_LEVEL, QUERY_LEVEL, propagate
from casmi.candidates.provenance import assert_no_shortcut_features

FEATURE_SCHEMA_VERSION = "casmi-v2-analog-features-1"
MASS_FEATURES = ["abs_mass_error_ppm"]
ANALOG_FEATURES = [f"analog_{c}" for c in CANDIDATE_LEVEL] + [f"analog_{c}" for c in QUERY_LEVEL]
ID_COLUMNS = ["query_id", "fold", "regime", "candidate_id", "connectivity_key", "is_true_candidate"]
assert_no_shortcut_features(MASS_FEATURES + ANALOG_FEATURES)


def feature_schema(features=None):
    feats = list(features or (MASS_FEATURES + ANALOG_FEATURES))
    return {"schema_version": FEATURE_SCHEMA_VERSION, "id_columns": ID_COLUMNS, "features": feats,
            "feature_hash": hashlib.sha256("\n".join(feats).encode("utf-8")).hexdigest()[:16]}


def candidate_pools(index, neutral_masses, ppm, exclude_mask=None, max_per_query=None):
    """CSR pools + a truncation log (never silent): `(offsets, ids, abs_ppm, n_truncated_queries)`."""
    offsets, ids, ap = index.search_ppm_batch(neutral_masses, ppm, exclude_mask=exclude_mask)
    if not max_per_query:
        return offsets, ids, ap, 0
    sizes = np.diff(offsets)
    pos = np.arange(len(ids)) - np.repeat(offsets[:-1], sizes)        # position inside each query's (ppm-ordered) pool
    keep = pos < max_per_query
    new_sizes = np.minimum(sizes, max_per_query)
    return np.concatenate([[0], np.cumsum(new_sizes)]), ids[keep], ap[keep], int((sizes > max_per_query).sum())


def build_feature_shard(queries, neighbors, offsets, cand_ids, cand_ppm, cand_rows, analog_rows, fp_get, cfg):
    """Features for one chunk of queries.

    queries:      DataFrame (query_id, fold, regime, true_connectivity_key, neutral_mass) -- same order as the pools
    neighbors:    neighbor table for these queries (`retrieval.NEIGHBOR_COLUMNS`)
    cand_rows:    universe rows indexed by candidate_id (connectivity_key, molecular_formula, representative_smiles)
    analog_rows:  universe rows indexed by connectivity_key for analog molecules (molecular_formula, exact_mass,
                  representative_smiles)
    fp_get:       callable(keys, smiles_by_key) -> (bits, valid)   (a `FingerprintCache.get` wrapper)
    cfg:          dict with support_tanimoto_threshold, similarity_weight_power, top_k_analogs"""
    tau, p, k = cfg["support_tanimoto_threshold"], cfg["similarity_weight_power"], cfg["top_k_analogs"]
    nb = {q: g.sort_values("analog_rank") for q, g in neighbors.groupby("query_id", sort=False)}
    smiles = {**cand_rows.set_index("connectivity_key")["representative_smiles"].to_dict(),
              **analog_rows["representative_smiles"].to_dict()}
    out = []
    for i, q in enumerate(queries.itertuples(index=False)):
        ids = cand_ids[offsets[i]:offsets[i + 1]]
        if not len(ids):
            continue
        cr = cand_rows.loc[ids]
        ckeys = cr["connectivity_key"].astype(str).to_numpy()
        cbits, cvalid = fp_get(ckeys, smiles)
        g = nb.get(q.query_id)
        if g is not None and len(g):
            ak = g["analog_connectivity_key"].astype(str).to_numpy()
            ar = analog_rows.reindex(ak)
            abits, avalid = fp_get(ak, smiles)
            feats = propagate(cbits, cvalid, ckeys, cr["molecular_formula"].to_numpy(), abits, avalid & ar["representative_smiles"].notna().to_numpy(),
                              g["modified_cosine"].to_numpy(), g["analog_rank"].to_numpy(), ak, ar["molecular_formula"].to_numpy(),
                              ar["exact_mass"].to_numpy(float), g["precursor_delta"].to_numpy(float), g["same_adduct"].to_numpy(bool),
                              float(q.neutral_mass), tau, p, k)
        else:
            feats = propagate(cbits, cvalid, ckeys, cr["molecular_formula"].to_numpy(), np.zeros((0, cbits.shape[1]), np.uint8),
                              np.zeros(0, bool), [], [], [], [], [], [], [], float(q.neutral_mass), tau, p, k)
        d = pd.DataFrame({"query_id": q.query_id, "fold": q.fold, "regime": q.regime, "candidate_id": ids, "connectivity_key": ckeys,
                          "is_true_candidate": ckeys == str(q.true_connectivity_key),
                          "abs_mass_error_ppm": cand_ppm[offsets[i]:offsets[i + 1]].astype(np.float32), **feats})
        out.append(d)
    cols = ID_COLUMNS + MASS_FEATURES + ANALOG_FEATURES
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def write_schema(out_dir, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = {**feature_schema(), **(extra or {})}
    (out_dir / "feature_schema.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s
