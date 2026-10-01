"""Gate-A computations for notebook 13 (kept out of the notebook so cells stay orchestration-sized).

Whether the resulting numbers count as Class-2 evidence is decided ONLY by `casmi.validation.c2_protocol`
(`assess_c2_protocol` / `gate_a_decision`); these functions just compute retrieval facts.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.candidates.recall import formula_element_class, recall_sweep, source_label, truth_reachable
from casmi.candidates.universe import keys_to_ids, read_candidates
from casmi.validation.metrics import candidate_stage_metrics
from casmi.validation.regimes import hidden_sets, removed_candidate_mask


def attach_truth_provenance(queries, universe_root, candidate_keys):
    """Q + truth_id, reach_universe, reach_external, formula_class, truth_sources (provenance of the truth only)."""
    q = queries.copy()
    q["truth_id"] = keys_to_ids(candidate_keys, q["true_connectivity_key"])
    tr = read_candidates(universe_root, q.loc[q.truth_id >= 0, "truth_id"].unique(), columns=["candidate_sources", "molecular_formula"])
    tr = tr.set_index("candidate_id").reindex(q["truth_id"]).reset_index(drop=True)
    tr["candidate_id"] = q["truth_id"].to_numpy()
    tr["candidate_sources"] = tr["candidate_sources"].apply(lambda s: s if isinstance(s, (list, np.ndarray)) else [])
    q["reach_universe"] = truth_reachable(tr, "universe")
    q["reach_external"] = truth_reachable(tr, "external_only")
    q["formula_class"] = tr["molecular_formula"].map(formula_element_class).to_numpy()
    q["truth_sources"] = tr["candidate_sources"].apply(lambda s: source_label(s, drop_train=True)).to_numpy()
    return q


def reference_flags(universe_root, n_candidates):
    """Bool over candidate ids: candidate owns reference spectra (TRAIN with spectra). Shortcut DIAGNOSTIC only."""
    root = Path(universe_root)
    off = json.loads((root / "bucket_offsets.json").read_text(encoding="utf-8"))["offsets"]
    has_ref = np.zeros(n_candidates, dtype=bool)
    for b, start in off.items():
        t = pd.read_parquet(root / "buckets" / f"bucket={b}.parquet", columns=["has_reference_spectrum"])
        has_ref[start:start + len(t)] = t["has_reference_spectrum"].fillna(False).to_numpy(bool)
    return has_ref


def recall_sweeps(queries, regimes, index, candidate_keys, has_ref, ppm_grid, log=print):
    """Per fold view (that fold's C3 truths removed, its hidden truths lose their reference flag): C1 + C2 sweeps in
    `external_only` and `universe` modes, plus the C3 leakage check at the widest tolerance.
    Returns (long sweep table, {fold: #C3 truths retrieved})."""
    sweeps, c3_retrieved = [], {}
    for f in sorted(queries["fold"].unique()):
        hs = hidden_sets(regimes, f)
        excl = removed_candidate_mask(candidate_keys, hs["removed_structure_keys"])
        has_ref_f = has_ref.copy()
        hid = keys_to_ids(candidate_keys, sorted(hs["hidden_reference_keys"]))
        has_ref_f[hid[hid >= 0]] = False
        qf = queries[(queries["fold"] == f) & queries["regime"].isin(["C1", "C2"])]
        for mode, col in (("external_only", "reach_external"), ("universe", "reach_universe")):
            s = recall_sweep(index, qf["neutral_mass"].to_numpy(float), qf["truth_id"].to_numpy(), ppm_grid, reachable=qf[col].to_numpy(),
                             exclude_mask=excl, query_ids=qf["query_id"].to_numpy(), has_reference_mask=has_ref_f)
            sweeps.append(s.assign(mode=mode, fold=f))
        q3 = queries[(queries["fold"] == f) & (queries["regime"] == "C3")]
        c3_retrieved[int(f)] = 0
        if len(q3):
            s3 = recall_sweep(index, q3["neutral_mass"].to_numpy(float), q3["truth_id"].to_numpy(), [max(ppm_grid)], exclude_mask=excl,
                              query_ids=q3["query_id"].to_numpy())
            c3_retrieved[int(f)] = int(s3["truth_rank"].notna().sum())
        log(f"fold {f}: {len(qf):,} C1/C2 queries swept")
    keep = ["query_id", "regime", "neutral_mass", "ionization_mode", "adduct", "instrument_type", "formula_class", "truth_sources",
            "true_connectivity_key"]
    sweep = pd.concat(sweeps, ignore_index=True).merge(queries[[c for c in keep if c in queries.columns]], on="query_id", how="left")
    return sweep, c3_retrieved


def per_fold_recall(sweep, ppm, k_values):
    rows = []
    for (mode, regime, fold), g in sweep[sweep["ppm"] == ppm].groupby(["mode", "regime", "fold"]):
        rows.append({"mode": mode, "regime": regime, "fold": fold, **candidate_stage_metrics(g, k_values=k_values)})
    return pd.DataFrame(rows)


def mass_accuracy(sweep, ppm, gross_error_ppm=1000.0, top=15):
    """Truth |ppm| error by instrument / adduct / ion mode in `universe` mode (the truth is always in the pool there,
    so recall < 1 is mass error alone). Errors > `gross_error_ppm` are adduct / charge labelling errors."""
    u = sweep[(sweep["mode"] == "universe") & (sweep["ppm"] == ppm)].copy()
    u["instrument"] = u["instrument_type"].astype(str).str.strip().str.lower()
    u["outside_10ppm"] = u["truth_abs_ppm"] > 10
    u["label_error"] = u["truth_abs_ppm"] > gross_error_ppm
    out = {}
    for c in ("instrument", "adduct", "ionization_mode"):
        out[c] = (u.groupby(c).agg(n=("query_id", "size"), median_ppm=("truth_abs_ppm", "median"),
                                   p95_ppm=("truth_abs_ppm", lambda x: x.quantile(0.95)), outside_10ppm=("outside_10ppm", "mean"),
                                   label_errors=("label_error", "mean")).sort_values("n", ascending=False).head(top).round(4))
    return out
