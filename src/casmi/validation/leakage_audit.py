"""Leakage audits for notebooks 11 / 13 / 14 -- assertions that run when the USER executes the notebook.

Each audit returns a check table (check, passed, detail, kind) and raises `LeakageAuditError` listing every failed
ASSERTION. Rows of kind "limitation" document a protection that does NOT exist (they never pass silently as a check).

Covered:
  * fold / connectivity isolation (train truths of folds != f never evaluated in fold f; one fold per connectivity)
  * query spectrum never its own reference / analog
  * C1: other references of the truth remain; T1 identity duplicates excluded (T2: documented limitation)
  * C2: every reference of the truth hidden; truth stays in the pool; external-source requirement under a valid protocol
  * C3: truth structure removed from the pool (never retrieved)
  * no TRAIN-provenance / reference-availability shortcut features in rankers
  * no stale analog cache (feature namespaces' identity == current fold / hidden set / query set)
  * visible competition test set never used (queries are training spectra)
  * protocol-invalid evaluations never reported as C2 evidence
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class LeakageAuditError(AssertionError):
    pass


def _table(rows):
    t = pd.DataFrame(rows, columns=["check", "passed", "detail", "kind"])
    bad = t[(t["kind"] == "assert") & ~t["passed"].astype(bool)]
    if len(bad):
        raise LeakageAuditError("LEAKAGE AUDIT FAILED:\n" + bad[["check", "detail"]].to_string(index=False))
    return t


def _row(check, passed, detail="", kind="assert"):
    return {"check": check, "passed": bool(passed), "detail": str(detail), "kind": kind}


# ---------------------------------------------------------------------------------------------------------------
# notebook 11
# ---------------------------------------------------------------------------------------------------------------

def audit_regimes(regimes, train_spectrum_ids, fold_table, reference_connectivity, universe_keys_sorted, regime_cfg, c2_mode,
                  rederived=None, unified_truths=None):
    """Regime-table audit. `rederived`: a second build from the same inputs (determinism); `unified_truths`: universe
    rows (connectivity_key, candidate_sources) of the truths, required when c2_mode == external_universe."""
    from casmi.validation.c2_protocol import C2_MODE_EXTERNAL
    from casmi.validation.regimes import assert_regime_integrity
    rows = []
    not_train = ~regimes["query_id"].astype(str).isin(set(map(str, train_spectrum_ids)))
    rows.append(_row("queries are training spectra (visible test set unused)", not not_train.any(), f"{int(not_train.sum())} non-training queries"))
    per_fold = []
    for f in sorted(regimes["fold"].dropna().unique()):
        try:
            per_fold.append(assert_regime_integrity(regimes, fold_table, reference_connectivity, universe_keys_sorted, f, regime_cfg))
        except AssertionError as e:
            rows.append(_row(f"fold {f}: regime integrity (isolation, C1 references, C2/C3 hiding, C3 removal)", False, str(e)[:500]))
    if per_fold:
        integ = pd.concat(per_fold, ignore_index=True)
        for check, ok in integ.groupby("check")["passed"].all().items():
            rows.append(_row(f"all folds: {check}", ok))
    if rederived is not None:
        a = rederived.set_index("query_id")["regime"].reindex(regimes["query_id"]).fillna("NA").to_numpy()
        same = (len(rederived) == len(regimes)) and bool((a == regimes["regime"].fillna("NA").to_numpy()).all())
        rows.append(_row("regime assignment deterministic (rebuild == persisted)", same))
    if c2_mode == C2_MODE_EXTERNAL:
        c2 = regimes.loc[regimes["regime"] == "C2", "true_connectivity_key"].unique()
        src = unified_truths.set_index("connectivity_key")["candidate_sources"].reindex(c2) if unified_truths is not None else pd.Series(dtype=object)
        ext = src.map(lambda s: isinstance(s, (list, tuple, np.ndarray)) and len(set(s) - {"TRAIN"}) > 0)
        rows.append(_row("C2 truths exist in a non-TRAIN source", len(c2) > 0 and bool(ext.all()), f"{int((~ext).sum())} of {len(c2)} lack one"))
    else:
        rows.append(_row("C2 external-source requirement", True, f"NOT ENFORCED: c2_mode={c2_mode} (preliminary, shortcut risk)", "limitation"))
    return _table(rows)


# ---------------------------------------------------------------------------------------------------------------
# notebook 13
# ---------------------------------------------------------------------------------------------------------------

def audit_gate_a(sweep, c3_retrieved_per_fold, protocol, gate_record, truth_rows=None):
    """Gate-A audit. `c3_retrieved_per_fold`: {fold: #C3 truths found in their own fold view} (must be 0).
    `truth_rows`: per-query truth provenance (columns reach_external, truth_sources) for the source check."""
    rows = [_row("C3 truths never retrievable in their fold view", all(v == 0 for v in c3_retrieved_per_fold.values()),
                 json.dumps({str(k): int(v) for k, v in c3_retrieved_per_fold.items()}))]
    if truth_rows is not None and len(truth_rows):
        bad = truth_rows["reach_external"].astype(bool) & (truth_rows["truth_sources"].astype(str) == "none")
        rows.append(_row("external_only reachability uses non-TRAIN provenance only", not bad.any(), f"{int(bad.sum())} inconsistent truths"))
    invalid_with_metrics = (not protocol["protocol_valid"]) and gate_record.get("metrics") is not None
    rows.append(_row("protocol-invalid Gate A carries no C2 metrics", not invalid_with_metrics, gate_record.get("decision")))
    if "pool_share_with_reference" in sweep.columns:
        share = sweep.loc[sweep["regime"] == "C2", "pool_share_with_reference"].median()
        rows.append(_row("C2 pool share with reference spectra (shortcut diagnostic)", True, f"median {share}", "diagnostic"))
    return _table(rows)


# ---------------------------------------------------------------------------------------------------------------
# notebook 14
# ---------------------------------------------------------------------------------------------------------------

def audit_analog(queries, positives_per_query, disjointness, features, score_cols, feature_cols, runs, regimes, t2_policy):
    """Analog-baseline audit. `runs`: run records of `run_analog_fold` (with features_dir); `regimes`: the regime
    table used (hidden sets are recomputed to check every feature namespace's identity)."""
    from casmi.validation.c2_protocol import assert_no_train_provenance_features
    from casmi.validation.regimes import hidden_sets
    from casmi.workspace.cache_identity import fingerprint_values, read_identity
    rows = []
    c3 = queries.loc[queries["regime"] == "C3", "query_id"].map(positives_per_query).fillna(0)
    rows.append(_row("C3 truths are never candidates", bool((c3 == 0).all()), f"{int((c3 > 0).sum())} C3 queries with a positive"))
    rows.append(_row("at most one positive (connectivity) per query", int(positives_per_query.max() if len(positives_per_query) else 0) <= 1))
    rows.append(_row("ranker training / evaluation connectivity-disjoint per fold", bool((disjointness["n_overlap"] == 0).all()),
                     f"{int(disjointness['n_overlap'].sum())} overlapping connectivities"))
    for c in score_cols:
        rows.append(_row(f"{c}: every row has an out-of-fold score", bool(features[c].notna().all()), f"{int(features[c].isna().sum())} NaN"))
    try:
        review = assert_no_train_provenance_features(feature_cols)
        rows.append(_row("no TRAIN-provenance / reference-availability ranking features", True, f"needs ablation before use: {review}"))
    except AssertionError as e:
        rows.append(_row("no TRAIN-provenance / reference-availability ranking features", False, str(e)))
    rows.append(_row("query never its own analog (asserted per fold in run_analog_fold)", True, "self matches raise in casmi.analog.pipeline"))
    rows.append(_row("C1 T1 identity duplicates excluded / T2 duplicates", True, t2_policy, "limitation"))
    for rec in runs:
        ident = read_identity(rec["features_dir"]) or {}
        f = rec["fold"]
        hs = hidden_sets(regimes, f)
        q = queries[queries["fold"] == f]
        ok = (ident.get("hidden_reference_keys") == fingerprint_values(hs["hidden_reference_keys"])
              and ident.get("removed_structure_keys") == fingerprint_values(hs["removed_structure_keys"])
              and ident.get("query_ids") == fingerprint_values(q["query_id"].astype(str)))
        rows.append(_row(f"fold {f}: feature cache identity matches current regimes / queries (no stale cache)", ok, rec["features_dir"]))
    loaded_dirs = {str(Path(r["features_dir"])) for r in runs}
    rows.append(_row("features loaded only from this run's identity namespaces", len(loaded_dirs) == len(runs), sorted(loaded_dirs)))
    return _table(rows)
