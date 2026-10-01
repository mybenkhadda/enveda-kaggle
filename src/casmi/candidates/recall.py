"""Phase 4 -- candidate recall (Gate A): is the truth RETRIEVABLE before any ranker is trained?

Two reachability modes, because they answer different questions:

    universe        truth counts as present if it is anywhere in the universe (TRAIN included). For dev
                    queries drawn from training spectra this is ~100% by construction: it measures only
                    precursor/adduct mass accuracy and pool density.
    external_only   truth counts as present only if a NON-TRAIN source (COCONUT, PubChem, ...) contains
                    its connectivity -- what a genuinely hidden Class-2 molecule experiences, since its
                    structure cannot come from the training library. THIS is the coverage number for Gate A.

Candidate order for Recall@k = the deterministic mass order of retrieval (abs ppm ASC, candidate_id ASC),
exactly `CandidateMassIndex.search_ppm`. Per-query output feeds
`casmi.validation.metrics.candidate_stage_metrics` and `casmi.validation.reporting`.
"""
import numpy as np
import pandas as pd

from casmi.validation.metrics import candidate_stage_metrics

REACHABILITY_MODES = ("universe", "external_only")


def formula_element_class(formula):
    """Coarse truth-formula class for breakdowns: CHO / CHNO / CHNOS / CHNOP / ... + '+X' for halogens."""
    from casmi.candidates.filters import formula_elements
    els = formula_elements(formula)
    if not els:
        return "unknown"
    core = "".join(e for e in ("C", "H", "N", "O", "S", "P") if e in els)
    hal = "+X" if els & {"F", "Cl", "Br", "I"} else ""
    other = "+other" if els - {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"} else ""
    return core + hal + other


def source_label(sources, drop_train=False):
    s = sorted(set(sources or ()) - ({"TRAIN"} if drop_train else set()))
    return "+".join(s) if s else "none"


def truth_reachable(truth_rows, mode="external_only"):
    """Boolean per truth row (v2 universe rows of the truths): `universe` -> present at all;
    `external_only` -> present in at least one non-TRAIN source."""
    if mode not in REACHABILITY_MODES:
        raise ValueError(f"mode must be one of {REACHABILITY_MODES}")
    present = truth_rows["candidate_id"].to_numpy() >= 0
    if mode == "universe":
        return present
    ext = truth_rows["candidate_sources"].map(lambda s: len(set(s or ()) - {"TRAIN"}) > 0 if isinstance(s, (list, tuple, np.ndarray)) else False)
    return present & ext.to_numpy(bool)


def recall_sweep(index, neutral_masses, truth_ids, ppm_grid, reachable=None, exclude_mask=None, query_ids=None, chunk=5000,
                 has_reference_mask=None):
    """Long per-query table: one row per (query, ppm) with truth_abs_ppm, truth_rank (mass order; NaN when
    outside the window, excluded or not reachable), pool_size (exact unique candidates) and, if
    `has_reference_mask` (bool over candidate ids) is given, `pool_share_with_reference`."""
    m = np.asarray(neutral_masses, dtype=float)
    t = np.asarray(truth_ids, dtype=np.int64)
    reach = np.ones(len(m), bool) if reachable is None else np.asarray(reachable, bool)
    qid = np.arange(len(m)) if query_ids is None else np.asarray(query_ids)
    t_eff = np.where(reach, t, -1)
    tppm = index.truth_abs_ppm(m, t_eff)
    rows = []
    for ppm in ppm_grid:
        for s in range(0, len(m), chunk):
            sl = slice(s, s + chunk)
            offsets, ids, _ = index.search_ppm_batch(m[sl], ppm, exclude_mask=exclude_mask)
            sizes = np.diff(offsets)
            rank = np.full(len(sizes), np.nan)
            tt = t_eff[sl]
            for i in range(len(sizes)):
                if tt[i] < 0:
                    continue
                hit = np.flatnonzero(ids[offsets[i]:offsets[i + 1]] == tt[i])
                if len(hit):
                    rank[i] = hit[0] + 1
            part = pd.DataFrame({"query_id": qid[sl], "ppm": ppm, "truth_abs_ppm": tppm[sl], "truth_rank": rank, "pool_size": sizes,
                                 "truth_reachable": reach[sl]})
            if has_reference_mask is not None:
                share = np.full(len(sizes), np.nan)
                nz = sizes > 0
                if nz.any() and len(ids):
                    sums = np.add.reduceat(np.asarray(has_reference_mask, dtype=float)[ids], offsets[:-1][nz])
                    share[nz] = sums / sizes[nz]
                part["pool_share_with_reference"] = share
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def recall_by_ppm(sweep, k_values=(25, 100, 500), group_cols=()):
    """`candidate_stage_metrics` per ppm (and optional extra group columns)."""
    out = []
    for keys, g in sweep.groupby(["ppm", *group_cols], sort=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        out.append({**dict(zip(["ppm", *group_cols], keys)), **candidate_stage_metrics(g, k_values=k_values)})
    return pd.DataFrame(out)


def gate_a_assessment(summary_external, ppm_ref=10, coverage_floor=0.80, density_gap=0.25):
    """HEURISTIC reading of the external_only recall table at `ppm_ref` (thresholds are configurable and
    NOT validated -- they only phrase the decision the user makes after looking at the numbers):

        recall_all < coverage_floor                       -> bottleneck: CANDIDATE COVERAGE (fix the universe)
        recall_all - recall_at_25 > density_gap           -> bottleneck: RANKING (pools too dense for mass alone)
        otherwise                                         -> both acceptable at this tolerance
    """
    row = summary_external[summary_external["ppm"] == ppm_ref]
    if row.empty:
        return {"decision": "UNDETERMINED", "reason": f"no row for ppm={ppm_ref}"}
    r = row.iloc[0]
    if r["recall_all"] < coverage_floor:
        d = "CANDIDATE_COVERAGE"
    elif r["recall_all"] - r["recall_at_25"] > density_gap:
        d = "RANKING"
    else:
        d = "BOTH_ACCEPTABLE"
    return {"decision": d, "ppm": ppm_ref, "recall_all": float(r["recall_all"]), "recall_at_25": float(r["recall_at_25"]),
            "coverage_floor": coverage_floor, "density_gap": density_gap, "label": "HEURISTIC -- thresholds not validated"}
