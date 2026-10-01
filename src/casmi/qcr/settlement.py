"""v4b cache settlement: are the persisted QCR/similarity-cache values materially different from a
FRESH recomputation under the current code?

Historical context (v4a.1, HISTORICAL REFERENCE): 22 / 1,000 sampled (query, reference) triples
differed from a fresh recomputation at a very strict tolerance (atol=1e-12). This module answers
the one remaining evidence-layer question -- are those differences material? -- with a single,
closed decision rule (`decide_cache`), and nothing more:

    ACCEPT_EXISTING   no tier change, no threshold crossing, every similarity |diff| <= 1e-9, and
                      the candidate-level aggregated features of every affected candidate are
                      unchanged within 1e-9.
    REBUILD_ONCE      anything else. The caller then rebuilds the evidence layer ONCE from an
                      empty cache with the current (deterministic top-N) code, re-runs the SAME
                      audit, and records `REBUILT_AND_VALIDATED` only if it now accepts.

Which thresholds matter is read from the similarity config the evidence was built with (the same
keys `casmi.qcr.builder.compute_reference_evidence` passes to
`casmi.spectra.deduplication.classify_identity_tier`), never re-typed here:

    T1  exact `peak_hash` equality (identity representation, no top-N truncation)
    T2  precursor diff <= tolerant_mirror_precursor_diff_da, matched fraction (both sides)
        >= tolerant_mirror_match_fraction, sqrt-intensity correlation >= tolerant_mirror_min_correlation
    T3  precursor diff < near_dup_precursor_diff_da AND identity binned cosine > near_dup_cosine_threshold

T1-T3 are computed on the UNTRUNCATED identity peaks, so the top-100 truncation rule can only move
the four similarity VALUES, never the tier. Tier changes are compared directly (cached tier vs
fresh tier). The similarity `cosine` is additionally checked for crossing
`near_dup_cosine_threshold` (0.95) because the spec asks for it explicitly -- that value is not
used by any eligibility rule, but a crossing would still mean the difference is not cosmetic.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.io.parquet import load_rows_by_offset
from casmi.ranking.features import compute_single_pair_scores
from casmi.spectra.binning import bin_spectrum
from casmi.spectra.deduplication import _match_peaks_greedy, classify_identity_tier
from casmi.spectra.preprocessing import (
    remove_invalid_peaks, top_n_boundary_diagnostics, truncate_top_peaks, truncate_top_peaks_legacy,
)
from casmi.spectra.reference_selection import MIRROR_TIERS, PROTOCOLS
from casmi.spectra.similarity import binned_cosine_similarity

SIMILARITY_METRICS = ("cosine", "modified_cosine", "peak_overlap_frac", "neutral_loss_cosine")
AGGREGATED_FEATURE_COLS = (
    "cosine_max", "cosine_top3_mean", "modified_cosine_max", "modified_cosine_top3_mean",
    "peak_overlap_frac_max", "peak_overlap_frac_top3_mean", "neutral_loss_cosine_max", "neutral_loss_cosine_top3_mean",
)
DECISION_ACCEPT = "ACCEPT_EXISTING"
DECISION_REBUILD = "REBUILD_ONCE"
DECISION_REBUILT_OK = "REBUILT_AND_VALIDATED"
DECISION_REBUILD_FAILED = "REBUILD_FAILED_VALIDATION"
ALLOWED_DOWNSTREAM_DECISIONS = (DECISION_ACCEPT, DECISION_REBUILT_OK)

DECISION_ATOL = 1e-9          # materiality tolerance of the decision rule (spec section 8)
STRICT_REPORT_ATOL = 1e-12    # the historical v4a.1 C3 tolerance -- used only to COUNT value mismatches
NEAR_THRESHOLD_MARGIN = 1e-6  # "fragile" identity-side margin, reported, not part of the decision

NEAR_DUP_TIERS = ("T1", "T2", "T3")  # excluded by near_dup_strict (MIRROR_TIERS + T3)


class CacheSettlementError(RuntimeError):
    """Raised when a downstream notebook asks for settled evidence that is missing, undecided,
    not accepted, or whose files changed after the settlement was written."""


# ---------------------------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------------------------

def identity_thresholds(similarity_config):
    """The exact tier thresholds in effect, read from the evidence's own similarity config."""
    return {
        "t2_precursor_diff_da": float(similarity_config["tolerant_mirror_precursor_diff_da"]),
        "t2_min_rel_intensity": float(similarity_config["tolerant_mirror_min_rel_intensity"]),
        "t2_ppm_tol": float(similarity_config["tolerant_mirror_ppm_tol"]),
        "t2_abs_tol_da": float(similarity_config["tolerant_mirror_abs_tol_da"]),
        "t2_match_fraction": float(similarity_config["tolerant_mirror_match_fraction"]),
        "t2_min_correlation": float(similarity_config["tolerant_mirror_min_correlation"]),
        "t3_precursor_diff_da": float(similarity_config["near_dup_precursor_diff_da"]),
        "t3_cosine": float(similarity_config["near_dup_cosine_threshold"]),
        "bin_width_da": float(similarity_config["bin_width_da"]),
    }


def similarity_value_thresholds(similarity_config):
    """Thresholds checked on the persisted similarity VALUES: `{metric: (threshold, ...)}`. The
    spec's cosine=0.95 is the config's `near_dup_cosine_threshold`, not a re-typed constant."""
    return {"cosine": (float(similarity_config["near_dup_cosine_threshold"]),)}


def crosses_threshold(a, b, threshold):
    """True iff `a` and `b` fall on different sides of `threshold` under EITHER convention
    (`> t` or `>= t`) -- so a value that lands exactly on the threshold on one side counts."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return bool(np.isfinite(a) != np.isfinite(b))
    return bool(((a > threshold) != (b > threshold)) or ((a >= threshold) != (b >= threshold)))


# ---------------------------------------------------------------------------------------------
# C3 sample
# ---------------------------------------------------------------------------------------------

C3_COLUMNS = ["query_id", "candidate_key", "is_true", "ref_spectrum_id", "compat_rank", "tier",
              *SIMILARITY_METRICS, *[f"accepted_{p}" for p in PROTOCOLS], *[f"eligible_{p}" for p in PROTOCOLS]]


def recreate_c3_sample(host_qcr, dev_qcr, n_pairs=1000, seed=42):
    """Recreate the v4a.1 C3 sample EXACTLY: `concat([host.assign(dataset="host"),
    dev.assign(dataset="dev")])` in persisted row order, then
    `np.random.RandomState(seed).choice(len, size=min(n_pairs, len), replace=False)`. Both QCR
    frames must be the persisted canonical tables, unfiltered and in their on-disk row order."""
    combined = pd.concat([host_qcr.assign(dataset="host"), dev_qcr.assign(dataset="dev")], ignore_index=True)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(combined), size=min(n_pairs, len(combined)), replace=False)
    keep = ["dataset"] + [c for c in C3_COLUMNS if c in combined.columns]
    sample = combined.iloc[idx][keep].copy()
    sample.insert(0, "sample_position", np.arange(len(sample)))
    return sample.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# fresh recomputation
# ---------------------------------------------------------------------------------------------

def offset_of(spectrum_id):
    return int(str(spectrum_id).split("_")[1])


def load_fresh_representations(peak_store_path, offsets, max_peaks):
    """Independently load raw peaks for `offsets` (bypassing every cache) and build the three
    representations the audit needs: identity (full cleaned peaks), current deterministic top-N,
    and the legacy argpartition top-N. Also returns per-offset top-N boundary diagnostics."""
    raw = load_rows_by_offset(peak_store_path, ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"], list(offsets))
    raw = raw.rename(columns={"_row_offset": "row_offset"})
    identity, stable, legacy, boundary = {}, {}, {}, {}
    for r in raw.itertuples(index=False):
        off = int(r.row_offset)
        mzs, ints = remove_invalid_peaks(r.ms2_mzs, r.ms2_normalized_intensities)
        identity[off] = {"mzs": mzs, "intensities": ints, "precursor_mz": r.precursor_mz}
        s_mz, s_int = truncate_top_peaks(mzs, ints, max_peaks=max_peaks)
        stable[off] = {"mzs": s_mz, "intensities": s_int, "precursor_mz": r.precursor_mz}
        l_mz, l_int = truncate_top_peaks_legacy(mzs, ints, max_peaks=max_peaks)
        legacy[off] = {"mzs": l_mz, "intensities": l_int, "precursor_mz": r.precursor_mz}
        diag = top_n_boundary_diagnostics(ints, max_peaks=max_peaks)
        diag["stable_vs_legacy_selection_differs"] = bool(
            len(s_mz) != len(l_mz) or not (np.array_equal(s_mz, l_mz) and np.array_equal(s_int, l_int)))
        boundary[off] = diag
    del raw
    return identity, stable, legacy, boundary


def _tier(q, r, cfg):
    return classify_identity_tier(
        q, r,
        tolerant_mirror_kwargs={
            "precursor_diff_da": cfg["tolerant_mirror_precursor_diff_da"],
            "min_relative_intensity": cfg["tolerant_mirror_min_rel_intensity"],
            "ppm_tol": cfg["tolerant_mirror_ppm_tol"], "abs_tol_da": cfg["tolerant_mirror_abs_tol_da"],
            "match_fraction": cfg["tolerant_mirror_match_fraction"], "min_correlation": cfg["tolerant_mirror_min_correlation"],
        },
        near_dup_cosine_threshold=cfg["near_dup_cosine_threshold"],
        near_dup_precursor_diff_da=cfg["near_dup_precursor_diff_da"],
        bin_width=cfg["bin_width_da"],
    )


def identity_margin_diagnostics(q, r, thresholds):
    """Distances of the identity-side quantities to their tier thresholds -- lets a reviewer see
    whether a pair sits on a knife edge even when its tier did not change."""
    precursor_diff = abs(float(q["precursor_mz"]) - float(r["precursor_mz"]))
    qb, qv = bin_spectrum(q["mzs"], q["intensities"], bin_width=thresholds["bin_width_da"])
    rb, rv = bin_spectrum(r["mzs"], r["intensities"], bin_width=thresholds["bin_width_da"])
    identity_cosine = binned_cosine_similarity(qb, qv, rb, rv)

    def _thr(p):
        mzs, ints = np.asarray(p["mzs"], float), np.asarray(p["intensities"], float)
        if len(ints) == 0 or ints.max() <= 0:
            return mzs[:0], ints[:0]
        keep = ints / ints.max() >= thresholds["t2_min_rel_intensity"]
        return mzs[keep], ints[keep]

    q_mz, q_int = _thr(q)
    r_mz, r_int = _thr(r)
    frac_q = frac_r = corr = float("nan")
    if len(q_mz) and len(r_mz):
        mq, mr = _match_peaks_greedy(q_mz, q_int, r_mz, r_int, ppm_tol=thresholds["t2_ppm_tol"], abs_tol_da=thresholds["t2_abs_tol_da"])
        frac_q, frac_r = len(mq) / len(q_mz), len(mq) / len(r_mz)
        if len(mq) >= 2:
            a, b = np.sqrt(q_int[mq]), np.sqrt(r_int[mr])
            if np.std(a) > 0 and np.std(b) > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
    m = NEAR_THRESHOLD_MARGIN
    near = {
        "near_t3_cosine": abs(identity_cosine - thresholds["t3_cosine"]) <= m,
        "near_t3_precursor": abs(precursor_diff - thresholds["t3_precursor_diff_da"]) <= m,
        "near_t2_precursor": abs(precursor_diff - thresholds["t2_precursor_diff_da"]) <= m,
        "near_t2_match_fraction": any(np.isfinite(f) and abs(f - thresholds["t2_match_fraction"]) <= m for f in (frac_q, frac_r)),
        "near_t2_correlation": bool(np.isfinite(corr) and abs(corr - thresholds["t2_min_correlation"]) <= m),
    }
    return {"precursor_diff_da": precursor_diff, "identity_cosine": identity_cosine, "t2_match_frac_query": frac_q,
            "t2_match_frac_ref": frac_r, "t2_sqrt_corr": corr, **near, "any_near_identity_threshold": any(near.values())}


def recompute_sample(sample, identity, stable, legacy, boundary, similarity_config):
    """One row per sampled triple: cached values, fresh values under the CURRENT code (identity
    tier + deterministic top-N similarities), fresh similarities under the LEGACY top-N rule,
    identity-side margins, and the top-N boundary diagnostics of both spectra."""
    thresholds = identity_thresholds(similarity_config)
    rows = []
    for row in sample.itertuples(index=False):
        q_off, r_off = offset_of(row.query_id), offset_of(row.ref_spectrum_id)
        fresh_tier = _tier(identity[q_off], identity[r_off], similarity_config)
        fresh = compute_single_pair_scores(stable[q_off], stable[r_off], bin_width=similarity_config["bin_width_da"],
                                           peak_tol_da=similarity_config["peak_tol_da"])
        legacy_vals = compute_single_pair_scores(legacy[q_off], legacy[r_off], bin_width=similarity_config["bin_width_da"],
                                                 peak_tol_da=similarity_config["peak_tol_da"])
        rec = {c: getattr(row, c) for c in sample.columns}
        rec["cached_tier"] = rec.pop("tier")
        rec["fresh_tier"] = fresh_tier
        for m, fv, lv in zip(SIMILARITY_METRICS, fresh, legacy_vals):
            rec[f"cached_{m}"] = float(rec.pop(m))
            rec[f"fresh_{m}"] = float(fv)
            rec[f"legacy_{m}"] = float(lv)
        rec.update(identity_margin_diagnostics(identity[q_off], identity[r_off], thresholds))
        for side, off in (("query", q_off), ("ref", r_off)):
            b = boundary[off]
            rec[f"{side}_n_peaks"] = b["n_peaks"]
            rec[f"{side}_top_n_truncated"] = b["truncated"]
            rec[f"{side}_intensity_at_n"] = b["intensity_at_n"]
            rec[f"{side}_intensity_at_n_plus_1"] = b["intensity_at_n_plus_1"]
            rec[f"{side}_top_n_exact_tie"] = b["exact_tie"]
            rec[f"{side}_top_n_near_tie"] = b["near_tie"]
            rec[f"{side}_top_n_selection_rule_sensitive"] = b["stable_vs_legacy_selection_differs"]
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# comparison + summary
# ---------------------------------------------------------------------------------------------

def annotate_differences(diag, similarity_config, strict_atol=STRICT_REPORT_ATOL):
    """Adds per-row `absdiff_<metric>`, `value_mismatch` (any metric beyond `strict_atol`, the
    historical C3 definition), `tier_changed`, mirror/near-dup eligibility changes, threshold
    crossings, and whether the legacy top-N rule reproduces the cached values (`explained_by_legacy_topn`)."""
    diag = diag.copy()
    value_thresholds = similarity_value_thresholds(similarity_config)
    mismatch_any = np.zeros(len(diag), dtype=bool)
    legacy_match_all = np.ones(len(diag), dtype=bool)
    crossing_count = np.zeros(len(diag), dtype=int)
    for m in SIMILARITY_METRICS:
        cached, fresh, legacy = diag[f"cached_{m}"].to_numpy(float), diag[f"fresh_{m}"].to_numpy(float), diag[f"legacy_{m}"].to_numpy(float)
        both_nan = np.isnan(cached) & np.isnan(fresh)
        absdiff = np.where(both_nan, 0.0, np.abs(cached - fresh))
        absdiff = np.where(np.isnan(absdiff), np.inf, absdiff)  # NaN on exactly one side = infinite discrepancy
        diag[f"absdiff_{m}"] = absdiff
        mismatch_any |= absdiff > strict_atol
        legacy_match_all &= np.isclose(cached, legacy, atol=strict_atol, rtol=0, equal_nan=True)
        for t in value_thresholds.get(m, ()):
            col = f"crosses_{m}_{t:g}"
            diag[col] = [crosses_threshold(a, b, t) for a, b in zip(cached, fresh)]
            crossing_count += diag[col].to_numpy(int)
    diag["value_mismatch"] = mismatch_any
    diag["tier_changed"] = diag["cached_tier"].astype(str) != diag["fresh_tier"].astype(str)
    diag["mirror_eligibility_changed"] = diag["cached_tier"].isin(MIRROR_TIERS) != diag["fresh_tier"].isin(MIRROR_TIERS)
    diag["near_dup_eligibility_changed"] = diag["cached_tier"].isin(NEAR_DUP_TIERS) != diag["fresh_tier"].isin(NEAR_DUP_TIERS)
    diag["n_value_threshold_crossings"] = crossing_count
    # a tier change IS an identity-threshold crossing by definition (tiers are thresholded quantities)
    diag["threshold_crossing"] = (crossing_count > 0) | diag["tier_changed"].to_numpy()
    diag["explained_by_legacy_topn"] = diag["value_mismatch"] & legacy_match_all
    return diag


def mismatch_table(diag, strict_atol=STRICT_REPORT_ATOL):
    """The compact per-metric table: metric | n_mismatch | median_abs_diff | p95_abs_diff |
    max_abs_diff (statistics over ALL checked pairs; `n_mismatch` at the strict historical atol)."""
    rows = []
    for m in SIMILARITY_METRICS:
        d = diag[f"absdiff_{m}"].to_numpy(float)
        rows.append({"metric": m, "n_mismatch": int((d > strict_atol).sum()),
                     "median_abs_diff": float(np.median(d)) if len(d) else float("nan"),
                     "p95_abs_diff": float(np.quantile(d, 0.95)) if len(d) else float("nan"),
                     "max_abs_diff": float(d.max()) if len(d) else float("nan")})
    return pd.DataFrame(rows)


def summarize_settlement(diag, aggregated_feature_mismatch_count):
    """The runtime fields spec section 9 requires (minus `decision`/`evidence_fingerprint`, which
    the caller adds)."""
    def _max(m):
        v = diag[f"absdiff_{m}"].to_numpy(float)
        return float(v.max()) if len(v) else float("nan")

    q_tie = diag["query_top_n_exact_tie"] | diag["query_top_n_near_tie"]
    r_tie = diag["ref_top_n_exact_tie"] | diag["ref_top_n_near_tie"]
    mism = diag["value_mismatch"]
    return {
        "n_pairs_checked": int(len(diag)),
        "n_value_mismatches": int(mism.sum()),
        "n_tier_changes": int(diag["tier_changed"].sum()),
        "n_mirror_eligibility_changes": int(diag["mirror_eligibility_changed"].sum()),
        "n_near_dup_eligibility_changes": int(diag["near_dup_eligibility_changed"].sum()),
        "n_cosine_crossing_0_95": int(sum(diag[c].sum() for c in diag.columns if c.startswith("crosses_cosine_"))),
        "n_threshold_crossings": int(diag["threshold_crossing"].sum()),
        "n_near_identity_threshold_pairs": int(diag["any_near_identity_threshold"].sum()),
        "max_abs_diff_cosine": _max("cosine"),
        "max_abs_diff_modified_cosine": _max("modified_cosine"),
        "max_abs_diff_peak_overlap": _max("peak_overlap_frac"),
        "max_abs_diff_neutral_loss": _max("neutral_loss_cosine"),
        "top100_tie_cases": int((mism & (q_tie | r_tie)).sum()),
        "top100_tie_cases_all_sampled": int((q_tie | r_tie).sum()),
        "top100_unstable_cases": int((mism & (diag["query_top_n_selection_rule_sensitive"] | diag["ref_top_n_selection_rule_sensitive"])).sum()),
        "n_mismatches_explained_by_legacy_topn": int(diag["explained_by_legacy_topn"].sum()),
        "aggregated_feature_mismatch_count": int(aggregated_feature_mismatch_count),
    }


def decide_cache(summary, atol=DECISION_ATOL):
    """THE decision rule (spec section 8). Pure function of the summary dict. Returns
    `(decision, reasons)`; `reasons` is empty iff the decision is ACCEPT_EXISTING. A NaN/missing
    max-diff is treated as a failure, never as a pass."""
    reasons = []
    if summary.get("n_pairs_checked", 0) <= 0:
        reasons.append("no pairs checked")
    if summary.get("n_tier_changes", 1) != 0:
        reasons.append(f"n_tier_changes={summary.get('n_tier_changes')}")
    if summary.get("n_threshold_crossings", 1) != 0:
        reasons.append(f"n_threshold_crossings={summary.get('n_threshold_crossings')}")
    for key in ("max_abs_diff_cosine", "max_abs_diff_modified_cosine", "max_abs_diff_peak_overlap", "max_abs_diff_neutral_loss"):
        v = summary.get(key)
        if v is None or not np.isfinite(v) or v > atol:
            reasons.append(f"{key}={v} > {atol}")
    if summary.get("aggregated_feature_mismatch_count", 1) != 0:
        reasons.append(f"aggregated_feature_mismatch_count={summary.get('aggregated_feature_mismatch_count')}")
    return (DECISION_ACCEPT if not reasons else DECISION_REBUILD), reasons


def decide_after_rebuild(post_rebuild_summary, atol=DECISION_ATOL):
    """After the ONE clean rebuild: the same rule must now accept, otherwise the evidence layer
    has a real problem that a rebuild does not fix -- never loop into a second rebuild."""
    decision, reasons = decide_cache(post_rebuild_summary, atol=atol)
    return (DECISION_REBUILT_OK if decision == DECISION_ACCEPT else DECISION_REBUILD_FAILED), reasons


# ---------------------------------------------------------------------------------------------
# aggregated candidate features of affected candidates
# ---------------------------------------------------------------------------------------------

def affected_candidates(diag):
    """(dataset, query_id, candidate_key) of every sampled triple with a value mismatch or tier change."""
    bad = diag[diag["value_mismatch"] | diag["tier_changed"]]
    return bad[["dataset", "query_id", "candidate_key"]].drop_duplicates().reset_index(drop=True)


def aggregated_feature_check(affected, qcr_by_dataset, feature_tables, fresh_values, protocols=PROTOCOLS, atol=DECISION_ATOL):
    """For every affected candidate and protocol: aggregate its ACCEPTED references' values
    (a) as persisted in QCR and (b) with FRESH values substituted for every accepted reference
    (`fresh_values[(dataset, query_id, ref_spectrum_id)] -> (c, m, o, n)`), and compare both to
    the persisted candidate feature table (`feature_tables[(dataset, protocol)]`).

    Returns `(mismatch_df, n_mismatches)`; a mismatch is any aggregated feature differing by more
    than `atol` between the fresh aggregation and the persisted table. Selection is held fixed
    here -- valid only when no tier changed (which `decide_cache` checks separately)."""
    from casmi.ranking.features import aggregate_pair_scores_from_values

    records = []
    for cand in affected.itertuples(index=False):
        qcr = qcr_by_dataset[cand.dataset]
        rows = qcr[(qcr["query_id"] == cand.query_id) & (qcr["candidate_key"] == cand.candidate_key)]
        for p in protocols:
            acc = rows[rows[f"accepted_{p}"]].sort_values("compat_rank")
            cached_vals = [acc[m].tolist() for m in SIMILARITY_METRICS]
            fresh_vals = [[], [], [], []]
            for rid in acc["ref_spectrum_id"]:
                fv = fresh_values[(cand.dataset, cand.query_id, rid)]
                for i in range(4):
                    fresh_vals[i].append(fv[i])
            agg_cached = aggregate_pair_scores_from_values(*cached_vals)
            agg_fresh = aggregate_pair_scores_from_values(*fresh_vals)
            table = feature_tables.get((cand.dataset, p))
            persisted = None
            if table is not None:
                hit = table[(table["query_id"] == cand.query_id) & (table["candidate_connectivity_key"] == cand.candidate_key)]
                persisted = hit.iloc[0] if len(hit) else None
            for f in AGGREGATED_FEATURE_COLS:
                pv = float(persisted[f]) if persisted is not None else float("nan")
                fv, cv = float(agg_fresh[f]), float(agg_cached[f])
                fresh_vs_persisted = 0.0 if (np.isnan(fv) and np.isnan(pv)) else abs(fv - pv)
                cached_vs_persisted = 0.0 if (np.isnan(cv) and np.isnan(pv)) else abs(cv - pv)
                records.append({"dataset": cand.dataset, "query_id": cand.query_id, "candidate_key": cand.candidate_key,
                                "protocol": p, "feature": f, "n_accepted_refs": len(acc), "persisted": pv,
                                "qcr_aggregated": cv, "fresh_aggregated": fv,
                                "absdiff_fresh_vs_persisted": fresh_vs_persisted if np.isfinite(fresh_vs_persisted) else np.inf,
                                "absdiff_qcr_vs_persisted": cached_vs_persisted if np.isfinite(cached_vs_persisted) else np.inf,
                                "persisted_row_found": persisted is not None})
    df = pd.DataFrame(records)
    if df.empty:
        return df, 0
    df["mismatch"] = (df["absdiff_fresh_vs_persisted"] > atol) | ~df["persisted_row_found"]
    return df, int(df["mismatch"].sum())


# ---------------------------------------------------------------------------------------------
# fingerprint + persistence + downstream gate
# ---------------------------------------------------------------------------------------------

def _file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def evidence_fingerprint(artifact_paths):
    """Content hash over every settled evidence artifact (`{logical_name: path}`) -- content, not
    mtime, so a copy is recognised and a silent rewrite is not. Returns `(combined, per_file)`."""
    per_file = {}
    for name in sorted(artifact_paths):
        p = Path(artifact_paths[name])
        per_file[name] = _file_sha256(p) if p.exists() else f"missing:{p}"
    combined = hashlib.sha256(json.dumps(per_file, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return combined, per_file


def write_settlement(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("written_at", datetime.now(timezone.utc).isoformat())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def load_settlement_or_refuse(path, verify_fingerprint=True):
    """Gate for notebook 01. Raises `CacheSettlementError` unless the settlement file exists, its
    `decision` is ACCEPT_EXISTING or REBUILT_AND_VALIDATED, it names its evidence artifacts, and
    (by default) those artifacts still hash to the recorded `evidence_fingerprint`."""
    path = Path(path)
    if not path.exists():
        raise CacheSettlementError(f"{path} not found -- run 10v4b_00_cache_settlement.ipynb first")
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    decision = record.get("decision")
    if decision not in ALLOWED_DOWNSTREAM_DECISIONS:
        raise CacheSettlementError(f"cache settlement decision is {decision!r}; notebook 01 requires one of {ALLOWED_DOWNSTREAM_DECISIONS}")
    artifacts = record.get("evidence_artifacts")
    if not artifacts:
        raise CacheSettlementError("cache settlement names no evidence_artifacts")
    if verify_fingerprint:
        current, _ = evidence_fingerprint(artifacts)
        if current != record.get("evidence_fingerprint"):
            raise CacheSettlementError(
                f"evidence artifacts changed since settlement (recorded {record.get('evidence_fingerprint')}, now {current}) -- re-run notebook 00")
    return record
