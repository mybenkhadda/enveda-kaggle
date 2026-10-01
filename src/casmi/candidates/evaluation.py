"""Candidate-generation evaluation: recall and candidate-count tradeoffs across a ppm
tolerance grid.

Deliberately does NOT require materializing a full (query, candidate) pool to answer "what is
recall at ppm X": since a query's true connectivity is either in the library or not (closed-
world, see the module docstring in `casmi.candidates`), whether it falls inside a ppm window
is fully determined by ONE number -- how far the query's derived neutral mass sits from the
true structure's own exact mass, in ppm. `query_candidate_summary` computes exactly that (plus
per-tolerance candidate COUNTS, which do need the mass index), and every function below works
off that compact per-query table.
"""
import numpy as np
import pandas as pd


def add_target_mass_error(queries, mass_col="neutral_mass", true_mass_col="exact_mass_true"):
    """Add `target_abs_mass_error_ppm`: how far each query's derived neutral mass sits from its
    OWN true structure's exact mass, in ppm. `nan` wherever either mass is missing (e.g. an
    unsupported adduct) -- never dropped, so it still counts against recall at every
    tolerance."""
    out = queries.copy()
    error_ppm = 1e6 * (out[true_mass_col] - out[mass_col]) / out[mass_col]
    out["target_mass_error_ppm"] = error_ppm
    out["target_abs_mass_error_ppm"] = error_ppm.abs()
    return out


def query_candidate_summary(queries, mass_index, tolerance_grid, query_id_col="query_id",
                             mass_col="neutral_mass", true_mass_col="exact_mass_true"):
    """One row per query: `candidate_count_<ppm>ppm` for every ppm in `tolerance_grid`, plus
    `target_abs_mass_error_ppm` when `true_mass_col` is present (development queries only --
    omit/rename it away for test queries, where the true mass is the prediction target).
    Compact by design (no candidate keys) so most analysis/plotting never has to touch the full
    candidate pool -- see item 39 of the notebook 03 spec this implements."""
    has_truth = true_mass_col in queries.columns
    out = add_target_mass_error(queries, mass_col=mass_col, true_mass_col=true_mass_col) if has_truth else queries.copy()

    masses = out[mass_col].to_numpy(dtype=float)
    counts = {ppm: np.zeros(len(out), dtype=int) for ppm in tolerance_grid}
    for i, m in enumerate(masses):
        if np.isnan(m):
            continue
        for ppm in tolerance_grid:
            counts[ppm][i] = mass_index.candidate_count(m, tolerance_ppm=ppm)

    for ppm in tolerance_grid:
        out[f"candidate_count_{ppm}ppm"] = counts[ppm]
    out["neutral_mass_supported"] = ~np.isnan(masses)
    return out


def tolerance_sweep(query_summary, tolerance_grid, target_col="target_abs_mass_error_ppm"):
    """The required ppm/recall/candidate-count table (notebook 03 item 19-20): one row per
    tolerance, with the FULL query set as denominator (a query with zero candidates, or an
    unsupported adduct, still counts against recall -- never silently dropped)."""
    n_queries = len(query_summary)
    has_truth = target_col in query_summary.columns
    rows = []
    for ppm in tolerance_grid:
        counts = query_summary[f"candidate_count_{ppm}ppm"]
        row = {
            "tolerance_ppm": ppm,
            "n_queries": n_queries,
            "mean_candidates": float(counts.mean()),
            "median_candidates": float(counts.median()),
            "p75_candidates": float(counts.quantile(0.75)),
            "p90_candidates": float(counts.quantile(0.90)),
            "p95_candidates": float(counts.quantile(0.95)),
            "p99_candidates": float(counts.quantile(0.99)),
            "max_candidates": int(counts.max()) if n_queries else 0,
            "zero_candidate_rate": float((counts == 0).mean()) if n_queries else float("nan"),
        }
        if has_truth:
            present = query_summary[target_col] <= ppm
            row.update({
                "n_target_present": int(present.sum()),
                "n_target_absent": int(n_queries - present.sum()),
                "candidate_recall": float(present.mean()) if n_queries else float("nan"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def select_operating_tolerance(sweep, target_recall=0.995, recall_col="candidate_recall", ppm_col="tolerance_ppm"):
    """The smallest tolerance in `sweep` achieving `target_recall`. If none does, returns the
    BEST TESTED (not "selected", not "final") recall and its tolerance instead of pretending
    the target was met -- `hit_target` tells the caller which case it is, and callers must not
    print or persist this as a final operating point when `hit_target` is False (see
    `classify_decision`, which is what actually decides FROZEN vs. PROVISIONAL)."""
    # `tolerance_ppm` is read off the DataFrame's own (typically int) column, not off a
    # per-row Series -- `.iloc[i]` on a mixed-dtype row upcasts everything (including an int
    # tolerance) to float64, and callers use this value to look up a
    # `candidate_count_<ppm>ppm` column by name, where `50.0` would miss where `50` hits.
    sweep = sweep.sort_values(ppm_col)
    meeting = sweep[sweep[recall_col] >= target_recall]
    chosen = meeting if len(meeting) else sweep.sort_values(recall_col, ascending=False)
    idx = chosen.index[0]
    return {
        "tolerance_ppm": sweep.loc[idx, ppm_col].item() if hasattr(sweep.loc[idx, ppm_col], "item") else sweep.loc[idx, ppm_col],
        "recall": float(sweep.loc[idx, recall_col]),
        "median_candidates": float(sweep.loc[idx, "median_candidates"]),
        "hit_target": len(meeting) > 0,
        "target_recall": target_recall,
    }


def tolerance_tradeoff_deltas(sweep, ppm_col="tolerance_ppm", recall_col="candidate_recall"):
    """One row per ADJACENT pair of tolerances in `sweep`: how much recall was bought, and how
    much it cost in extra candidates -- the quantified version of "20->50ppm: small recall
    gain, large candidate-count increase" rather than an eyeballed claim."""
    s = sweep.sort_values(ppm_col).reset_index(drop=True)
    rows = []
    for i in range(1, len(s)):
        prev, cur = s.iloc[i - 1], s.iloc[i]
        recall_gain_pp = 100 * (cur[recall_col] - prev[recall_col]) if recall_col in s.columns else float("nan")
        median_increase = cur["median_candidates"] - prev["median_candidates"]
        p90_increase = cur["p90_candidates"] - prev["p90_candidates"]
        rows.append({
            "from_ppm": prev[ppm_col], "to_ppm": cur[ppm_col],
            "recall_gain_pp": recall_gain_pp,
            "median_candidate_increase": median_increase,
            "p90_candidate_increase": p90_increase,
            "candidates_per_recall_point": (median_increase / recall_gain_pp) if recall_gain_pp else float("inf"),
        })
    return pd.DataFrame(rows)


def evaluate_variable_tolerance(queries, mass_index, tolerance_col, mass_col="neutral_mass",
                                 true_mass_col="exact_mass_true"):
    """Like `query_candidate_summary`, but each row uses its OWN tolerance (`tolerance_col`)
    instead of a shared grid -- what an adaptive/per-group policy needs to be scored with.
    Adds `candidate_count` (at that row's own tolerance) and, when `true_mass_col` is present,
    `target_abs_mass_error_ppm` and `target_present` (whether that row's own tolerance actually
    covers its true target)."""
    has_truth = true_mass_col in queries.columns
    out = add_target_mass_error(queries, mass_col=mass_col, true_mass_col=true_mass_col) if has_truth else queries.copy()

    masses = out[mass_col].to_numpy(dtype=float)
    tolerances = out[tolerance_col].to_numpy(dtype=float)
    counts = np.zeros(len(out), dtype=int)
    for i in range(len(out)):
        if np.isnan(masses[i]):
            continue
        counts[i] = mass_index.candidate_count(masses[i], tolerance_ppm=tolerances[i])
    out["candidate_count"] = counts
    out["neutral_mass_supported"] = ~np.isnan(masses)
    if has_truth:
        out["target_present"] = out["target_abs_mass_error_ppm"] <= out[tolerance_col]
    return out


def variable_tolerance_summary(evaluated, count_col="candidate_count", target_present_col="target_present"):
    """Aggregate `evaluate_variable_tolerance`'s per-row output into the same shape as one
    `tolerance_sweep` row, for direct comparison against global-tolerance policies in a policy
    comparison table."""
    n = len(evaluated)
    row = {
        "n_queries": n,
        "median_candidates": float(evaluated[count_col].median()),
        "p90_candidates": float(evaluated[count_col].quantile(0.90)),
        "p95_candidates": float(evaluated[count_col].quantile(0.95)),
        "p99_candidates": float(evaluated[count_col].quantile(0.99)),
        "zero_candidate_rate": float((evaluated[count_col] == 0).mean()) if n else float("nan"),
    }
    if target_present_col in evaluated.columns:
        row["candidate_recall"] = float(evaluated[target_present_col].mean()) if n else float("nan")
    return row


def classify_decision(global_result, adaptive_result=None):
    """The revised, non-fabricated final decision (spec section 61/79): classifies into
    exactly one of GLOBAL_POLICY_VALIDATED / ADAPTIVE_POLICY_VALIDATED / TARGET_NOT_ACHIEVED,
    and sets `status` to "frozen" only in the first two cases -- a target that was never
    reached leaves candidate generation `"provisional"`, never silently promoted to final.

    `global_result`/`adaptive_result`: dicts shaped like `select_operating_tolerance`'s return
    (`adaptive_result` is optional -- pass `None` if no adaptive policy was evaluated, or it
    wasn't fold-safe enough to trust).
    """
    if global_result["hit_target"]:
        return {"decision": "GLOBAL_POLICY_VALIDATED", "status": "frozen", "policy": "global",
                "tolerance_ppm": global_result["tolerance_ppm"], "recall": global_result["recall"]}
    if adaptive_result is not None and adaptive_result.get("hit_target"):
        return {"decision": "ADAPTIVE_POLICY_VALIDATED", "status": "frozen", "policy": "adaptive",
                "recall": adaptive_result["recall"]}
    return {
        "decision": "TARGET_NOT_ACHIEVED", "status": "provisional", "policy": "best-effort global",
        "tolerance_ppm": global_result["tolerance_ppm"], "recall": global_result["recall"],
        "best_adaptive_recall": adaptive_result["recall"] if adaptive_result is not None else None,
    }
