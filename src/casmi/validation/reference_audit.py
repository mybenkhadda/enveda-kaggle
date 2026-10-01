"""Auditing the exact lazy reference walk (`casmi.spectra.reference_selection`): a brute-force
reimplementation to prove the lazy walk's early stopping never changes which references get
selected, plus reusable aggregation for the coverage/walk-depth audits every notebook that uses
the walk needs to report.
"""
import pandas as pd

from casmi.spectra.reference_selection import MAX_REFS_PER_PROTOCOL, PROTOCOLS, is_eligible


def brute_force_reference_selection(ranked_reference_ids, query_meta, ref_meta_lookup, classify_fn,
                                     protocols=PROTOCOLS, max_refs=MAX_REFS_PER_PROTOCOL):
    """Independent (non-lazy) reimplementation: classify EVERY reference in
    `ranked_reference_ids` (no early stopping), then per protocol filter to eligible references
    and take the first `max_refs` of the (already compat-rank-ordered) full list. Exists ONLY to
    verify `casmi.spectra.reference_selection.walk_references`'s early-stopping selects the
    exact same references as full evaluation would -- see `assert_lazy_matches_brute_force`.
    Returns `(accepted, classified)`: `accepted[protocol]` is the selected id list,
    `classified[reference_id]` is `classify_fn`'s raw return tuple."""
    classified = {rid: classify_fn(rid) for rid in ranked_reference_ids}
    accepted = {}
    for p in protocols:
        eligible_ids = [rid for rid in ranked_reference_ids if is_eligible(p, ref_meta_lookup[rid], query_meta, classified[rid][0])]
        accepted[p] = eligible_ids[:max_refs]
    return accepted, classified


def assert_lazy_matches_brute_force(lazy_result, brute_accepted, protocols=PROTOCOLS):
    """Raises `AssertionError` with full mismatch detail (not just the first one found) if any
    protocol's lazy-walk selection differs from the brute-force selection. Returns silently
    (never truthy) on a match -- call it for its side effect, like a normal `assert`."""
    mismatches = {
        p: {"lazy": lazy_result.accepted[p], "brute": brute_accepted[p]}
        for p in protocols
        if lazy_result.accepted[p] != brute_accepted[p]
    }
    if mismatches:
        raise AssertionError(f"lazy walk != brute force for protocol(s): {mismatches}")


def reference_coverage_summary(walk_results, is_true_by_key, protocols=PROTOCOLS, max_refs=MAX_REFS_PER_PROTOCOL):
    """`walk_results`: `{(query_id, candidate_key): WalkResult}`. `is_true_by_key`: same keys ->
    bool. Coverage thresholds (`>=1`/`>=3`/`>=5`) are EXACT under the lazy walk's stopping rule
    (see `WalkResult.has_at_least`) -- never a lower bound, regardless of early stopping. Returns
    one row per (protocol, group) with `frac_ge1`/`frac_ge3`/`frac_ge5`/`n_zero`/`n`."""
    rows = []
    for p in protocols:
        for group_name, want_true in [("true_candidates", True), ("decoys", False)]:
            keys = [k for k, is_true in is_true_by_key.items() if is_true == want_true]
            if not keys:
                continue
            counts = [len(walk_results[k].accepted[p]) for k in keys]
            rows.append({
                "protocol": p, "group": group_name, "n": len(counts),
                "frac_ge1": sum(c >= 1 for c in counts) / len(counts),
                "frac_ge3": sum(c >= 3 for c in counts) / len(counts),
                "frac_ge5": sum(c >= max_refs for c in counts) / len(counts),
                "n_zero": sum(c == 0 for c in counts),
            })
    return pd.DataFrame(rows)


def walk_depth_summary(walk_results, protocols=PROTOCOLS):
    """Per-protocol walk-depth distribution (`compat_rank` at which the 5th eligible reference
    was accepted; `NaN` when never reached) and the fraction of walks reaching beyond
    rank 15/30/50/100 -- the decisive proof that no fixed-position cap is silently in effect."""
    rows = []
    for p in protocols:
        depths = pd.Series([wr.walk_depth[p] for wr in walk_results.values()], dtype=float)
        exhausted_before_5 = sum(1 for wr in walk_results.values() if not wr.reached5[p] and wr.exhausted)
        rows.append({
            "protocol": p,
            "median_depth": depths.median(), "p90_depth": depths.quantile(0.9),
            "p95_depth": depths.quantile(0.95), "p99_depth": depths.quantile(0.99), "max_depth": depths.max(),
            "frac_gt_15": (depths > 15).mean(), "frac_gt_30": (depths > 30).mean(),
            "frac_gt_50": (depths > 50).mean(), "frac_gt_100": (depths > 100).mean(),
            "frac_exhausted_before_5": exhausted_before_5 / len(walk_results) if walk_results else float("nan"),
        })
    return pd.DataFrame(rows)
