"""Fold-stability decision rule (research decision framework: KEEP / REJECT / NEEDS_MORE_EVIDENCE).

A variant beats a baseline only if the improvement is REPRODUCIBLE across folds, not just on the pooled mean:

    KEEP                 delta > `min_mean_delta` pooled AND positive on >= `min_positive_folds` folds
                         (default: all folds but one)
    NEEDS_MORE_EVIDENCE  pooled delta > 0 but not stable enough across folds (or too few common folds)
    REJECT               pooled delta <= 0

The rule only reads metrics computed elsewhere; it never decides which variant "is best" on its own.
"""
import pandas as pd


def keep_reject(per_fold, variant, baseline, metric="mrr_at_25", fold_col="fold", variant_col="variant", min_positive_folds=None,
                min_mean_delta=0.0, min_folds=3):
    """`per_fold`: long table (variant_col, fold_col, metric) -- one row per (variant, fold), already filtered to the
    regime the decision is about. Returns (decision, reason, stats)."""
    piv = per_fold.pivot_table(index=fold_col, columns=variant_col, values=metric, aggfunc="first")
    if variant not in piv.columns or baseline not in piv.columns:
        return "NEEDS_MORE_EVIDENCE", f"{variant!r} or {baseline!r} has no per-fold rows", {}
    delta = (piv[variant] - piv[baseline]).dropna()
    n, n_pos = int(len(delta)), int((delta > 0).sum())
    mean = float(delta.mean()) if n else float("nan")
    need = int(min_positive_folds) if min_positive_folds is not None else max(n - 1, 1)
    stats = {"n_folds": n, "n_positive": n_pos, "mean_delta": mean, "min_delta": float(delta.min()) if n else None,
             "max_delta": float(delta.max()) if n else None, "required_positive": need,
             "per_fold_delta": {str(k): float(v) for k, v in delta.items()}}
    reason = f"d{metric} vs {baseline}: {mean:+.4f} pooled-of-folds, positive on {n_pos}/{n} folds"
    if n < min_folds:
        return "NEEDS_MORE_EVIDENCE", reason + f" (fewer than {min_folds} common folds)", stats
    if n_pos >= need and mean > min_mean_delta:
        return "KEEP", reason, stats
    if mean > 0:
        return "NEEDS_MORE_EVIDENCE", reason, stats
    return "REJECT", reason, stats


def decision_table(per_fold, baseline, metric="mrr_at_25", variant_col="variant", **kw):
    """keep_reject for every variant against `baseline` -> DataFrame(variant, decision, reason, n_positive, ...)."""
    rows = []
    for v in sorted(per_fold[variant_col].dropna().unique()):
        if v == baseline:
            rows.append({"variant": v, "decision": "BASELINE", "reason": ""})
            continue
        d, r, s = keep_reject(per_fold, v, baseline, metric=metric, variant_col=variant_col, **kw)
        rows.append({"variant": v, "decision": d, "reason": r, **{k: s.get(k) for k in ("n_folds", "n_positive", "mean_delta")}})
    return pd.DataFrame(rows)
