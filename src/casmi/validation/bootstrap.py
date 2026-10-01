"""Paired bootstrap confidence intervals for ranking-metric comparisons -- so "0.608 > 0.589"
is never reported as a meaningful improvement without an uncertainty estimate. "Paired" means
both methods are evaluated on the SAME queries (the common case here: two rankers scored over
the same dev/test query set), so resampling QUERIES together (not resampling each method's
values independently) correctly preserves the shared per-query variance.
"""
import numpy as np


def paired_bootstrap_metric_difference(values_a, values_b, n_bootstrap=5000, seed=42):
    """`values_a`/`values_b`: arrays of the SAME per-query metric value (e.g. reciprocal rank
    per query, or a 0/1 hit indicator per query) for two methods, over the SAME queries in the
    SAME order -- their DIFFERENCE's distribution under resampling queries with replacement is
    the paired bootstrap. Returns `{mean_diff, ci_low, ci_high, prob_a_greater}` where
    `mean_diff = mean(a) - mean(b)` and `prob_a_greater` is the fraction of bootstrap
    resamples where A's mean exceeds B's -- a direct, assumption-light substitute for a p-value.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired inputs must be the same length (got {len(a)} and {len(b)})")
    n = len(a)
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "prob_a_greater": float("nan")}

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()

    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_a_greater": float((diffs > 0).mean()),
    }


def paired_bootstrap_report(metric_name, method_a_name, values_a, method_b_name, values_b, n_bootstrap=5000, seed=42):
    """`paired_bootstrap_metric_difference`, wrapped into one printable/tabulable row --
    the shape section 13/59's comparison table wants."""
    result = paired_bootstrap_metric_difference(values_a, values_b, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "metric": metric_name, "method_a": method_a_name, "method_b": method_b_name,
        "mean_a": float(np.mean(values_a)) if len(values_a) else float("nan"),
        "mean_b": float(np.mean(values_b)) if len(values_b) else float("nan"),
        "mean_diff": result["mean_diff"], "ci_low": result["ci_low"], "ci_high": result["ci_high"],
        "prob_a_greater": result["prob_a_greater"],
        "significant_at_95": bool(result["ci_low"] > 0 or result["ci_high"] < 0),
    }
