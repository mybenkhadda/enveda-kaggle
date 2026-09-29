"""Connectivity-level (cluster) bootstrap -- the unit resampled is the MOLECULE (true
connectivity key), not the spectrum/query row: spectra of one molecule are correlated, so a
row-level bootstrap understates the uncertainty (see the v4a notes: the earlier CI was
spectrum-level). Paired version for model comparisons: both models are evaluated on the same
queries, and each bootstrap replicate resamples connectivities with replacement and keeps ALL
queries of every drawn connectivity for BOTH models.
"""
import numpy as np
import pandas as pd


def _cluster_sums(values, clusters):
    codes, uniq = pd.factorize(pd.Series(clusters), sort=True)
    if (codes < 0).any():
        raise ValueError("cluster labels contain nulls")
    n_c = len(uniq)
    sums = np.bincount(codes, weights=np.asarray(values, dtype=float), minlength=n_c)
    counts = np.bincount(codes, minlength=n_c).astype(float)
    return sums, counts, n_c


def _replicate_weights(n_clusters, n_boot, rng, batch=500):
    for start in range(0, n_boot, batch):
        b = min(batch, n_boot - start)
        draws = rng.integers(0, n_clusters, size=(b, n_clusters))
        w = np.zeros((b, n_clusters))
        rows = np.repeat(np.arange(b), n_clusters)
        np.add.at(w, (rows, draws.ravel()), 1.0)
        yield w


def cluster_bootstrap_mean(values, clusters, n_boot=2000, seed=42, ci=0.95):
    """Mean of per-query `values` with a connectivity-cluster percentile CI."""
    values = np.asarray(values, dtype=float)
    sums, counts, n_c = _cluster_sums(values, clusters)
    rng = np.random.default_rng(seed)
    reps = np.concatenate([(w @ sums) / (w @ counts) for w in _replicate_weights(n_c, n_boot, rng)])
    lo, hi = np.quantile(reps, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_queries": int(len(values)),
            "n_clusters": int(n_c), "n_boot": int(n_boot)}


def paired_cluster_bootstrap(values_a, values_b, clusters, n_boot=2000, seed=42, ci=0.95):
    """Delta = mean(a) - mean(b) over the same queries, CI from resampling connectivities."""
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if len(a) != len(b) or len(a) != len(clusters):
        raise ValueError("values_a, values_b and clusters must be aligned and equal length")
    sums_d, counts, n_c = _cluster_sums(a - b, clusters)
    rng = np.random.default_rng(seed)
    reps = np.concatenate([(w @ sums_d) / (w @ counts) for w in _replicate_weights(n_c, n_boot, rng)])
    lo, hi = np.quantile(reps, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(a.mean() - b.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "prob_delta_gt_0": float((reps > 0).mean()),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_queries": int(len(a)), "n_clusters": int(n_c), "n_boot": int(n_boot)}


def paired_comparison_row(name_a, pq_a, name_b, pq_b, cluster_of_query, metric="rr", **kw):
    """Align two per-query metric frames on `query_id` (outer-join is an error: both must cover
    exactly the same queries) and run the paired cluster bootstrap."""
    a = pq_a.set_index("query_id")[metric]
    b = pq_b.set_index("query_id")[metric]
    if set(a.index) != set(b.index):
        raise ValueError(f"{name_a} and {name_b} were evaluated on different query sets")
    b = b.reindex(a.index)
    clusters = a.index.map(cluster_of_query)
    if pd.isna(clusters).any():
        raise ValueError("some queries have no connectivity cluster")
    res = paired_cluster_bootstrap(a.to_numpy(), b.to_numpy(), np.asarray(clusters), **kw)
    return {"comparison": f"{name_a} vs {name_b}", "model_a": name_a, "model_b": name_b, "metric": metric, **res}
