"""Per-spectrum temperature calibration of ranker scores (11_02).

p(candidate | spectrum) = softmax(score / T) within the spectrum's candidate pool. T is fitted by
minimizing the mean negative log probability of the TRUE candidate over spectra whose truth is in
the pool, with log(T) bounded to [-3, 3] (golden-section search -- deterministic, no SciPy).
Fit data must be DEV / TL-like OOF predictions; HOST is refused by the caller's guard.
Calibration quality (NLL, Brier, ECE, reliability) is DIAGNOSTIC; molecule MRR selects the aggregator.
"""
import numpy as np
import pandas as pd

from casmi_infer.aggregation import softmax_by_group

LOG_T_BOUNDS = (-3.0, 3.0)


def truth_nll(scores, groups, is_true, temperature):
    p = softmax_by_group(scores, groups, temperature)
    t = np.asarray(is_true, dtype=bool)
    return float(-np.mean(np.log(np.clip(p[t], 1e-300, None)))) if t.any() else float("nan")


def fit_temperature(df, score_col="score", group_col="query_id", truth_col="is_true_candidate", bounds=LOG_T_BOUNDS, tol=1e-6, max_iter=200):
    """Golden-section search on log T. Only spectra with their truth in the pool contribute."""
    has_truth = df.groupby(group_col)[truth_col].transform("any")
    d = df[has_truth]
    s, g, y = d[score_col].to_numpy(float), d[group_col].to_numpy(), d[truth_col].to_numpy(bool)
    f = lambda lt: truth_nll(s, g, y, np.exp(lt))
    a, b = bounds
    gr = (np.sqrt(5) - 1) / 2
    c, e = b - gr * (b - a), a + gr * (b - a)
    fc, fe = f(c), f(e)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, e, fe
            e = a + gr * (b - a)
            fe = f(e)
    log_t = (a + b) / 2
    return {"temperature": float(np.exp(log_t)), "log_temperature": float(log_t), "nll": f(log_t), "nll_at_T1": f(0.0),
            "n_spectra": int(d[group_col].nunique()), "at_bound": bool(min(abs(log_t - bounds[0]), abs(log_t - bounds[1])) < 1e-3)}


def calibration_report(df, temperature, score_col="score", group_col="query_id", truth_col="is_true_candidate", n_bins=10):
    """NLL, multi-class Brier (per spectrum, sum over its candidates), top-1 ECE and reliability table."""
    d = df.assign(prob=softmax_by_group(df[score_col], df[group_col], temperature))
    y = d[truth_col].astype(float)
    brier = float(((d["prob"] - y) ** 2).groupby(d[group_col]).sum().mean())
    has_truth = d.groupby(group_col)[truth_col].transform("any")
    nll = float(-np.mean(np.log(np.clip(d.loc[d[truth_col].astype(bool) & has_truth, "prob"], 1e-300, None))))
    top = d.sort_values([group_col, "prob"], ascending=[True, False]).drop_duplicates(group_col)
    conf, acc = top["prob"].to_numpy(), top[truth_col].astype(float).to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rel = pd.DataFrame({"bin": b, "conf": conf, "acc": acc}).groupby("bin").agg(n=("conf", "size"), mean_confidence=("conf", "mean"),
                                                                                 accuracy=("acc", "mean")).reindex(range(n_bins)).reset_index()
    rel["bin_lo"], rel["bin_hi"] = edges[:-1], edges[1:]
    rel["n"] = rel["n"].fillna(0).astype(int)
    ece = float(np.nansum(rel["n"] / max(len(top), 1) * np.abs(rel["accuracy"] - rel["mean_confidence"])))
    brier_top1 = float(np.mean((conf - acc) ** 2)) if len(top) else float("nan")      # v6: (top-1 confidence - top-1 correct)^2
    return {"temperature": temperature, "nll": nll, "brier": brier, "brier_top1": brier_top1, "ece_top1": ece, "n_spectra": int(len(top))}, rel
