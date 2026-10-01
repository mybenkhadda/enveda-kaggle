"""Plain, readable plots for the v2 regimes / recall / ranking tables. Each function takes the tables
produced by `casmi.validation.reporting` / `casmi.candidates.recall` and returns the matplotlib Figure
(save with `fig.savefig`). No styling beyond what readability needs."""
import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def metric_by_regime(summary, metric="mrr_at_25", title=None):
    """Bar chart of one metric per regime (`regime_summary` output, ALL row included)."""
    plt = _plt()
    d = summary.dropna(subset=[metric])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(d["regime"].astype(str), d[metric])
    for x, v, n in zip(range(len(d)), d[metric], d["n_queries"]):
        ax.text(x, v, f"{v:.3f}\n(n={int(n)})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(1.0, float(d[metric].max()) * 1.15) if len(d) else 1)
    ax.set_title(title or f"{metric} by regime")
    fig.tight_layout()
    return fig


def recall_vs_ppm(recall_table, metrics=("recall_all", "recall_at_500", "recall_at_100", "recall_at_25"), group_col=None, title=None):
    """Recall curves over the ppm grid (`recall_by_ppm` output); one line per metric (and per group)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    groups = [(None, recall_table)] if group_col is None else list(recall_table.groupby(group_col))
    for g, d in groups:
        d = d.sort_values("ppm")
        for m in metrics:
            if m in d:
                ax.plot(d["ppm"], d[m], marker="o", label=m if g is None else f"{g}: {m}")
    ax.set_xscale("log")
    ax.set_xticks(sorted(recall_table["ppm"].unique()))
    from matplotlib.ticker import ScalarFormatter
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("mass tolerance (ppm)")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title(title or "candidate recall vs ppm")
    fig.tight_layout()
    return fig


def pool_size_distribution(sweep, ppm_values=None, title=None):
    """Histogram (log x) of candidate-pool size per query, one series per ppm."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for ppm in (ppm_values or sorted(sweep["ppm"].unique())):
        s = sweep.loc[sweep["ppm"] == ppm, "pool_size"].to_numpy()
        ax.hist(np.log10(s + 1), bins=50, histtype="step", label=f"{ppm} ppm")
    ax.set_xlabel("log10(pool size + 1)")
    ax.set_ylabel("queries")
    ax.legend(fontsize=8)
    ax.set_title(title or "candidate pool size")
    fig.tight_layout()
    return fig


def truth_rank_distribution(per_query, rank_col="truth_rank", by="regime", max_rank=100, title=None):
    """Truth-rank histogram (ranks > max_rank and misses shown as separate bars)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    groups = [("ALL", per_query)] if by is None or by not in per_query else list(per_query.groupby(by))
    bins = np.arange(1, max_rank + 2)
    for g, d in groups:
        r = d[rank_col].to_numpy(float)
        ax.hist(np.clip(r[np.isfinite(r)], 1, max_rank + 1), bins=bins, histtype="step", label=f"{g} (miss {np.mean(~np.isfinite(r)):.1%})")
    ax.set_xlabel(f"truth rank (>{max_rank} clipped)")
    ax.set_ylabel("queries")
    ax.legend(fontsize=8)
    ax.set_title(title or "truth rank")
    fig.tight_layout()
    return fig


def score_margins(margins, title=None):
    """Histogram of (truth score - best false score) per query."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    m = np.asarray(margins, float)
    ax.hist(m[np.isfinite(m)], bins=60)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("truth score - best false score")
    ax.set_ylabel("queries")
    ax.set_title(title or "score margins")
    fig.tight_layout()
    return fig


def channel_ablation(ablation, metric="mrr_at_25", regime_col="regime", variant_col="variant", title=None):
    """Grouped bars: one group per variant (channel set), one bar per regime."""
    plt = _plt()
    variants = list(dict.fromkeys(ablation[variant_col]))
    regimes = sorted(ablation[regime_col].unique())
    x = np.arange(len(variants))
    w = 0.8 / max(len(regimes), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(variants) * 1.2), 4))
    for i, r in enumerate(regimes):
        d = ablation[ablation[regime_col] == r].set_index(variant_col).reindex(variants)
        ax.bar(x + i * w, d[metric], width=w, label=r)
    ax.set_xticks(x + w * (len(regimes) - 1) / 2)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.set_title(title or f"channel ablation ({metric})")
    fig.tight_layout()
    return fig


def same_formula_errors(table, title=None):
    """Bar chart of error categories (counts) -- e.g. the isomer-panel failure table."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.barh(table["category"].astype(str), table["n"])
    ax.set_xlabel("queries")
    ax.set_title(title or "error categories")
    fig.tight_layout()
    return fig


def validation_vs_leaderboard(points, x="C2_MRR25", y="leaderboard", label="experiment_id", title=None):
    """Scatter of a validation metric vs leaderboard score per experiment (only rows with both values)."""
    plt = _plt()
    d = points.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(d[x], d[y])
    for _, r in d.iterrows():
        ax.annotate(str(r[label]), (r[x], r[y]), fontsize=7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or "validation vs leaderboard")
    fig.tight_layout()
    return fig
