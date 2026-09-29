"""v5 training-scale experiment: common-population fold-mean scoring, the pre-registered scale
decision / headline rules, and the density-reweighted HOST estimate.

TL_EVAL selects, HOST confirms. Every scale model scores the SAME TL_EVAL and HOST queries with
the MEAN of its 5 fold models (fair size comparison -- never each model's own OOF population).
The rules below are written to `outputs/v5/preregistration_v5.json` (via
`casmi.ranking.selection.write_preregistration`) before any metric is displayed.
"""
import numpy as np
import pandas as pd

from casmi.ranking.rank_eval import per_query_metrics, rank_by_keys
from casmi.ranking.selection import HostLeakError, guard_no_host

KEY = "candidate_connectivity_key"
SCALE_MODELS = ("V1_RND_1K", "V1_TL_1K", "V1_TL_3K", "V1_TL_10K")
HEADLINE_CANDIDATES = ("V1_TL_1K", "V1_TL_3K", "V1_TL_10K")
MODEL_MANIFEST = {"V1_RND_1K": "RND_1K", "V1_TL_1K": "TL_1K", "V1_TL_3K": "TL_3K", "V1_TL_10K": "TL_10K"}
NESTED_ORDER = ("V1_TL_1K", "V1_TL_3K", "V1_TL_10K")
PAIRED_COMPARISONS = (("V1_TL_3K", "V1_TL_1K"), ("V1_TL_10K", "V1_TL_3K"), ("V1_TL_10K", "V1_TL_1K"), ("V1_TL_1K", "V1_RND_1K"))

V5_RULE = {
    "rule_id": "v5-scale-1",
    "selection_population": "TL_EVAL (all queries; zero-candidate / truth-absent queries contribute 0)",
    "confirmation_population": "HOST (descriptive only; never enters selection)",
    "training_recipe": "exact v4b V1 recipe: BASE_FEATURES order, LGBM_PARAMS, lambdarank, seed 42, one group per query, 5 connectivity folds, mirror_aware",
    "scoring": "mean of the 5 fold-model predictions on the common population; ties: score DESC, abs_mass_error_ppm ASC, connectivity_key ASC",
    "scale_decision": ("largest available nested pair (TL_10K vs TL_3K, else TL_3K vs TL_1K): CONTINUE_SCALING iff "
                       "delta TL-EVAL MRR@25 > 0 AND its 95% paired connectivity-bootstrap CI excludes 0; otherwise PLATEAU"),
    "headline_rule": ("among V1_TL_1K / V1_TL_3K / V1_TL_10K: highest TL-EVAL MRR@25; every model within 0.005 of the best "
                      "is a contender and the SMALLEST training set among contenders wins"),
    "headline_margin": 0.005,
    "rnd_role": "V1_RND_1K is a composition control, never headline",
    "b5_role": "frozen v4b B5 (DEV-selected alpha, DEV standardizer, v4b mass model) scored on TL_EVAL; nothing refit on TL_EVAL",
    "density_reweighting": "HOST per-pool-size-bin MRR weighted by the visible TEST pool-size bin proportions; label DENSITY-REWEIGHTED CLASS-1 ESTIMATE; bins with n_host < 20 flagged",
    "bootstrap": "connectivity-cluster bootstrap, n_boot=2000, seed=42; paired deltas share the SAME cluster draws (same seed, same cluster set)",
}

HEADLINE_ALLOWED_COLUMNS = ("model_id", "tl_eval_mrr", "n_train_queries", "status")


def fold_mean_scores(models, df, feature_cols):
    """Mean prediction of every fold model (Booster or sklearn ranker) on the SAME rows."""
    if not len(df):
        return np.zeros(0)
    X = df[list(feature_cols)]
    preds = np.column_stack([np.asarray(m.predict(X), dtype=float) for m in models.values()])
    return preds.mean(axis=1)


def score_population(models, df, feature_cols, all_query_ids):
    """Per-query metrics of fold-mean scores with the canonical tie rule."""
    scored = df.assign(score=fold_mean_scores(models, df, feature_cols))
    ranked = rank_by_keys(scored, [("score", False), ("abs_mass_error_ppm", True), (KEY, True)])
    return per_query_metrics(ranked, all_query_ids), ranked


def scale_decision(paired_rows, available, nested_order=NESTED_ORDER):
    """`paired_rows`: {(a, b): paired_cluster_bootstrap result dict}; `available`: set of trained
    nested models. Applies the pre-registered rule to the largest available nested pair."""
    nested = [m for m in nested_order if m in available]
    if len(nested) < 2:
        return {"scaling_status": "INSUFFICIENT_MODELS", "pair": None, "reason": f"only {nested} available"}
    pair = (nested[-1], nested[-2])
    r = paired_rows.get(pair)
    if r is None:
        raise KeyError(f"paired comparison {pair} missing")
    cont = bool(r["delta"] > 0 and r["ci_low"] > 0)
    return {"scaling_status": "CONTINUE_SCALING" if cont else "PLATEAU", "pair": list(pair), "delta": r["delta"],
            "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "reason": "delta > 0 and CI excludes 0" if cont else "delta <= 0 or CI includes 0"}


def select_scale_headline(tl_eval_table, margin=V5_RULE["headline_margin"], candidates=HEADLINE_CANDIDATES):
    """TL-EVAL-only headline rule. Input must contain only `HEADLINE_ALLOWED_COLUMNS`; any HOST-
    named or unexpected column raises `HostLeakError`. Returns `(model_id, ranked_table)`."""
    guard_no_host(tl_eval_table, "select_scale_headline")
    extra = sorted(set(tl_eval_table.columns) - set(HEADLINE_ALLOWED_COLUMNS))
    if extra:
        raise HostLeakError(f"select_scale_headline: unexpected input columns {extra}")
    t = tl_eval_table[tl_eval_table["model_id"].isin(candidates)].copy()
    if "status" in t:
        t = t[t["status"] == "OK"]
    if t.empty:
        raise RuntimeError("no trained headline candidate")
    best = t["tl_eval_mrr"].max()
    t["contender"] = t["tl_eval_mrr"] >= best - margin
    t = t.sort_values(["contender", "n_train_queries", "tl_eval_mrr", "model_id"], ascending=[False, True, False, True], kind="mergesort")
    t["selection_order"] = np.arange(1, len(t) + 1)
    return str(t.iloc[0]["model_id"]), t.reset_index(drop=True)


POOL_BIN_EDGES = (0, 50, 100, 200, 400, 800, np.inf)


def pool_bin(sizes, edges=POOL_BIN_EDGES):
    labels = [f"{int(a)}-{int(b) - 1}" if np.isfinite(b) else f">={int(a)}" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(pd.Series(sizes, dtype=float), list(edges), right=False, labels=labels).astype(str).to_numpy(), labels


def pool_size_stratification(per_query, edges=POOL_BIN_EDGES, low_n=20):
    b, labels = pool_bin(per_query["n_candidates"].to_numpy(), edges)
    d = per_query.assign(pool_bin=b)
    t = d.groupby("pool_bin").agg(n_queries=("rr", "size"), mrr=("rr", "mean"), hit_at_1=("hit_at_1", "mean"),
                                  median_pool=("n_candidates", "median")).reindex(labels).reset_index()
    t["n_queries"] = t["n_queries"].fillna(0).astype(int)
    t["low_n"] = t["n_queries"] < low_n
    return t


def density_reweighted_mrr(host_per_query, test_pool_sizes, edges=POOL_BIN_EDGES, low_n=20):
    """DENSITY-REWEIGHTED CLASS-1 ESTIMATE: sum_b w_test(b) * MRR_host(b). A test bin with no HOST
    queries cannot be estimated -- its weight is reported as uncovered and the estimate is
    renormalized over covered bins (and flagged). Not a leaderboard prediction."""
    hb, labels = pool_bin(host_per_query["n_candidates"].to_numpy(), edges)
    tb, _ = pool_bin(np.asarray(test_pool_sizes, dtype=float), edges)
    host = host_per_query.assign(pool_bin=hb).groupby("pool_bin")["rr"].agg(["size", "mean"]).reindex(labels)
    w_test = pd.Series(tb).value_counts(normalize=True).reindex(labels).fillna(0.0)
    tab = pd.DataFrame({"pool_bin": labels, "test_weight": w_test.to_numpy(), "n_host": host["size"].fillna(0).astype(int).to_numpy(),
                        "host_mrr": host["mean"].to_numpy()})
    tab["low_n_host"] = tab["n_host"] < low_n
    tab["uncovered"] = (tab["n_host"] == 0) & (tab["test_weight"] > 0)
    covered = ~tab["uncovered"] & (tab["test_weight"] > 0)
    cov_w = float(tab.loc[covered, "test_weight"].sum())
    est = float((tab.loc[covered, "test_weight"] * tab.loc[covered, "host_mrr"]).sum() / cov_w) if cov_w > 0 else float("nan")
    return {"label": "DENSITY-REWEIGHTED CLASS-1 ESTIMATE", "estimate": est, "raw_host_mrr": float(host_per_query["rr"].mean()),
            "test_weight_covered": cov_w, "test_weight_uncovered": float(tab.loc[tab["uncovered"], "test_weight"].sum()),
            "test_weight_in_low_n_bins": float(tab.loc[covered & tab["low_n_host"], "test_weight"].sum()),
            "caveat": "HOST bins with n_host < 20 are extrapolations; uncovered test weight is excluded and renormalized; not a leaderboard prediction",
            "table": tab}


# ---------------------------------------------------------------------------------------------
# v5.1: regime-safe Mode-A scaling (pre-registered here, BEFORE any v5.1 execution)
# ---------------------------------------------------------------------------------------------

MODE_A_SCALE_MODELS = ("V1_RND_1K_MODE_A", "V1_TL_1K_MODE_A", "V1_TL_3K_MODE_A", "V1_TL_10K_MODE_A")
MODE_A_HEADLINE_CANDIDATES = ("V1_TL_1K_MODE_A", "V1_TL_3K_MODE_A", "V1_TL_10K_MODE_A")
MODE_A_NESTED_ORDER = MODE_A_HEADLINE_CANDIDATES
MODE_A_MODEL_MANIFEST = {"V1_RND_1K_MODE_A": "RND_1K_MODE_A", "V1_TL_1K_MODE_A": "TL_1K_MODE_A",
                         "V1_TL_3K_MODE_A": "TL_3K_MODE_A", "V1_TL_10K_MODE_A": "TL_10K_MODE_A"}
MODE_A_PAIRED_COMPARISONS = (("V1_TL_3K_MODE_A", "V1_TL_1K_MODE_A"), ("V1_TL_10K_MODE_A", "V1_TL_3K_MODE_A"),
                             ("V1_TL_10K_MODE_A", "V1_TL_1K_MODE_A"), ("V1_TL_1K_MODE_A", "V1_RND_1K_MODE_A"))
EVAL_POPULATIONS = ("TL_EVAL_MODE_A", "TL_EVAL_ALL", "TL_EVAL_STANDARD_ONLY", "TL_EVAL_REF_ABSENT", "TL_EVAL_POOL_ABSENT")

V51_RULE = {
    "rule_id": "v5.1-mode-a-scale-1",
    "training_population": "*_MODE_A manifests: original TL_1K/TL_3K/TL_10K/RND_1K queries whose truth is in the pool with >=1 eligible mirror_aware reference (derived after QCR; originals unchanged; nesting asserted)",
    "selection_population": "TL_EVAL_MODE_A (TL_EVAL queries with regime MODE_A_MIRROR)",
    "stress_population": "TL_EVAL_ALL (every TL_EVAL query; zero-candidate / truth-absent / reference-absent queries count)",
    "diagnostic_populations": ["TL_EVAL_STANDARD_ONLY", "TL_EVAL_REF_ABSENT", "TL_EVAL_POOL_ABSENT"],
    "confirmation_population": "HOST (never enters selection)",
    "training_recipe": "exact v4b V1 recipe (BASE_FEATURES order, LGBM_PARAMS, lambdarank, seed 42, one group per query, 5 connectivity folds, mirror_aware); no reference-count / availability / source / regime feature",
    "scoring": "mean of the 5 fold predictions; ties score DESC, abs_mass_error_ppm ASC, connectivity_key ASC",
    "scale_decision": ("delta = TL-EVAL_MODE_A MRR@25 of V1_TL_10K_MODE_A minus V1_TL_3K_MODE_A (else largest available nested pair); "
                       "CONTINUE_SCALING iff delta > 0 AND 95% paired connectivity-cluster bootstrap CI excludes 0; else PLATEAU"),
    "headline_rule": "among V1_TL_{1K,3K,10K}_MODE_A: highest TL_EVAL_MODE_A MRR@25; within 0.005 of the best -> the smallest training set",
    "headline_margin": 0.005,
    "freeze_report": "TL_EVAL_MODE_A, TL_EVAL_ALL, TL_EVAL_STANDARD_ONLY, TL_EVAL_REF_ABSENT, HOST and density-reweighted HOST MRR side by side",
    "k_stress": "MODE_A_MIRROR queries only; REF_ABSENT k-stress is NOT_APPLICABLE (true reference count is zero)",
    "bootstrap": "connectivity-cluster bootstrap, n_boot=2000, seed=42; paired deltas share cluster draws",
}

MODE_A_ALLOWED_COLUMNS = ("model_id", "tl_eval_mode_a_mrr", "n_train_queries", "status")


def select_mode_a_headline(table, margin=V51_RULE["headline_margin"], candidates=MODE_A_HEADLINE_CANDIDATES):
    """Mode-A headline from TL_EVAL_MODE_A ONLY. Any other metric column -- HOST, ALL, REF_ABSENT,
    STANDARD_ONLY, anything -- raises `HostLeakError` (it is the wrong selection population)."""
    guard_no_host(table, "select_mode_a_headline")
    extra = sorted(set(table.columns) - set(MODE_A_ALLOWED_COLUMNS))
    if extra:
        raise HostLeakError(f"select_mode_a_headline: only TL_EVAL_MODE_A metrics may select; unexpected columns {extra}")
    t = table.rename(columns={"tl_eval_mode_a_mrr": "tl_eval_mrr"})
    t = t[t["model_id"].isin(candidates)]
    if "status" in t:
        t = t[t["status"] == "OK"]
    if t.empty:
        raise RuntimeError("no trained Mode-A headline candidate")
    best = t["tl_eval_mrr"].max()
    t = t.assign(contender=t["tl_eval_mrr"] >= best - margin)
    t = t.sort_values(["contender", "n_train_queries", "tl_eval_mrr", "model_id"], ascending=[False, True, False, True], kind="mergesort")
    t["selection_order"] = np.arange(1, len(t) + 1)
    return str(t.iloc[0]["model_id"]), t.rename(columns={"tl_eval_mrr": "tl_eval_mode_a_mrr"}).reset_index(drop=True)


def population_ids(regimes, population):
    """Query ids of an evaluation population (TL_EVAL_ALL = every regime row)."""
    if population == "TL_EVAL_ALL":
        return regimes["query_id"].tolist()
    reg = population.replace("TL_EVAL_", "")
    reg = {"MODE_A": "MODE_A_MIRROR"}.get(reg, reg)
    return regimes.loc[regimes["regime"] == reg, "query_id"].tolist()


# ---------------------------------------------------------------------------------------------
# v5.2: protocol-parameterized scale configuration (one pipeline, protocol as configuration)
# ---------------------------------------------------------------------------------------------

PROTOCOL_MANIFEST_SUFFIX = {"mirror_aware": "_MODE_A", "test_simulated_strict": "_TESTSIM_STRICT", "test_simulated_relaxed": "_TESTSIM_RELAXED"}


def protocol_scale_config(protocol):
    """Model ids, manifests, nested order and paired comparisons for one protocol. mirror_aware
    reproduces the v5.1 constants exactly."""
    sfx = PROTOCOL_MANIFEST_SUFFIX[protocol]
    manifests = {f"V1_{b}{sfx}": f"{b}{sfx}" for b in ("RND_1K", "TL_1K", "TL_3K", "TL_10K")}
    heads = tuple(f"V1_{b}{sfx}" for b in ("TL_1K", "TL_3K", "TL_10K"))
    pairs = ((heads[1], heads[0]), (heads[2], heads[1]), (heads[2], heads[0]), (heads[0], f"V1_RND_1K{sfx}"))
    return {"protocol": protocol, "suffix": sfx, "model_manifest": manifests, "scale_models": tuple(manifests), "headline_candidates": heads,
            "nested_order": heads, "paired_comparisons": pairs, "selection_population": f"TL_EVAL queries with >=1 truth reference under {protocol}"}


# ---------------------------------------------------------------------------------------------
# v5.3: test_simulated_strict scaling + final spectrum freeze (pre-registered here, before execution)
# ---------------------------------------------------------------------------------------------

V53_PROTOCOL = "test_simulated_strict"
V53_MODELS = {"V1_TL_1K_TESTSIM_STRICT": "TL_1K_TESTSIM_STRICT", "V1_TL_3K_TESTSIM_STRICT": "TL_3K_TESTSIM_STRICT",
              "V1_TL_10K_TESTSIM_STRICT": "TL_10K_TESTSIM_STRICT"}
V53_NESTED_ORDER = tuple(V53_MODELS)
V53_PAIRS = (("V1_TL_3K_TESTSIM_STRICT", "V1_TL_1K_TESTSIM_STRICT"), ("V1_TL_10K_TESTSIM_STRICT", "V1_TL_3K_TESTSIM_STRICT"),
             ("V1_TL_10K_TESTSIM_STRICT", "V1_TL_1K_TESTSIM_STRICT"))
V53_RULE = {
    "rule_id": "v5.3-testsim-strict-scale-1",
    "protocol": V53_PROTOCOL,
    "training": "TL_{1K,3K,10K}_TESTSIM_STRICT: original nested manifests filtered to truth-in-pool with >=1 eligible test_simulated_strict reference (nesting asserted; originals immutable)",
    "recipe": "exact V1 (BASE_FEATURES order, LGBM_PARAMS, lambdarank, seed 42, one group per query, 5 connectivity folds); only the training-set size changes",
    "selection_population": "full fixed TL_EVAL (every query; truth absent / no evidence -> RR 0) with test_simulated_strict evidence",
    "scoring": "mean of the 5 fold predictions; ties score DESC, abs_mass_error_ppm ASC, connectivity_key ASC",
    "scale_decision": ("delta = TL_EVAL MRR@25 of V1_TL_10K_TESTSIM_STRICT minus V1_TL_3K_TESTSIM_STRICT; CONTINUE_SCALING iff delta > 0 "
                       "AND its 95% paired connectivity-cluster bootstrap CI excludes 0; else PLATEAU"),
    "headline_rule": "highest TL_EVAL MRR@25 among 1K/3K/10K; any within 0.005 of the best -> the smallest training set; locked before HOST",
    "headline_margin": 0.005,
    "bootstrap": "connectivity-cluster, n_boot=2000, seed=42; paired deltas share the same cluster resamples",
    "host_role": "confirmation only (test_simulated_strict primary; mirror_aware / standard HOST labelled sensitivities)",
    "sensitivities": {"standard": "OPTIMISTIC / DUPLICATE-PERMITTING SENSITIVITY", "mirror_aware": "conservative cross-library sensitivity"},
}
