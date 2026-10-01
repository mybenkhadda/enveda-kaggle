"""v4b LambdaMART fold training / scoring.

Contract:
  * one LightGBM ranking group == one query; rows of a query are contiguous and never split;
  * fold models are trained on TRAINING-SET queries whose connectivity fold != f and score the
    EVALUATION queries of fold f (DEV OOF) -- training and evaluation tables may differ (reference
    dropout, 3k/10k training sets, source/structure filters) but the evaluation set is fixed;
  * HOST queries are scored by the model that HELD OUT their connectivity fold (the V0 convention
    carried forward), so no HOST structure is ever a training positive of the model scoring it;
  * zero-candidate queries simply have no rows -- they never create an empty group, and they are
    added back with RR=0 by `casmi.ranking.rank_eval.per_query_metrics`;
  * identical parameters, seed and objective for every variant (spec: no retuning per variant).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LGBM_PARAMS = dict(objective="lambdarank", metric="ndcg", n_estimators=300, learning_rate=0.05,
                   num_leaves=31, min_child_samples=20, random_state=42, verbosity=-1)

BASE_FEATURES = ["abs_mass_error_ppm", "cosine_max", "cosine_top3_mean", "modified_cosine_max",
                 "modified_cosine_top3_mean", "peak_overlap_frac_max", "peak_overlap_frac_top3_mean",
                 "neutral_loss_cosine_max", "neutral_loss_cosine_top3_mean"]
MASS_LIK_FEATURES = BASE_FEATURES + ["mass_loglik"]

# never allowed into a main-model feature list (popularity / availability / provenance)
FORBIDDEN_MODEL_FEATURES = ("n_reference_spectra", "has_reference_spectrum", "eligible_reference_count",
                            "eligible_reference_saturated", "eligible_reference_count_is_exact", "popularity_level",
                            "source", "query_source", "is_true_candidate", "fold")


def default_model_factory(params=None):
    import lightgbm as lgb
    return lgb.LGBMRanker(**(params or LGBM_PARAMS))


def assert_feature_list_allowed(feature_cols):
    bad = sorted(set(feature_cols) & set(FORBIDDEN_MODEL_FEATURES))
    assert not bad, f"forbidden model features: {bad}"


def make_groups(df, query_col="query_id", group_col=None):
    """Sort by the group column (default: `query_col`) and return `(sorted_df, group_sizes)`.
    Asserts every group is contiguous and non-empty. `group_col` lets reference-dropout replicas
    of the same query form separate groups."""
    gcol = group_col or query_col
    d = df.sort_values([gcol], kind="mergesort").reset_index(drop=True)
    sizes = d.groupby(gcol, sort=False).size()
    assert (sizes > 0).all(), "empty ranking group"
    # contiguity: after the stable sort each group id appears in exactly one run
    runs = (d[gcol] != d[gcol].shift()).sum()
    assert runs == len(sizes), "ranking group rows are not contiguous"
    return d, sizes.to_numpy()


def fit_one(train_df, feature_cols, model_factory=None, params=None, label_col="is_true_candidate", group_col=None):
    assert_feature_list_allowed(feature_cols)
    d, groups = make_groups(train_df, group_col=group_col)
    model = (model_factory or default_model_factory)(params)
    model.fit(d[feature_cols], d[label_col].astype(int), group=groups)
    return model


def fit_fold_models(train_df, feature_cols, fold_col="fold", folds=None, model_factory=None, params=None,
                    group_col=None, eval_query_ids_by_fold=None):
    """`{fold: model}`; model f trained on `train_df[fold != f]`. If `eval_query_ids_by_fold` is
    given, asserts that none of fold f's evaluation queries (or their rows) leak into model f's
    training rows."""
    folds = sorted(train_df[fold_col].dropna().unique()) if folds is None else list(folds)
    models, audit = {}, []
    for f in folds:
        tr = train_df[train_df[fold_col] != f]
        if eval_query_ids_by_fold is not None:
            leaked = set(tr["query_id"]) & set(eval_query_ids_by_fold.get(f, ()))
            assert not leaked, f"fold {f}: {len(leaked)} evaluation queries present in training rows"
        models[f] = fit_one(tr, feature_cols, model_factory=model_factory, params=params, group_col=group_col)
        audit.append({"fold": f, "n_train_rows": int(len(tr)), "n_train_queries": int(tr["query_id"].nunique()),
                      "n_train_groups": int(tr[group_col or "query_id"].nunique())})
    return models, pd.DataFrame(audit)


def predict_by_fold(models, df, feature_cols, fold_col="fold"):
    """Score each row with `models[row fold]` (OOF for DEV; fold-matched for HOST). Rows whose
    fold has no model get NaN (and are reported by the caller, never silently ranked)."""
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    for f, model in models.items():
        mask = df[fold_col] == f
        if mask.any():
            scores.loc[mask] = model.predict(df.loc[mask, feature_cols])
    return scores


def connectivity_disjointness(train_queries, eval_queries, fold_col="fold", key_col="true_connectivity_key"):
    """Per fold: overlap between model f's training truth connectivities and fold f's evaluation
    truth connectivities -- must be 0 everywhere."""
    rows = []
    for f in sorted(eval_queries[fold_col].dropna().unique()):
        tr = set(train_queries.loc[train_queries[fold_col] != f, key_col])
        ev = set(eval_queries.loc[eval_queries[fold_col] == f, key_col])
        rows.append({"fold": f, "n_train_keys": len(tr), "n_eval_keys": len(ev), "n_overlap": len(tr & ev)})
    return pd.DataFrame(rows)


def save_fold_models(models, out_dir, feature_cols, params, training_query_ids, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f, m in models.items():
        m.booster_.save_model(str(out_dir / f"fold_{f}.txt"))
    ids = sorted(str(q) for q in training_query_ids)
    meta = {"folds": [int(f) if isinstance(f, (int, np.integer)) else f for f in models],
            "feature_cols": list(feature_cols), "params": params,
            "n_training_queries": len(ids),
            "training_query_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            **(extra or {})}
    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return out_dir


def load_fold_models(out_dir):
    """Boosters (not sklearn wrappers) keyed by fold, plus the saved metadata. A Booster's
    `.predict(frame)` has the same signature `predict_by_fold` uses."""
    import lightgbm as lgb
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))
    models = {f: lgb.Booster(model_file=str(out_dir / f"fold_{f}.txt")) for f in meta["folds"]}
    return models, meta
