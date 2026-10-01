"""Mode-B ("unseen-connectivity") shortcut audit.

Motivating hypothesis (see `casmi.spectra.library.unseen_connectivity_reference_ids`): under
unseen-connectivity CV, the TRUE candidate structurally has every spectral feature NaN --
its own connectivity's reference spectra are, by construction, all in the held-out fold. If a
large share of FALSE candidates share that exact all-NaN pattern (because THEY also happen to
belong to a held-out fold, e.g. another dev query's true structure reused as a mass-ambiguous
decoy), then "all spectral features missing" can act as a fold-membership fingerprint that a
flexible ranker exploits via LightGBM's native missing-value branching -- a leakage shortcut
invisible to `casmi.ranking.feature_policy`'s explicit-column governance, since that governance
only blocks `has_reference_spectrum`/`n_reference_spectra` as RAW features; it says nothing
about a model branching on the missingness of features it IS allowed to see.

Every function here takes plain pandas frames (or an already-trained, duck-typed `.predict`
model) and returns plain dicts/DataFrames -- no notebook-only globals -- so the same audit can
run from a notebook, a script, or a test.
"""
import numpy as np
import pandas as pd

from casmi.ranking.evaluation import rank_candidates, ranking_metrics
from casmi.ranking.feature_policy import SPECTRAL_RANKING_FEATURE_POLICY

# The 8 spectral-similarity features (everything in the shared model-feature policy except the
# mass feature) -- imported, not redefined, so this audit can never silently drift out of sync
# with what notebook 04/09 actually train on.
SPECTRAL_FEATURES = [f for f in SPECTRAL_RANKING_FEATURE_POLICY.model_features if f != "abs_mass_error_ppm"]


def summarize_feature_missingness(pair_df, spectral_features=SPECTRAL_FEATURES, query_id_col="query_id",
                                   is_true_col="is_true_candidate", n_folds=None, fold_tolerance=0.05):
    """Section 1.3. Returns `(augmented_df, summary)`:

    `augmented_df` -- `pair_df` plus `spectral_missing_count` / `all_spectral_nan` /
    `any_spectral_nan` columns.

    `summary` -- `P(all/any spectral NaN | true)` vs. `P(... | false)`, plus per-query
    `fraction_candidates_all_nan` (median/p25/p75/p90). When `n_folds` is given,
    `fold_membership_shortcut_flag` is True whenever the median per-query all-NaN fraction sits
    within `fold_tolerance` of `1 / n_folds` -- the signature of "no reference spectrum" mostly
    just meaning "same fold as the query", not "genuinely evidence-free"."""
    out = pair_df.copy()
    out["spectral_missing_count"] = out[spectral_features].isna().sum(axis=1)
    out["all_spectral_nan"] = out[spectral_features].isna().all(axis=1)
    out["any_spectral_nan"] = out[spectral_features].isna().any(axis=1)

    is_true = out[is_true_col].astype(bool)
    per_query_frac = out.groupby(query_id_col)["all_spectral_nan"].mean()

    summary = {
        "n_rows": len(out),
        "n_queries": int(out[query_id_col].nunique()),
        "p_all_nan_given_true": float(out.loc[is_true, "all_spectral_nan"].mean()) if is_true.any() else float("nan"),
        "p_all_nan_given_false": float(out.loc[~is_true, "all_spectral_nan"].mean()) if (~is_true).any() else float("nan"),
        "p_any_nan_given_true": float(out.loc[is_true, "any_spectral_nan"].mean()) if is_true.any() else float("nan"),
        "p_any_nan_given_false": float(out.loc[~is_true, "any_spectral_nan"].mean()) if (~is_true).any() else float("nan"),
        "per_query_all_nan_fraction_median": float(per_query_frac.median()) if len(per_query_frac) else float("nan"),
        "per_query_all_nan_fraction_p25": float(per_query_frac.quantile(0.25)) if len(per_query_frac) else float("nan"),
        "per_query_all_nan_fraction_p75": float(per_query_frac.quantile(0.75)) if len(per_query_frac) else float("nan"),
        "per_query_all_nan_fraction_p90": float(per_query_frac.quantile(0.90)) if len(per_query_frac) else float("nan"),
    }

    if n_folds:
        expected = 1.0 / n_folds
        summary["expected_fold_fraction"] = expected
        summary["fold_membership_shortcut_flag"] = bool(
            np.isfinite(summary["per_query_all_nan_fraction_median"])
            and abs(summary["per_query_all_nan_fraction_median"] - expected) <= fold_tolerance
        )
    else:
        summary["expected_fold_fraction"] = None
        summary["fold_membership_shortcut_flag"] = None

    return out, summary


def evaluate_missingness_shortcut(pair_df_with_missingness, all_query_ids=None, query_id_col="query_id",
                                   is_true_col="is_true_candidate", mass_col="abs_mass_error_ppm", k_values=(1, 5, 25)):
    """Section 1.4. Restrict candidates to the `all_spectral_nan` subset (the subset the true
    candidate always falls into) and rank it by mass alone (`-abs_mass_error_ppm`). If this
    restricted mass-only MRR explains a large share of a suspicious full Mode-B MRR, most of
    that MRR is "mass ranks well among fold-mates", not real spectral generalization to unseen
    structures. Requires `pair_df_with_missingness` to already carry `all_spectral_nan`
    (from `summarize_feature_missingness`).

    `all_query_ids`: the FULL end-to-end denominator (e.g. notebook 09's `all_dev_query_ids`) --
    pass it explicitly whenever comparing against another end-to-end MRR (e.g. the registered
    Mode-B LambdaMART score), since a query with zero unseen-connectivity candidates never
    appears in `pair_df_with_missingness` at all and would otherwise be silently dropped from
    the denominator, inflating this MRR relative to a metric computed over every intended query.
    Defaults to `pair_df_with_missingness[query_id_col].unique()` ONLY as a same-table fallback."""
    if all_query_ids is None:
        all_query_ids = pair_df_with_missingness[query_id_col].unique().tolist()
    else:
        all_query_ids = list(all_query_ids)
    subset = pair_df_with_missingness[pair_df_with_missingness["all_spectral_nan"]].copy()
    subset["score"] = -subset[mass_col]
    ranked = rank_candidates(subset, query_id_col=query_id_col, score_col="score", ascending=False)
    metrics = ranking_metrics(ranked, all_query_ids, query_id_col=query_id_col, is_true_col=is_true_col, k_values=k_values)
    return {"n_rows_in_subset": len(subset), "n_queries_total": len(all_query_ids), **metrics}


def _oof_predict(fold_models, df, model_features, fold_col):
    scores = pd.Series(np.nan, index=df.index)
    for fold, model in fold_models.items():
        mask = df[fold_col] == fold
        if mask.any():
            scores.loc[mask] = model.predict(df.loc[mask, model_features])
    assert scores.notna().all(), (
        f"{scores.isna().sum()} rows have no held-out model for their fold -- "
        f"fold_models keys {sorted(fold_models.keys())} vs. data folds {sorted(df[fold_col].unique())}"
    )
    return scores


def constant_spectral_feature_control(fold_models, pair_df, model_features, spectral_features=SPECTRAL_FEATURES,
                                       constant_value=0.0, fold_col="fold", query_id_col="query_id",
                                       is_true_col="is_true_candidate", k_values=(1, 5, 25), all_query_ids=None):
    """Section 1.5. `fold_models`: `{fold_value: already-trained model}`, one per held-out fold
    (e.g. notebook 09's `fold_models`) -- every row is scored by the model that held its fold
    out, exactly like a real OOF pass. Replaces every spectral feature, for every candidate,
    with the SAME finite constant (never NaN -- that would just recreate the pattern under
    test) and rescores with the SAME trained models. If performance collapses toward the
    mass-only baseline, the real-feature score was substantially driven by spectral
    value/missingness STRUCTURE, not genuine spectral discrimination.

    `all_query_ids`: the FULL end-to-end denominator -- pass it explicitly (e.g. notebook 09's
    `all_dev_query_ids`) so this is comparable to another end-to-end MRR computed over every
    intended query, not just the ones that happen to appear in `pair_df`. Defaults to
    `pair_df[query_id_col].unique()` ONLY as a same-table fallback."""
    if all_query_ids is None:
        all_query_ids = pair_df[query_id_col].unique().tolist()
    else:
        all_query_ids = list(all_query_ids)

    real_scores = _oof_predict(fold_models, pair_df, model_features, fold_col)
    real_ranked = rank_candidates(pair_df.assign(score=real_scores), query_id_col=query_id_col, score_col="score", ascending=False)
    real_metrics = ranking_metrics(real_ranked, all_query_ids, query_id_col=query_id_col, is_true_col=is_true_col, k_values=k_values)

    constant_df = pair_df.copy()
    constant_df[spectral_features] = float(constant_value)
    constant_scores = _oof_predict(fold_models, constant_df, model_features, fold_col)
    constant_ranked = rank_candidates(constant_df.assign(score=constant_scores), query_id_col=query_id_col, score_col="score", ascending=False)
    constant_metrics = ranking_metrics(constant_ranked, all_query_ids, query_id_col=query_id_col, is_true_col=is_true_col, k_values=k_values)

    return {
        "real_features_mrr_at_25": real_metrics["end_to_end"]["mrr_at_25"],
        "constant_spectral_features_mrr_at_25": constant_metrics["end_to_end"]["mrr_at_25"],
        "difference": real_metrics["end_to_end"]["mrr_at_25"] - constant_metrics["end_to_end"]["mrr_at_25"],
        "real_metrics": real_metrics,
        "constant_metrics": constant_metrics,
    }


def missingness_only_baseline(pair_df, spectral_features=SPECTRAL_FEATURES, mass_col="abs_mass_error_ppm",
                               fold_col="fold", query_id_col="query_id", is_true_col="is_true_candidate",
                               k_values=(1, 5, 25), lgbm_params=None, seed=42, all_query_ids=None):
    """Section 1.6. An intentionally crippled leave-one-fold-out ranker trained on ONLY
    `abs_mass_error_ppm` + `spectral_missing_count` + `all_spectral_nan` -- explicitly no real
    spectral-similarity values. Same connectivity-safe folds the real model uses. If this
    recovers a large share of the real model's Mode-B MRR, the real model's Mode-B "skill" is
    mostly the missingness pattern, not spectral discrimination -- `SHORTCUT CONFIRMED` per
    section 1.6. Requires `lightgbm` (imported lazily so importing this module never does).

    `all_query_ids`: the FULL end-to-end denominator -- pass it explicitly (e.g. notebook 09's
    `all_dev_query_ids`) for a fair comparison against another end-to-end MRR. Defaults to
    `pair_df[query_id_col].unique()` ONLY as a same-table fallback."""
    import lightgbm as lgb

    df = pair_df.copy()
    df["spectral_missing_count"] = df[spectral_features].isna().sum(axis=1)
    df["all_spectral_nan"] = df[spectral_features].isna().all(axis=1).astype(int)
    features = [mass_col, "spectral_missing_count", "all_spectral_nan"]

    params = lgbm_params or dict(objective="lambdarank", metric="ndcg", n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, min_child_samples=20, random_state=seed, verbosity=-1)

    if all_query_ids is None:
        all_query_ids = df[query_id_col].unique().tolist()
    else:
        all_query_ids = list(all_query_ids)
    oof_scores = pd.Series(np.nan, index=df.index)
    for held_out_fold in sorted(df[fold_col].unique()):
        tr = df[df[fold_col] != held_out_fold].sort_values(query_id_col)
        va = df[df[fold_col] == held_out_fold].sort_values(query_id_col)
        tr_group = tr.groupby(query_id_col, sort=True).size().to_numpy()
        model = lgb.LGBMRanker(**params)
        model.fit(tr[features], tr[is_true_col].astype(int), group=tr_group)
        oof_scores.loc[va.index] = model.predict(va[features])

    assert oof_scores.notna().all(), "every row must receive an out-of-fold prediction exactly once"
    ranked = rank_candidates(df.assign(score=oof_scores), query_id_col=query_id_col, score_col="score", ascending=False)
    metrics = ranking_metrics(ranked, all_query_ids, query_id_col=query_id_col, is_true_col=is_true_col, k_values=k_values)
    return {"features_used": features, "metrics": metrics, "mrr_at_25": metrics["end_to_end"]["mrr_at_25"]}


def candidate_order_control(model, pair_df, model_features, seed=999, atol=1e-9):
    """Section 13.3, applied to a Mode-B-scoring model: shuffle candidate row order within the
    frame handed to `.predict` and confirm every row's score is unchanged. A tree ensemble
    scores each row from its own feature values only; failure here would point at something
    else (e.g. an accidental groupby/rank step leaking row-order information)."""
    original = pd.Series(model.predict(pair_df[model_features]), index=pair_df.index).sort_index()
    shuffled = pair_df.sample(frac=1.0, random_state=seed)
    shuffled_scores = pd.Series(model.predict(shuffled[model_features]), index=shuffled.index).sort_index()
    max_abs_diff = float(np.abs(original.to_numpy() - shuffled_scores.to_numpy()).max())
    return {"passed": bool(np.allclose(original.to_numpy(), shuffled_scores.to_numpy(), atol=atol)), "max_abs_diff": max_abs_diff}


def spectrum_mask_control(model, pair_df, model_features, spectral_features=SPECTRAL_FEATURES, sample_n=None, seed=42):
    """Take candidates that DO have real spectral evidence, mask their spectral features to NaN
    (simulating "no reference spectrum"), and check the trained model's score shifts
    meaningfully. Confirms the model is actually reading spectral values rather than ignoring
    them in favor of some other leaked signal."""
    has_evidence = pair_df[spectral_features].notna().any(axis=1)
    sample = pair_df[has_evidence]
    if sample_n:
        sample = sample.sample(n=min(sample_n, len(sample)), random_state=seed)
    real_preds = model.predict(sample[model_features])
    masked = sample.copy()
    masked[spectral_features] = np.nan
    masked_preds = model.predict(masked[model_features])
    mean_abs_shift = float(np.abs(real_preds - masked_preds).mean())
    return {"n_sampled": int(len(sample)), "mean_abs_shift": mean_abs_shift, "sensitive": bool(mean_abs_shift > 1e-6)}
