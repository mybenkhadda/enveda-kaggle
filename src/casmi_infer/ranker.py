"""Fold-mean LightGBM scoring + deterministic per-spectrum ranking.

score = mean over the fold boosters (LightGBM text format) of their predictions on the frozen
feature order. Ties: score DESC, abs_mass_error_ppm ASC, connectivity_key ASC (== conn_idx ASC,
since conn_idx indexes the key-sorted connectivity table).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_infer.features import model_matrix


class FoldMeanRanker:
    def __init__(self, boosters, feature_names, model_id=None):
        if not boosters:
            raise ValueError("no fold boosters")
        self.boosters = list(boosters)
        self.feature_names = list(feature_names)
        self.model_id = model_id
        for b in self.boosters:
            names = list(b.feature_name())
            if names != self.feature_names:
                raise ValueError(f"booster feature order {names} != frozen feature_names {self.feature_names}")

    @classmethod
    def load(cls, models_dir):
        import lightgbm as lgb
        d = Path(models_dir)
        feature_names = json.loads((d / "feature_names.json").read_text(encoding="utf-8"))
        info = json.loads((d / "model_info.json").read_text(encoding="utf-8")) if (d / "model_info.json").exists() else {}
        files = sorted(d.glob("*_fold*.txt"))
        return cls([lgb.Booster(model_file=str(f)) for f in files], feature_names, info.get("model_id"))

    def predict_folds(self, features_df):
        """(n_folds, n_rows) per-fold predictions on the frozen feature order."""
        if not len(features_df):
            return np.zeros((len(self.boosters), 0))
        X = model_matrix(features_df, self.feature_names)
        return np.asarray([np.asarray(b.predict(X), dtype=float) for b in self.boosters])

    def predict(self, features_df):
        """Fold-mean score (== predict_folds(...).mean(axis=0), the same arithmetic the self-test freezes)."""
        if not len(features_df):
            return np.zeros(0)
        return self.predict_folds(features_df).mean(axis=0)


def rank_spectrum(features_df, scores):
    """Adds `score` and 1-based `rank` with the canonical tie rule."""
    d = features_df.assign(score=np.asarray(scores, dtype=float))
    order = np.lexsort((d["conn_idx"].to_numpy(), d["abs_mass_error_ppm"].to_numpy(), -d["score"].to_numpy()))
    d = d.iloc[order].reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    return d
