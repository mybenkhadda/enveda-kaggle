"""Calibrated mass-residual likelihood `log P(|mass_error_ppm| | instrument, adduct)` (v4b B1L/B4/B5/V4).

Model (per group): the TRUE candidate's signed ppm error `e = 1e6 (exact_mass_true - neutral_mass)
/ neutral_mass` -- the same definition `casmi.candidates.generator` uses for `mass_error_ppm` --
is a robust Gaussian N(mu, s) (mu = median, s = 1.4826 * MAD, floored at `min_scale_ppm`) mixed
with a uniform outlier component over the candidate window [-tol, tol]. Candidates only carry
|e|, so the density is FOLDED onto [0, tol]:

    f(a) = (1 - eps) * [N(a; mu, s) + N(-a; mu, s)] + eps / tol

(the folded form is invariant to the sign convention of e, so a flipped convention upstream
cannot silently break it). Decoy errors are ~uniform over the window, so within one query,
ranking by log f(a) is ranking by the truth-vs-decoy likelihood ratio.

Back-off hierarchy: instrument+adduct -> instrument -> adduct -> global, each level used only
with >= `min_n` fitting samples. Fitting data must be TRAIN/DEV-safe: `fit` refuses rows from
the HOST source and records exactly which rows it used.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GLOBAL_KEY = ("__global__",)
LEVELS = ("instrument_adduct", "instrument", "adduct", "global")


def signed_truth_ppm(neutral_mass, exact_mass_true):
    neutral_mass = np.asarray(neutral_mass, dtype=float)
    return 1e6 * (np.asarray(exact_mass_true, dtype=float) - neutral_mass) / neutral_mass


def _robust_params(e, min_scale_ppm):
    e = np.asarray(e, dtype=float)
    mu = float(np.median(e))
    s = float(1.4826 * np.median(np.abs(e - mu)))
    return {"mu": mu, "s": max(s, min_scale_ppm), "n": int(len(e))}


@dataclass
class MassLikelihoodModel:
    tol_ppm: float = 50.0
    eps: float = 0.02
    min_n: int = 200
    min_scale_ppm: float = 0.1
    params: dict = field(default_factory=dict)     # level -> {key tuple: {"mu","s","n"}}
    fit_info: dict = field(default_factory=dict)

    def fit(self, df, err_col="truth_ppm", instrument_col="instrument_type", adduct_col="adduct",
            source_col="source", forbidden_sources=()):
        """`df`: one row per fitting spectrum with its signed truth ppm error. Rows with |e| > tol
        are dropped (they could never be candidates). Raises if any row comes from a forbidden
        (HOST) source -- the mass model must never see HOST."""
        if source_col in df.columns and len(forbidden_sources):
            leaked = df[source_col].isin(list(forbidden_sources))
            if leaked.any():
                raise ValueError(f"mass-likelihood fit data contains {int(leaked.sum())} rows from forbidden source(s) {forbidden_sources}")
        d = df[[err_col, instrument_col, adduct_col]].dropna(subset=[err_col])
        d = d[np.abs(d[err_col]) <= self.tol_ppm]
        inst = d[instrument_col].astype(str).to_numpy()
        add = d[adduct_col].astype(str).to_numpy()
        e = d[err_col].to_numpy(float)
        self.params = {lvl: {} for lvl in LEVELS}
        frame = pd.DataFrame({"i": inst, "a": add, "e": e})
        for (i, a), g in frame.groupby(["i", "a"]):
            if len(g) >= self.min_n:
                self.params["instrument_adduct"][(i, a)] = _robust_params(g["e"], self.min_scale_ppm)
        for i, g in frame.groupby("i"):
            if len(g) >= self.min_n:
                self.params["instrument"][(i,)] = _robust_params(g["e"], self.min_scale_ppm)
        for a, g in frame.groupby("a"):
            if len(g) >= self.min_n:
                self.params["adduct"][(a,)] = _robust_params(g["e"], self.min_scale_ppm)
        if len(frame) == 0:
            raise ValueError("no fitting rows left after dropping |e| > tol")
        self.params["global"][GLOBAL_KEY] = _robust_params(frame["e"], self.min_scale_ppm)
        self.fit_info = {"n_rows_fit": int(len(frame)), "n_groups": {lvl: len(v) for lvl, v in self.params.items()},
                         "tol_ppm": self.tol_ppm, "eps": self.eps, "min_n": self.min_n, "min_scale_ppm": self.min_scale_ppm}
        return self

    def resolve(self, instrument, adduct):
        """`(params, level)` for one (instrument, adduct) via the back-off hierarchy."""
        i, a = str(instrument), str(adduct)
        for lvl, key in (("instrument_adduct", (i, a)), ("instrument", (i,)), ("adduct", (a,)), ("global", GLOBAL_KEY)):
            p = self.params.get(lvl, {}).get(key)
            if p is not None:
                return p, lvl
        raise RuntimeError("model not fitted")

    def _folded_logpdf(self, a, mu, s):
        a = np.asarray(a, dtype=float)
        z1, z2 = (a - mu) / s, (-a - mu) / s
        norm = 1.0 / (s * np.sqrt(2 * np.pi))
        core = norm * (np.exp(-0.5 * z1 * z1) + np.exp(-0.5 * z2 * z2))
        return np.log((1.0 - self.eps) * core + self.eps / self.tol_ppm)

    def loglik(self, df, abs_err_col="abs_mass_error_ppm", instrument_col="instrument_type", adduct_col="adduct"):
        """Vectorized over candidate rows. Returns `(loglik array, level array)`."""
        keys = df[[instrument_col, adduct_col]].astype(str)
        uniq = keys.drop_duplicates()
        lut = {}
        for i, a in uniq.itertuples(index=False):
            lut[(i, a)] = self.resolve(i, a)
        mu = np.empty(len(df))
        s = np.empty(len(df))
        level = np.empty(len(df), dtype=object)
        for j, (i, a) in enumerate(keys.itertuples(index=False)):
            p, lvl = lut[(i, a)]
            mu[j], s[j], level[j] = p["mu"], p["s"], lvl
        return self._folded_logpdf(df[abs_err_col].to_numpy(float), mu, s), level

    @classmethod
    def from_dict(cls, d):
        """Inverse of `to_dict` -- reload a FROZEN model (e.g. v4b's DEV-safe fit) without refitting."""
        fi = d["fit_info"]
        m = cls(tol_ppm=fi["tol_ppm"], eps=fi["eps"], min_n=fi["min_n"], min_scale_ppm=fi["min_scale_ppm"])
        m.params = {lvl: {tuple(k.split("|")): v for k, v in grp.items()} for lvl, grp in d["params"].items()}
        m.fit_info = dict(fi)
        return m

    def to_dict(self):
        return {"fit_info": self.fit_info,
                "params": {lvl: {"|".join(k): v for k, v in grp.items()} for lvl, grp in self.params.items()}}
