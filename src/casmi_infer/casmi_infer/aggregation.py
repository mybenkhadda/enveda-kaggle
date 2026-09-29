"""Molecule-level aggregation of per-spectrum candidate rankings behind one `Aggregator` interface.

Input (long, one row per (spectrum, candidate)): `molecule_id`, `spectrum_id`, the candidate column
(`conn_idx` at inference, `candidate_connectivity_key` in development), `score`, `rank` (1-based,
per spectrum), `abs_mass_error_ppm`, and `prob` for calibrated aggregators.

Production (v6.3+) uses `MostConfidentSpectrum` (A7, config name MOST_CONFIDENT_SPECTRUM), locked on MOL_DEV
before HOST. `RRFAggregator` (k=60) is kept as an EXPLICIT baseline only: RRF(c) = sum_s 1 / (k + rank_s(c)); a
candidate absent from a spectrum contributes 0; ties RRF DESC, best individual rank ASC, candidate ASC.
The calibrated aggregators (A1-A8, selected by 11_02 on DEV) break ties by aggregator score DESC,
mean abs ppm ASC (mean mass evidence over the spectra where the candidate appears), candidate ASC.
`make_aggregator(config)` rebuilds any of them from the bundle's `config.json`, so swapping the
aggregator is a config change, not a code change.
"""
import numpy as np
import pandas as pd

DEFAULT_CAND = "conn_idx"


def softmax_by_group(scores, groups, temperature=1.0):
    """Per-spectrum softmax of `scores / T` (numerically stable)."""
    s = pd.Series(np.asarray(scores, dtype=float) / float(temperature))
    g = pd.Series(np.asarray(groups))
    z = s - s.groupby(g).transform("max")
    e = np.exp(z)
    return (e / e.groupby(g).transform("sum")).to_numpy()


class Aggregator:
    name = "base"
    simplicity = 99          # lower = simpler (the selection rule prefers simpler within CI)
    needs_prob = False

    def __init__(self, cand_col=DEFAULT_CAND):
        self.cand_col = cand_col

    def scores(self, df):
        """-> DataFrame[molecule_id, <cand>, agg_score, secondary] (secondary: smaller is better)."""
        raise NotImplementedError

    def _mean_ppm(self, df):
        return df.groupby(["molecule_id", self.cand_col], sort=False)["abs_mass_error_ppm"].mean().rename("secondary")

    def rank(self, df):
        """Molecule-level ranking with this aggregator's tie rule; one row per (molecule, candidate)
        -- candidates are deduplicated BEFORE any truncation."""
        if self.needs_prob and "prob" not in df.columns:
            raise ValueError(f"{self.name} needs calibrated `prob`")
        s = self.scores(df).reset_index()
        s = s.sort_values(["molecule_id", "agg_score", "secondary", self.cand_col], ascending=[True, False, True, True], kind="mergesort")
        s["mol_rank"] = s.groupby("molecule_id", sort=False).cumcount() + 1
        assert not s.duplicated(["molecule_id", self.cand_col]).any()
        return s.reset_index(drop=True)

    def config(self):
        return {"name": self.name}


class RRFAggregator(Aggregator):
    """A6 / Submission-1 aggregator."""
    name, simplicity = "A6_rrf", 2

    def __init__(self, k=60, cand_col=DEFAULT_CAND):
        super().__init__(cand_col)
        self.k = int(k)

    def scores(self, df):
        g = df.assign(_rrf=1.0 / (self.k + df["rank"].astype(float))).groupby(["molecule_id", self.cand_col], sort=False)
        return pd.DataFrame({"agg_score": g["_rrf"].sum(), "secondary": g["rank"].min().astype(float)})

    def config(self):
        return {"name": self.name, "k": self.k}


class MaxScore(Aggregator):
    """A1: max raw ranker score over the molecule's spectra."""
    name, simplicity = "A1_max_score", 1

    def scores(self, df):
        return pd.concat([df.groupby(["molecule_id", self.cand_col], sort=False)["score"].max().rename("agg_score"), self._mean_ppm(df)], axis=1)


class MaxProb(Aggregator):
    """A2: max calibrated probability."""
    name, simplicity, needs_prob = "A2_max_prob", 3, True

    def scores(self, df):
        return pd.concat([df.groupby(["molecule_id", self.cand_col], sort=False)["prob"].max().rename("agg_score"), self._mean_ppm(df)], axis=1)


class MeanProb(Aggregator):
    """A3: mean calibrated probability over ALL the molecule's spectra (absent -> 0)."""
    name, simplicity, needs_prob = "A3_mean_prob", 4, True

    def scores(self, df):
        n_spec = df.groupby("molecule_id")["spectrum_id"].nunique()
        s = df.groupby(["molecule_id", self.cand_col], sort=False)["prob"].sum()
        mol = s.index.get_level_values("molecule_id")
        return pd.concat([(s / n_spec.reindex(mol).to_numpy()).rename("agg_score"), self._mean_ppm(df)], axis=1)


class SumLogProb(Aggregator):
    """A4: sum over ALL spectra of log(max(p, eps)); absent -> log(eps). eps is a fixed DEV-chosen option."""
    name, simplicity, needs_prob = "A4_sum_log_prob", 6, True

    def __init__(self, eps=1e-6, cand_col=DEFAULT_CAND):
        super().__init__(cand_col)
        self.eps = float(eps)

    def scores(self, df):
        n_spec = df.groupby("molecule_id")["spectrum_id"].nunique()
        g = df.assign(_lp=np.log(np.maximum(df["prob"].to_numpy(float), self.eps))).groupby(["molecule_id", self.cand_col], sort=False)
        lp_sum, n_present = g["_lp"].sum(), g["_lp"].size()
        mol = lp_sum.index.get_level_values("molecule_id")
        agg = lp_sum + (n_spec.reindex(mol).to_numpy() - n_present) * np.log(self.eps)
        return pd.concat([agg.rename("agg_score"), self._mean_ppm(df)], axis=1)

    def config(self):
        return {"name": self.name, "eps": self.eps}


class LogMeanExp(Aggregator):
    """A5: log-mean-exp of log p over the spectra where the candidate is PRESENT (absent spectra are
    ignored -- this is what distinguishes A5 from A3, whose absent-as-0 mean is log-monotone-equal
    to a log-mean-exp over ALL spectra)."""
    name, simplicity, needs_prob = "A5_logmeanexp_present", 5, True

    def scores(self, df):
        g = df.groupby(["molecule_id", self.cand_col], sort=False)["prob"]
        return pd.concat([np.log(g.mean().clip(lower=1e-300)).rename("agg_score"), self._mean_ppm(df)], axis=1)


class MostConfidentSpectrum(Aggregator):
    """A7 (production since v6.3): per molecule, the spectrum whose top-1 calibrated probability is highest
    (ties: spectrum_id ASC, stable string order); its own candidate ranking is the molecule ranking (candidates
    absent from it are not ranked), ordered by its calibrated probability DESC, abs ppm ASC, candidate ASC --
    i.e. the spectrum's own frozen rank order, since softmax is monotone within a spectrum."""
    name, simplicity, needs_prob = "A7_most_confident_spectrum", 1, True

    @staticmethod
    def selected_spectra(df):
        """One row per molecule: molecule_id, spectrum_id (the selected one), confidence (its top-1 prob)."""
        top1 = df.groupby(["molecule_id", "spectrum_id"])["prob"].max().rename("confidence").reset_index()
        top1["spectrum_id"] = top1["spectrum_id"].astype(str)
        return (top1.sort_values(["molecule_id", "confidence", "spectrum_id"], ascending=[True, False, True], kind="mergesort")
                .drop_duplicates("molecule_id").reset_index(drop=True))

    def scores(self, df):
        sel = self.selected_spectra(df)[["molecule_id", "spectrum_id"]]
        sub = df.assign(spectrum_id=df["spectrum_id"].astype(str)).merge(sel, on=["molecule_id", "spectrum_id"])
        return sub.set_index(["molecule_id", self.cand_col])[["prob", "abs_mass_error_ppm"]].rename(
            columns={"prob": "agg_score", "abs_mass_error_ppm": "secondary"})


class FirstSpectrum(Aggregator):
    """A8: legacy baseline -- the molecule's first spectrum (smallest `spectrum_order`, else
    spectrum_id ASC) and its own ranking."""
    name, simplicity = "A8_first_spectrum", 0

    def scores(self, df):
        key = "spectrum_order" if "spectrum_order" in df.columns else "spectrum_id"
        first = df.sort_values(["molecule_id", key]).drop_duplicates("molecule_id")[["molecule_id", "spectrum_id"]]
        sub = df.merge(first, on=["molecule_id", "spectrum_id"])
        out = sub.set_index(["molecule_id", self.cand_col])
        return pd.DataFrame({"agg_score": -out["rank"].astype(float), "secondary": out["abs_mass_error_ppm"]})


AGGREGATORS = {"A1_max_score": MaxScore, "A2_max_prob": MaxProb, "A3_mean_prob": MeanProb, "A4_sum_log_prob": SumLogProb,
               "A5_logmeanexp_present": LogMeanExp, "A6_rrf": RRFAggregator, "A7_most_confident_spectrum": MostConfidentSpectrum,
               "A8_first_spectrum": FirstSpectrum}


# production names (bundle config) <-> development ids; an alias resolves to exactly one class
PRODUCTION_NAMES = {"A7_most_confident_spectrum": "MOST_CONFIDENT_SPECTRUM", "A6_rrf": "RRF", "A1_max_score": "MAX_SCORE"}
ALIASES = {v: k for k, v in PRODUCTION_NAMES.items()}


def canonical_aggregator_id(name):
    """'MOST_CONFIDENT_SPECTRUM' / 'A7_most_confident_spectrum' -> 'A7_most_confident_spectrum'; unknown -> ValueError."""
    aid = ALIASES.get(name, name)
    if aid not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; known: {sorted(AGGREGATORS) + sorted(ALIASES)}")
    return aid


def make_aggregator(cfg, cand_col=DEFAULT_CAND):
    """From the bundle config: {"name": ..., plus "k" (RRF) / "eps" (A4)}. There is NO default: a missing or
    unknown name raises, so RRF (or anything else) can never be activated implicitly -- RRF runs only when the
    config says "RRF" / "A6_rrf". Temperature is applied by the pipeline (`softmax_by_group`) before a
    prob-based aggregator runs; the pipeline refuses a prob-based aggregator without a calibrated temperature."""
    if not isinstance(cfg, dict) or not cfg.get("name"):
        raise ValueError(f"aggregator config must name its aggregator explicitly (no implicit default), got {cfg!r}")
    aid = canonical_aggregator_id(cfg["name"])
    params = {**cfg.get("params", {}), **{k: cfg[k] for k in ("k", "eps") if k in cfg}}
    if aid == "A6_rrf":
        return RRFAggregator(k=params.get("k", 60), cand_col=cand_col)
    if aid == "A4_sum_log_prob":
        return SumLogProb(eps=params.get("eps", 1e-6), cand_col=cand_col)
    return AGGREGATORS[aid](cand_col=cand_col)
