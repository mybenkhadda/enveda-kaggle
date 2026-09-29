"""End-to-end closed-world inference, one molecule at a time (never a global pair table).

Stages per spectrum: clean -> neutral mass -> candidate retrieval (+ logged fallbacks) -> reference
selection + similarities + features -> fold-mean score -> deterministic rank. Molecules are then
aggregated with the configured `Aggregator` (production: A7 MOST_CONFIDENT_SPECTRUM with the frozen
temperature; RRF only when the config names it explicitly).

Robustness: any exception inside a spectrum is CAUGHT, LOGGED (molecule_id, spectrum_id, exception
class, message, fallback type) and replaced by a deterministic nearest-mass ranking -- never
swallowed silently. The optional emergency pool cap (closest N candidates by abs ppm) is OFF by
default; when the runtime projection exceeds the budget and the cap is enabled, its activation,
N and the number of affected spectra go into the run report.
"""
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_infer.adducts import AdductRules
from casmi_infer.aggregation import make_aggregator, softmax_by_group
from casmi_infer.features import candidate_features
from casmi_infer.mass_search import MassSearch
from casmi_infer.ranker import FoldMeanRanker, rank_spectrum
from casmi_infer.reference_index import ReferenceLibrary
from casmi_infer.spectrum import clean_query, query_meta, query_peak_hash
from casmi_infer.validation import verify_aggregator_contract, verify_bundle

TEST_REQUIRED = ("molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
                 "ionization_mode", "instrument_type")


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except Exception:
        return float("nan")


@dataclass
class RunLog:
    stage_seconds: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    fallbacks: list = field(default_factory=list)
    exceptions: list = field(default_factory=list)
    unsupported_adducts: dict = field(default_factory=dict)
    emergency_cap: dict = field(default_factory=lambda: {"enabled": False, "activated": False, "n": None, "n_spectra_affected": 0})
    peak_rss_gb: float = 0.0
    t0: float = field(default_factory=time.time)

    def tick_rss(self):
        self.peak_rss_gb = float(np.nanmax([self.peak_rss_gb, _rss_gb()]))

    def stage(self, name, seconds):
        self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + float(seconds)

    def bump(self, key, n=1):
        self.counters[key] = self.counters.get(key, 0) + n

    def to_dict(self):
        return {"stage_seconds": self.stage_seconds, "counters": self.counters, "fallbacks": self.fallbacks,
                "exceptions": self.exceptions, "unsupported_adducts": self.unsupported_adducts, "emergency_cap": self.emergency_cap,
                "peak_rss_gb": self.peak_rss_gb, "wall_seconds": time.time() - self.t0}


class Bundle:
    def __init__(self, bundle_dir, verify=True, mmap=True):
        d = Path(bundle_dir)
        self.dir = d
        if verify:
            self.manifest, self.config, problems = verify_bundle(d)
            if problems:
                raise RuntimeError("bundle verification failed:\n  " + "\n  ".join(problems))
        else:
            self.manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            self.config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self.ppm = json.loads((d / "ppm_windows.json").read_text(encoding="utf-8"))
        self.adducts = AdductRules.load(d / "adducts.json")
        structures = pd.read_parquet(d / "structures.parquet")
        self.conn_table = pd.read_parquet(d / "connectivities.parquet")
        masses = np.load(d / "structures_mass.npy")
        if not np.array_equal(masses, structures["neutral_monoisotopic_mass"].to_numpy(np.float64)):
            raise RuntimeError("structures_mass.npy does not match structures.parquet order")
        self.search = MassSearch(masses, structures["conn_idx"].to_numpy(), self.conn_table["connectivity_key"].to_numpy())
        self.smiles = self.conn_table["representative_smiles"].astype(str).to_numpy()
        self.lib = ReferenceLibrary(d, mmap=mmap)
        self.ranker = FoldMeanRanker.load(d / "models")
        # v6.3: explicit aggregator + frozen calibration contract (raises; RRF is never implied, T never refit)
        self.aggregator_contract = verify_aggregator_contract(self.config, d)
        self.aggregator = make_aggregator(self.config["aggregator"])
        self.temperature = self.aggregator_contract["temperature"]
        if self.config.get("feature_order") is not None and list(self.config["feature_order"]) != self.ranker.feature_names:
            raise RuntimeError("config feature_order != the shipped models' feature order")
        if self.config.get("model_id") is not None and self.config["model_id"] != self.ranker.model_id:
            raise RuntimeError(f"config model_id {self.config['model_id']!r} != models/model_info.json {self.ranker.model_id!r}")
        self.sim_cfg = {**self.config["similarity"], "top_reference_count": self.config["top_reference_count"]}

    def primary_ppm(self, adduct):
        return float(self.ppm["per_adduct_primary_ppm"].get(str(adduct), self.ppm["default_primary_ppm"]))


def resolve_neutral_mass(bundle, row, log):
    adduct = str(row["adduct"])
    nm = bundle.adducts.neutral_mass(row["precursor_mz"], adduct)
    if nm is None:
        used = bundle.adducts.polarity_default(row.get("ionization_mode"))
        log.unsupported_adducts[adduct] = log.unsupported_adducts.get(adduct, 0) + 1
        log.fallbacks.append({"molecule_id": row["molecule_id"], "spectrum_id": row["spectrum_id"], "fallback_type": "unsupported_adduct_polarity_default",
                              "adduct": adduct, "adduct_used": used})
        return bundle.adducts.neutral_mass(row["precursor_mz"], used), used
    return nm, adduct


def nearest_mass_ranking(bundle, neutral_mass, n):
    ci, ap = bundle.search.nearest(neutral_mass, n) if neutral_mass is not None else (np.zeros(0, np.int64), np.zeros(0))
    return pd.DataFrame({"conn_idx": ci, "abs_mass_error_ppm": ap, "score": -ap, "rank": np.arange(1, len(ci) + 1)})


def process_spectrum(bundle, row, log, exclude_sources=frozenset(), exclude_ids=frozenset(), cap_n=None):
    """Ranked candidates for one spectrum (+ its retrieval level)."""
    t = time.time()
    identity, sim = clean_query(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"], bundle.sim_cfg["max_peaks_similarity"])
    qhash = query_peak_hash(identity)
    qm = query_meta(row["adduct"], row.get("ionization_mode"), row.get("instrument_type"), row.get("collision_energy_ev"),
                    source=row.get("source"))
    log.stage("preprocess", time.time() - t)

    t = time.time()
    nm, adduct_used = resolve_neutral_mass(bundle, row, log)
    ci, ap, level = bundle.search.search(nm, bundle.primary_ppm(adduct_used), tuple(bundle.ppm["fallback_ppm"]), bundle.ppm["nearest_n"])
    if level != "primary":
        log.fallbacks.append({"molecule_id": row["molecule_id"], "spectrum_id": row["spectrum_id"], "fallback_type": f"retrieval_{level}"})
    log.bump(f"retrieval_level_{level}")
    if cap_n is not None and len(ci) > cap_n:
        ci, ap = ci[:cap_n], ap[:cap_n]  # search() returns abs-ppm order, so this keeps the closest N
        log.emergency_cap["n_spectra_affected"] += 1
    log.bump("candidates", len(ci))
    log.stage("retrieval", time.time() - t)

    t = time.time()
    feats = candidate_features(bundle.lib, ci, ap, sim, qm, qhash, bundle.sim_cfg, exclude_sources, exclude_ids, counters=log.counters)
    log.stage("spectral_evidence", time.time() - t)

    t = time.time()
    ranked = rank_spectrum(feats, bundle.ranker.predict(feats))
    log.stage("ranking", time.time() - t)
    return ranked, {"retrieval_level": level, "neutral_mass": nm, "adduct_used": adduct_used, "n_candidates": int(len(ci))}


def run_inference(bundle, test_df, log, time_budget_s=None, emergency_cap_enabled=False, emergency_cap_n=300,
                  exclude_sources_fn=None, exclude_ids_fn=None, progress_every=50):
    """Per-spectrum rankings for every test spectrum, molecule by molecule. Returns the long frame
    (molecule_id, spectrum_id, spectrum_order, conn_idx, score, rank, abs_mass_error_ppm, ...).
    `exclude_*_fn(row)` exist ONLY for HOST benchmark mode (11_01); hidden test uses empty sets."""
    missing = [c for c in TEST_REQUIRED if c not in test_df.columns]
    if missing:
        raise KeyError(f"test frame missing columns {missing}")
    log.emergency_cap["enabled"] = bool(emergency_cap_enabled)
    parts, n_total, n_done, cap_n = [], len(test_df), 0, None
    t_start = time.time()
    df = test_df.reset_index(drop=True).assign(spectrum_order=lambda d: np.arange(len(d)))
    for mid, g in df.groupby("molecule_id", sort=True):
        for _, row in g.iterrows():
            row = row.to_dict()
            try:
                ranked, info = process_spectrum(bundle, row, log, exclude_sources=frozenset(exclude_sources_fn(row)) if exclude_sources_fn else frozenset(),
                                                exclude_ids=frozenset(exclude_ids_fn(row)) if exclude_ids_fn else frozenset(), cap_n=cap_n)
            except Exception as e:  # logged + replaced, never swallowed
                log.exceptions.append({"molecule_id": mid, "spectrum_id": row.get("spectrum_id"), "exception_class": type(e).__name__,
                                       "message": str(e)[:500], "traceback_tail": traceback.format_exc().splitlines()[-3:],
                                       "fallback_type": "nearest_mass"})
                try:
                    nm, _ = resolve_neutral_mass(bundle, row, log)
                except Exception:
                    nm = None
                ranked, info = nearest_mass_ranking(bundle, nm, bundle.ppm["nearest_n"]), {"retrieval_level": "exception_nearest_mass"}
            ranked = ranked.assign(molecule_id=mid, spectrum_id=row["spectrum_id"], spectrum_order=row["spectrum_order"],
                                   retrieval_level=info["retrieval_level"], model_scored=info["retrieval_level"] != "exception_nearest_mass")
            parts.append(ranked)
            n_done += 1
        log.tick_rss()
        if time_budget_s and n_done:
            projected = (time.time() - t_start) / n_done * n_total
            log.counters["projected_total_seconds"] = projected
            if projected > time_budget_s and emergency_cap_enabled and cap_n is None:
                cap_n = int(emergency_cap_n)
                log.emergency_cap.update(activated=True, n=cap_n, activated_after_spectra=n_done, projected_seconds_at_activation=projected)
        if progress_every and n_done % progress_every < len(g):
            print(f"[infer] {n_done}/{n_total} spectra  {time.time() - t_start:.0f}s  rss={_rss_gb():.1f}GB")
    log.counters["n_spectra"], log.counters["n_molecules"] = n_total, int(df["molecule_id"].nunique())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def calibrated_probabilities(bundle, per_spectrum):
    """prob = softmax(score / T) over each spectrum's full candidate list, T = the bundle's FROZEN temperature."""
    if bundle.temperature is None:
        raise RuntimeError(f"{bundle.aggregator.name} needs the frozen calibration temperature; the bundle has none")
    return per_spectrum.assign(prob=softmax_by_group(per_spectrum["score"], per_spectrum["spectrum_id"], bundle.temperature))


def model_scored_only(per_spectrum, counters=None):
    """Probability aggregators rank calibrated MODEL scores. A spectrum ranked by the nearest-mass fallback
    (`model_scored` False: score = -ppm, no spectral evidence) is therefore not a candidate for A7's
    'most confident spectrum' whenever its molecule has at least one model-scored spectrum; a molecule whose
    spectra ALL fell back keeps the existing nearest-mass fallback ranking. Counted, never silent."""
    if "model_scored" not in per_spectrum.columns:
        return per_spectrum
    ms = per_spectrum["model_scored"].astype(bool)
    has_model = ms.groupby(per_spectrum["molecule_id"]).transform("any")
    drop = has_model & ~ms
    if counters is not None:
        counters["fallback_spectra_excluded_from_prob_aggregation"] = int(per_spectrum.loc[drop, "spectrum_id"].nunique())
        counters["molecules_all_spectra_fallback"] = int((~has_model).groupby(per_spectrum["molecule_id"]).first().sum())
    return per_spectrum[~drop]


def aggregate_molecules(bundle, per_spectrum, counters=None):
    """Molecule rankings with the CONFIGURED aggregator (A7 in production; RRF only if the config says so)."""
    agg = bundle.aggregator
    df = per_spectrum
    if agg.needs_prob:
        df = calibrated_probabilities(bundle, model_scored_only(df, counters))
    return agg.rank(df)


def selected_spectra(bundle, per_spectrum, counters=None):
    """A7 audit trail: the spectrum each molecule's ranking came from (+ its confidence); None for other aggregators."""
    if not hasattr(bundle.aggregator, "selected_spectra"):
        return None
    return bundle.aggregator.selected_spectra(calibrated_probabilities(bundle, model_scored_only(per_spectrum, counters)))
