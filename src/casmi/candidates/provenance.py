"""Candidate provenance (which sources contain a connectivity, under which source ids), the
source-PRIOR schema (metadata only -- no values invented, no ranking weights), and the future
candidate-EVIDENCE schema with its shortcut-feature guard.

Provenance is never lost by deduplication: every connectivity keeps the sorted set of canonical
sources and every original `SOURCE:source_id`.
"""
import numpy as np
import pandas as pd

NP_SOURCES = ("COCONUT", "LOTUS", "NPATLAS")
SOURCE_PRIOR_COLUMNS = ["connectivity_key", "in_train", "in_coconut", "in_lotus", "in_npatlas", "in_pubchem", "number_of_np_sources",
                        "literature_count", "organism_evidence"]


def aggregate_provenance(records, key="connectivity_key"):
    """`records`: one row per (source, source_id, connectivity_key). Returns one row per connectivity
    with `candidate_sources` (sorted unique), `source_ids` (sorted unique 'SOURCE:id'), `n_source_records`."""
    r = records[[key, "source", "source_id"]].dropna(subset=[key]).copy()
    r["_sid"] = r["source"].astype(str) + ":" + r["source_id"].astype(str)
    g = r.groupby(key, sort=True)
    return pd.DataFrame({"candidate_sources": g["source"].agg(lambda s: sorted(set(map(str, s)))),
                         "source_ids": g["_sid"].agg(lambda s: sorted(set(s))), "n_source_records": g.size()}).reset_index()


def source_prior_frame(unified):
    """Schema for FUTURE priors, filled only with what is known: source membership flags and the count
    of NP sources. `literature_count` / `organism_evidence` stay NA until a real source provides them."""
    srcs = unified["candidate_sources"].map(lambda s: set(s) if isinstance(s, (list, tuple, set, np.ndarray)) else set())
    out = pd.DataFrame({"connectivity_key": unified["connectivity_key"].to_numpy()})
    for s in ("TRAIN", "COCONUT", "LOTUS", "NPATLAS", "PUBCHEM"):
        out[f"in_{s.lower()}"] = srcs.map(lambda x, s=s: s in x).to_numpy()
    out["number_of_np_sources"] = srcs.map(lambda x: sum(s in x for s in NP_SOURCES)).to_numpy()
    out["literature_count"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["organism_evidence"] = pd.array([pd.NA] * len(out), dtype="string")
    return out[SOURCE_PRIOR_COLUMNS]


# ---------------------------------------------------------------------------------------------
# future unified candidate evidence (architecture only -- nothing is trained here)
# ---------------------------------------------------------------------------------------------

EVIDENCE_SCHEMA = {
    "direct_reference_score": "frozen Mode-A spectrum ranker score / its 8 similarity aggregates; NaN when the candidate has no eligible reference",
    "analog_transfer_score": "similarity to reference spectra of STRUCTURAL ANALOGS of the candidate (future)",
    "fingerprint_likelihood": "log-likelihood of the candidate fingerprint under a spectrum->fingerprint model (future; not implemented in v6)",
    "mass_evidence": "abs_mass_error_ppm (or a calibrated mass likelihood) of the candidate given the precursor/adduct",
    "formula_evidence": "agreement with a VALIDATED formula predictor (future; formula is metadata until then)",
    "np_prior": "source-prior features derived from SOURCE_PRIOR_COLUMNS (future; no weights before validation)",
    "candidate_source_metadata": "candidate_sources / source ids for audit and stratified reporting (not a model feature by default)",
}
# Raw reference-AVAILABILITY encodings: in Class-2 the truth has no reference by construction, so these
# would let a ranker learn 'no reference => truth' (or the reverse in closed-world data). Use evidence
# STRENGTHS (max library similarity, spectral likelihood, analog score) instead.
FORBIDDEN_SHORTCUT_FEATURES = ("has_reference", "has_reference_spectrum", "n_reference_spectra", "n_refs_total", "n_selected_refs",
                               "eligible_reference_count", "reference_available", "in_train", "is_train_structure", "n_train_spectra")


def assert_no_shortcut_features(feature_names):
    bad = [f for f in feature_names if f in FORBIDDEN_SHORTCUT_FEATURES or str(f).startswith("has_ref")]
    if bad:
        raise AssertionError(f"reference-availability shortcut features are not allowed in the unified ranker: {bad}")
    return True


def empty_evidence_frame(candidates, key="connectivity_key"):
    """Placeholder evidence table (all NaN) with the future columns -- lets downstream code be written
    against the schema before any evidence model exists."""
    out = pd.DataFrame({key: candidates[key].to_numpy()})
    for c in EVIDENCE_SCHEMA:
        if c != "candidate_source_metadata":
            out[c] = np.nan
    return out
