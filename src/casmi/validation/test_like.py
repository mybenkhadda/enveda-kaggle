"""Test-shaped validation: three structure-availability modes, primary host-like holdout
detection, and full-vs-test-like CV framing. Deliberately reuses ALREADY-COMPUTED spectral
pair features (`casmi.ranking.features`) wherever possible -- Mode A and Mode B are just two
different REFERENCE-SET restrictions over the same underlying similarity scores (already
computed by notebook 04's `pair_features_dev_known_spectrum` / `..._unseen_connectivity`), and
Mode C is analytically determined (see below), so this module never re-runs expensive
spectral-similarity computation.
"""
import pandas as pd

MODE_DESCRIPTIONS = {
    "A_reference_available": "True structure may have same-connectivity reference spectra (exact query spectrum excluded). Simulates a library-known case.",
    "B_database_only": "True structure is a valid candidate, but ALL same-connectivity reference spectra are hidden. Simulates: structure known, spectra unavailable.",
    "C_structure_absent": "True structure is removed from the candidate pool entirely. Tests whether ANYTHING (analogue transfer, generation -- notebook 08, not implemented this session) could recover it. With no such mechanism active, this mode's candidate availability is 0% by construction -- not a bug, the honest ceiling of the current pipeline.",
}


def detect_primary_holdout(train_metadata, source_col="ingest_lib", holdout_source="enveda-np-examples"):
    """Look for a locally available primary host-like holdout (spec section 9) BEFORE falling
    back to a synthetic protocol -- never downloads or fabricates one. Returns
    `(found: bool, subset_or_None)`. Prints nothing itself; the caller decides how to report
    "not available locally" so the message reaches wherever the caller wants it."""
    if source_col not in train_metadata.columns:
        return False, None
    subset = train_metadata[train_metadata[source_col] == holdout_source]
    if len(subset) == 0:
        return False, None
    return True, subset


def mode_a_candidate_availability(pair_features_known, all_query_ids, is_true_col="is_true_candidate"):
    """Mode A candidate availability: fraction of `all_query_ids` whose true candidate appears
    in `pair_features_known` (notebook 04's known-spectrum pair table) at all -- independent of
    whether it ended up with a usable spectral score (see `target_score_coverage` for that)."""
    present = set(pair_features_known.loc[pair_features_known[is_true_col], "query_id"])
    return len(present & set(all_query_ids)) / len(all_query_ids) if len(all_query_ids) else float("nan")


def mode_b_candidate_availability(pair_features_unseen, all_query_ids, is_true_col="is_true_candidate"):
    """Mode B candidate availability: same idea, over the unseen-connectivity pair table (the
    true candidate IS present as a ROW -- mass-based candidate generation doesn't know about
    reference-spectrum hiding -- it just structurally never has `has_reference_spectrum=True`)."""
    present = set(pair_features_unseen.loc[pair_features_unseen[is_true_col], "query_id"])
    return len(present & set(all_query_ids)) / len(all_query_ids) if len(all_query_ids) else float("nan")


def mode_c_candidate_availability(all_query_ids):
    """Mode C candidate availability is 0.0 for every query, ALWAYS, by construction: the true
    structure was removed from the candidate pool before scoring, so no retrieval-based method
    (mass, cosine, or any classical similarity) can find it. This function exists so the 0.0
    is computed and labeled explicitly, not silently omitted from a results table."""
    return 0.0 if len(all_query_ids) else float("nan")


def summarize_modes(pair_features_known, pair_features_unseen, all_query_ids):
    """The three-row candidate-availability table (spec section 10/58): one row per mode,
    reported SEPARATELY -- never averaged into one blended number."""
    rows = [
        {"mode": "A_reference_available", "candidate_availability": mode_a_candidate_availability(pair_features_known, all_query_ids),
         "description": MODE_DESCRIPTIONS["A_reference_available"]},
        {"mode": "B_database_only", "candidate_availability": mode_b_candidate_availability(pair_features_unseen, all_query_ids),
         "description": MODE_DESCRIPTIONS["B_database_only"]},
        {"mode": "C_structure_absent", "candidate_availability": mode_c_candidate_availability(all_query_ids),
         "description": MODE_DESCRIPTIONS["C_structure_absent"]},
    ]
    return pd.DataFrame(rows)


def failure_stage_table(query_summary, pair_features, all_query_ids, target_col="target_abs_mass_error_ppm",
                         tolerance_ppm=None, query_id_col="query_id", is_true_col="is_true_candidate"):
    """Section 5's diagnostic table: separates "missing from candidate pool" (mass retrieval
    never found it) from "retrieved but no usable direct reference score" (candidate-generation
    succeeded, spectral evidence didn't) from "retrieved and scored" -- three DIFFERENT failure
    stages the previous notebook's single `candidate_recall` number conflated."""
    all_query_ids = list(all_query_ids)
    if tolerance_ppm is not None:
        in_pool = set(query_summary.loc[query_summary[target_col] <= tolerance_ppm, query_id_col])
    else:
        in_pool = set(query_summary[query_id_col])
    scored = set(pair_features.loc[pair_features[is_true_col] & pair_features["has_reference_spectrum"], "query_id"])

    missing = [q for q in all_query_ids if q not in in_pool]
    retrieved_unscored = [q for q in all_query_ids if q in in_pool and q not in scored]
    retrieved_scored = [q for q in all_query_ids if q in in_pool and q in scored]

    n = len(all_query_ids)
    return pd.DataFrame([
        {"failure_stage": "missing from candidate pool", "count": len(missing), "fraction": len(missing) / n if n else float("nan")},
        {"failure_stage": "candidate retrieved, direct reference unavailable", "count": len(retrieved_unscored), "fraction": len(retrieved_unscored) / n if n else float("nan")},
        {"failure_stage": "candidate retrieved and scored", "count": len(retrieved_scored), "fraction": len(retrieved_scored) / n if n else float("nan")},
    ])
