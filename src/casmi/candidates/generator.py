"""Mass-window candidate generation: precursor m/z + adduct -> a connectivity-deduplicated
pool of mass-plausible structures. A retrieval stage, not a ranking stage -- see the module
docstring in `casmi.candidates` for what this pipeline deliberately does NOT do yet.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CandidateGenerationConfig:
    tolerance_grid_ppm: tuple = (1, 2, 3, 5, 10, 20, 50)
    max_tolerance_ppm: float = 50.0
    target_recall: float = 0.995

    # A "lightweight query table for development" (not literally every one of the 2.5M train
    # spectra): MassIndex lookups are cheap (microsecond-scale binary search), but MATERIALIZING
    # one candidate row per (query, candidate) at max_tolerance_ppm for millions of queries
    # produces a hundreds-of-millions-of-rows table for no analytical benefit -- percentile
    # estimates from a large stratified sample are indistinguishable from the full-population
    # ones. Sampling is stratified by connectivity fold so every fold stays represented.
    dev_sample_size: int = 60_000

    # A second, PURPOSE-BUILT sample for molecule-level (union/intersection) evaluation: the
    # flat dev_sample_size draw above is spectrum-stratified, so most sampled connectivity_keys
    # end up with exactly one spectrum in the sample (average ~9.3 spectra/molecule in train,
    # but a 60k/2.5M ~2.4% sampling rate leaves most molecules under-sampled for a meaningful
    # union-vs-intersection comparison). This instead samples MOLECULES first, then takes up to
    # `dev_molecule_max_spectra_per_molecule` of their actual spectra -- mirroring how test
    # naturally has ~3 spectra/molecule.
    dev_molecule_sample_size: int = 4_000
    dev_molecule_max_spectra_per_molecule: int = 10

    # Adaptive (per-group) mass-tolerance policy, evaluated in `casmi.candidates.adaptive`
    # against the global tolerance grid -- not assumed better, only tested.
    adaptive_target_quantile: float = 0.995
    adaptive_min_group_size: int = 100

    mass_dtype: str = "float64"
    save_candidate_pools: bool = True
    force_recompute: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.max_tolerance_ppm < max(self.tolerance_grid_ppm):
            raise ValueError("max_tolerance_ppm must be >= every tolerance in tolerance_grid_ppm")


class CandidateGenerator:
    """Wraps a `MassIndex`: turns per-query neutral masses into per-(query, candidate) rows."""

    def __init__(self, mass_index):
        self.mass_index = mass_index

    def generate(self, neutral_mass, tolerance_ppm):
        """Single-query API: `(candidate_keys, candidate_masses)`, closest-first."""
        return self.mass_index.query_with_masses(neutral_mass, tolerance_ppm=tolerance_ppm)

    def generate_dataframe(self, queries, tolerance_ppm=None, tolerance_col=None, query_id_col="query_id",
                            mass_col="neutral_mass", true_key_col="true_connectivity_key", extra_cols=()):
        """Batch API: one row per (query, candidate) pair within tolerance.

        Exactly one of `tolerance_ppm` (a single value shared by every query) or
        `tolerance_col` (a column name in `queries` -- each query brings its own tolerance,
        e.g. from `casmi.candidates.adaptive.apply_adaptive_mass_windows`) must be given.

        `queries`: a DataFrame with at least `query_id_col`/`mass_col`. Rows with a null mass
        (e.g. an unsupported adduct) are skipped -- they simply contribute zero candidate rows,
        never a crash. `true_key_col`, if present in `queries`, is carried onto every candidate
        row as `is_true_candidate`/`true_connectivity_key` for development-only recall scoring;
        omit it (or pass a column that doesn't exist) for test queries, where the true
        connectivity is the prediction target and genuinely unknown.

        Iterates query-by-query over plain numpy arrays (not `DataFrame.iterrows`/`to_dict`) --
        each step is one `np.searchsorted` binary search, so this comfortably handles tens of
        thousands of queries; ragged per-query candidate counts are the reason this can't be a
        single vectorized array op.
        """
        if (tolerance_ppm is None) == (tolerance_col is None):
            raise ValueError("give exactly one of tolerance_ppm or tolerance_col")

        has_true_key = true_key_col in queries.columns
        query_ids = queries[query_id_col].to_numpy()
        masses = queries[mass_col].to_numpy(dtype=float)
        per_row_tolerance = queries[tolerance_col].to_numpy(dtype=float) if tolerance_col else None
        true_keys = queries[true_key_col].to_numpy() if has_true_key else None
        extra_arrays = {c: queries[c].to_numpy() for c in extra_cols if c in queries.columns}

        rows = {
            "query_id": [], "candidate_connectivity_key": [], "query_neutral_mass": [],
            "candidate_exact_mass": [], "tolerance_ppm": [],
        }
        if has_true_key:
            rows["true_connectivity_key"] = []
        for c in extra_arrays:
            rows[c] = []

        for i in range(len(queries)):
            mass = masses[i]
            if np.isnan(mass):
                continue
            this_tolerance = tolerance_ppm if tolerance_col is None else per_row_tolerance[i]
            cand_keys, cand_masses = self.mass_index.query_with_masses(mass, tolerance_ppm=this_tolerance)
            n = len(cand_keys)
            if n == 0:
                continue
            rows["query_id"].extend([query_ids[i]] * n)
            rows["candidate_connectivity_key"].extend(cand_keys)
            rows["query_neutral_mass"].extend([mass] * n)
            rows["candidate_exact_mass"].extend(cand_masses)
            rows["tolerance_ppm"].extend([this_tolerance] * n)
            if has_true_key:
                rows["true_connectivity_key"].extend([true_keys[i]] * n)
            for c, arr in extra_arrays.items():
                rows[c].extend([arr[i]] * n)

        pool = pd.DataFrame(rows)
        if len(pool) == 0:
            pool["mass_error_da"] = pd.Series(dtype=float)
            pool["mass_error_ppm"] = pd.Series(dtype=float)
            pool["abs_mass_error_ppm"] = pd.Series(dtype=float)
        else:
            pool["mass_error_da"] = pool["candidate_exact_mass"] - pool["query_neutral_mass"]
            pool["mass_error_ppm"] = 1e6 * pool["mass_error_da"] / pool["query_neutral_mass"]
            pool["abs_mass_error_ppm"] = pool["mass_error_ppm"].abs()
        if has_true_key:
            pool["is_true_candidate"] = pool["candidate_connectivity_key"] == pool["true_connectivity_key"]

        assert not pool[["query_id", "candidate_connectivity_key"]].duplicated().any(), (
            "generate_dataframe should never emit more than one row per (query, candidate) -- "
            "MassIndex.query_with_masses already returns each connectivity_key at most once."
        )
        return pool


def dedupe_pool_to_connectivity(pool, variant_to_connectivity, variant_col="candidate_connectivity_key"):
    """Collapse a MASS-VARIANT-level candidate pool (one row per (query, mass_variant_id) --
    what you get from `CandidateGenerator.generate_dataframe` when its `MassIndex` was built
    over `molecule_mass_variants` rather than one-mass-per-connectivity) down to one row per
    (query, connectivity_key), the competition's actual candidate unit.

    Keeps the BEST (lowest `abs_mass_error_ppm`) variant's row for each (query, connectivity),
    and adds `mass_variant_support_count` -- how many of that connectivity's mass variants fell
    within the tolerance window (almost always 1; >1 only for the rare isotope-ambiguous keys).
    `variant_col` names the column currently holding `mass_variant_id` values (defaults to
    `candidate_connectivity_key`, i.e. the raw output of `generate_dataframe` before this
    function renames it).
    """
    pool = pool.copy()
    true_variant_col = "candidate_mass_variant_id" if variant_col == "candidate_connectivity_key" else variant_col
    if variant_col == "candidate_connectivity_key":
        pool = pool.rename(columns={variant_col: true_variant_col})
    pool["candidate_connectivity_key"] = pool[true_variant_col].map(variant_to_connectivity)
    if "true_connectivity_key" in pool.columns:
        # is_true_candidate was computed against mass_variant_id values at generation time
        # (always False, since a variant id like "KEY__v0" never equals "KEY") -- recompute it
        # now that candidate_connectivity_key holds real connectivity keys again.
        pool["is_true_candidate"] = pool["candidate_connectivity_key"] == pool["true_connectivity_key"]

    if len(pool) == 0:
        pool["mass_variant_support_count"] = pd.Series(dtype=int)
        return pool

    support = pool.groupby(["query_id", "candidate_connectivity_key"]).size().rename("mass_variant_support_count")
    best_idx = pool.groupby(["query_id", "candidate_connectivity_key"])["abs_mass_error_ppm"].idxmin()
    deduped = pool.loc[best_idx].merge(support.reset_index(), on=["query_id", "candidate_connectivity_key"])

    assert not deduped[["query_id", "candidate_connectivity_key"]].duplicated().any(), (
        "dedupe_pool_to_connectivity should never emit more than one row per (query, connectivity_key)"
    )
    return deduped.reset_index(drop=True)
