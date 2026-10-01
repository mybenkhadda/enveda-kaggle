"""Unified OPEN structure universe: TRAIN (the exact training candidate library) + external sources
(COCONUT first), one row per competition connectivity, provenance preserved.

Rules (deterministic, documented -- no implementation default decides them):
  * TRAIN connectivities keep the TRAINING representative SMILES (first mass variant by
    mass_variant_id, as the bundle export does), formula and mass variants: the training retrieval
    contract is not altered for structures the closed-world model already knows.
  * External-only connectivities: representative = normalized SMILES of the first record by
    (source order, source_id); mass variants = the distinct standardized masses rounded to 5 decimals
    (the training `build_molecule_mass_variants` precision).
  * Shared connectivities additionally record `open_representative_smiles` and the |TRAIN - external|
    mass difference (audit only).
  * `has_reference_spectrum` is METADATA (TRAIN connectivities with >= 1 training spectrum). It is on
    `FORBIDDEN_SHORTCUT_FEATURES` and must never become a raw ranker feature.
"""
import numpy as np
import pandas as pd

from casmi.candidates.provenance import aggregate_provenance

SOURCE_ORDER = {"TRAIN": 0, "COCONUT": 1, "LOTUS": 2, "NPATLAS": 3, "PUBCHEM": 4}
MASS_PRECISION = 5
UNIFIED_COLUMNS = ["connectivity_key", "representative_smiles", "open_representative_smiles", "formula", "exact_mass", "in_train",
                   "has_reference_spectrum", "candidate_sources", "source_ids", "n_source_records", "n_mass_variants", "name",
                   "train_vs_open_mass_diff_da"]


def train_structures(mass_variants):
    """`molecule_mass_variants` -> (structures one row per connectivity, variants (connectivity_key, exact_mass), records)."""
    mv = mass_variants.dropna(subset=["exact_mass"]).copy()
    rep = mv.sort_values(["connectivity_key", "mass_variant_id"], kind="mergesort").drop_duplicates("connectivity_key")
    n_spec = mv.groupby("connectivity_key")["n_train_spectra"].sum() if "n_train_spectra" in mv.columns else None
    st = pd.DataFrame({"connectivity_key": rep["connectivity_key"].to_numpy(), "train_smiles": rep["representative_smiles"].to_numpy(),
                       "train_formula": rep["molecular_formula"].to_numpy(), "train_mass": rep["exact_mass"].to_numpy(float)})
    st["has_reference_spectrum"] = (st["connectivity_key"].map(n_spec).fillna(0).to_numpy() > 0) if n_spec is not None else True
    variants = mv[["connectivity_key", "exact_mass"]].assign(variant_source="TRAIN")
    records = pd.DataFrame({"connectivity_key": st["connectivity_key"], "source": "TRAIN", "source_id": st["connectivity_key"]})
    return st, variants, records


def _external_representatives(ext):
    e = ext.assign(_o=ext["source"].map(SOURCE_ORDER).fillna(99)).sort_values(["connectivity_key", "_o", "source_id"], kind="mergesort")
    first = e.drop_duplicates("connectivity_key")
    return pd.DataFrame({"connectivity_key": first["connectivity_key"].to_numpy(), "open_representative_smiles": first["representative_smiles"].to_numpy(),
                         "open_formula": first["molecular_formula"].to_numpy(), "open_mass": first["neutral_monoisotopic_mass"].to_numpy(float),
                         "name": first["name"].to_numpy() if "name" in first else None})


def unify(mass_variants, external_standardized):
    """Returns (unified, variants). `external_standardized`: rows from `standardize_records` (any number
    of sources concatenated)."""
    st, tv, trec = train_structures(mass_variants)
    ext = external_standardized.dropna(subset=["connectivity_key"])
    prov = aggregate_provenance(pd.concat([trec, ext[["connectivity_key", "source", "source_id"]]], ignore_index=True))
    u = prov.merge(st, on="connectivity_key", how="left").merge(_external_representatives(ext), on="connectivity_key", how="left")
    u["in_train"] = u["train_smiles"].notna()
    u["representative_smiles"] = np.where(u["in_train"], u["train_smiles"], u["open_representative_smiles"])
    u["formula"] = np.where(u["in_train"], u["train_formula"], u["open_formula"])
    u["exact_mass"] = np.where(u["in_train"], u["train_mass"], u["open_mass"]).astype(float)
    u["has_reference_spectrum"] = u["has_reference_spectrum"].fillna(False).astype(bool)
    u["train_vs_open_mass_diff_da"] = (u["train_mass"] - u["open_mass"]).abs()

    ext_only = ext[~ext["connectivity_key"].isin(set(st["connectivity_key"]))]
    ev = (ext_only.assign(_m=ext_only["neutral_monoisotopic_mass"].astype(float).round(MASS_PRECISION))
          .groupby(["connectivity_key", "_m"], sort=True)["neutral_monoisotopic_mass"].mean().reset_index()
          .rename(columns={"neutral_monoisotopic_mass": "exact_mass"}).drop(columns="_m").assign(variant_source="EXTERNAL"))
    variants = pd.concat([tv, ev], ignore_index=True).sort_values(["connectivity_key", "exact_mass"], kind="mergesort").reset_index(drop=True)
    u["n_mass_variants"] = u["connectivity_key"].map(variants.groupby("connectivity_key").size()).fillna(0).astype(int)
    if "name" not in u:
        u["name"] = None
    u = u[UNIFIED_COLUMNS].sort_values("connectivity_key", kind="mergesort").reset_index(drop=True)
    assert u["connectivity_key"].is_unique, "unified universe must hold one row per connectivity"
    assert u["representative_smiles"].notna().all(), "every connectivity needs a representative SMILES"
    return u, variants


def source_summary(unified, standardized, rejected, n_input_records):
    """Counts per source + TRAIN overlap (user inspects after running)."""
    rows = []
    for s in sorted(set(standardized["source"]) | set(rejected["source"])):
        sub = standardized[standardized["source"] == s]
        keys = set(sub["connectivity_key"])
        in_train = set(unified.loc[unified["in_train"], "connectivity_key"])
        rows.append({"source": s, "n_input_records": int(n_input_records.get(s, 0)), "n_standardized": int(len(sub)),
                     "n_rejected": int((rejected["source"] == s).sum()), "n_unique_connectivities": len(keys),
                     "n_overlap_with_train": len(keys & in_train), "n_new_vs_train": len(keys - in_train)})
    tr = unified["in_train"]
    rows.append({"source": "UNIFIED", "n_unique_connectivities": int(len(unified)), "n_overlap_with_train": int(tr.sum()),
                 "n_new_vs_train": int((~tr).sum())})
    return pd.DataFrame(rows)
