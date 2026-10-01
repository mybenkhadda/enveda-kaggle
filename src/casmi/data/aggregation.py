"""Spectrum-level -> molecule-level aggregation.

The prediction unit differs by split: train's is `connectivity_key` (built from
`normalized_smiles` via the structure table), never a fabricated train molecule id; test's is
its own official `molecule_id`.
"""
import pandas as pd


def build_molecule_metadata(train_metadata, structure_table, smiles_col="normalized_smiles",
                             adduct_col="adduct", instrument_col="instrument_type",
                             ion_mode_col="ionization_mode", precursor_col="precursor_mz"):
    """One row per train `connectivity_key` (NOT per spectrum, NOT a fabricated per-row id).

    `train_metadata`: from `casmi.data.metadata.load_train_metadata`.
    `structure_table`: from `casmi.chemistry.structures.build_structure_table`, run over
        `train_metadata[smiles_col]` -- provides the `smiles -> connectivity_key` mapping.
    """
    smiles_to_key = structure_table.set_index("smiles")["connectivity_key"]
    df = train_metadata.copy()
    df["connectivity_key"] = df[smiles_col].map(smiles_to_key)

    n_unmapped = df["connectivity_key"].isna().sum()
    if n_unmapped:
        df = df.dropna(subset=["connectivity_key"])

    g = df.groupby("connectivity_key")
    out = pd.DataFrame({
        "n_spectra": g.size(),
        "n_adducts": g[adduct_col].nunique(),
        "n_instruments": g[instrument_col].nunique(),
        "n_ion_modes": g[ion_mode_col].nunique(),
        "precursor_mz_min": g[precursor_col].min(),
        "precursor_mz_max": g[precursor_col].max(),
    }).reset_index()
    out["precursor_mz_range"] = out["precursor_mz_max"] - out["precursor_mz_min"]
    out.attrs["n_unmapped_spectra"] = int(n_unmapped)
    return out


def build_molecule_mass_variants(train_metadata, connectivity_col="connectivity_key", mass_col="exact_mass",
                                  smiles_col="normalized_smiles", formula_col="molecular_formula",
                                  inchikey_col="inchikey", inchikey14_col="inchikey14", mass_precision=5):
    """One row per (connectivity_key, distinct exact mass) -- a `connectivity_key` is the first
    14 characters of a tautomer-canonical InChIKey, which encodes constitution only, NOT
    isotopic substitution. A small number of keys are shared by a compound and its
    deuterium-labeled internal-standard analog, whose exact mass differs by several Da; a single
    `groupby("connectivity_key")["exact_mass"].median()` silently produces a chemically
    meaningless mass for those keys. This keeps every real mass as its own variant instead.

    `mass_precision`: masses are rounded to this many decimal places before grouping, so two
    SMILES for the same tautomer that RDKit happened to sum in a different floating-point order
    (differing at the ~1e-9 Da level) collapse into one variant, while a genuine isotope-label
    difference (multiple Da) does not.
    """
    df = train_metadata[[connectivity_col, mass_col, smiles_col, formula_col, inchikey_col, inchikey14_col]].copy()
    df["_mass_rounded"] = df[mass_col].round(mass_precision)

    g = df.groupby([connectivity_col, "_mass_rounded"])
    variants = g.agg(
        exact_mass=(mass_col, "mean"),
        representative_smiles=(smiles_col, "first"),
        molecular_formula=(formula_col, "first"),
        n_train_spectra=(mass_col, "size"),
        provided_inchikey=(inchikey_col, "first"),
        provided_inchikey14=(inchikey14_col, "first"),
    ).reset_index().drop(columns="_mass_rounded")
    variants = variants.sort_values([connectivity_col, "exact_mass"]).reset_index(drop=True)

    n_variants_per_key = variants.groupby(connectivity_col)[connectivity_col].transform("size")
    variants["is_isotope_variant"] = n_variants_per_key > 1
    variants["mass_variant_id"] = (
        variants[connectivity_col] + "__v" + variants.groupby(connectivity_col).cumcount().astype(str)
    )
    return variants[["mass_variant_id", connectivity_col, "exact_mass", "representative_smiles",
                      "molecular_formula", "is_isotope_variant", "n_train_spectra",
                      "provided_inchikey", "provided_inchikey14"]]


def build_test_molecule_map(test_metadata, molecule_id_col="molecule_id", spectrum_id_col="spectrum_id",
                             adduct_col="adduct", instrument_col="instrument_type",
                             ion_mode_col="ionization_mode", precursor_col="precursor_mz"):
    """One row per test `molecule_id`, aggregating its constituent spectra -- the mapping
    later stages use to go from a per-molecule submission row back to the spectra that
    produced it."""
    g = test_metadata.groupby(molecule_id_col)
    out = pd.DataFrame({
        "n_spectra": g.size(),
        "spectrum_ids": g[spectrum_id_col].apply(list),
        "n_adducts": g[adduct_col].nunique(),
        "adducts": g[adduct_col].apply(lambda s: sorted(s.unique())),
        "n_instruments": g[instrument_col].nunique(),
        "n_ion_modes": g[ion_mode_col].nunique(),
        "precursor_mz_min": g[precursor_col].min(),
        "precursor_mz_max": g[precursor_col].max(),
    }).reset_index()
    out["precursor_mz_range"] = out["precursor_mz_max"] - out["precursor_mz_min"]
    return out
