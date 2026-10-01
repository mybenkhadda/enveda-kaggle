"""The CASMI 2026 train/test schemas, encoded explicitly -- confirmed by direct pyarrow
inspection of the real competition files, not inferred by fuzzy auto-detection. If the
competition data changes shape, `validate_train_schema`/`validate_test_schema` are the
tripwire, not a silent downstream KeyError three modules later.

Important, and easy to get backwards: **train has no official `spectrum_id` or
`molecule_id`** (those are minted in `casmi.data.metadata`); **test has no `smiles` /
`inchikey` / `molecular_formula`** (that's the prediction target).
"""
from casmi.io.parquet import get_parquet_columns
from casmi.validation.checks import check_required_columns

TRAIN_REQUIRED_COLUMNS = [
    "ingest_lib",
    "normalized_smiles",
    "inchikey",
    "inchikey14",
    "molecular_formula",
    "ionization_mode",
    "instrument_type",
    "adduct",
    "adduct_orig",
    "precursor_mz",
    "precursor_error_ppm",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "num_peaks",
    "base_peak_intensity",
    "collision_energy_ev",
    "collision_energy_orig",
    "collision_energy_orig_units",
]

TEST_REQUIRED_COLUMNS = [
    "molecule_id",
    "spectrum_id",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "base_peak_intensity",
    "adduct",
    "ionization_mode",
    "instrument_type",
    "precursor_mz",
    "collision_energy_orig",
    "collision_energy_ev",
    "collision_energy_orig_units",
]

# Columns present in train but NOT test (the prediction target + train-only provenance).
TRAIN_ONLY_COLUMNS = ["normalized_smiles", "inchikey", "inchikey14", "molecular_formula", "ingest_lib", "adduct_orig", "precursor_error_ppm"]
# Columns present in test but NOT train (test's own identifiers).
TEST_ONLY_COLUMNS = ["molecule_id", "spectrum_id"]

# Peak / acquisition columns shared by both splits, used directly by casmi.spectra.*.
SHARED_SPECTRAL_COLUMNS = [
    "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity", "adduct",
    "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_ev",
    "collision_energy_orig", "collision_energy_orig_units",
]


def validate_train_schema(path):
    columns = get_parquet_columns(path)
    return check_required_columns(columns, TRAIN_REQUIRED_COLUMNS, name="train schema")


def validate_test_schema(path):
    columns = get_parquet_columns(path)
    return check_required_columns(columns, TEST_REQUIRED_COLUMNS, name="test schema")
