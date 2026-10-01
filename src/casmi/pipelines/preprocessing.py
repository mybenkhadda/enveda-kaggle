"""The top-level preprocessing orchestrator.

`run_preprocessing(paths, config)` is the one function notebook 01 calls; everything it does
is implemented in `casmi.io` / `casmi.data` / `casmi.chemistry` / `casmi.spectra` /
`casmi.validation`, never inline here -- this module's job is sequencing and provenance, not
domain logic.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from casmi.chemistry.adducts import build_adduct_reference, neutral_mass_from_precursor, precursor_from_neutral_mass
from casmi.chemistry.structures import build_structure_table
from casmi.data.aggregation import build_molecule_metadata, build_test_molecule_map
from casmi.data.metadata import (
    load_test_metadata,
    load_train_metadata,
    optimize_metadata_dtypes,
    summarize_collision_energy,
)
from casmi.data.schema import validate_test_schema, validate_train_schema
from casmi.io.artifacts import artifact_is_valid, artifact_paths, load_artifact, save_artifact, save_artifact_metadata
from casmi.io.files import file_fingerprint, validate_required_files
from casmi.spectra.streaming import process_spectrum_file
from casmi.utils.logging import get_logger
from casmi.utils.timing import timer
from casmi.validation.checks import (
    CheckResult,
    check_no_row_loss,
    check_structure_mapping,
    check_test_ids,
    check_unique,
    summarize_checks,
)

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    artifacts: dict
    checks: dict
    summary: dict
    runtime_seconds: float


def _maybe_cached(name, directory, config, force, build_fn, input_fingerprints=None, description=""):
    """Load `<name>` from `directory` if it's already valid for this exact config + inputs
    (and `force=False`); otherwise build it, save it, and return the fresh DataFrame. Always
    returns `(df, path, was_cached)`."""
    path, _ = artifact_paths(name, directory)
    if not force and artifact_is_valid(name, directory, config=config, input_fingerprints=input_fingerprints):
        logger.info("cache hit: %s", name)
        return load_artifact(name, directory), path, True
    df = build_fn()
    save_artifact(df, name, directory, config=config, description=description, input_fingerprints=input_fingerprints)
    return df, path, False


def run_preprocessing(paths, config, force=False):
    """Run the full preprocessing pipeline and return a `PreprocessingResult`.

    Idempotent by default (`force=False`): every artifact is skipped and loaded from disk
    instead of rebuilt if it already matches the current `config` and its recorded input
    fingerprints (see `casmi.io.artifacts.artifact_is_valid`). Pass `force=True` to rebuild
    everything regardless.
    """
    t0 = time.perf_counter()
    checks = []
    artifacts = {}

    # 1-2. validate raw files + schemas ----------------------------------------------------
    with timer("validate raw files + schemas", logger_=logger):
        validate_required_files(paths)
        checks.append(validate_train_schema(paths.train))
        checks.append(validate_test_schema(paths.test))

    train_fp = file_fingerprint(paths.train)
    test_fp = file_fingerprint(paths.test)

    # 3-5. lightweight metadata: load, summarize CE, optimize dtypes -----------------------
    with timer("load + optimize metadata", logger_=logger):
        train_meta = optimize_metadata_dtypes(summarize_collision_energy(load_train_metadata(paths)))
        test_meta = optimize_metadata_dtypes(summarize_collision_energy(load_test_metadata(paths)))
        checks.append(check_unique(train_meta["train_spectrum_id"], name="train_spectrum_id unique"))
        checks.append(check_test_ids(test_meta))

    # 6. audit structure labels (logged, not an artifact) -----------------------------------
    n_unique_smiles = int(train_meta["normalized_smiles"].nunique())
    n_unique_provided_inchikey14 = int(train_meta["inchikey14"].nunique())
    logger.info("audit: %d unique normalized_smiles, %d unique provided inchikey14 (raw, pre-tautomer)",
                n_unique_smiles, n_unique_provided_inchikey14)

    # 7. build/load structure table ---------------------------------------------------------
    with timer("build_structure_table", n_items=n_unique_smiles, logger_=logger):
        structure_table, structure_path, _ = _maybe_cached(
            "structure_table", paths.interim, config, force,
            build_fn=lambda: build_structure_table(
                train_meta["normalized_smiles"], n_jobs=config.n_jobs,
                max_tautomers=config.tautomer_max_tautomers, max_transforms=config.tautomer_max_transforms,
            ),
            input_fingerprints={"train": train_fp},
            description="One row per unique train normalized_smiles: identity + descriptors",
        )
    artifacts["structure_table"] = structure_path
    checks.append(CheckResult(
        name="structure_table parse rate",
        passed=bool(structure_table["parse_ok"].mean() > 0.99),
        detail=f"{100 * structure_table['parse_ok'].mean():.2f}% parsed",
    ))

    # 8-9. validate connectivity + map structure identity back to train ---------------------
    with timer("map connectivity_key onto train", logger_=logger):
        smiles_to_key = structure_table.set_index("smiles")["connectivity_key"]
        train_meta = train_meta.assign(
            connectivity_key=train_meta["normalized_smiles"].map(smiles_to_key),
            exact_mass=train_meta["normalized_smiles"].map(structure_table.set_index("smiles")["exact_mass"]),
        )
        checks.append(check_structure_mapping(train_meta, "connectivity_key"))

    # 10. build molecule metadata -------------------------------------------------------------
    molecule_metadata, molecule_metadata_path, _ = _maybe_cached(
        "molecule_metadata", paths.processed, config, force,
        build_fn=lambda: build_molecule_metadata(train_meta, structure_table),
        input_fingerprints={"train": train_fp},
        description="One row per train connectivity_key",
    )
    artifacts["molecule_metadata"] = molecule_metadata_path

    # 11. build adduct reference --------------------------------------------------------------
    adduct_reference, adduct_reference_path, _ = _maybe_cached(
        "adduct_reference", paths.processed, config, force,
        build_fn=lambda: build_adduct_reference(pd.concat([train_meta["adduct"], test_meta["adduct"]], ignore_index=True)),
        input_fingerprints={"train": train_fp, "test": test_fp},
        description="Parsed adduct info + frequency, train+test combined",
    )
    artifacts["adduct_reference"] = adduct_reference_path
    n_total = int(adduct_reference["n_occurrences"].sum())
    n_supported = int(adduct_reference.loc[adduct_reference["supported"], "n_occurrences"].sum())
    checks.append(CheckResult(
        name="adduct support rate",
        passed=(n_total > 0 and n_supported / n_total > 0.999),
        detail=f"{n_supported}/{n_total} spectra have a supported adduct",
    ))

    # 12. validate adduct algebra (round-trip: mz -> neutral -> mz) --------------------------
    with timer("validate adduct algebra", logger_=logger):
        sample = train_meta.dropna(subset=["precursor_mz", "adduct"])
        sample = sample.sample(min(2000, len(sample)), random_state=config.seed) if len(sample) else sample
        errors = []
        for mz, adduct in zip(sample["precursor_mz"], sample["adduct"]):
            neutral = neutral_mass_from_precursor(mz, adduct)
            if neutral is None:
                continue
            back = precursor_from_neutral_mass(neutral, adduct)
            errors.append(abs(back - mz))
        max_err = max(errors) if errors else float("nan")
        checks.append(CheckResult(
            name="adduct algebra round-trip",
            passed=bool(errors) and max_err < 1e-6,
            detail=f"max abs error over {len(errors)} sampled spectra: {max_err:.2e} Da",
        ))

    # 13-14. train neutral mass + mass error ---------------------------------------------------
    with timer("train neutral mass / mass error", logger_=logger):
        expected_mz = [precursor_from_neutral_mass(m, a) for m, a in zip(train_meta["exact_mass"], train_meta["adduct"])]
        train_meta = train_meta.assign(expected_precursor_mz=expected_mz)
        train_meta["mass_error_da"] = train_meta["precursor_mz"] - train_meta["expected_precursor_mz"]
        train_meta["mass_error_ppm"] = 1e6 * train_meta["mass_error_da"] / train_meta["expected_precursor_mz"]

    # 15. test neutral mass -------------------------------------------------------------------
    with timer("test neutral mass", logger_=logger):
        test_meta = test_meta.assign(
            neutral_mass=[neutral_mass_from_precursor(mz, a) for mz, a in zip(test_meta["precursor_mz"], test_meta["adduct"])]
        )

    train_meta_path = save_artifact(
        train_meta, "train_spectrum_metadata", paths.processed, config=config, input_fingerprints={"train": train_fp},
        description="Non-peak train columns + minted train_spectrum_id + CE summary + connectivity_key + mass error",
    )
    test_meta_path = save_artifact(
        test_meta, "test_spectrum_metadata", paths.processed, config=config, input_fingerprints={"test": test_fp},
        description="Non-peak test columns + CE summary + neutral mass",
    )
    artifacts["train_spectrum_metadata"] = train_meta_path
    artifacts["test_spectrum_metadata"] = test_meta_path

    # 16-17. stream train/test spectra -> per-spectrum features --------------------------------
    train_features_path, _ = artifact_paths("train_spectrum_features", paths.processed)
    if force or not artifact_is_valid("train_spectrum_features", paths.processed, config=config, input_fingerprints={"train": train_fp}):
        with timer("stream train spectrum features", n_items=len(train_meta), logger_=logger):
            process_spectrum_file(
                paths.train, train_features_path, mz_col="ms2_mzs", intensity_col="ms2_normalized_intensities",
                precursor_col="precursor_mz", id_cols=None, batch_size=config.peak_batch_size,
                fragment_above_precursor_da=config.fragment_above_precursor_da,
                extreme_fragment_mz=config.extreme_fragment_mz, float_dtype=config.float_dtype,
            )
        save_artifact_metadata("train_spectrum_features", paths.processed, row_count=len(train_meta),
                                columns=["row_index"], config=config, input_fingerprints={"train": train_fp},
                                description="Per-train-spectrum validation flags + features, row_index-keyed (no official spectrum_id)")
    else:
        logger.info("cache hit: train_spectrum_features")
    artifacts["train_spectrum_features"] = train_features_path

    test_features_path, _ = artifact_paths("test_spectrum_features", paths.processed)
    if force or not artifact_is_valid("test_spectrum_features", paths.processed, config=config, input_fingerprints={"test": test_fp}):
        with timer("stream test spectrum features", n_items=len(test_meta), logger_=logger):
            process_spectrum_file(
                paths.test, test_features_path, mz_col="ms2_mzs", intensity_col="ms2_normalized_intensities",
                precursor_col="precursor_mz", id_cols=["spectrum_id", "molecule_id"], batch_size=config.peak_batch_size,
                fragment_above_precursor_da=config.fragment_above_precursor_da,
                extreme_fragment_mz=config.extreme_fragment_mz, float_dtype=config.float_dtype,
            )
        save_artifact_metadata("test_spectrum_features", paths.processed, row_count=len(test_meta),
                                columns=["row_index", "spectrum_id", "molecule_id"], config=config,
                                input_fingerprints={"test": test_fp},
                                description="Per-test-spectrum validation flags + features, keyed by spectrum_id/molecule_id")
    else:
        logger.info("cache hit: test_spectrum_features")
    artifacts["test_spectrum_features"] = test_features_path

    # 18. build test molecule map ------------------------------------------------------------
    test_molecule_map, test_molecule_map_path, _ = _maybe_cached(
        "test_molecule_map", paths.processed, config, force,
        build_fn=lambda: build_test_molecule_map(test_meta),
        input_fingerprints={"test": test_fp},
        description="One row per test molecule_id, aggregating its constituent spectra",
    )
    artifacts["test_molecule_map"] = test_molecule_map_path

    # 19. artifacts are already saved incrementally above; nothing further to do here.

    # 20. final validation checks -------------------------------------------------------------
    checks.append(check_no_row_loss(n_unique_smiles, len(structure_table), name="structure_table row count matches unique SMILES"))
    checks.append(CheckResult(
        name="test_molecule_map covers every test spectrum",
        passed=int(test_molecule_map["n_spectra"].sum()) == len(test_meta),
        detail=f"{int(test_molecule_map['n_spectra'].sum())}/{len(test_meta)} test spectra accounted for",
    ))
    all_passed, checks_dict = summarize_checks(checks)

    runtime = time.perf_counter() - t0
    summary = {
        "n_train_spectra": len(train_meta),
        "n_test_spectra": len(test_meta),
        "n_unique_train_smiles": n_unique_smiles,
        "n_connectivity_keys": len(molecule_metadata),
        "n_test_molecules": len(test_molecule_map),
        "adduct_support_rate": n_supported / n_total if n_total else float("nan"),
        "median_spectra_per_train_molecule": float(molecule_metadata["n_spectra"].median()) if len(molecule_metadata) else float("nan"),
        "median_spectra_per_test_molecule": float(test_molecule_map["n_spectra"].median()) if len(test_molecule_map) else float("nan"),
        "all_checks_passed": all_passed,
        "runtime_seconds": runtime,
    }

    logger.info("run_preprocessing complete in %.1fs; all_checks_passed=%s", runtime, all_passed)
    return PreprocessingResult(artifacts=artifacts, checks=checks_dict, summary=summary, runtime_seconds=runtime)
