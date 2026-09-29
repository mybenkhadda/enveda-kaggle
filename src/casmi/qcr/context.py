"""Shared v4b context: artifact locations, the similarity configs, and the per-spectrum metadata
lookups every QCR builder call needs -- defined ONCE so 10v4b_00, 10v4b_01 and
`scripts/v4b_build_scale_features.py` cannot drift apart.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from casmi.spectra.library import build_reference_index
from casmi.spectra.reference_selection import PROTOCOLS
from casmi.utils.hashing import stable_config_hash

HOST_INGEST_LIB = "enveda-np-examples"
SEED = 42

# EXACT config the persisted v4a.1 evidence was built with (config hash 93f7f9fa61d4, HISTORICAL
# REFERENCE) -- used to recompute "fresh" values under identical parameters.
V4A1_SIMILARITY_CONFIG = {
    "bin_width_da": 0.1, "peak_tol_da": 0.02, "max_peaks_similarity": 100,
    "near_dup_cosine_threshold": 0.95, "near_dup_precursor_diff_da": 0.01,
    "tolerant_mirror_precursor_diff_da": 0.005, "tolerant_mirror_min_rel_intensity": 0.01,
    "tolerant_mirror_ppm_tol": 5.0, "tolerant_mirror_abs_tol_da": 0.002,
    "tolerant_mirror_match_fraction": 0.95, "tolerant_mirror_min_correlation": 0.99,
    "peak_hash_mz_decimals": 4, "peak_hash_intensity_decimals": 3, "peak_hash_precursor_decimals": 3,
    "code_version": "10v4a1-1",
}

# Identical numerical parameters; only the code_version tag differs, so a v4b rebuild's cache
# keys (config_hash) can never collide with -- or silently hit -- v4a.1 cache entries.
V4B_REBUILD_SIMILARITY_CONFIG = {**V4A1_SIMILARITY_CONFIG, "code_version": "10v4b-deterministic-topn-1"}


def config_hash(similarity_config):
    return stable_config_hash(similarity_config)


@dataclass(frozen=True)
class V4bPaths:
    root: Path
    out: Path                 # outputs/v4b
    host_processed: Path      # data/processed/host_holdout
    host_interim: Path        # data/interim/host_holdout
    dev_processed: Path       # data/processed/dev_qcr
    dev_interim: Path         # data/interim/dev_qcr
    rebuild_dir: Path         # data/processed/v4b_rebuild (REBUILD_ONCE output, never overwrites v4a.1)
    scale_dir: Path           # data/processed/v4b_scale
    settlement_json: Path
    mismatch_parquet: Path

    def subdir(self, name):
        p = self.out / name
        p.mkdir(parents=True, exist_ok=True)
        return p


def v4b_paths(project_paths):
    out = project_paths.outputs / "v4b"
    out.mkdir(parents=True, exist_ok=True)
    return V4bPaths(
        root=project_paths.root, out=out,
        host_processed=project_paths.processed / "host_holdout", host_interim=project_paths.interim / "host_holdout",
        dev_processed=project_paths.processed / "dev_qcr", dev_interim=project_paths.interim / "dev_qcr",
        rebuild_dir=project_paths.processed / "v4b_rebuild", scale_dir=project_paths.processed / "v4b_scale",
        settlement_json=out / "cache_settlement.json", mismatch_parquet=out / "cache_mismatch_diagnostics.parquet",
    )


V5_SUBDIRS = ("manifests", "qcr", "features", "models", "metrics", "bootstrap", "figures", "freeze", "parity")


def v5_paths(project_paths):
    """`{name: Path}` for outputs/v5/<subdir> (created) plus "root", "bundle" (repo/bundle) and
    "submissions" (outputs/submissions)."""
    root = project_paths.outputs / "v5"
    d = {k: root / k for k in V5_SUBDIRS}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    d["root"] = root
    d["bundle"] = project_paths.root / "bundle"
    d["submissions"] = project_paths.outputs / "submissions"
    return d


def existing_evidence_artifacts(vp):
    """The persisted v4a.1 evidence layer, `{logical_name: path}`."""
    arts = {
        "host_qcr": vp.host_processed / "host_qcr.parquet",
        "host_qcr_pairs": vp.host_processed / "host_qcr_pairs.parquet",
        "dev_qcr": vp.dev_processed / "dev_qcr.parquet",
        "dev_qcr_pairs": vp.dev_processed / "dev_qcr_pairs.parquet",
    }
    for p in PROTOCOLS:
        arts[f"host_features_{p}"] = vp.host_interim / f"host_pair_features_{p}.parquet"
        arts[f"dev_features_{p}"] = vp.dev_interim / f"dev_pair_features_{p}.parquet"
    return arts


def rebuilt_evidence_artifacts(vp):
    """Where the one-time REBUILD_ONCE branch writes the replacement evidence layer."""
    d = vp.rebuild_dir
    arts = {"host_qcr": d / "host_qcr.parquet", "host_qcr_pairs": d / "host_qcr_pairs.parquet",
            "dev_qcr": d / "dev_qcr.parquet", "dev_qcr_pairs": d / "dev_qcr_pairs.parquet"}
    for p in PROTOCOLS:
        arts[f"host_features_{p}"] = d / f"host_pair_features_{p}.parquet"
        arts[f"dev_features_{p}"] = d / f"dev_pair_features_{p}.parquet"
    return arts


class SpectrumLookups:
    """`ref_meta_of`, `offset_of` and `reference_index` exactly as the v4a.1 builder defined them."""

    def __init__(self, train_meta):
        tm = train_meta.set_index("train_spectrum_id")
        self._adduct = tm["adduct"].to_dict()
        self._ce = tm["ce_mean"].to_dict()
        self._ion = tm["ionization_mode"].to_dict()
        self._inst = tm["instrument_type"].to_dict()
        self._src = tm["ingest_lib"].to_dict()
        self.reference_index = build_reference_index(train_meta, connectivity_col="connectivity_key", id_col="train_spectrum_id")

    @staticmethod
    def offset_of(spectrum_id):
        return int(str(spectrum_id).split("_")[1])

    def ref_meta_of(self, spectrum_id):
        ce = self._ce.get(spectrum_id)
        ce = ce if pd.notna(ce) else None
        return {"spectrum_id": spectrum_id, "adduct": self._adduct.get(spectrum_id), "ion_mode": self._ion.get(spectrum_id),
                "ce": ce, "ce_unit": "eV" if ce is not None else None, "instrument": self._inst.get(spectrum_id),
                "source": self._src.get(spectrum_id)}

    def source_of(self, spectrum_id):
        return self._src.get(spectrum_id)
