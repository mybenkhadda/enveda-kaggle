"""Typed pipeline configuration.

A `PreprocessingConfig` instance is threaded through every pipeline stage and hashed
(`casmi.utils.hashing.stable_config_hash`) into every artifact's provenance metadata, so a
changed knob invalidates dependent caches rather than silently reusing stale output.
"""
from dataclasses import dataclass, field


@dataclass
class PreprocessingConfig:
    seed: int = 42
    n_jobs: int = 8

    # --- structure processing (chemistry.structures / chemistry.connectivity) -------------
    tautomer_max_tautomers: int | None = None   # None = RDKit default (closest to the
                                                 # competition's own canonicalizer)
    tautomer_max_transforms: int | None = None

    # --- spectrum streaming / features (spectra.streaming, spectra.features) --------------
    peak_batch_size: int = 50_000
    fragment_above_precursor_da: float = 5.0
    extreme_fragment_mz: float = 5000.0
    float_dtype: str = "float32"

    # --- adduct / mass physics (chemistry.adducts) -----------------------------------------
    mass_tolerance_grid_ppm: list[float] = field(
        default_factory=lambda: [1, 2, 3, 5, 10, 20, 50]
    )

    # --- molecule-level aggregation (data.aggregation) -------------------------------------
    n_mass_bins: int = 10

    def __post_init__(self):
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")
        if self.peak_batch_size < 1:
            raise ValueError(f"peak_batch_size must be >= 1, got {self.peak_batch_size}")
        if self.float_dtype not in ("float32", "float64"):
            raise ValueError(f"float_dtype must be 'float32' or 'float64', got {self.float_dtype!r}")
