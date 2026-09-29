"""casmi_infer -- minimal, offline, RDKit-free Mode-A inference for CASMI 2026 (Kaggle).

CLOSED-WORLD / CLASS-1 ONLY: candidates are the training structure library; no external candidate
expansion, no Mode B. The first submission is an anchor, not the final solution.

Shared-source packaging: every algorithm that must match training (peak cleaning, deterministic
top-N truncation, peak hashing, compatibility ordering, protocol eligibility, the four similarity
kernels, reference aggregation, adduct arithmetic) is IMPORTED from the `casmi` package subset that
`casmi.bundle_export` copies verbatim into `bundle/code/` (see `SHARED_CASMI_MODULES`). The only
re-implementations here are (a) the collision-energy mean (`spectrum.ce_mean`) and (b) a vectorized
form of `compat_sort_key` (`compat.compat_order`); both have parity tests.
"""
__version__ = "0.1.0"
BUNDLE_FORMAT = "casmi-modeA-bundle-1"

# modules of the `casmi` package the inference path needs -- exported verbatim, nothing else
SHARED_CASMI_MODULES = (
    "casmi/__init__.py",
    "casmi/chemistry/__init__.py", "casmi/chemistry/adducts.py",
    "casmi/spectra/__init__.py", "casmi/spectra/binning.py", "casmi/spectra/similarity.py",
    "casmi/spectra/neutral_loss.py", "casmi/spectra/preprocessing.py", "casmi/spectra/deduplication.py",
    "casmi/spectra/reference_selection.py",
    "casmi/ranking/__init__.py", "casmi/ranking/aggregation.py", "casmi/ranking/features.py",
)
