"""CASMI 2026 preprocessing package.

Notebooks orchestrate; this package holds the reusable logic:

    casmi.io          parquet/file/artifact I/O
    casmi.data        train/test schema + metadata + aggregation
    casmi.chemistry   SMILES/structure/adduct chemistry
    casmi.spectra     peak validation/preprocessing/features/streaming
    casmi.validation  reusable checks + retrieval metrics
    casmi.pipelines   the top-level orchestrator (run_preprocessing)
    casmi.utils       logging/timing/memory/hashing helpers
"""
__version__ = "0.1.0"
