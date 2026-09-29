"""casmi_runtime -- ORCHESTRATION for running the frozen CASMI bundle on Colab / Kaggle (GPU-first, parity-safe).

Code authority:
    casmi_runtime                 this package (GitHub runtime repo): discovery, accelerator detection, backend
                                  selection, the run sequence, reports, result storage. It never implements inference.
    <bundle>/code/casmi_infer     the FROZEN inference implementation bound to the bundle's CONFIG_HASH. Loaded by
                                  `frozen.load_frozen_code`, which puts it first on sys.path and proves every module
                                  came from the bundle. Nothing here imports `casmi` / `casmi_infer` at module level.

GPU policy: a GPU backend is used only if the FROZEN bundle code implements it, the bundle config declares it
parity-validated, a GPU is present, AND the bundled self-test passes on it. Bundle v2-A7 ships no GPU backend,
so it runs on the best CPU backend that passes the self-test (numba, else numpy) and records why.
"""
__version__ = "1.0.0"
