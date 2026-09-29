# enveda-casmi-runtime: frozen CASMI 2026 inference on Colab / Kaggle

This is the lightweight runtime and orchestration code for running the **frozen** CASMI 2026 pipeline:

* model `V1_TL_1K_TESTSIM_STRICT`;
* aggregator `MOST_CONFIDENT_SPECTRUM` (A7) with its frozen calibration temperature;
* bundle `v2-A7`.

It runs on Google Colab or Kaggle, GPU-first but parity-safe, and never needs the Windows training workspace.

```text
LOCAL WINDOWS (training repo)      training, 11_02 aggregation (A7 locked), 11_00 bundle export, 11_01 parity, v63 check
        │
        ├── GITHUB (this repo)       lightweight runtime / orchestration source: src/casmi_runtime, src/casmi_infer,
        │                            whitelisted src/casmi, scripts, notebooks 00-03, tests
        ├── GOOGLE DRIVE             the complete, immutable bundle/ + competition files (Colab)
        └── KAGGLE DATASETS          the complete bundle/ + this repo as a "runtime source" dataset (+ optional lightgbm wheel)
                │
                ▼
        COLAB / KAGGLE   environment check -> verified bundle -> self-test -> validated backend -> inference -> submission.csv
```

| component | role |
|---|---|
| `src/casmi_runtime/` | **orchestration**: accelerator detection, frozen-code loading, validated backend selection, the run sequence, the run report, immutable results |
| `<bundle>/code/casmi_infer` | the **frozen inference implementation**, bound to `CONFIG_HASH`. It is the only code that runs the self-test and inference |
| `src/casmi_infer`, `src/casmi` (whitelist) | the current source, for tests and development. Inference never imports it: `casmi_runtime.frozen` removes it and proves the frozen origin |
| `notebooks/00_colab_setup` | Colab environment check only |
| `notebooks/01_colab_inference` | the primary Colab run (Drive → local SSD → self-test → inference → Drive results) |
| `notebooks/02_kaggle_inference` | the Kaggle run (internet OFF, GPU when available, no Drive, no clone) |
| `notebooks/03_runtime_benchmark` | engineering benchmark of **validated** backends only |
| `kaggle/kaggle_submit.ipynb` | the self-contained v6.3 anchor notebook (bundle dataset only), kept as a fallback path |

## GPU policy

GPU priority means **using a GPU where it is parity-safe**. It does not mean rewriting the frozen model just to
force GPU use.

Backend order, with a record of every step:
1. A GPU backend, only if the frozen bundle code implements it, the bundle config declares it parity-validated
   (inside `CONFIG_HASH`), and it passes the bundle self-test.
2. Otherwise numba on CPU, if it passes the self-test.
3. Otherwise numpy on CPU, if it passes the self-test.
4. Otherwise **STOP**, with no submission.

Every skipped or failed step is written to `run_report.json` as `requested_device`, `actual_backend`, `gpu_used`,
`numpy_fallback_used` and `fallback_reason`.

**Bundle v2-A7 ships no GPU backend,** so a GPU runtime runs on the validated CPU backend and says so. Why no GPU
backend yet:
* The frozen LightGBM text boosters predict on CPU.
* The spectral-evidence cost is millions of tiny per-pair computations (≤ 100 peaks, ≤ 5 references per
  candidate, a greedy modified-cosine match). A GPU version needs a batched re-implementation.
* That re-implementation would have to be proven identical (exact candidate sets and ranks, features and fold
  scores at 1e-9, identical A7 Top-25) against the frozen fixture, and then shipped in a **re-exported** bundle.
* It is never injected at runtime.

Correctness comes before GPU use.

## Tolerances (from the frozen fixture; never loosened)

| check | tolerance |
|---|---|
| candidate sets and ranks | exact |
| A7 selected spectrum and Top-25 order | exact |
| features, fold scores, fold-mean score and calibrated probabilities | 1e-9 |

See `COLAB_RUN_GUIDE.md` and `KAGGLE_RUN_GUIDE.md`. Submitting to Kaggle is always manual.
