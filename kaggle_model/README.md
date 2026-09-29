# Kaggle Model package — ENVEDA CASMI v2-A7

This directory contains metadata templates and packaging helpers for publishing the complete
frozen CASMI inference stack as a **private Kaggle Model**.

## Model layout

The uploaded model variation payload is:

```text
payload/
├── PACKAGE_INFO.json
├── kaggle_model_inventory.json
├── bundle/
│   ├── models/                  # five frozen LightGBM folds + metadata
│   ├── aggregation/             # A7 selection + calibration
│   ├── selftest/                # functional parity fixture
│   ├── code/
│   │   ├── casmi_infer/         # authoritative frozen inference implementation
│   │   └── casmi/               # frozen shared runtime dependencies
│   ├── reference arrays/indexes
│   ├── structures/connectivities
│   ├── config.json
│   └── manifest.json
└── runtime/
    ├── src/casmi_runtime/       # lightweight orchestration
    ├── notebooks/02_kaggle_inference.ipynb
    ├── requirements-kaggle.txt
    └── KAGGLE_RUN_GUIDE.md
```

The competition files `test.parquet` and `sample_submission.csv` stay separate; they are
provided by the Kaggle competition input.

## Build on Windows PowerShell

From the Enveda repo:

```powershell
cd C:\Users\myben\OneDrive\Documents\Enveda
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_kaggle_model.ps1 -KaggleUsername YOUR_KAGGLE_USERNAME -Force
```

The script prefers NTFS hard links for large frozen files and falls back to copies, so staging
does not unnecessarily duplicate multi-GB arrays when hard links are available.

## Create the private parent model once

Kaggle Models use a parent model plus a framework/variation. The prepared metadata uses:

- model: `enveda-casmi-v2-a7`
- framework: `sklearn`
- variation: `full-e2e`

Create the parent through the Kaggle UI, or with the Kaggle CLI:

```powershell
kaggle models create -p .\kaggle_model_build\metadata
```

## Upload the complete E2E variation

```powershell
$PY = "$env:USERPROFILE\.conda\envs\casmi2026\python.exe"
& $PY .\scripts\upload_kaggle_model.py --handle YOUR_KAGGLE_USERNAME/enveda-casmi-v2-a7/sklearn/full-e2e
```

The uploader uses `kagglehub.model_upload`, which supports nested model directories. The model
is intended to remain private.

## In Kaggle

Attach the model to the notebook and also attach the competition. Kaggle exposes attached model
files under `/kaggle/input`; the inference notebook resolves the model payload by filenames rather
than by a hard-coded mount slug.

The scientific code always comes from `bundle/code/casmi_infer`; the `runtime/src/casmi_runtime`
package only orchestrates loading, self-test, inference, validation and reporting.
