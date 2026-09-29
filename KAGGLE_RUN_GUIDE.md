# Kaggle offline anchor — Kaggle Model workflow

The recommended Kaggle path uses **one private Kaggle Model** containing the entire frozen E2E inference payload plus the lightweight orchestration code.

## Attachments

Attach only:

1. **Private Kaggle Model**: `enveda-casmi-v2-a7 / sklearn / full-e2e`
   - `bundle/`: complete frozen scientific package, models, reference library, aggregation artifacts, self-test fixture and authoritative `bundle/code`.
   - `runtime/src/casmi_runtime/`: lightweight orchestration.
   - `runtime/notebooks/02_kaggle_inference.ipynb`: offline anchor notebook.
2. **Competition input**
   - `test.parquet`
   - `sample_submission.csv`

No Google Drive, no Git clone, no runtime download, no Kaggle API call and no internet access are required in the inference notebook.

## Prepare the Kaggle Model locally

From Windows PowerShell:

```powershell
cd C:\Users\myben\OneDrive\Documents\Enveda
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_kaggle_model.ps1 -KaggleUsername YOUR_KAGGLE_USERNAME -Force
```

The payload is written under:

```text
kaggle_model_build/
├── metadata/
│   ├── model-metadata.json
│   └── model-instance-metadata.json
└── payload/
    ├── PACKAGE_INFO.json
    ├── kaggle_model_inventory.json
    ├── bundle/
    └── runtime/
```

The builder validates the required filenames but intentionally does not perform SHA-based bundle verification.

## Create the private parent model once

Create the private model in the Kaggle UI, or:

```powershell
kaggle models create -p .\kaggle_model_build\metadata
```

Prepared identity:

- model slug: `enveda-casmi-v2-a7`
- framework: `sklearn`
- variation: `full-e2e`

## Upload the E2E model payload

```powershell
$PY = "$env:USERPROFILE\.conda\envs\casmi2026\python.exe"
& $PY -m pip install kagglehub
& $PY .\scripts\upload_kaggle_model.py --handle YOUR_KAGGLE_USERNAME/enveda-casmi-v2-a7/sklearn/full-e2e
```

The model should remain private.

## Run the Kaggle offline anchor

1. Import `notebooks/02_kaggle_inference.ipynb`.
2. Settings -> Internet -> **OFF**.
3. Accelerator may be GPU or CPU. Current frozen `v2-A7` has no validated GPU spectral backend, so the expected scientific backend is NumBA when available, otherwise validated NumPy.
4. Add the private Kaggle Model.
5. Add the competition input.
6. Run All / Save Version.
7. Inspect:
   - `/kaggle/working/run_report.json`
   - `/kaggle/working/kaggle_anchor_report.json`
   - `/kaggle/working/submission.csv`
8. Continue only when `anchor_status == "PASS"` and `submission_validation_status == "PASS"`.
9. Submit `submission.csv` manually.

## Filename-only contract

The notebook resolves the model and competition input by filenames under `/kaggle/input`; it does not hard-code the model mount slug.

Scientific authority is always:

```text
bundle/code/casmi_infer
```

The runtime package is orchestration only:

```text
runtime/src/casmi_runtime
```

The full inference path is:

```text
test.parquet
  -> spectrum preprocessing
  -> neutral mass / adduct handling
  -> candidate retrieval
  -> compatible reference selection
  -> spectral evidence
  -> frozen 9-feature matrix
  -> five frozen LightGBM fold predictions
  -> spectrum ranking / calibrated confidence
  -> A7 MOST_CONFIDENT_SPECTRUM
  -> connectivity deduplication
  -> Top-25 representative SMILES
  -> submission validation
  -> submission.csv
```

## Environment policy

The notebook performs no installs. The expected Kaggle environment is documented in `requirements-kaggle.txt`, but exact package versions are not the parity gate. The frozen functional self-test is the compatibility gate.

The successful Colab anchor used LightGBM 4.6.0 with exact downstream outputs, so LightGBM 4.7.0 is not a mandatory condition.

## Legacy notebook

`kaggle/kaggle_submit.ipynb` is legacy and is not the recommended anchor path. Use `notebooks/02_kaggle_inference.ipynb`.
