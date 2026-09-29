# Kaggle run guide (internet OFF)

`notebooks/02_kaggle_inference.ipynb` needs only attached Kaggle inputs:

| input | content | how |
|---|---|---|
| **runtime source** dataset (e.g. `casmi-runtime-src`) | this repo: `src/`, `scripts/`, `notebooks/`, ... | upload the `colab_repo` folder built by `scripts\prepare_runtime_repo.ps1` |
| **frozen bundle** dataset (e.g. `casmi-bundle-v2-a7`) | the **complete** `bundle/` directory | upload the whole local `bundle\` folder |
| competition data | `test.parquet`, `sample_submission.csv` | "Add Input" → the competition |
| optional **wheels** dataset (e.g. `casmi-wheels`) | `lightgbm-4.7.0-*.whl` | only if the Kaggle image's lightgbm is not 4.7.0 |

Inputs are discovered by **schema**, never by dataset name:
* exactly one `manifest.json` with a CASMI `bundle_format`;
* exactly one parquet with the test columns;
* exactly one `molecule_id,smiles` CSV;
* exactly one `src/casmi_runtime`.

## Windows: prepare the datasets

```powershell
cd C:\Users\myben\OneDrive\Documents\Enveda
# runtime source (same folder as the GitHub repo)
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_runtime_repo.ps1 -Force
& "$env:USERPROFILE\.conda\envs\casmi2026\python.exe" .\scripts\check_runtime_repo.py --repo .\colab_repo

# optional offline wheel for the lightgbm parity pin (platform wheel; no install happens here)
New-Item -ItemType Directory -Force -Path .\kaggle_wheels | Out-Null
& "$env:USERPROFILE\.conda\envs\casmi2026\python.exe" -m pip download lightgbm==4.7.0 --no-deps --only-binary=:all: `
    --platform manylinux_2_28_x86_64 -d .\kaggle_wheels
Get-ChildItem .\kaggle_wheels
```

Upload in the Kaggle web UI (Datasets → New Dataset, or New Version for an existing one), keeping every dataset
**private**:
1. `colab_repo\` → `casmi-runtime-src`. Upload the folder, not a single file. Exclude the hidden `.git` folder.
2. `bundle\` → `casmi-bundle-v2-a7`. Upload the whole folder: every file listed in `manifest.json` must be present.
3. `kaggle_wheels\` → `casmi-wheels` (optional).

After a Kaggle upload, check the dataset file count against the local bundle. Bundle verification in the notebook
fails on any missing or altered file.

## Kaggle notebook

1. Upload `notebooks/02_kaggle_inference.ipynb` (File → Import notebook), or paste its cells into a new notebook.
2. **Settings → Accelerator → GPU** (T4 / P100). CPU also works.
3. **Settings → Internet → OFF.**
4. **Add Input:** the competition, `casmi-bundle-v2-a7`, `casmi-runtime-src` and, optionally, `casmi-wheels`.
5. **Run all**, or **Save Version → Save & Run All** for a code-competition submission.
6. **Inspect `/kaggle/working/run_report.json`** (the last cell prints the summary):
   * `self_test_status` and `submission_validation_status` must both be `PASS`;
   * `casmi_infer_source` must be inside the bundle dataset's `code/`;
   * `CONFIG_HASH` must equal your export;
   * check `actual_backend`, `gpu_used` and `fallback_reason`.
7. **Submit `submission.csv` manually** (Output → Submit, or the competition's code-submission flow). Nothing in the
   notebook submits.
8. Record the public score locally with `casmi.submissions.log_submission` (see `COLAB_RUN_GUIDE.md`).

`kaggle/kaggle_submit.ipynb` (the v6.3 anchor notebook) remains a self-contained fallback. It needs only the bundle
dataset and the competition data, and runs the same frozen code.
