# Colab run guide

## A. Windows: build the runtime repo, push it, put the bundle on Drive

The training repo is `C:\Users\myben\OneDrive\Documents\Enveda` and the conda environment is `casmi2026`.
The bundle must already be exported and checked (`RUN_GUIDE_V6.md`: 11_00 → `v63_bundle_check.py` → 11_01).

```powershell
cd C:\Users\myben\OneDrive\Documents\Enveda
$PY = "$env:USERPROFILE\.conda\envs\casmi2026\python.exe"

# 1. build colab_repo (add -OutDir C:\dev\enveda-casmi-runtime to keep .git out of OneDrive; -Force to rebuild)
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_runtime_repo.ps1

# 2. safety check -- must print RUNTIME REPO CHECK: PASS
& $PY .\scripts\check_runtime_repo.py --repo .\colab_repo

# 3-6. private GitHub repo
cd .\colab_repo
git init -b main
git add .
git commit -m "Initial CASMI GPU runtime"
gh auth status
gh repo create enveda-casmi-runtime `
    --private `
    --source . `
    --remote origin `
    --push
gh repo view --web
cd ..

# 7. the ENTIRE bundle + competition files to Drive (Google Drive for desktop; adjust the drive letter)
$DRIVE = "G:\My Drive\EnvedaCASMI"
New-Item -ItemType Directory -Force -Path "$DRIVE\bundle", "$DRIVE\competition", "$DRIVE\results" | Out-Null
robocopy .\bundle "$DRIVE\bundle" /MIR /XD __pycache__ /NFL /NDL /NP        # exit codes 0-7 = success
Copy-Item .\data\test.parquet, .\data\sample_submission.csv -Destination "$DRIVE\competition" -Force
(Get-ChildItem .\bundle -Recurse -File | Measure-Object Length -Sum).Sum
(Get-ChildItem "$DRIVE\bundle" -Recurse -File | Measure-Object Length -Sum).Sum     # must match once sync is complete
```

Without the desktop client, upload the whole `bundle` folder in the Drive web UI (New → Folder upload).
Never pick individual files, and never edit a bundle file on Drive.

For a private repo, create a **fine-grained token** on GitHub with read-only Contents access to
`enveda-casmi-runtime` only.

Drive layout:

```text
MyDrive/EnvedaCASMI/
├── bundle/              (complete v2-A7 export: manifest.json, config.json, structures*, ref_*, models/, aggregation/, selftest/, code/ ...)
├── competition/         test.parquet, sample_submission.csv
└── results/             one NEW <UTC timestamp>_<CONFIG_HASH>/ folder per run (never overwritten)
```

## B. Colab

1. **Runtime → Change runtime type → GPU** (T4 or better; High-RAM helps).
2. Key icon (Secrets) → add `GITHUB_TOKEN` → enable notebook access. Private repo only; never paste the token into a cell.
3. File → Open notebook → GitHub → `notebooks/00_colab_setup.ipynb`. Set `GITHUB_USER` and run all cells. It must
   end with `SETUP OK`.
4. Open `notebooks/01_colab_inference.ipynb`, set `GITHUB_USER`, then **Run all**.
5. **Inspect the self-test:** cell 11 must end with `BUNDLE CHECK: PASS`. The inference run repeats the self-test
   in-process for each candidate backend.
6. **Inspect the actual backend:** the report shows `actual_backend`, `gpu_used` and `fallback_reason`. For v2-A7
   the fallback reason says the frozen bundle implements no GPU backend.
7. **Inspect `run_report.json`:**
   * `CONFIG_HASH` must equal your export;
   * `casmi_infer_source` must be under `.../input/bundle/code`;
   * `self_test_status` and `submission_validation_status` must both be `PASS`.
8. **Retrieve the submission:** open `MyDrive/EnvedaCASMI/results/<timestamp>_<CONFIG_HASH>/` and download
   `submission.csv`. That folder also holds `run_report.json` and `provenance.json`.
9. Submit manually on Kaggle, then record the score locally:

   ```powershell
   & $PY -c "from casmi.submissions import fields_from_run_report, log_submission; log_submission('outputs/submissions/submission_log.jsonl', submission_name='colab-v2-A7', public_score=None, notes='Colab run', **fields_from_run_report(r'<path to run_report.json>'))"
   ```

Optional: `notebooks/03_runtime_benchmark.ipynb` in the same session. It times validated backends only.

## What stops a run

Each of these raises. No `submission.csv` is written, and `run_report_failed.json` names the stage:

* a missing or incomplete bundle;
* a manifest sha256 or `CONFIG_HASH` mismatch;
* a missing or mismatched calibration;
* the wrong model, aggregator or bundle version, or a pinned `EXPECTED_CONFIG_HASH` that doesn't match;
* a self-test that fails on every backend;
* `REQUIRE_GPU=True` without a validated GPU backend;
* a failed submission validation;
* a work folder that already holds outputs.
