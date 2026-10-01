# Colab + Google Drive migration

**Code lives in GitHub, data and artifacts live on Drive, and heavy reads use the Colab SSD.**

| What | Where |
|---|---|
| Source, notebooks, configs, tests, docs | GitHub `mybenkhadda/enveda-kaggle` → Colab clones it to `/content/Enveda` |
| Datasets, frozen bundle, external databases, candidates, features, embeddings, checkpoints, predictions, reports, caches, runs, exports | Google Drive `MyDrive/EnvedaCASMI/` |
| Staged copies, temporaries | `/content/enveda_work` (disposable) |

The repository is **not** copied into Drive. Generated research artifacts are **never** moved back into
Git: `.gitignore` covers them with root-anchored patterns.

## 0. Before you start

1. Install and start **Google Drive for desktop**, and sign in.
2. Commit and push the v2 code so that Colab can clone it (run this yourself in the repository):

   ```
   git add src configs notebooks tests docs scripts .gitignore
   git commit -m "CASMI v2 Colab layout"
   git push origin main
   ```

   If the repository is private, add a Colab secret named `GITHUB_TOKEN`. Notebook 10 reads it.

## 1. Windows: find the Drive mount

```
powershell -ExecutionPolicy Bypass -File ".\scripts\find_google_drive.ps1"
```

- **One root found:** it is printed. Use it as `-DriveRoot` below.
- **Several found:** the script lists them and stops. Pick one and pass it explicitly.
- **None found:** start Drive for desktop, or pass your synced folder explicitly.

The commands below use `"G:\My Drive"` as an example only. Replace it with the printed root. A
French Windows typically shows `"G:\Mon Drive"`.

## 2. Create the Drive folder tree (directories only, safe to re-run)

```
powershell -ExecutionPolicy Bypass -File ".\scripts\create_colab_drive_architecture.ps1" -DriveRoot "G:\My Drive"
```

## 3. Preview the copy (nothing is created or copied)

```
powershell -ExecutionPolicy Bypass -File ".\scripts\copy_colab_assets_to_drive.ps1" -RepoRoot "." -DriveRoot "G:\My Drive" -WhatIf
```

## 4. Copy (COPY only: sources are never deleted or moved)

```
powershell -ExecutionPolicy Bypass -File ".\scripts\copy_colab_assets_to_drive.ps1" -RepoRoot "." -DriveRoot "G:\My Drive" -Yes
```

Options:

| Option | Effect |
|---|---|
| `-SkipBundle` | skip the 4.5 GB bundle (e.g. when it is already on Drive) |
| `-SkipExternal` | skip COCONUT / PubChem |
| `-IncludeRaw` | also copy `data\train.parquet`, `test.parquet` and `sample_submission.csv` (not needed by notebooks 10–14) |
| `-Overwrite` | replace Drive files whose size differs. Without it, existing Drive files are never replaced |

Without `-Yes`, the script asks you to type `YES`.

After copying, the script verifies every file by relative name and size, without hashing. It
writes `EnvedaCASMI\exports\manifests\local_to_drive_copy_manifest.json` plus a timestamped copy. Each
asset gets one of these statuses: `COPIED`, `ALREADY_EXISTS`, `SKIPPED`, `MISSING_OPTIONAL` or `ERROR`.

Re-running is safe. Files already present with the same size are not copied again.

## 5. Wait for Google Drive to finish syncing

About 4.7 GB is uploaded. Watch the Drive for desktop tray icon until it says everything is up to date.

## 6. Verify on drive.google.com

Check `My Drive/EnvedaCASMI/`:

- `bundle/` contains `config.json`, `models/v1_fold0..4.txt` and the `ref_peaks_*.npy` / `_ref_peaks_*.f8` arrays.
- `data/processed/` contains the five parquet files.
- `exports/manifests/local_to_drive_copy_manifest.json` exists.

## 7–8. Colab: run notebook 10

Open `notebooks/10_colab_v2_setup.ipynb` from GitHub in Colab (File → Open notebook → GitHub) and
choose a GPU runtime. It mounts Drive, clones or pulls the repo into `/content/Enveda`, checks the
required Drive inputs and the bundle identity, reports the GPU, creates `/content/enveda_work`, and prints
`READY`.

## 9. Continue with notebooks 11–14

Run order and the authoritative workflow are in [CASMI_V2_COLAB_RUNBOOK.md](CASMI_V2_COLAB_RUNBOOK.md).

Each notebook's generated bootstrap cell clones the repository if `/content/Enveda` is missing and otherwise
fast-forwards it to GitHub `main`. All outputs go straight to Drive (locations: `casmi.workspace.artifact_registry`):

| Output | Drive location |
|---|---|
| Regimes | `validation/regimes/` |
| Universe | `candidates/` (universe root: `candidate_keys.npy`, `bucket_offsets.json`, `universe_manifest.json`, `index/`, `formula_index/`, `buckets/`, `variants/`), reports in `candidates/manifests/` |
| Recall / Gate A | `reports/candidate_recall/` (incl. `gate_a_decision.json`, `mass_calibration/`) |
| Analog | `cache/analog_neighbors/`, `features/analog/` (identity-namespaced), `cache/fingerprints/`, `checkpoints/ranker/analog/`, `predictions/validation/`, `reports/analog/` |
| Experiment tracker | `runs/records/*.json` (append-only) + derived `runs/experiments.parquet`, `runs/leaderboard.parquet` |

A universe built by an earlier notebook version under `candidates/indexes/` is no longer read; notebook 12 rebuilds
the universe at the canonical root (stage A chunks are reused).

Heavy, repeatedly read files (the reference-library arrays and the mass index) are staged to
`/content/enveda_work` with `casmi.workspace.staging`. Staging decides whether a file changed by its
name, size and modification time, with no hashing.

## Later

- Put COCONUT / PubChem exports in `data\external\coconut\` or `data\external\pubchem\` locally, then re-run step 4, which copies only what is new. Alternatively, upload them directly to Drive `EnvedaCASMI/data/external/...`.
- After you have verified the Drive copy, you may delete redundant local files yourself. The scripts never do.
