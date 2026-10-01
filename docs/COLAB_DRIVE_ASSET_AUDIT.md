# Colab Drive asset audit

Local root: `C:\Users\myben\OneDrive\Documents\Enveda`. Inspected on 2026-10-01 with file listing only:
nothing was hashed, copied or executed.

Sizes are **recorded values, not fresh measurements**. They come from the size fields that artifact
`.meta.json` sidecars store for their inputs, or from the 2026-09-30 Kaggle ZIP packaging run.
`copy_colab_assets_to_drive.ps1 -WhatIf` prints the current sizes.

Drive targets are relative to `MyDrive/EnvedaCASMI/`.

| Asset | Resolved local path | Exists | Size | Required | Drive target | Copy policy |
|---|---|---|---|---|---|---|
| Frozen bundle v2-A7 | `bundle\` | yes | ≈ 4.55 GB total. 94 files incl. `__pycache__`; 64 files after excluding transient content (63 listed in `manifest.json`, plus the manifest) | **yes** | `bundle/` | complete directory (robocopy `/E /Z`), excluding only `__pycache__`, `*.pyc`, `.pytest_cache`, `.ipynb_checkpoints`; identity checked; verified by name + size |
| train_spectrum_metadata | `data\processed\train_spectrum_metadata.parquet` | yes | 121,386,819 B | **yes** | `data/processed/train_spectrum_metadata.parquet` | file copy + size check |
| connectivity_folds | `data\processed\connectivity_folds.parquet` | yes | 4,113,418 B | **yes** | `data/processed/connectivity_folds.parquet` | file copy + size check |
| structure_table | `data\interim\structure_table.parquet` | yes | not recorded (277,566 rows) | **yes** | `data/processed/structure_table.parquet` | file copy + size check |
| dev_queries | `data\interim\candidate_generation\dev_queries.parquet` | yes | 3,225,719 B | **yes** | `data/processed/dev_queries.parquet` | file copy + size check |
| molecule_mass_variants | `data\interim\candidate_generation\molecule_mass_variants.parquet` | yes | 24,659,108 B | **yes** | `data/processed/molecule_mass_variants.parquet` | file copy + size check |
| `.meta.json` sidecars of the five files | next to each file | yes | a few KB | no | next to the Drive copy | optional provenance |
| Competition train (raw spectra + metadata) | `data\train.parquet` | yes | 3,033,286,496 B | no | `data/raw/competition/train.parquet` | **opt-in** (`-IncludeRaw`); notebooks 10–14 do not read it |
| Competition test | `data\test.parquet` | yes | not recorded | no | `data/raw/competition/test.parquet` | opt-in (`-IncludeRaw`); **never a validation input** |
| Sample submission | `data\sample_submission.csv` | yes | not recorded | no | `data/raw/competition/sample_submission.csv` | opt-in (`-IncludeRaw`) |
| Separate raw spectra files | — | **no** | — | no | `data/raw/spectra/` (empty) | peaks live inside `train.parquet` |
| External folder | `data\external\` | **no** | — | no | `data/external/` | — |
| COCONUT export | `data\external\coconut\` | **no** (only the template `configs\v6\coconut_source.json`) | — | no | `data/external/coconut/` | copied when present; otherwise "NOT PROVIDED YET" |
| PubChem export / subset | `data\external\pubchem\` | **no** (only the template `configs\v6\pubchem_source.json`) | — | no | `data/external/pubchem/` | copied when present; otherwise "NOT PROVIDED YET" |
| Other structure metadata | `data\processed\molecule_metadata.parquet` | yes | not recorded | no | — | not copied (not used by v2 notebooks) |
| Candidate / spectral-ranking caches (v3–v6) | `data\interim\candidate_generation\*`, `data\interim\spectral_ranking\*`, `data\processed\*\` | yes | various | no | — | not copied (v1 history; v2 regenerates what it needs on Drive) |
| Kaggle offline ZIP | `kaggle_upload\enveda_casmi_offline_payload.zip` | yes | 4,545,218,897 B | no | — | **not copied** (duplicates `bundle/`) |
| `outputs\`, `kaggle_dryrun\`, `colab_repo\`, `artifacts\` | repo root | yes | various | no | — | not copied |
| Source code, notebooks, configs, tests, docs | `src\`, `notebooks\`, `configs\`, `tests\`, `docs\`, `scripts\` | yes | small | — | **GitHub** `mybenkhadda/enveda-kaggle` | never copied to Drive; Colab clones to `/content/Enveda` |

## Not found locally

- `data\external\` (no COCONUT / PubChem export yet)
- separate raw spectra or metadata files outside `train.parquet`

## Notes

- `data\` is the local root for competition files. The v2 inputs are spread over `data\processed\` and `data\interim\`, and all five go to Drive `data/processed/`.
- The bundle identity was checked on 2026-09-30: `bundle_version` v2-A7, `CONFIG_HASH` 60174e39a2a3c6b4, `model_id` V1_TL_1K_TESTSIM_STRICT. The copy script re-checks it at the source and on Drive.
