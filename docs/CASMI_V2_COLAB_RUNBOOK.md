# CASMI v2 — Colab runbook (first pass)

## 1. Drive layout (upload once)

```
MyDrive/EnvedaCASMI/                    <- ENVEDA_DRIVE_ROOT (override with the env var)
├── repo/                               <- this repository (upload, or set REPO_GIT_URL in notebook 10)
├── bundle/                             <- the frozen v2-A7 bundle folder (reference library peaks + connectivities)
├── data/
│   ├── raw/train.parquet               (optional for this pass; not read by notebooks 10–14)
│   ├── processed/train_spectrum_metadata.parquet
│   ├── processed/connectivity_folds.parquet
│   ├── interim/structure_table.parquet
│   ├── interim/candidate_generation/dev_queries.parquet
│   ├── interim/candidate_generation/molecule_mass_variants.parquet
│   └── external/coconut.csv            <- obtained MANUALLY (and optionally a curated PubChem subset)
└── (created by the notebooks) candidates/ cache/ reports/ predictions/ checkpoints/ experiments/ embeddings/
```

Copy the `data/…` files from the local repository's `data/` folder; their `.meta.json` sidecars are optional.
Check the column names in `configs/v6/coconut_source.json` (or `pubchem_source.json`) against the
header of your file, and list the file under `universe.sources` in `configs/casmi_v2_colab.yaml`.

## 2. Run order

| # | Notebook | Runtime | Writes |
|---|---|---|---|
| 10 | `10_colab_v2_setup` | any | `reports/environment/*.json`, Drive folders |
| 11 | `11_colab_hidden_like_validation` | CPU | `data/processed/validation_regimes.parquet` (+ metadata), `reports/validation_summary.{parquet,json}` |
| 12 | `12_colab_candidate_universe` | CPU, high RAM preferred | `candidates/universe/…` (buckets, index, formula index, manifest) |
| 11 again | with `REBUILD = True` | CPU | final (non-PRELIMINARY) regimes with external-source C2 eligibility |
| 13 | `13_colab_candidate_recall` | CPU | `reports/c2_candidate_recall.{parquet,json}` + breakdowns. **Gate A** |
| 14 | `14_colab_analog_baseline` | GPU optional | `cache/analog_neighbors/`, `data/processed/analog_features/`, `checkpoints/analog_ranker/`, `reports/analog_baseline/`, `predictions/analog_baseline/` |

Notebook 11 can run before 12, but its C2 regime is then PRELIMINARY: C2 eligibility needs to know
which truths exist in an external source. Re-run 11 with `REBUILD = True` after 12. Notebooks 13
and 14 warn when the regimes are preliminary.

## 3. Restarts

Every expensive step resumes from Drive: universe chunks and buckets (`_done/` markers), analog
neighbours and feature shards (`part-*.done`), fingerprints (append-only shards) and the binned
library matrices (`cache/analog_library_*`). After a disconnect, re-run the notebook from the top.

## 4. Gate A (before any deep learning)

Read `reports/c2_candidate_recall.json → gate_a`. The decision is a heuristic with thresholds you set.
If C2 `external_only` Recall@All at a sensible ppm is low, the bottleneck is **candidate coverage**:
add or curate sources before building rankers.
