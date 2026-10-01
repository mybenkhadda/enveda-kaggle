# CASMI v2 — Colab runbook (authoritative workflow)

## 1. Layout

| layer | location | rule |
|---|---|---|
| source | GitHub `mybenkhadda/enveda-kaggle` → `/content/Enveda` | cloned if missing, otherwise fetched and fast-forwarded (`git pull --ff-only origin main`) by every notebook's bootstrap cell; never stored on Drive |
| persistence | `MyDrive/EnvedaCASMI/` | inputs, frozen bundle, every artifact (tree: `configs/drive_asset_map.yaml`) |
| scratch | `/content/enveda_work` | staged copies, temporaries; disposable |

**Inputs.** Copy the required local assets with the Windows scripts in [COLAB_DRIVE_MIGRATION.md](COLAB_DRIVE_MIGRATION.md):

- the frozen `bundle/`;
- the five canonical inputs in Drive `data/processed/`: `train_spectrum_metadata`, `connectivity_folds`, `structure_table`, `dev_queries` and `molecule_mass_variants` (all `.parquet`).

The v2 notebooks read only these five files, never `data/interim`.

**External candidate sources.** These are user-supplied and never downloaded by code:

- **COCONUT** goes at Drive **`data/external/coconut/coconut.csv`**. Check its columns against `configs/v6/coconut_source.json`.
- **PubChem** subsets go under `data/external/pubchem/`. Enable them under `universe.sources` in `configs/casmi_v2_colab.yaml`.

**Frozen bundle.** The bundle is identified by its config labels only: `bundle_version = v2-A7`, `CONFIG_HASH = 60174e39a2a3c6b4`, `model_id = V1_TL_1K_TESTSIM_STRICT`. `CONFIG_HASH` is an identity label; no file is hashed.

## 2. One path API, one artifact registry, one bootstrap

| concept | where | used in notebooks as |
|---|---|---|
| physical roots | `casmi.workspace.config.V2Paths`: `drive_root, repo_root, raw_data_dir, processed_dir, interim_dir, external_dir, bundle_dir, candidate_db_dir, embeddings_dir, checkpoints_dir, predictions_dir, reports_dir, cache_dir, experiments_dir, scratch_dir` | `P.<field>` |
| semantic artifact locations | `casmi.workspace.artifact_registry.ArtifactRegistry` | `ARTIFACTS.<name>`, e.g. `ARTIFACTS.validation_regimes`, `ARTIFACTS.universe_root`, `ARTIFACTS.gate_a_decision` |
| versions | `casmi.workspace.versions` | `bootstrap(..., notebook_api='casmi-v2-notebooks-4')` |

The briefly used aliases `P.universe`, `P.regimes`, `P.reports`, `P.candidates`, `P.formula_index`, `P.candidate_manifests`, `P.scratch_root` and `P.runs` are **retired**. `config.DEPRECATED_PATH_ALIASES` maps each to its replacement for error messages only; they are never resolved.

Every v2 notebook starts with two cells:

1. **The generated bootstrap cell** (`casmi.workspace.notebook_cells`, version `casmi-v2-bootstrap-2`). It:
   - mounts Drive and sets `ENVEDA_DRIVE_ROOT` / `ENVEDA_REPO_ROOT`;
   - clones the repository if missing, otherwise runs fetch, checkout `main` and `pull --ff-only`;
   - refuses a clone that predates `versions.py`;
   - drops already-imported `casmi` modules and puts `src/` on `sys.path`.
2. **`CTX = bootstrap(NOTEBOOK, notebook_api=..., uses_paths=..., uses_artifacts=..., requires=..., signatures=...)`**. It:
   - checks the notebook API and bootstrap-cell versions;
   - installs only missing packages;
   - loads the config, with a schema-version check;
   - builds `P` and `ARTIFACTS`;
   - runs **preflight**, which checks:
     - the expected config schema, paths API and artifact-registry API versions;
     - the declared `P` fields and `ARTIFACTS` names;
     - upstream artifacts, each named with its producer notebook;
     - function signatures;
     - repository state: branch, HEAD, dirty files, and whether the clone is behind `origin/main`;
     - external sources, where a notebook requires them;
     - bundle identity labels, where a notebook requires them.

   Every problem is reported as `ERROR: … / Fix: …` before any work starts, and there is never a silent fallback.

When a `V2Paths` field, registry name, config layout or the bootstrap cell changes, bump the version in `casmi.workspace.versions` (and `config.py`). Re-stamp the notebooks and update `notebook_api` in the same commit.

## 3. Run order

Notebook 10 prints the pipeline state (`artifact_registry.pipeline_status`) and the next notebook.

**First setup / preliminary**

| # | notebook | notes |
|---|---|---|
| 10 | `10_colab_v2_setup` | inputs, bundle identity labels, external sources, pipeline state, optional self-tests |
| 11 | `11_colab_hidden_like_validation` (preliminary) | without an external universe: `C2_MODE = PRELIMINARY_no_universe` / `PRELIMINARY_train_only_universe` -- setup only |
| 12 | `12_colab_candidate_universe` | TRAIN + COCONUT (+ PubChem). Without a source file the universe is TRAIN-only and `C2_PROTOCOL_VALID = False` |

**After a real external universe exists** (notebook 12 reports `UNIVERSE_STATUS = external`)

| # | notebook | notes |
|---|---|---|
| 11 | `REBUILD = True` | `C2_MODE = external_universe`: every C2 truth exists in a non-TRAIN source, all its references hidden |
| 13 | `13_colab_candidate_recall` (Gate A) | C2 Recall@All at 2 / 5 / 10 ppm, Recall@100 and Recall@25 at 5 ppm, candidate-count median / p90 / p99 -- **only** when the protocol is valid; otherwise `FAIL_PROTOCOL_INVALID` with the reason and no C2 metric |

**Only if Gate A is protocol-valid**

| # | notebook | notes |
|---|---|---|
| 14 | `14_colab_analog_baseline` | stops unless `gate_a_decision.json` is protocol-valid **for the current universe** (C1-only diagnostic with `ALLOW_PROTOCOL_INVALID_DIAGNOSTIC = True`, recorded as `PRELIMINARY`) |

**Later GPU work** (do **not** start before Gate A and the analog baseline have been evaluated): 20 fingerprint · 21 contrastive · 22 candidate embeddings · 30 unified ranker · 31 aggregation ablation · 32 error analysis · 40 fragmentation rerank.

> **A TRAIN-only candidate universe is NOT a valid C2 external-universe evaluation.** Hidden C2 truths would be the only reference-less TRAIN structures in their pools (a perfect shortcut), and external-only recall is undefined. `casmi.validation.c2_protocol` encodes this, and notebooks 11 / 13 / 14, `pipeline_status` and the leaderboard (which excludes `PROTOCOL_INVALID` runs) all use it.

## 4. Universe layout (canonical)

`P.candidate_db_dir` (`candidates/`) is the universe root and contains:

- `candidate_keys.npy`, `bucket_offsets.json` and `universe_manifest.json`. The manifest embeds `source_summary` with TRAIN-only, external-only and shared counts.
- `index/` (mass), `formula_index/`, `buckets/` and `variants/`.
- `standardized/`, `rejected/`, `filtered_out/` and `_done/` from stage A.
- `manifests/` (reports).

The canonical call is `finalize_universe(universe_root, buckets)`.

Bucket done-markers record a signature of their stage-A inputs. Adding COCONUT later therefore rebuilds the buckets instead of silently keeping the TRAIN-only ones.

A universe built by an earlier notebook version under `candidates/indexes/` is not read any more; notebook 12 rebuilds it at the root.

## 5. Restarts and cache identity

- **Universe.** Stage-A chunks resume via `_done/<source>-chunk-*.json`. Stage-B buckets resume via `_done/bucket-*.json` only for the same stage-A inputs.
- **Analog caches** (notebook 14) are **identity-namespaced** (`casmi.workspace.cache_identity`): `<base>/fold=f/ns-<hash>/identity.json + part-*`.
  - The neighbor identity covers: fold, query set, hidden-reference set, search config and reference-library identity (bundle labels plus `ref_meta` size).
  - The feature identity adds: the query→regime assignment, the removed (C3) set, the full analog config, the universe identity, the feature schema and the chunk size.
  - Any change selects a new namespace. A tampered or legacy directory raises `StaleCacheError`.
  - Features are loaded only from the namespaces returned by the run.
- **Fingerprint cache.** New fingerprints are buffered in memory and written as **one shard per chunk** (`flush()`). Nothing is re-read from disk after construction, and more than 32 shards are compacted into one on load.
- **Staging.** Files are staged to `/content/enveda_work` when their name, size or modification time changes (no hashing).

## 6. Experiment tracking (append-only)

- `runs/records/<timestamp>-<uuid>-<experiment>.json` holds one immutable record per run, and is never overwritten. Each record carries:
  - the run id, timestamp, notebook and git commit;
  - the config identity, regime identity and universe identity;
  - the metrics, artifact paths, status and decision.
- `runs/experiments.parquet` and `runs/leaderboard.parquet` are **derived** views (`rebuild_index`, `write_leaderboard`).
- The leaderboard excludes `PROTOCOL_INVALID` runs.

## 7. Local scorer

`casmi.validation.official_metric` computes MRR@25 on the RDKit tautomer-canonical InChIKey14, using the project's training canonicalizer. Where the official rules are ambiguous it is conservative:

- invalid or repeated SMILES still occupy their slot;
- entries after the 25th are ignored;
- nothing is compacted before scoring.

`dedupe_by_connectivity` removes wasted slots before a submission is written.

## 8. Before committing a notebook

```bash
python scripts/stamp_notebooks.py           # re-stamp the canonical bootstrap cell (+ deterministic cell ids)
python scripts/audit_notebooks.py           # must report 0 errors
python -m pytest -q tests/test_notebook_audit.py tests/test_preflight.py tests/test_artifact_registry.py tests/test_c2_protocol.py
```
