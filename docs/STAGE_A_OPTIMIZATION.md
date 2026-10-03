# Stage A optimization (external candidate universe)

Stage A turns an external structure file (COCONUT: `data/external/coconut.csv`, ~670 MB) into standardized, filtered,
bucketed chunks. It used to take hours on a 2-core Colab runtime. This document describes what was changed for speed,
what was deliberately **not** changed, and how to verify parity yourself. No timing in this document is a measurement:
every performance statement is an expectation to be confirmed with `scripts/benchmark_stage_a.py`.

## 1. Why Stage A is CPU-bound and why the GPU is not used

Per unique SMILES, Stage A runs RDKit parsing, canonical SMILES, `TautomerEnumerator.Canonicalize`, InChI/InChIKey,
three descriptors and a molecular formula. All are single-threaded CPU code in RDKit. The tautomer step dominates.

| Operation | Device | GPU benefit |
|---|---|---|
| CSV parsing | CPU / I/O | no meaningful benefit |
| formula prefilter | CPU | no |
| RDKit parsing | CPU | no |
| tautomer canonicalization | CPU | no identity-safe GPU equivalent |
| canonical SMILES | CPU | no |
| InChI / InChIKey | CPU | no |
| identity descriptors (mass, MW, charge, formula) | CPU | no |
| bucket sort | CPU | usually no (small per chunk) |
| parquet write | CPU / I/O | no |
| later: spectrum models, contrastive training, embedding inference | GPU | yes (later notebooks) |

A GPU chemistry stack would compute a *different* canonical form and could change `connectivity_key`, which is the
competition identity. Policy (`casmi.candidates.stage_a.GPU_STAGE_A_POLICY`): **GPU detected but not used for Stage A.**
Notebook 12 prints this decision in the plan table.

## 2. Immutable scientific contract

These are unchanged and guarded by `tests/test_stage_a_optimization.py` (legacy vs optimized on fixtures):

- `canonicalize_smiles` → `normalized_smiles` / `representative_smiles`.
- `competition_connectivity_key_detailed` (Canonicalize → InChIKey[:14]), using the training tautomer caps from `PreprocessingConfig`.
- `CalcExactMolWt`, `MolWt` and `GetFormalCharge` on `MolFromSmiles(raw)`. These come from one shared helper, `_identity_values`, so both profiles call the same code.
- `molecular_formula(normalized SMILES)`, `n_fragments` and `is_charged`. There is no salt stripping or neutralization, as before.
- The formula prefilter, `apply_filters`, reject reasons and filter reasons.
- Provenance: every source record keeps its own `source_id`, `name`, `source_formula` and `source_metadata`. Dedup applies to canonicalizer calls, never to records.
- The bucket layout: sorted by bucket, one row group per bucket, the same Arrow schema.

## 3. What changed (performance only)

| Area | Before | Now |
|---|---|---|
| Input | read from Drive FUSE | staged once to the local SSD (`casmi.workspace.staging.stage_external_file`); reused when the name and size match; never hashed; the Drive original is untouched |
| CSV columns | every column parsed | only the configured columns (`usecols`), same `dtype=str` / NA semantics; a missing column still raises `SourceConfigError` |
| Workers | a joblib/loky dispatch per chunk | ONE persistent `multiprocessing` pool per source (`spawn`), batches of `standardize_batch_records` unique SMILES, native threads pinned to 1 |
| Profile | full EDA descriptors (Murcko, AddHs, TPSA, logP, counts) | `universe_minimal`: only the columns filters / merge / v2 schema read. `full` remains available and is identical to legacy |
| Formula | computed serially in the parent | computed in the workers by the same function |
| Prefilter | formula parsed per chunk | per-run formula → (mass, organic) cache, same functions |
| Bucket write | one pandas → Arrow conversion per bucket | one Arrow table, contiguous slices: same schema, row groups and values |
| Outputs | written straight to Drive | written to local scratch, persisted (`.part` + rename + size check), and only then marked done |
| Cross-chunk dedup | none | optional SQLite cache keyed by (canonicalizer contract, exact raw SMILES) |
| Visibility | silent | per-chunk progress, structures/s, rows/s, ETA from a rough row estimate, RAM / disk telemetry |

### Minimal profile

`universe_minimal` runs exactly the identity calls listed above and nothing else. It also writes no unused columns
(`UNIVERSE_STANDARDIZED_COLUMNS`). `compute_tautomer_hit_cap: true` (the default) keeps the diagnostic
`tautomer_hit_cap` column, which needs one extra `Enumerate()`. Setting it to `false` skips only that call. The key always
comes from `Canonicalize`, so it does not change; the column becomes null. Because the column differs, this flag is part
of the build identity.

### Dedup design

- **Within a chunk:** unique raw SMILES are canonicalized once, as before. The accounting is now visible in each chunk marker: `n_unique_raw_smiles`, `n_duplicate_raw_smiles_saved`, `n_standardizer_calls`, `n_cache_hits`.
- **Across chunks (optional):** `cache/standardization/std_cache.sqlite` lives on local scratch. The key is (contract fingerprint, exact raw SMILES string). The contract includes the RDKit version, the profile, the hit-cap flag, the tautomer caps and `STANDARDIZATION_CONTRACT_VERSION`. Any SQLite error disables the cache and the rows are recomputed.
- **Dedup by exact raw string only.** SMILES are never pre-canonicalized for dedup purposes, because that would itself be a chemistry decision.
- **Records are never merged.** The structure row is mapped back to every record (`assemble_standardized`, `validate="many_to_one"`).

## 4. Resume identity (Stage A → B → C)

**Stage A.** `build_id = fingerprint(stage_a_identity(...))` covers:

- the source file name and size (the Drive file, not the staged copy);
- the column mapping and `max_records`;
- `chunk_records` and `bucket_prefix_len`;
- the canonicalizer contract (contract version, profile, hit-cap flag, tautomer caps, RDKit version);
- the filters and the prefilter settings.

Outputs go to `candidates/universe_work/stage_a/<build_id>/{standardized,rejected,filtered_out,_done}/<SOURCE>/`.

A chunk marker records `row_start`, `row_end`, `n_input` and the persisted file sizes. A marker is trusted only if all of
these match and the files are intact. Workers, batch size, cache, staging and progress settings do **not** change the id.
Old builds are kept but never mixed in.

**Stage B.** `stage_b_id` covers:

- for each source, its Stage-A build id, chunk files and sizes;
- the TRAIN mass variants and structure-table metadata;
- the merge config.

Bucket markers live under `candidates/universe/_done/stage_b/<stage_b_id>/`. As a result, a TRAIN-only bucket built
before COCONUT existed is **rebuilt** once COCONUT has a complete Stage-A build.

**Stage C.** `finalize_universe(..., stage_b_id=...)` raises `StageBIncompleteError` unless every bucket was merged for
that id. The manifest records:

- `build_id` and `build_info`;
- `sources` (present and candidate counts per source);
- `external_source_present`, `n_train_candidates`, `n_external_only` and `n_train_and_external`.

`universe_source_status(OUT)` returns `absent`, `train_only`, `external` or `unknown` from the manifest alone.

## 5. Resuming after a Colab disconnect

Re-run notebook 12 from the top. Staging reuses the local copy if the VM survived; otherwise it copies again.

Stage A skips every chunk whose marker matches. A chunk that was being processed has no marker, so it is recomputed.
A chunk whose files were persisted but whose marker was not yet written is recomputed and overwritten. Stage B skips
buckets that are already merged for the same `stage_b_id`.

## 6. High-CPU runtime / workstation (CLI)

```bash
# Colab terminal or any VM with the Drive tree mounted / mirrored
python scripts/build_external_universe.py --stage A --source COCONUT --n-jobs auto
python scripts/build_external_universe.py --stage BC

# local 16-core workstation with a local copy of the Drive tree
python scripts/build_external_universe.py --drive-root D:/EnvedaCASMI --scratch-root D:/scratch --stage all --n-jobs 16
```

The CLI calls the same functions as notebook 12 and writes to the same `candidates/` tree. A build made on a
workstation is accepted by notebook 12 only if the identity matches, which means the same file size, config and RDKit
version. Otherwise notebook 12 starts a new build.

| Logical CPUs | `n_jobs` | `standardize_batch_records` | `chunk_records` | RAM guidance |
|---|---|---|---|---|
| 2 (standard Colab) | `auto` (→ 2) | 2000 | 100000 | ≥ 8 GB |
| 4 | `auto` | 2000 | 100000 | ≥ 12 GB |
| 8 | `auto` | 2000–4000 | 100000–200000 | ≥ 16 GB |
| 16 | `auto` or 16 | 4000 | 200000 | ≥ 32 GB |

`chunk_records` is a checkpoint size and is part of the build identity. Changing it starts a new build. Larger chunks
mean fewer markers and fewer Drive writes, but more work is lost after a disconnect. `ram_per_worker_gb` caps `auto`
when RAM is low.

## 7. Benchmark and parity validation

```bash
python scripts/benchmark_stage_a.py \
    --source-file /content/drive/MyDrive/EnvedaCASMI/data/external/coconut.csv \
    --template configs/v6/coconut_source.json --sizes 1000,10000,25000 --n-jobs auto --skip-hit-cap \
    --json /content/drive/MyDrive/EnvedaCASMI/reports/universe/stage_a_benchmark.json
```

- The script takes the first N records, which is deterministic.
- It runs the legacy `standardize_records` and the optimized engine (`full` and `universe_minimal`, plus `+no_hit_cap` with `--skip-hit-cap`).
- It compares the standardized, rejected, kept and filtered tables position by position on the identity columns.
- It prints a speedup **only when there are zero mismatches**. If any mismatch is found, it exits with status 1 and prints the first rows that differ.

Run the unit tests (including the RDKit parity tests) with:

```bash
pytest tests/test_stage_a_optimization.py tests/test_v2_candidate_universe.py -q
```

## 8. User workflow

1. Start a fresh Colab runtime and run notebook 10 (setup).
2. Upload `coconut.csv` to `MyDrive/EnvedaCASMI/data/external/coconut.csv`.
3. Optional but recommended: run the benchmark (section 7). Continue only if it reports **identical**.
4. Run notebook 12 top to bottom: plan table, Stage A (resumable), Stage B, Stage C.
5. Check the Stage C output. It must show `universe status: external` and `sources.COCONUT.n_candidates > 0`. `train_only` means the C2 external-universe protocol is **not valid**.
6. Re-run notebook 11 with `REBUILD = True` to get the final C2 regimes.
7. Run notebook 13 for Gate A.
8. Run notebook 14 only after a valid Gate A on an `external` universe.

## 9. Known limits

- The row estimate used for the ETA is rough: newline density of the first 4 MB. It is never used for correctness.
- The SQLite cache is per VM (local scratch). It helps when duplicates recur across chunks within a session; it never changes outputs.
- Parity is asserted on fixtures and by the benchmark. It is not proven for every COCONUT molecule until the benchmark (or a full parallel legacy run) is executed on the real file.
