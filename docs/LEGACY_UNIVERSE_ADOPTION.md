# Legacy universe adoption

Notebook: `notebooks/11a_colab_legacy_universe_adoption.ipynb`. Script: `scripts/adopt_legacy_universe_manifest.py`.
Module: `casmi.candidates.legacy_adoption`. Status: **authored, not executed.**

## Problem

The TRAIN ∪ COCONUT universe on Drive (`candidates/universe/`) was finalized by the pre-v3 code (commit 85d43d9). The
user's run reported these counts, recorded here as inputs, not recomputed:

| | count |
|---|---|
| TRAIN | 274,195 |
| COCONUT | 446,794 |
| TRAIN ∩ COCONUT | 24,542 |
| TRAIN ∪ COCONUT | 696,447 |

Its `universe_manifest.json` holds only `universe_schema_version`, `n_candidates`, `n_buckets`, `mass_index`,
`formula_index` and `finalized_at`. Without `external_source_present` and `sources`, `universe_source_status` returns
`unknown`. The v1 protocol check then read empty sources and reported "COCONUT has no candidates", even though every
bucket carries `candidate_sources` with COCONUT.

## Adoption (no rebuild)

`verify_legacy_universe` reads the persisted artifacts only, in one pyarrow pass over the buckets with no RDKit, and
checks these invariants:

| invariant | detail |
|---|---|
| manifest readable | |
| `n_candidates` consistent | manifest == `bucket_offsets.json` == `len(candidate_keys.npy)` == sum of bucket rows |
| offsets | contiguous in key order |
| keys | `candidate_keys.npy` strictly sorted; each bucket's keys equal its slice |
| index sizes | the mass index and the formula index `n_candidates` equal the universe size |
| provenance consistent | `in_train == ('TRAIN' in candidate_sources)` on every row |
| expected counts | `validation_v2.legacy_adoption.expected_counts` (the four numbers above) match **exactly** |

The status is one of:

- `LEGACY_EXTERNAL_VERIFIED`
- `LEGACY_TRAIN_ONLY` (refused)
- `VERIFICATION_FAILED` (refused)
- `MODERN_MANIFEST` (nothing to do)
- `UNIVERSE_ABSENT`

On `LEGACY_EXTERNAL_VERIFIED`, `adopt_legacy_universe` writes `candidates/universe/universe_manifest_v3_adopted.json`
with these fields:

- `adopted_from_legacy: true`, `adoption_policy_version`, `adopted_build_id`
  (`adopted-<digest of candidate_keys.npy, bucket_offsets.json, the legacy manifest and the counts>`);
- the counts: per source, overlap, external-only, and TRAIN ∩ COCONUT;
- the mass and formula index counts;
- `legacy_source_identity`: sha256 of the legacy manifest, `metadata.json`, candidate keys and offsets; the legacy
  notebook-12 config identity; the legacy Stage-A summary;
- `verification_checks` and `limitations`.

**Two separate facts.** `contents_verified: true` and `reproducible_build_id_available: false`. The adopted id is a
**content** digest, not a recipe. The universe can be measured read-only and used for regimes. A rebuild (notebook 12
Stage B/C) is required only for identity-dependent reuse.

**Safety:**

- The legacy manifest and the buckets are never rewritten.
- A sidecar with the same adopted id is left untouched, so adoption is idempotent.
- A different sidecar is refused unless `--force`. Even then the old sidecar is renamed `*.superseded-<timestamp>.json`
  and never deleted.

`effective_universe_identity` then reports `kind = legacy_adopted`. If the keys, offsets or legacy manifest change
later, the kind becomes `contradiction` (protocol case D, INVALID).

## Legacy Stage-A discovery (audit only)

The legacy code wrote markers and outputs in this layout:

- markers: `universe_work/_done/COCONUT-chunk-NNNNN.json`, each with `n_input`, `n_prefiltered`, `n_rejected`,
  `n_filtered`, `n_kept` and `finished_at`;
- outputs: `universe_work/{standardized,rejected,filtered_out}/COCONUT/chunk-NNNNN.parquet`.

`stage_a.active_stage_a_builds` reads only the modern `stage_a/ACTIVE/*.json` pointers. That is why notebook 13 showed
`ACTIVE={}` and 0 lost records.

`discover_legacy_stage_a` normalizes the legacy markers into one record per source with these fields:

- `source`, `status` (`complete_contiguous`, `gaps` or `outputs_missing`), `legacy=true`;
- `chunk_ids` and the n_* totals;
- `namespace` and `paths`;
- `inferred_identity` (kind `INFERRED_FROM_MARKERS`, not a reproducible build id).

`discover_stage_a_builds` merges modern and legacy records. `select_stage_a_namespaces` picks the legacy records for a
legacy or adopted universe and the modern complete builds for a modern one. The selection is always reported.

## Commands

The notebook is the preferred way. The script equivalent:

```
python scripts/adopt_legacy_universe_manifest.py \
  --universe /content/drive/MyDrive/EnvedaCASMI/candidates/universe \
  --work /content/drive/MyDrive/EnvedaCASMI/candidates/universe_work \
  --expect-n-candidates 696447 --expect-train 274195 --expect-coconut 446794 --expect-train-and-coconut 24542
```

Exit codes: 0 adopted or nothing to do, 2 refused, 3 universe absent. Add `--dry-run` to verify only.
