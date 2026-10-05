# Validation regimes v2

Notebook: `notebooks/11b_colab_validation_regimes_v2.ipynb`. Module: `casmi.validation.regimes_v2`.
Status: **authored, not executed.** No regime has been assigned and no count below is a result.

## 1. Why the v1 table was stale (mechanism)

The persisted `data/processed/validation_regimes.parquet` was built by notebook 11 (v1) on 2026-10-01, before the external
universe existed. Four defects combined:

1. **Wrong universe.** It was built against the TRAIN-only universe (274,195 candidates). C2 eligibility required a
   non-TRAIN source, which did not exist yet, so the table was PRELIMINARY.
2. **Nobody checked it.** Notebook 11 wrote a signature (`regime_config_hash`, `c2_mode`, `universe_n_candidates`,
   `n_dev_queries`), but notebook 13 read the parquet directly and filled missing regimes with `UNASSIGNED`. When the
   universe grew to 696,447 candidates, nothing noticed.
3. **Imposed shares.** Eligible truths were split 34 / 33 / 33 by a seeded hash. A truth with real reference evidence
   could be put in C3, so the regime shares said nothing about the data.
4. **Wrong C1 authority.** C1 meant "≥ 2 spectra of the truth" under `test_simulated_strict`, not the mirror-aware
   protocol that is the project's reference-availability authority.

## 2. v2 assignment

Regimes are assigned per truth **connectivity**. The first matching rule wins. There are no fractions, no hash and no
random split, so the counts are natural.

| regime | rule | reason code |
|---|---|---|
| C1 | ≥ 1 eligible **mirror-aware** reference of the truth | `C1_MIRROR_AWARE_REFERENCE` |
| UNRESOLVED | no mirror-aware evidence for any query of the connectivity | `UNRESOLVED_MIRROR_AWARE_EVIDENCE_MISSING` |
| C2 | no mirror-aware reference, but externally reachable | `C2_NO_MIRROR_REFERENCE_EXTERNALLY_REACHABLE` |
| C3 | neither | `C3_NO_MIRROR_REFERENCE_NOT_EXTERNALLY_REACHABLE` |

- **Mirror-aware evidence.** Eligibility excludes same-source references and T1/T2 mirrors
  (`casmi.spectra.reference_selection.is_eligible`). The flag is the exact lazy-walk QCR truth flag
  `has_ge1_mirror_aware`, read from the configured files (`validation_v2.mirror_aware_evidence`):
  - `data/processed/dev_qcr/dev_qcr_pairs.parquet`
  - `data/processed/host_holdout/host_qcr_pairs.parquet`

  Copy both from the local repository to Drive. They are a few MB.
- **Several queries per connectivity.** Mirror-aware eligibility depends on the query's source library, so a
  connectivity counts as having a mirror-aware reference when **any** of its evaluated queries does. The counts
  `n_queries_mirror_evaluated` and `n_queries_with_mirror_reference` are recorded with it.
- **UNRESOLVED.** No evidence means the query was not walked by QCR, or its truth lay outside the walked pool.
  UNRESOLVED is always reported, and never folded into C2 or C3.
- **`test_simulated_strict`** is a sensitivity diagnostic only (`has_test_simulated_strict_reference`). It is
  `NOT_AVAILABLE` unless `validation_v2.test_simulated_strict_evidence.files` is set, and it never decides C1.
- **External sources** come from `validation_v2.external_sources` and are currently `[COCONUT]`. TRAIN provenance never
  counts. Adding `PUBCHEM` requires a built PubChem membership index, and it changes the identity, so notebook 11b
  must re-sign the regimes.

### Per-molecule record (§18)

`validation_regime_records_v2.parquet` has one row per (panel, connectivity):

| column | content |
|---|---|
| `connectivity_key`, `panel`, `fold`, `n_queries` | the molecule and its panel |
| `has_mirror_aware_reference`, `n_queries_mirror_evaluated`, `n_queries_with_mirror_reference` | C1 evidence |
| `has_test_simulated_strict_reference` | sensitivity evidence |
| `coconut_reachable`, `pubchem_membership_estimate`, `external_reachable` | reachability; `external_sources` lists the sources used |
| `regime`, `regime_reason` | the assignment |

## 3. Identity-bound manifest

`data/processed/validation_regimes_v2.manifest.json` holds the `identity` below and a `regime_signature`, which is a
digest of every field except `git_commit` and `built_at`:

| field | content |
|---|---|
| `protocol_version` | `casmi-validation-regime-2` |
| `git_commit` | recorded and compared. A mismatch is a warning; it forces a rebuild only with `validation_v2.strict_git_commit: true` |
| `dev_queries_identity`, `split_identity` | file identity plus a digest of (query_id, truth key, fold) and of (connectivity_key, fold) |
| `universe_build_id`, `universe_identity_kind` | the modern Stage-C id, or the **adopted** legacy digest |
| `external_sources`, `membership_identity` | the PubChem index identity, when PUBCHEM is listed |
| `mirror_aware_protocol_identity` | `reference_selection` source hash plus the evidence files (name, size, sha256, rows) |
| `panel_manifest_identity`, `host_source_identity` | the panels and the HOST file |
| `policy_identity`, `assignment_code_identity` | the assignment rules and the source of `regimes_v2` |
| `seed`, `config` | `validation_v2` and `panels` |

Notebook 11b signs the manifest. Notebook 13 recomputes the same identity with the same function,
`regime_identity_from_run`. Any mismatch stops notebook 13 with `REGIME_REBUILD_REQUIRED`, and the
`coverage_gate_summary.json` it writes then says `completed: false`. Notebook 14 re-checks the signature against the
gate.

## 4. Safe replacement of the v1 table

`archive_legacy_regimes` **copies** these files into `data/processed/legacy/`:

- `validation_regimes.parquet` → `validation_regimes_pre_external_universe.parquet`
- `validation_regimes.metadata.json` → `validation_regimes_pre_external_universe.metadata.json`

It also writes `validation_regimes.STALE.json` next to the original. It never deletes anything, and it refuses to
overwrite a different existing archive. Notebook 11 (v1) is labelled LEGACY and superseded.

## 5. Integrity

- **Definitional invariants** (`assert_regime_v2_consistency`, raises on failure):
  - C1 ⇔ mirror evidence True;
  - UNRESOLVED ⇔ evidence missing;
  - C2 ⇒ reachable;
  - C3 ⇒ not reachable;
  - one regime per connectivity.
- **Leakage and masking invariants.** The v1 `assert_regime_integrity` runs on ALL_DEV for every fold: train ∩ val = ∅,
  one fold per connectivity, C2/C3 truths hidden from the reference library, and C3 removed from the pool.

## 6. Exclusion manifests (support only)

`data/processed/exclusion_manifests/` contains:

- `host_connectivities.parquet`, `np_panel_connectivities.parquet`, `c2_connectivities.parquet`,
  `c3_connectivities.parquet`;
- `model_training_exclusions.metadata.json`, which carries the regime signature and the panel manifest identity.

A model evaluated on panel P or regime R must exclude the matching list from its training data.
`casmi.ranking.fingerprint_contract` builds and checks a fingerprint dataset manifest against these lists. It does no
training.

## 7. Experiment binding

`casmi.workspace.experiments.log_bound_experiment` requires these fields:

- timestamp, notebook, git_commit (an empty value is refused), universe_build_id, regime_signature,
  panel_manifest_identity, policy_identity, model_name and status;
- for ranking rows, also feature_contract and checkpoint_identity.

Rows written by the old `log_experiment` are kept and labelled `legacy_unbound=True`. They are never comparable
(`comparable_experiments`).
