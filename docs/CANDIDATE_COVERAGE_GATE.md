# External Candidate Coverage Gate (Gate A, redesigned)

Notebook: `notebooks/13_candidate_coverage.ipynb`. Modules: `casmi.validation.external_coverage`,
`casmi.validation.protocol_v2`, `casmi.validation.regimes_v2`, `casmi.validation.panels`, `casmi.candidates.legacy_adoption`.
Status: **authored, not executed.** Every number in this document comes from the user's brief or from static code
reading. No coverage value has been computed.

> **Validation protocol v2** (branch `feature/validation-protocol-v2`). Notebook 13 now requires the identity-bound
> regime manifest of notebook 11b (else it stops with `REGIME_REBUILD_REQUIRED`), recognizes the adopted legacy universe
> (notebook 11a), evaluates the rules on the HOST / NP_PANEL decision panel (never TIMSTOF_DEV, the former `host_like`),
> refuses a final decision while the rule B/C/D thresholds are null (`BLOCKED_A8_THRESHOLDS`), audits legacy Stage-A
> drops, and diagnoses every round-trip failure. See `docs/VALIDATION_REGIME_V2.md`, `docs/LEGACY_UNIVERSE_ADOPTION.md`
> and `docs/NP_PANEL_PROTOCOL.md`.

## 1. Why the old Recall@All could be trivial

Validation molecules are training molecules. The finalized universe is TRAIN ∪ COCONUT. Its counts below come from the
user's run of notebook 12:

| | connectivities |
|---|---|
| TRAIN | 274,195 |
| COCONUT (unique) | 446,794 |
| TRAIN ∩ COCONUT | 24,542 |
| TRAIN ∪ COCONUT (finalized) | 696,447 |

COCONUT Stage-A accounting:

- 738,794 input records;
- 28,583 prefiltered;
- 2 rejected (parse_failed);
- 13,337 filtered (≈ 9,981 charged, 577 mass_out_of_range + charged, 1,821 multi_component);
- 696,872 source records kept.

Two mechanisms made the old notebook-13 number uninformative about database coverage:

1. **TRAIN provenance.** A held-out truth is always in the universe through its TRAIN record.
   `recall.truth_reachable(mode='universe')` is therefore ~100% by construction, and only measures mass accuracy.
2. **Selection on the outcome.** C2 eligibility (`class2_split.class2_eligible`) *requires* a non-TRAIN source. C2
   `external_only` Recall@All was therefore conditioned on external reachability, the very property it should measure.

Neither answers the real question: *could this structure have been found from external databases if it had never
appeared in training?*

## 2. The external-only protocol

The population is all held-out truth connectivities of the grouped (connectivity-level) validation split. No random
molecule split is used. For every truth `k`:

```
candidate_sources_without_train(k) = candidate_sources(k) − {TRAIN}          (remove_train_provenance_for_truth)
COCONUT_REACHABLE   "COCONUT" ∈ candidate_sources_without_train(k)          (finalized universe, current filter policy)
PUBCHEM_REACHABLE   k ∈ local PubChem key-membership index                   (PUBCHEM_KEY_MEMBERSHIP_ESTIMATE)
EXTERNAL_REACHABLE  any external source remains
EXTERNAL_UNREACHABLE none remains (TRAIN-only truth)
TRAIN_REACHABLE     diagnostic only: NEVER counted as reachability
```

- **C2 (external protocol)** = held-out truths that ARE externally reachable. A TRAIN-only truth is never C2.
  `legacy_c2_train_only` flags any legacy C2 truth without an external source; it must be 0.
- **No per-query rebuild.** TRAIN provenance is removed per truth. The external-only candidate pool is one boolean mask
  over candidate ids (`external_candidate_mask`), because TRAIN-only candidates would not exist without training.
- **Regression contract** (`tests/test_external_coverage_gate.py`):
  - A = TRAIN only: unreachable;
  - B = TRAIN + COCONUT: reachable after masking;
  - C = COCONUT: reachable;
  - D = PubChem membership: reachable.

## 3. Coverage is separated from ranking

Gate A measures **coverage only**:

- external reachability;
- whether the truth's mass falls inside the ±2/5/10 ppm window;
- pool sizes;
- filter losses;
- representative-SMILES round trip.

The frozen ranker was built around candidates with reference spectra, and external COCONUT/PubChem candidates lack
those features. Recall@25, Recall@100, rank and MRR are therefore **not** Gate-A numbers. The old notebook
`13_colab_candidate_recall.ipynb` remains as `LEGACY_DIAGNOSTIC_ONLY`. Ranking external candidates starts in notebook 14.

## 4. Database coverage failure vs mass retrieval failure

`evaluate_mass_window_coverage` labels each query with one of three outcomes:

| outcome | meaning | fix |
|---|---|---|
| `database_coverage_failure` | not externally reachable | the DATABASE (sources, filters, rescue) |
| `mass_retrieval_failure` | reachable, but outside the ppm window | precursor, adduct or mass handling |
| `retrieved` | reachable and inside the window | ranking comes later |

Pool sizes reported per query:

- `full_current_universe_pool_size` (finalized TRAIN ∪ COCONUT);
- `external_only_pool_size`;
- `projected_pubchem_tier1_pool_size`: RAW CID rows, only when a Tier-1 table exists. These are never mixed with
  finalized connectivity counts.

The truth's mass comes from the universe mass variants. For TRAIN ∩ COCONUT keys these are the TRAIN variants
(`merge.unify`), and `train_vs_open_mass_diff_da` audits the difference.

## 5. PubChem key-membership estimate (local files only)

- **Input.** The user supplies a `CID → InChIKey` file under `MyDrive/EnvedaCASMI/data/external/pubchem/`, for example
  PubChem's `CID-InChI-Key` export (tsv, gzip, no header). The layout is described in
  `configs/v6/pubchem_membership_source.json`, and URLs are refused. There is no downloader, no network access and no
  PubChem API.
- **Build** (authored, not run):
  ```
  python scripts/build_pubchem_membership_index.py --input <local file> --out-dir <Drive>/candidates/pubchem_membership --work-dir /content/enveda_work/pubchem_spool
  ```
  The build is chunked and resumable. It writes sorted parquet shards by key prefix, each holding `connectivity_key`,
  `n_records` and `min_cid`.
- **Lookup.** `PubChemMembershipIndex.contains(keys)` answers batch membership and loads only the shards it needs.
- **Caveat.** `connectivity_key` is the first block of PubChem's *stored* InChIKey. PubChem's tautomer, charge and salt
  representation can differ from the competition RDKit tautomer-canonical key, so the metric is always labelled
  **PUBCHEM_KEY_MEMBERSHIP_ESTIMATE**, never canonical parity.

## 6. PubChem Tier-1 architecture (`casmi.candidates.pubchem_tier1`, architecture only)

```
PubChem metadata (local) → metadata prefilter (config thresholds) → Tier-1 table (sorted by mass)
  → mass / formula index → query mass-window retrieval → competition RDKit canonicalization of the retained subset
  → connectivity dedup
```

- **Columns are optional.** Supported columns are exact mass, formula, formal charge, component count, literature count,
  patent count and NP flag. A rule whose column is absent is skipped and reported.
- **No invented thresholds.** Literature and patent thresholds default to `null`, meaning disabled.
- **Tier 2** is the remaining PubChem space, only if Tier 1 misses too much coverage.
- **Nothing runs** while `coverage.pubchem_tier1.enabled: false`. All of PubChem is never standardized.

## 7. Charge and salt rescue (`casmi.candidates.rescue`)

Two named policies:

- `current_policy`: the default. It built the current universe and changes nothing.
- `rescue_policy`, applied to external records only:

  1. Parse the record with RDKit, the same toolkit as the canonicalizer.
  2. Multi-component records: RDKit MolStandardize `LargestFragmentChooser(preferOrganic=True)`. The project has no
     salt-stripping standardizer of its own, so this documented RDKit chooser is used.
  3. Charged records: RDKit MolStandardize `Uncharger`.
  4. Still charged afterwards: **retained** with `permanently_charged=True` and
     `charge_cleanup_status=permanent_charge_retained`. The effective filters drop `neutral_only` for this policy.
  5. Provenance is kept: `original_source_smiles`, `original_component_count`, `selected_parent_smiles`,
     `salt_stripped`, `uncharged`, `formal_charge_before_cleanup`. `source` and `source_id` are untouched.

Rules that keep the two policies separate:

- Under the rescue policy, the formula prefilter drops only *organic* formulas outside the mass range, because salt
  formulas are not the parent's formula.
- TRAIN chemistry is never touched.
- An active policy enters the Stage-A identity (a new build) and the Stage-B identity. The existing COCONUT Stage-A build
  keeps its build id under `current_policy`.
- Notebook 13 compares the policies on the records the current policy lost: `rescue_lost_records`,
  `audit_filter_losses`, `filter_policy_comparison`. It never rebuilds the production universe.
- A rescue rebuild happens only when the user sets `universe.cleanup_policy: rescue_policy` after the decision
  `REBUILD_COCONUT_WITH_RESCUE_POLICY`.

**Adducts.** `adduct_charge_audit` lists only adducts present in the data. A `[M]+` / `[M]-` style adduct (M is the ion
itself) is flagged `permanently_charged_compatible`; none is assumed.

## 8. Representative-SMILES round trip

`casmi.validation.smiles_roundtrip` checks the following chain:

```
representative_smiles → competition_connectivity_key (training tautomer caps) → recomputed key == stored key ?
```

- It uses the exact project canonicalizer. Results are stratified as TRAIN only, COCONUT only and TRAIN+COCONUT.
- Failures are saved with `candidate_id`, `stored_connectivity_key`, `representative_smiles`,
  `recomputed_connectivity_key`, `candidate_sources` and `source_ids`.
- The check is resumable per bucket. An optional deterministic per-bucket sample uses a stable hash, never randomness.

**Precedence when a connectivity is in TRAIN and COCONUT** (`merge.unify`, unchanged):

- the TRAIN representative (first mass variant by `mass_variant_id`) is kept;
- `open_representative_smiles` = the first external record by (source order, `source_id`).

Determinism is tested by permuting the input rows.

**Diagnosis of failures (v2, `casmi.validation.roundtrip_diagnostics`).** The earlier full run reported 696,425 OK and
22 failures, all COCONUT-only. Notebook 13 cell `c-roundtrip-diagnose` re-canonicalizes the Stage-A source records of
every failing key and assigns one outcome: `REPRESENTATIVE_SMILES_FIX_ONLY`, `SOURCE_CANONICALIZATION_FIX`,
`MERGE_REPRESENTATIVE_SELECTION_FIX`, `EXPECTED_CANONICALIZER_VARIATION` or `UNRESOLVED`, with a `proposed_smiles`
when one maps back to the stored key. Nothing is replaced and nothing is blacklisted.

**Deployment gate** (`protocol_v2.roundtrip_deployment_check`): `ROUNDTRIP_DEPLOYMENT_OK` only when every emitted
candidate round-trips (failure rate 0) or is in an explicit blacklist passed deliberately by the user. A sampled round
trip (`coverage.roundtrip.max_per_bucket` set) can never certify deployment. A ranking decision needs this gate.

## 9. Official metric and public baseline

- **Official metric.** No competition-provided metric or canonicalizer exists in the repository. `src/metrics.py` is a
  project MRR helper, and `official_metric.py` on the unmerged `fix/v2-colab-bootstrap-artifacts` branch is a local
  reproduction built on the project canonicalizer. Notebook 13 therefore writes
  `OFFICIAL_METRIC_PARITY = NOT_AVAILABLE` unless `coverage.official_metric.module_path` points to a user-supplied file.
- **Public baseline.** No local artifact of the public notebook exists in the repository or on Drive. The interface is
  `casmi.validation.public_baseline.PublicBaselineSpec` (`coverage.public_baseline`: path, query_id / key / SMILES /
  rank columns, label, provenance, ppm). Without a file the output is `PUBLIC_BASELINE_COVERAGE = NOT_AVAILABLE`; a
  public leaderboard score is never typed into code or config.

## 10. Stage-B / Stage-C invalidation (fixed before any new source)

**Stage B** (`STAGE_B_VERSION = casmi-stage-b-3`): `stage_b_id` is a digest of metadata only, never a file hash. It covers:

- the universe schema;
- per source: Stage-A build id, file name, byte size, column mapping, canonicalizer contract (including RDKit version
  and profile), filters, prefilter, chunk size, cleanup policy, and chunk file names with sizes;
- TRAIN mass-variant and structure-table identity;
- the merge config (filters, prefilter, cleanup).

Bucket markers live under `_done/stage_b/<id>/bucket-X.json`. Each records:

- `build_id`, `bucket`, `created_at`;
- `source_build_ids`;
- `n_connectivities`, `n_variants`, `n_train`, `n_external_records`;
- file sizes.

A marker is reused only when the id, bucket and intact files all match. Older markers stay in place and are never deleted.

**Stage C** (`manifest_version 3`) writes these manifest fields:

- `build_id` (the Stage-C id) and `parent_stage_b_build_id`;
- `source_list`, `sources` counts and `overlap_counts`;
- filter, prefilter and cleanup policy;
- canonicalizer contracts;
- `mass_index_count`, `formula_index_count`, `external_source_present`.

`finalize_universe` refuses any bucket whose marker belongs to another Stage-B id. `universe_manifest_status` reports
`stale` and `legacy_manifest_detected`.

**Legacy.** The existing 696,447-candidate universe has a pre-v3 manifest without source counts, so
`universe_source_status` says `unknown` and the v1 protocol check reported "COCONUT has no candidates". Protocol v2
fixes this without a rebuild: notebook 11a (`casmi.candidates.legacy_adoption`) verifies the persisted artifacts and
writes the sidecar `universe_manifest_v3_adopted.json`; `effective_universe_identity` then reports `legacy_adopted`
with an adopted (content-digest) build id. See `docs/LEGACY_UNIVERSE_ADOPTION.md`. Nothing is rebuilt or deleted.

**Legacy Stage A.** The legacy run wrote `universe_work/_done/COCONUT-chunk-NNNNN.json` markers, which
`active_stage_a_builds` (ACTIVE pointers only) did not see; hence `ACTIVE={}` and 0 lost records. Notebook 13 now uses
`discover_stage_a_builds` + `select_stage_a_namespaces`: a legacy / adopted universe audits the legacy Stage-A drops.

## 11. Decision rules (pre-registered in `coverage.decision`)

The rules are evaluated on the **decision panel**: HOST (enveda-np-examples file) when available, else NP_PANEL
(structure-only NP-likeness with a user threshold). TIMSTOF_DEV (the former `host_like`, a timsTOF instrument subset)
never governs a decision.

| rule | condition (decision panel) | decision |
|---|---|---|
| A | PubChem adds ≥ 10 pp external reachability over COCONUT | `BUILD_PUBCHEM_TIER1_NEXT` |
| B | reachability ≥ `rule_b_min_reachability` AND pool statistic at ±5 ppm ≥ `rule_b_large_pool_threshold` | `BUILD_FORMULA_PRUNING_BEFORE_RANKING` |
| C | rescue recovers ≥ `rule_c_min_rescued_fraction` of the truths reachable before filters (requires the Stage-B/C fix, present) | `REBUILD_COCONUT_WITH_RESCUE_POLICY` |
| D | COCONUT reachability ≥ `rule_d_min_coconut_reachability` AND lost fraction ≤ `rule_d_max_lost_fraction` | `PROCEED_TO_ANALOG_BASELINE` |

- Only the 10 pp threshold is given by the project. The repository was searched for "Prompt A" / A8 threshold values:
  "A8" is the molecule aggregator (`casmi_infer.aggregation`), and no value for "large", "meaningful", "high" or
  "immaterial" exists. The B / C / D thresholds stay `null`: each rule is NOT_EVALUATED and the decision is
  **`BLOCKED_A8_THRESHOLDS`** (notebook 13 shows every rule but refuses a final decision). **Set them before running,
  never after looking at results.** The policy identity (`policy_identity_v2`) covers the thresholds, the A8 status and
  the decision-panel order.
- Priority is A, C, B, D. The first fired rule is the mechanical decision.
- Other outcomes: `PROTOCOL_INVALID` (protocol cases A / D), `DECISION_PANEL_UNAVAILABLE`, `REGIME_REBUILD_REQUIRED`
  (stopped before any coverage number), `UNDETERMINED_THRESHOLDS_OR_INPUTS_MISSING`, or `NO_RULE_FIRED`.
- Protocol validity v2: case A no external source → INVALID; case B legacy verified but not adopted →
  VALID_FOR_READ_ONLY_COVERAGE; case C modern or adopted → VALID; case D contradiction → INVALID.
- Notebook 14 runs only when the gate completed, the protocol is VALID, the regimes are current and bound to the same
  universe, the decision permits ranking, the round-trip deployment check is OK and a decision panel exists
  (`require_coverage_gate` + `require_protocol_v2_ready`, explicit logged override only).

## 12. Runtime order (user)

1. Fresh Colab. Run `10_colab_v2_setup.ipynb`: it clones branch `feature/validation-protocol-v2` into `/content/Enveda`,
   with scratch at `/content/enveda_work`.
2. Run `11a_colab_legacy_universe_adoption.ipynb` (legacy universe only), then `11b_colab_validation_regimes_v2.ipynb`.
3. Optional: build the PubChem membership index, add `PUBCHEM` to `validation_v2.external_sources`, re-run 11b.
4. Run `13_candidate_coverage.ipynb` top to bottom. Do **not** re-run notebook 12 automatically.
5. Follow the decision:
   - `BUILD_PUBCHEM_TIER1_NEXT`: build the PubChem Tier-1 table.
   - `REBUILD_COCONUT_WITH_RESCUE_POLICY`: set `universe.cleanup_policy: rescue_policy`, re-run 12, then 13.
   - `BUILD_FORMULA_PRUNING_BEFORE_RANKING`: pruning work before any ranker.
   - `PROCEED_TO_ANALOG_BASELINE`: run `14_colab_analog_baseline.ipynb`.

Optional, before notebook 13: build the PubChem membership index (section 5) and set any thresholds you want evaluated.

## 13. Artifacts and locations

Runtime artifacts (future; written by notebook 13) go to `MyDrive/EnvedaCASMI/reports/candidate_coverage/`:

- reachability:
  - `external_reachability_by_molecule.parquet`
  - `external_reachability_summary.csv`
  - `reachability_by_mass_bin.csv`
- pool density: `mass_pool_density.parquet`, `mass_pool_density_summary.csv`, `mass_window_outcomes.csv`
- filter loss: `filter_loss_by_molecule.parquet`, `filter_loss_summary.csv`, `rescue_attempts.parquet`
- round trip: `smiles_roundtrip_failures.parquet`, `smiles_roundtrip_summary.json`, `roundtrip_parts/`,
  `smiles_roundtrip_diagnosis.parquet`, `smiles_roundtrip_diagnosis.json` (with the deployment check)
- regimes / panels: `regime_counts_by_panel.csv`, `panel_status.json`
- PubChem: `pubchem_membership_summary.json`, `pubchem_incremental_coverage.csv`
- official metric and public baseline:
  - `official_metric_parity.json`
  - `public_baseline_coverage.json`
  - `public_baseline_coverage_comparison.csv`
- decision: `decision_rules.json` (declared, then evaluated), `coverage_gate_summary.json`
- audits: `mass_range_audit.csv`, `inference_mass_range.json`, `adduct_charge_audit.csv`
- other: `figures/`, `metadata.json`

Code lives on GitHub `mybenkhadda/enveda-kaggle`, branch `feature/validation-protocol-v2` (based on
`feature/external-coverage-gate`). The operational notebooks are in `MyDrive/EnvedaCASMI/notebooks/`, and a mirror of
every file changed by the protocol-v2 task is under `MyDrive/EnvedaCASMI/code_delivery/validation_protocol_v2/`
(repository-relative paths).
