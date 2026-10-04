# External Candidate Coverage Gate (Gate A, redesigned)

Notebook: `notebooks/13_candidate_coverage.ipynb`. Module: `casmi.validation.external_coverage`.
Status: **authored, not executed.** Every number in this document comes from the user's brief or from static code
reading. No coverage value has been computed.

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

## 9. Official metric and public baseline

- **Official metric.** No competition-provided metric or canonicalizer exists in the repository. `src/metrics.py` is a
  project MRR helper, and `official_metric.py` on the unmerged `fix/v2-colab-bootstrap-artifacts` branch is a local
  reproduction built on the project canonicalizer. Notebook 13 therefore writes
  `OFFICIAL_METRIC_PARITY = NOT_AVAILABLE` unless `coverage.official_metric.module_path` points to a user-supplied file.
- **Public baseline.** No local artifacts of the public ~0.335 notebook were found in the repository or on Drive. The
  output is `PUBLIC_BASELINE_COVERAGE = NOT_AVAILABLE` until `coverage.public_baseline.path` is set (query_id +
  connectivity_key or SMILES). The adapter then separates a coverage gap from a ranking gap.

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

**Legacy.** The existing 696,447-candidate universe has a v2 manifest. It stays readable: notebook 13 only reads keys,
buckets and indexes, and reports `legacy_manifest_detected = true`. The next notebook-12 Stage B/C run rebuilds the
buckets under the v3 identity; the Stage-A COCONUT build is reused. Nothing is deleted.

## 11. Decision rules (pre-registered in `coverage.decision`)

| rule | condition (NP panel = `natural_product_panel`, default `host_like` = timsTOF dev queries) | decision |
|---|---|---|
| A | PubChem adds ≥ 10 pp external reachability over COCONUT | `BUILD_PUBCHEM_TIER1_NEXT` |
| B | reachability ≥ `rule_b_min_reachability` AND pool statistic at ±5 ppm ≥ `rule_b_large_pool_threshold` | `BUILD_FORMULA_PRUNING_BEFORE_RANKING` |
| C | rescue recovers ≥ `rule_c_min_rescued_fraction` of the truths reachable before filters (requires the Stage-B/C fix, present) | `REBUILD_COCONUT_WITH_RESCUE_POLICY` |
| D | COCONUT reachability ≥ `rule_d_min_coconut_reachability` AND lost fraction ≤ `rule_d_max_lost_fraction` | `PROCEED_TO_ANALOG_BASELINE` |

- Only the 10 pp threshold is given by the project. The other thresholds are `null`, so the rule is NOT_EVALUATED; no
  value for "large", "meaningful", "high" or "immaterial" was invented. **Set them before running, never after looking
  at results.**
- Priority is A, C, B, D. The first fired rule is the decision.
- Other outcomes: `PROTOCOL_INVALID` (no external source, or no population),
  `UNDETERMINED_THRESHOLDS_OR_INPUTS_MISSING`, or `NO_RULE_FIRED`.
- Notebook 14 runs only when `coverage_gate_summary.json` says completed, protocol valid and `PROCEED_TO_ANALOG_BASELINE`.
  `require_coverage_gate` enforces this, with an explicit override flag that is logged.

## 12. Runtime order (user)

1. Fresh Colab. Run `10_colab_v2_setup.ipynb`: it clones branch `feature/external-coverage-gate` into `/content/Enveda`,
   with scratch at `/content/enveda_work`.
2. Run `13_candidate_coverage.ipynb` top to bottom. Do **not** re-run notebook 12 automatically.
3. Follow the decision:
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
- round trip: `smiles_roundtrip_failures.parquet`, `smiles_roundtrip_summary.json`, `roundtrip_parts/`
- PubChem: `pubchem_membership_summary.json`, `pubchem_incremental_coverage.csv`
- official metric and public baseline:
  - `official_metric_parity.json`
  - `public_baseline_coverage.json`
  - `public_baseline_coverage_comparison.csv`
- decision: `decision_rules.json` (declared, then evaluated), `coverage_gate_summary.json`
- audits: `mass_range_audit.csv`, `inference_mass_range.json`, `adduct_charge_audit.csv`
- other: `figures/`, `metadata.json`

Code lives on GitHub `mybenkhadda/enveda-kaggle`, branch `feature/external-coverage-gate`. The notebook's operational
copy is at `MyDrive/EnvedaCASMI/notebooks/13_candidate_coverage.ipynb`, and a mirror of every file changed by this task
is under `MyDrive/EnvedaCASMI/code_delivery/external_coverage_gate/` (repository-relative paths).
