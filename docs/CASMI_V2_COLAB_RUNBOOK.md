# CASMI v2 — Colab runbook (first pass)

## 1. Drive layout (upload once)

```
MyDrive/EnvedaCASMI/                    <- ENVEDA_DRIVE_ROOT (override with the env var)
├── notebooks/                          <- operational .ipynb copies opened in Colab (10, 11a, 11b, 12, 13_candidate_coverage, 14)
├── bundle/                             <- the frozen v2-A7 bundle folder (reference library peaks + connectivities)
├── data/
│   ├── raw/train.parquet               (optional for this pass; not read by notebooks 10–14)
│   ├── processed/train_spectrum_metadata.parquet
│   ├── processed/connectivity_folds.parquet
│   ├── processed/dev_qcr/dev_qcr_pairs.parquet            <- protocol v2: mirror-aware C1 evidence (copy from local)
│   ├── processed/host_holdout/host_qcr_pairs.parquet      <- protocol v2: mirror-aware evidence of HOST queries
│   ├── processed/host_holdout/host_holdout_spectra.parquet <- protocol v2: HOST panel (enveda-np-examples)
│   ├── interim/structure_table.parquet
│   ├── interim/candidate_generation/dev_queries.parquet
│   ├── interim/candidate_generation/molecule_mass_variants.parquet
│   ├── external/coconut.csv            <- obtained MANUALLY
│   └── external/pubchem/               <- optional, MANUAL: PubChem CID -> InChIKey file for the membership estimate
└── (created by the notebooks) candidates/ cache/ reports/ predictions/ checkpoints/ experiments/ embeddings/
```

The source code is NOT on Drive: notebook 10 clones GitHub `mybenkhadda/enveda-kaggle` (branch `ENVEDA_GIT_BRANCH`,
default `feature/validation-protocol-v2`) to `/content/Enveda`. The one canonical scratch root is `/content/enveda_work`
(`ENVEDA_SCRATCH_ROOT` = `paths.scratch_dir`). It is disposable and never part of a build identity.

Copy the `data/…` files from the local repository's `data/` folder; their `.meta.json` sidecars are optional.
Check the column names in `configs/v6/coconut_source.json` (or `pubchem_source.json`) against the
header of your file, and list the file under `universe.sources` in `configs/casmi_v2_colab.yaml`.

## 2. Run order

| # | Notebook | Runtime | Writes |
|---|---|---|---|
| 10 | `10_colab_v2_setup` | any | `reports/environment/*.json`, Drive folders |
| 11 | `11_colab_hidden_like_validation` | CPU | LEGACY v1 (superseded by 11b for Gate A): `data/processed/validation_regimes.parquet` (+ metadata), `reports/validation_summary.{parquet,json}` |
| 11a | `11a_colab_legacy_universe_adoption` | CPU | adoption sidecar next to the legacy universe manifest (verification only; nothing rebuilt) |
| 11b | `11b_colab_validation_regimes_v2` | CPU | identity-bound v2 regimes, records, panel manifest, exclusion manifests |
| 12 | `12_colab_candidate_universe` | CPU, high RAM preferred | `candidates/universe/…` (buckets, index, formula index, manifest) |
| 11 again | with `REBUILD = True` | CPU | final (non-PRELIMINARY) regimes with external-source C2 eligibility |
| 13 | `13_candidate_coverage` | CPU | `reports/candidate_coverage/*`. **Gate A = external COVERAGE** (no ranking) and the decision |
| (13 legacy) | `13_colab_candidate_recall` | CPU | `reports/c2_candidate_recall.*`. **LEGACY_DIAGNOSTIC_ONLY**, never a Gate-A decision |
| 14 | `14_colab_analog_baseline` | GPU optional | only if notebook 13 decided `PROCEED_TO_ANALOG_BASELINE`; `cache/analog_neighbors/`, `data/processed/analog_features/`, `checkpoints/analog_ranker/`, `reports/analog_baseline/`, `predictions/analog_baseline/` |

**Validation protocol v2, with the current (legacy) TRAIN ∪ COCONUT universe already built:**

| step | notebook / command | writes |
|---|---|---|
| 1 | `10_colab_v2_setup` (fresh Colab) | environment record |
| 2 | `11a_colab_legacy_universe_adoption` (or `scripts/adopt_legacy_universe_manifest.py`) | `candidates/universe/universe_manifest_v3_adopted.json`, `reports/universe/legacy_adoption_report.json` |
| 3 | `11b_colab_validation_regimes_v2` | `data/processed/validation_regimes_v2.{parquet,manifest.json}`, `validation_regime_records_v2.parquet`, `panels/`, `exclusion_manifests/`, `legacy/validation_regimes_pre_external_universe.*` |
| 4 (optional) | upload `data/external/pubchem/CID-InChI-Key.gz`, run `python scripts/build_pubchem_membership_index.py`, add `PUBCHEM` to `validation_v2.external_sources`, **re-run 11b** | `candidates/pubchem_membership/` |
| 5 | `13_candidate_coverage` | `reports/candidate_coverage/*` (stops with `REGIME_REBUILD_REQUIRED` if 11b is stale) |
| 6 | read the decision | `BLOCKED_A8_THRESHOLDS` until the rule B/C/D thresholds are declared |
| 7 | `14_colab_analog_baseline` | only if `permits_ranking` (protocol VALID, regimes current, round-trip deployment OK) |

Before step 3, upload the protocol-v2 inputs listed in section 1 (mirror-aware evidence, HOST file) and, if wanted,
declare `panels.NP_PANEL.threshold`. Do NOT re-run 12 unless the decision is `REBUILD_COCONUT_WITH_RESCUE_POLICY`, or a
new external universe (e.g. PubChem Tier 1) is built deliberately. See `docs/CANDIDATE_COVERAGE_GATE.md`,
`docs/VALIDATION_REGIME_V2.md`, `docs/LEGACY_UNIVERSE_ADOPTION.md` and `docs/NP_PANEL_PROTOCOL.md`.

Notebook 11 can run before 12, but its C2 regime is then PRELIMINARY: C2 eligibility needs to know
which truths exist in an external source. Re-run 11 with `REBUILD = True` after 12. Notebooks 13
and 14 warn when the regimes are preliminary.

### Notebook 12 (candidate universe): optimized Stage A

Stage A is CPU-bound, and RDKit canonicalization dominates. The GPU is **not** used for it (see
`docs/STAGE_A_OPTIMIZATION.md`). Settings live in `universe.performance` in `configs/casmi_v2_colab.yaml`.
Worker count is `universe.n_jobs` (`auto`, or an integer override).

1. Optional: run the parity benchmark (`scripts/benchmark_stage_a.py`). Continue only if it reports *identical*.
2. Run notebook 12 top to bottom. The plan table shows the source size, staging, workers, the GPU decision and the build id.
3. After Stage C, `universe status` must be `external` and `sources.COCONUT.n_candidates` must be > 0.
   `train_only` means the C2 external-universe protocol is not valid.
4. Then re-run notebook 11 with `REBUILD = True`, then run notebook 13 (Gate A). Run notebook 14 only after a valid Gate A.

On a high-CPU VM or workstation, use `python scripts/build_external_universe.py --stage all --n-jobs auto`. It calls the
same functions and writes the same `candidates/` tree.

## 3. Restarts

Stage-A chunk markers carry a build identity: source file name and size, column mapping, chunk size, canonicalizer
contract (including the RDKit version), filters, and the cleanup policy when it is not `current_policy`.

Stage-B bucket markers (`casmi-stage-b-3`) carry the full Stage-B identity: per-source build id, file name and size,
columns, canonicalizer, filters, prefilter and cleanup policy, plus the TRAIN artifacts and merge config. A changed
source or config never reuses old chunks or buckets, and a TRAIN-only bucket is rebuilt once COCONUT is added.

The Stage-C manifest (v3) records `parent_stage_b_build_id`. A Stage C built for another Stage B is reported `stale`, and
a pre-v3 manifest is reported `legacy_manifest_detected` (readable; rebuild only when identity-dependent reuse is needed).
After this change the next notebook-12 run rebuilds Stage B and C once; the COCONUT Stage-A build is reused.

Every expensive step resumes from Drive: universe chunks and buckets (`_done/` markers), analog
neighbours and feature shards (`part-*.done`), fingerprints (append-only shards) and the binned
library matrices (`cache/analog_library_*`). After a disconnect, re-run the notebook from the top.

## 4. Gate A (before any ranking or deep learning)

Read `reports/candidate_coverage/coverage_gate_summary.json`. Gate A measures **external coverage**: TRAIN provenance
never counts as reachable. It also reports mass-window coverage, pool density, filter losses (current vs rescue policy)
and the representative-SMILES round trip.

The decision rules are declared in `coverage.decision` **before** running. Unset thresholds are reported NOT_EVALUATED,
never guessed. The old `reports/c2_candidate_recall.json` is LEGACY_DIAGNOSTIC_ONLY. Details are in
`docs/CANDIDATE_COVERAGE_GATE.md`.
