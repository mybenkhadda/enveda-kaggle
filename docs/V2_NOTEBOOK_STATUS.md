# CASMI v2 notebook status

Code status only. "Authored" means the code was written and statically reviewed; it has **not** been executed. Runtime
results are recorded by the user after running.

| Notebook | Status | Notes |
|---|---|---|
| 10 `10_colab_v2_setup` | **authored (validation-protocol-v2), not executed** | Cells c1–c9. `c6-verify` is a real code cell: required inputs (`missing_required_inputs`), frozen bundle identity, import checks. One canonical scratch root, `/content/enveda_work`. Branch from `ENVEDA_GIT_BRANCH` (default `feature/validation-protocol-v2`). Next step points to 11a → 11b → 13 |
| 11 `11_colab_hidden_like_validation` | **LEGACY v1, superseded for Gate A** | 34 / 33 / 33 fraction split, `test_simulated_strict` C1. Its table is archived by 11b and marked STALE; notebook 13 no longer reads it |
| 11a `11a_colab_legacy_universe_adoption` | **NEW, authored, not executed** | Verifies the legacy TRAIN ∪ COCONUT universe from persisted artifacts. Writes `universe_manifest_v3_adopted.json` only if every invariant holds. Discovers legacy Stage-A markers |
| 11b `11b_colab_validation_regimes_v2` | **NEW, authored, not executed** | Eligibility-based C1 / C2 / C3 / UNRESOLVED (mirror-aware C1 authority, natural counts), four panels, identity-bound manifest + signature, v1 archive, exclusion manifests |
| 12 `12_colab_candidate_universe` | unchanged on this branch | Re-run only on a notebook-13 decision |
| 13 `13_candidate_coverage` | **authored (protocol v2), not executed** | Stops with `REGIME_REBUILD_REQUIRED` unless the 11b manifest matches. Protocol validity A–D, panels with HOST / NP_PANEL decision panel, regimes per panel, legacy Stage-A filter-loss audit, round-trip diagnosis + deployment check, public-baseline interface, `BLOCKED_A8_THRESHOLDS` while B/C/D thresholds are null, bound experiment row |
| 13 `13_colab_candidate_recall` | **LEGACY_DIAGNOSTIC_ONLY** | Never a Gate-A decision |
| 14 `14_colab_analog_baseline` | **authored (preflight v2), not executed** | `c-coverage-gate` runs `require_coverage_gate` + `require_protocol_v2_ready`: VALID protocol, current regimes bound to the same universe, decision permits ranking, round-trip deployment OK, decision panel present |

## First run: what to verify

1. **11a.** `status: LEGACY_EXTERNAL_VERIFIED`, every check passed, and the counts equal the expected 696,447 / 274,195 /
   446,794 / 24,542. Then `kind = legacy_adopted`.
2. **11b.**
   - The panel table: HOST is `AVAILABLE` only if its file was uploaded. NP_PANEL is `NOT_AVAILABLE`
     (`NP_THRESHOLD_NOT_SET`) until a threshold is declared.
   - The mirror-aware evidence files were found.
   - The natural regime counts per panel, including the UNRESOLVED share.
   - `ALL REGIME CHECKS PASSED: True`.
3. **13.**
   - `c-regimes` prints `REGIME_CURRENT`.
   - `c-universe` prints protocol case C (`VALID`).
   - `c-filter-loss-load` selects the legacy COCONUT namespace.
   - `c-roundtrip-diagnose` lists one outcome per failure.
   - `c-decision` prints `BLOCKED_A8_THRESHOLDS` until the B/C/D thresholds are declared.

## Open items (user inputs)

- Upload to Drive:
  - `data/processed/dev_qcr/dev_qcr_pairs.parquet`
  - `data/processed/host_holdout/host_qcr_pairs.parquet`
  - `data/processed/host_holdout/host_holdout_spectra.parquet`
- Declare the rule B / C / D thresholds in `coverage.decision`. No project value exists.
- Optionally declare `panels.NP_PANEL.threshold` and its provenance.
- Optionally upload `data/external/pubchem/CID-InChI-Key.gz`, build the membership index, add `PUBCHEM` to
  `validation_v2.external_sources`, and re-run 11b.
- The 22 round-trip failures need a fix chosen from their diagnosed outcome, or a deliberate blacklist, before any
  ranking deployment.
- Merging `feature/validation-protocol-v2` into `main` is the user's decision.
