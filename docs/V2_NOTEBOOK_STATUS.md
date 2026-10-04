# CASMI v2 notebook status

Code status only. "Authored" means the code was written and statically reviewed; it has **not** been executed. Runtime
results are recorded by the user after running.

| Notebook | Status | Notes |
|---|---|---|
| 10 `10_colab_v2_setup` | **authored (external-coverage-gate), not executed** | Rebuilt from the operational Drive version (cells c1–c9). `c6-verify` is now a real code cell: required inputs via `missing_required_inputs`, frozen bundle identity, import checks. One canonical scratch root, `/content/enveda_work` (`ENVEDA_SCRATCH_ROOT` = `paths.scratch_dir`). Branch from `ENVEDA_GIT_BRANCH` (default `feature/external-coverage-gate`). Next step points to notebook 13 |
| 11 `11_colab_hidden_like_validation` | **authored (small fix), not executed** | C2 mode is `external_universe` only when `universe_source_status == external`, no longer when a manifest merely exists. Truth rows are read with pyarrow (pandas-3 safe). Repo root resolves to the notebook-10 clone |
| 12 `12_colab_candidate_universe` | unchanged on this branch | Scratch now comes from the config (`/content/enveda_work`). Stage B/C identities are stronger (`casmi-stage-b-3`, manifest v3), so the next run rebuilds Stage B + C once and reuses Stage A. Re-run only on a notebook-13 decision |
| 13 `13_candidate_coverage` | **NEW, authored, not executed** | External Candidate Coverage Gate: external-only C2 protocol, COCONUT and PubChem membership, mass windows, pool density, filter loss with rescue comparison, adducts, round trip, official metric / public baseline (NOT_AVAILABLE without local artifacts), pre-registered decision rules |
| 13 `13_colab_candidate_recall` | **LEGACY_DIAGNOSTIC_ONLY** | Title, gate label and experiment id relabelled; never a Gate-A decision |
| 14 `14_colab_analog_baseline` | **authored (precondition), not executed** | New cell `c-coverage-gate`: `require_coverage_gate` needs a completed gate, a valid protocol and `PROCEED_TO_ANALOG_BASELINE`. It loads notebook-13 artifacts and never recomputes coverage |

## Notebook 13: what to verify on the first run

1. `c-universe` prints `universe_status: external` and the legacy flag. `legacy_manifest_detected = true` is expected
   for the universe built before this branch.
2. `c-populations` shows `dev_grouped` and a non-empty `host_like` panel. If the panel is empty, adjust
   `coverage.panels.host_like.values` to the printed `instrument_type` values **before** reading any result.
3. `c-membership` prints `legacy C2 truths without any external source (must be 0)`.
4. `c-decision` prints the decision and every rule's status.

## Open items

- Set the `coverage.decision` thresholds that should be evaluated (B, C, D) **before** running notebook 13.
- Optional PubChem membership: upload the local CID→InChIKey file and run `scripts/build_pubchem_membership_index.py`.
- Merge `feature/external-coverage-gate` into `main` (user decision), then set `ENVEDA_GIT_BRANCH=main`, or edit notebook 10's default.
