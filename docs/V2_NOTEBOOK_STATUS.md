# CASMI v2 notebooks 10–14: implementation status

> **Execution status of this revision: NOT EXECUTED.** The refactor below was authored and statically reviewed
> only. No notebook, test, pipeline or git command was run for it. Every "status" entry describes code, not results.

## Root cause of the `AttributeError: 'V2Paths' object has no attribute 'universe' / 'regimes' / 'candidates'` errors

These were not isolated notebook bugs. The cause was **two path vocabularies in flight at once**:

- GitHub `main` (`85d43d9`) exposed the `*_dir` fields.
- An uncommitted local revision introduced aliases (`P.universe`, `P.regimes`, `P.reports`, …) that some notebooks had started to use.

Notebooks also cloned the repository only when it was absent, so a stale `/content/Enveda` kept running old code.

**Resolution (global).**

1. `V2Paths` = the `*_dir` physical roots (`casmi-v2-paths-4`), the only path API. Retired aliases are mapped to their replacement for error messages only (`config.DEPRECATED_PATH_ALIASES`).
2. `ArtifactRegistry` (`casmi.workspace.artifact_registry`, `casmi-v2-artifacts-2`) is the only place semantic locations are derived.
3. One generated bootstrap cell (`casmi-v2-bootstrap-2`) clones or fast-forwards the repository and purges stale imports. `bootstrap()` / preflight then check every version, field, artifact, signature, the repository state, external sources and bundle labels before work starts.
4. `scripts/audit_notebooks.py` statically rejects deprecated aliases, unknown fields and artifacts, obsolete signatures, imports before bootstrap, ad-hoc git, hard-coded paths, visible-test references and missing leakage audits.

## Status table

| Notebook | Purpose | Code status (this revision) | Key changes | Required inputs | Produces |
|---|---|---|---|---|---|
| 10 `colab_v2_setup` | environment, Drive inputs, bundle identity (labels), external sources, pipeline state, run order | authored, not executed | shared bootstrap; `missing_required(P)`; bundle identity by labels (no hashing); external-source table; `pipeline_status` (TRAIN-only universe = BLOCKED, protocol-invalid Gate A blocks 14); environment record in `reports/environment/` | 5 processed inputs, `bundle/config.json` | `reports/environment/*.json` |
| 11 `hidden_like_validation` | C1/C2/C3 regimes | authored, not executed | C2 mode from universe **provenance** (`assess_universe`), never from manifest existence; exposes `UNIVERSE_STATUS`, `EXTERNAL_SOURCE_PRESENT`, `C2_PROTOCOL_VALID`, `C2_MODE`; signature includes universe identity (stale tables refused); `audit_regimes` leakage audit | 5 inputs (+ universe) | `validation/regimes/*`, `reports/validation/*` |
| 12 `candidate_universe` | TRAIN + COCONUT (+ PubChem) universe | authored, not executed | canonical root `candidates/`; `finalize_universe(root, buckets)` (obsolete `formula_dir` / `manifest_dir` removed); manifest embeds `source_summary`; bucket markers keyed by stage-A input signature (adding COCONUT rebuilds buckets); pandas-2/3 reader in `casmi.io.parquet` | mass variants, structure table, external files | `candidates/**` |
| 13 `candidate_recall` (Gate A) | retrievability before ranking | authored, not executed | `assess_c2_protocol` → Gate-A metrics only if valid, else `FAIL_PROTOCOL_INVALID` + reason, no metric; requested metric set from `recall.gate_a`; universe-mode + mass-accuracy diagnostics labelled as such; out-of-fold mass calibration; `audit_gate_a`; append-only logging with `status` | regimes, universe | `reports/candidate_recall/**`, `gate_a_decision.json` |
| 14 `analog_baseline` | first spectrum-informed baseline | authored, not executed | refuses to run without a protocol-valid Gate A for the current universe (explicit diagnostic override); identity-namespaced neighbor / feature caches; buffered fingerprint cache; molecule-level fusion strategies compared; `decision_table` fold-stability rule; `audit_analog` (stale-cache, provenance, T2 limitation) | regimes, Gate A, universe, bundle | `cache/analog_neighbors`, `features/analog`, `reports/analog`, `checkpoints/ranker/analog`, `predictions/validation` |

## Remaining blockers

1. **No external candidate source exists yet.** The Drive universe observed on 2026-10-01 contained exactly the 274,195 TRAIN connectivities. Until COCONUT is at `data/external/coconut/coconut.csv`, the following hold:
   - Gate A is `FAIL_PROTOCOL_INVALID`.
   - Notebook 14 is blocked.
   - Every C2 number is preliminary.
2. **The existing Drive universe under `candidates/indexes/` is not read by this revision.** Notebook 12 must be re-run; it rebuilds the universe at `candidates/`.
3. **The regime table on Drive predates the universe-identity signature.** Notebook 11 will refuse it until `REBUILD = True`.
4. **T2 duplicate exclusion is not applied in the analog channel.** Identity peaks are not shipped in the bundle. This is documented as a limitation row in the notebook-14 leakage audit, not as a passed check.

## Earlier observations (previous code version, for re-measurement only)

A local run of the *previous* notebook versions on 2026-10-01 suggested the following:

- timsTOF truths are almost always within 5 ppm.
- There may be a systematic signed offset of about +1.6 ppm overall, and about +2.2 ppm for `[M+H]+`.
- Mass-only top-25 ranking is far weaker than raw retrieval coverage.

These are **not** hard-coded anywhere. `casmi.validation.mass_calibration` re-estimates offsets per fold, instrument and adduct, on training folds only, and notebook 13 reports the held-out before/after effect.

## Pre-existing test failures outside the v2 path

Earlier reporting listed 7 failures and 1 error in v4b / v5 / v6 / runtime-packaging tests. They are treated as pre-existing and were not modified by this refactor.
