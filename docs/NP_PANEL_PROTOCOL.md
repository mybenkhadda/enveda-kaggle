# Panels and the NP-panel protocol

Module: `casmi.validation.panels`. Config: top-level `panels:` in `configs/casmi_v2_colab.yaml`.
Status: **authored, not executed.**

## The four panels

| panel | definition | role |
|---|---|---|
| `ALL_DEV` | every grouped-validation dev query (formerly `dev_grouped`) | reporting |
| `TIMSTOF_DEV` | dev queries with `instrument_type` in `[timsTOF]` (formerly `host_like`; deprecated alias kept with a warning) | instrument diagnostic, **never a decision panel** |
| `HOST` | file-backed: `data/processed/host_holdout/host_holdout_spectra.parquet` (enveda-np-examples, 250 connectivities / 1,184 spectra in the local metadata) | decision panel (primary) |
| `NP_PANEL` | dev queries whose truth has a structure-only NP-likeness score ≥ a user threshold | decision panel (fallback) |

Panels are never merged. Every query row carries its panel.

## Why `host_like` was wrong

`host_like` filtered dev queries by `instrument_type == timsTOF`. That is the hidden test's instrument, not a
natural-product population. Using it as `natural_product_panel` meant Rule A ("PubChem adds ≥ 10 pp on the NP panel")
was evaluated on an instrument subset.

## HOST

The file is not on Drive by default. Copy it from the local repository, `data/processed/host_holdout/`, to
`MyDrive/EnvedaCASMI/data/processed/host_holdout/`.

- If the file is absent, HOST is `NOT_AVAILABLE` with reason `HOST_PANEL_SOURCE_MISSING`. It is never derived from
  another column and never fabricated.
- The source identity (name, size, sha256) is recorded in the panel manifest.
- A molecule count other than 250 is reported as a warning.

## NP_PANEL (non-circular, structure-only)

NP_PANEL membership is a function of the **truth structure only**. It is never defined by:

- COCONUT or PubChem membership;
- external reachability;
- a rank, a score, or any outcome measured on the panel.

Any of those would make Rule A circular.

**Methods** (`panels.NP_PANEL.method`):

- **`rdkit_contrib_np_score`.** The Ertl NP-likeness model shipped in RDKit's `Contrib/NP_Score`. Availability is
  checked at run time. If the installed RDKit lacks Contrib, the result is `NP_METHOD_UNAVAILABLE`. The repository had
  no NP-likeness implementation: the universe column `natural_product_score` is NaN by design.
- **`file`.** A user-supplied per-connectivity score table (`score_file`, `score_column`).

**Threshold.** `panels.NP_PANEL.threshold` is a **user decision**, declared before running notebook 11b. Its provenance
goes in `threshold_provenance`. While it is `null`, NP_PANEL is `NOT_AVAILABLE` with reason `NP_THRESHOLD_NOT_SET`. No
default threshold is invented.

**Scores.** They are computed in notebook 11b from the TRAIN structure-table SMILES, restricted to the dev truths. They
are cached in `data/processed/panels/np_scores.parquet` plus a `.json` sidecar holding the method, RDKit version and
model sha. Notebook 13 reads only the cache.

## Decision panel

`panels.decision_panel_order` is `[HOST, NP_PANEL]`. The first `AVAILABLE`, non-empty panel governs Gate A.
`TIMSTOF_DEV` and `ALL_DEV` are refused, both in the config validation and in `select_decision_panel`. With no
decision panel, the decision is `DECISION_PANEL_UNAVAILABLE`.

## Panel manifest

`data/processed/panels/panel_manifest.json` records:

- the panel config and its identity;
- per-panel status, reason, sizes, HOST source identity, and NP method and threshold;
- the decision panel.

The panel manifest identity is part of the regime identity, so any panel change forces notebook 11b to re-sign the
regimes.
