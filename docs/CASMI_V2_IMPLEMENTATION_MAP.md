# CASMI v2 implementation map

This map covers the v2 effort: Class-1 + Class-2 identification, with the existing closed-world system
kept as `DIRECT_LIBRARY_CHANNEL`. The frozen v1 production path (`bundle/`, `casmi_infer`,
`casmi_runtime`, Kaggle packaging) is **not modified**.

Legend: **R** reused as is · **E** extended (backward compatible) · **N** new · **F** frozen, untouched.

## Existing components

| Area | Module / file | Responsibility | Reusable? | v2 action |
|---|---|---|---|---|
| Data loading | `data/schema.py`, `data/metadata.py`, `pipelines/preprocessing.py` | train/test schemas, minted `train_spectrum_id`, CE summary, neutral mass | yes | **R**. v2 reads their artifacts (`train_spectrum_metadata`, `dev_queries`, `molecule_mass_variants`, `structure_table`) from Drive |
| Canonicalization | `chemistry/connectivity.py` (`competition_connectivity_key`), `chemistry/structures.py` (`build_structure_table`) | tautomer-canonical InChIKey first block = connectivity | yes | **R**. External structures use the same function via `candidates/standardize.py`. No second canonicalizer |
| Adducts / mass | `chemistry/adducts.py` | adduct grammar, precursor ↔ neutral mass | yes | **R** (`_formula_mass` also drives the cheap source-formula prefilter) |
| Grouped splitting | `validation/folds.py` (GroupKFold by connectivity), `connectivity_folds.parquet` | fold per connectivity | yes | **R**. Regimes are built on top; leakage is re-asserted per fold |
| Class-2 masking | `validation/class2_split.py` | C2 eligibility (non-TRAIN provenance), TRAIN-provenance stripping, reference masks | yes | **R**, wrapped by the new `validation/regimes.py` |
| Spectral preprocessing | `spectra/preprocessing.py`, `spectra/binning.py`, `spectra/neutral_loss.py` | cleaning, deterministic top-N, binning, losses | yes | **R**. The analog prefilter is a vectorized, row-wise `bin_spectrum` (parity-tested) |
| Similarity / QCR | `spectra/similarity.py`, `qcr/*`, `spectra/reference_selection.py`, `spectra/deduplication.py` | cosine / modified cosine / NL cosine, compat walk, identity tiers | yes | **R**. Analog rescoring calls `modified_cosine_similarity`; T1 exclusion by peak hash |
| Reference library | `bundle/` + `casmi_infer/reference_index.py` | memory-mapped cleaned peaks, connectivity → refs CSR | yes | **R** (read-only). `analog/retrieval.AnalogLibrary` inverts the CSR to ref → connectivity |
| Candidate retrieval | `candidates/mass_index.py` (`MassIndex`, `OpenMassIndex`, `FormulaIndex`), `candidates/generator.py` | sorted-search mass windows | yes | **E**: new `CandidateMassIndex` (int ids, mmap, batch, exclusion mask, truth rank) using the inference arithmetic |
| External sources | `candidates/sources.py`, `standardize.py`, `merge.py`, `provenance.py` | local readers, standardization + reject table, TRAIN+external unify, provenance, shortcut-feature guard | yes | **R**, orchestrated at scale by the new `candidates/universe.py` |
| Fingerprints | `chemistry/similarity.py` (Morgan, Tanimoto) | RDKit Morgan bits | yes | **R**. The new `chemistry/fingerprint_store.py` packs bits and caches them by connectivity |
| Learning to rank | `ranking/lambdamart.py`, `ranking/rank_eval.py` | LightGBM lambdarank fold models, per-query ranks over all intended queries | yes | **R** in notebook 14 (`fit_fold_models`, `predict_by_fold`, `per_query_metrics`) |
| Aggregation / calibration | `ranking/aggregation.py`, `ranking/calibration.py`, frozen A7 | molecule aggregation | yes | **R** later (Phase 12). Untouched now |
| Metrics | `validation/metrics.py` | MRR@k / Hit@k / candidate recall (list based) | yes | **E**: vectorized per-query candidate-stage + ranking metrics, PROVISIONAL composite |
| Bundle / runtime | `bundle/`, `src/casmi_infer`, `src/casmi_runtime`, `kaggle_upload/` | frozen v2-A7 inference | — | **F** |
| Notebooks | `src/*.ipynb` (v1–v6 history), `notebooks/0x_*` | exploration, freezes | read-only | v2 notebooks are new, under `notebooks/10_…14_…` |

## New in the first pass

| Module | Purpose | New dependency |
|---|---|---|
| `workspace/config.py`, `configs/casmi_v2_colab.yaml` | Colab/Drive paths (`ENVEDA_DRIVE_ROOT`, `ENVEDA_REPO_ROOT`), all config | pyyaml (already listed) |
| `workspace/environment.py` | GPU/VRAM report, bf16/fp16 policy, seeds, batch-size presets, `ensure_packages` | torch, psutil (optional; imported lazily) |
| `workspace/artifacts.py`, `workspace/experiments.py` | `metadata.json` per artifact folder; `experiments/experiments.parquet` | — |
| `validation/regimes.py` | C1/C2/C3 builder, fold-scoped masks, leakage assertions | — |
| `validation/reporting.py` | stratified tables (regime × ion mode / adduct / mass bin / instrument / #spectra / pool size) | — |
| `candidates/filters.py` | mass / organic / neutral / single-component filters with reasons | — |
| `candidates/universe.py` | resumable bucketed universe build (stages A–C) | pyarrow (already used) |
| `candidates/formula_index.py` | compact formula → candidate ids CSR | — |
| `candidates/recall.py` | Gate-A recall sweep (`external_only` vs `universe` reachability) | — |
| `chemistry/fingerprint_store.py` | packed Morgan bits, popcount Tanimoto, append-only cache | — |
| `analog/{retrieval,propagation,features,pipeline}.py` | analog-propagation channel | scipy (already listed); torch optional for the GPU prefilter |
| `viz/plots_v2.py` | reusable plots | matplotlib (already listed) |

## Key design decisions

* **Two separations are never confused.** First, *model leakage*: train ∩ validation connectivities = ∅ (folds). Second, *regime masks*: what fold f may see, namely the library minus hidden C2/C3 references and the universe minus C3 structures.
* **C2 requires an external source for the truth** (`class2_split`). Otherwise "in TRAIN but has no spectra" would single out the truth. Notebook 11 labels tables built before the universe exists as PRELIMINARY.
* **Gate A uses `external_only` reachability.** Dev truths always exist in TRAIN, so plain universe recall measures only mass accuracy.
* **Analog T2 policy = production reading.** T1 is excluded by peak hash. T2 is not applied because identity peaks are not shipped in the bundle.
* **Visible `test.parquet` is never a validation input.** Notebook 11 asserts that every query is a training spectrum.
