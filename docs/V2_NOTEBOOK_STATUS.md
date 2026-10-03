# CASMI v2 notebook status

Code status only. "Authored" means the code was written and statically reviewed; it has **not** been executed. Runtime
results are recorded by the user after running.

| Notebook | Status | Notes |
|---|---|---|
| 10 `10_colab_v2_setup` | unchanged | environment + Drive folders |
| 11 `11_colab_hidden_like_validation` | unchanged | still infers the C2 external regime from the universe manifest's existence. `casmi.candidates.universe.universe_source_status(OUT)` now gives an honest `train_only` / `external` answer; wiring it into 11 is pending |
| 12 `12_colab_candidate_universe` | **authored (Stage A optimization), not executed** | optimized resumable Stage A (`run_stage_a`), identity-checked Stage B (`run_stage_b`), Stage C refusing foreign buckets, manifest source counts, pandas-3-safe list-column reads |
| 13 `13_colab_candidate_recall` | unchanged | Gate A is meaningful only on an `external` universe |
| 14 `14_colab_analog_baseline` | unchanged | run only after a valid Gate A |

## Notebook 12: what to verify on the first run

1. The plan table shows `exists=True` for COCONUT, a staging action, the worker count, `gpu_used_for_stage_a = NO`, and a build id.
2. Stage A chunk lines appear: rows read, unique raw SMILES, standardizer calls, kept count, rows/s, ETA.
3. The status printed by Stage B shows `{'COCONUT': 'complete', ...}`.
4. The Stage C output shows `universe status: external` and `sources.COCONUT.n_candidates > 0`.
5. Optional: `scripts/benchmark_stage_a.py` reports `identical` for every size and profile.

## Open items

- Notebook 11: switch C2 eligibility to `universe_source_status` (not manifest existence).
- First real run: record the benchmark JSON and the Stage A chunk throughput in `reports/universe/`.
