# Public-leaderboard governance (CASMI 2026)

The public leaderboard is a **coarse external sanity check**. It is not a validation set. It is
computed on a small, fixed, unknown part of the hidden test. Anything tuned against it is
over-fitted to that part.

## Allowed uses

The public score may be used **only** for these purposes:

1. **Catching a major validation mismatch.** For example, the leaderboard contradicts the v6
   test-match resemblance status by a wide margin.
2. **Checking pipeline direction.** For example, the anchor submission scores in a plausible
   range, or an engineering change did not break inference.
3. **Evaluating a major candidate-universe change.** For example, closed-world TRAIN library
   versus TRAIN + COCONUT.

## Forbidden uses

The public score must **never** be used for:

* tuning thresholds (ppm windows, similarity cut-offs, resemblance-rule thresholds, …);
* choosing between minor model variants (hyperparameters, feature subsets, training-set sizes);
* tuning aggregator hyperparameters (RRF k, temperature, ε of the log-probability aggregator, …);
* repeated score chasing, meaning resubmitting small variations to climb the leaderboard.

Those choices are made on development populations only. The spectrum model uses TL_EVAL and the
molecule aggregator uses MOL_DEV. HOST is confirmation only, and each rule is pre-registered
before its metric exists.

## Procedure

1. Every submission is **manual**. No code in this repository calls a Kaggle API or submits
   anything.
2. Before submitting, record what is being submitted. After submitting, type in the public score
   by hand:

   ```python
   from casmi.submissions import fields_from_run_report, log_submission, resemblance_status_from
   f = fields_from_run_report('run_report.json')          # downloaded from the Kaggle notebook output
   f['validation_resemblance_status'] = resemblance_status_from('outputs/v6/test_match/validation_resemblance.json')
   log_submission('outputs/submissions/submission_log.jsonl', submission_name='anchor_v1_tl1k_strict_rrf',
                  candidate_universe_version='closed_world_train_library_v1', public_score=None,   # fill in manually
                  notes='anchor: frozen V1 1k + production inference + RRF k=60', **f)
   ```

3. Each submission states its purpose in `notes`, which must be one of the three allowed uses. A
   submission with no allowed purpose is not made.
4. A leaderboard observation that suggests a change is written into the decision log as
   `category='leaderboard_observation'`. The change is then **validated on development data**
   before it is adopted. The leaderboard score alone never justifies it.

## Anchor submission (v6)

Configuration: frozen `V1_TL_1K_TESTSIM_STRICT`, the current closed-world TRAIN candidate universe,
production inference semantics (`exclude_sources = {}`, T1 by peak hash, T2 not applied, waived
pending evidence) and RRF k = 60 molecule aggregation.

Its purpose is an **engineering and external sanity check**. It is not scientific model
selection.
