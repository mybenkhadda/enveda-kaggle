# Roadmap: the reference-absent route (NOT implemented)

This route is taken if `outputs/v5/project_route.json` says `PRIORITIZE_REFERENCE_ABSENT`, meaning
the TL_EVAL `MODE_A_MIRROR` share is below 0.50. The running TL_3K / TL_10K builds and the Mode-A
scale comparison still finish first.

Candidate directions for evidence that exists when a candidate has **zero direct reference spectra**:

- **External candidate universe** (the planned Notebook 06). Candidates beyond the training
  structure library, which is also a prerequisite for a trustworthy Mode B.
- **Formula / exact-mass constraints.** Formula plausibility against the precursor, isotope and
  adduct consistency.
- **Analogue-spectrum transfer** (the planned Notebook 08). Evidence from reference spectra of
  structurally similar connectivities.
- **Molecular-fingerprint evidence.** Predicted fingerprints from the query spectrum, compared
  with candidate fingerprints.
- Any evidence usable when the direct reference count is 0, evaluated on the `REF_ABSENT` and
  `STANDARD_ONLY` TL_EVAL populations produced by `10v5_03_tl_regime_audit.ipynb`.

Nothing here is built yet. Scope and design are decided after the v5.1 audit results are reviewed.
