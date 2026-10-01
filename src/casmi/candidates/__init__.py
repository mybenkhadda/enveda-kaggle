"""Candidate generation: given a query's neutral mass (from precursor m/z + adduct), which
train structures are plausible? A retrieval stage, not a ranking stage -- spectral similarity,
learned rerankers, and generative models are all explicitly out of scope here (see notebook
03). Candidate generation only answers "is the correct structure available to a downstream
model at all", never "can that model rank it near 1".

    casmi.candidates.mass_index    sorted-array exact-mass index (MassIndex), versioned cache +
                                    brute-force validation
    casmi.candidates.generator     CandidateGenerationConfig, CandidateGenerator (fixed OR
                                    per-row/adaptive tolerance)
    casmi.candidates.evaluation    tolerance-sweep recall/candidate-count tradeoff, marginal-
                                    cost deltas, decision classification
    casmi.candidates.diagnostics   subgroup/fold/failure/mass-error/mass-bin/density-shift
                                    breakdowns, true-structure-in-library verification
    casmi.candidates.adaptive      fold-safe per-group mass-tolerance policy fitting/evaluation
    casmi.candidates.molecule      spectrum-level candidate pools -> molecule-level union/
                                    intersection/support
    casmi.candidates.cache         cache-first helpers ([CACHE HIT]/[CACHE MISS]) with real
                                    upstream-artifact fingerprinting (`upstream_fingerprints`)

v6 OPEN candidate universe (infrastructure only; nothing is downloaded):

    casmi.candidates.sources       config-driven LOCAL readers (csv/tsv/parquet/smi/sdf) -> canonical records
    casmi.candidates.standardize   external SMILES through the TRAINING structure table (same
                                    connectivity contract) + explicit reject table
    casmi.candidates.provenance    per-connectivity source/id provenance, source-prior schema,
                                    future evidence schema + reference-availability shortcut guard
    casmi.candidates.merge         unified TRAIN + external universe (one row per connectivity)
    casmi.candidates.mass_index    + OpenMassIndex (inference retrieval arithmetic, deterministic,
                                    npy persistence, brute-force parity) and FormulaIndex (metadata only)

v2 (Colab, PubChem-scale; see docs/CASMI_V2_IMPLEMENTATION_MAP.md):

    casmi.candidates.filters       mass / organic / neutral / single-component filters (reasons kept)
    casmi.candidates.universe      resumable bucketed universe build -> v2 schema, global candidate ids
    casmi.candidates.mass_index    + CandidateMassIndex (int ids, mmap, batch search, exclusion masks)
    casmi.candidates.formula_index compact persisted formula -> candidate ids
    casmi.candidates.recall        Gate-A recall sweep (external_only vs universe reachability)

CLOSED-WORLD CAVEAT: `MassIndex` is built from `molecule_metadata`, which covers every unique
train connectivity_key regardless of CV fold. So a development query's true structure is
*always* present in the library it's being searched against (that's what "closed-world /
library-contained candidate retrieval" means) -- this measures how well precursor/adduct
physics alone retrieves a known structure, NOT coverage of genuinely novel molecules absent
from the candidate database. That's a separate, harder problem this notebook does not attempt.
"""
