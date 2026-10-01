"""Structure standardization for the OPEN candidate universe -- the SAME contract the training
structure table used, not a second canonicalization pipeline.

Training built its identities with `casmi.chemistry.structures.build_structure_table` (one row per
unique SMILES; RDKit canonical isomeric SMILES, the competition connectivity key = tautomer-canonical
InChIKey14 with the `PreprocessingConfig` tautomer caps, and `compute_molecular_descriptors`). This
module calls THAT function on external SMILES and only adds:

  * `molecular_formula` (RDKit `CalcMolFormula`; the training library took its formula from the
    competition file, so formula here is candidate METADATA, never a filter),
  * `n_fragments` / `is_charged` flags (no salt stripping or neutralization -- the training contract
    did neither; such structures are flagged, not silently altered),
  * an explicit REJECT table (`failure_reason`) instead of silent drops.

Output identity columns:
    raw_smiles             as supplied by the source
    normalized_smiles      RDKit canonical isomeric SMILES of raw_smiles (training `canonical_smiles`)
    representative_smiles  = normalized_smiles for an external record (a TRAIN connectivity keeps the
                           training representative in the unified universe, see `casmi.candidates.merge`)
    connectivity_key       competition connectivity (tautomer-canonical InChIKey14)
    neutral_monoisotopic_mass, molecular_weight, formal_charge (+ a few descriptors)
"""
import numpy as np
import pandas as pd

from casmi.chemistry.structures import build_structure_table

REJECT_COLUMNS = ["source", "source_id", "raw_smiles", "failure_reason"]
STANDARDIZED_COLUMNS = ["source", "source_id", "raw_smiles", "normalized_smiles", "representative_smiles", "connectivity_key",
                        "plain_inchikey14", "tautomer_hit_cap", "molecular_formula", "neutral_monoisotopic_mass", "molecular_weight",
                        "formal_charge", "n_fragments", "is_charged", "num_heavy_atoms", "num_rings", "num_aromatic_rings", "hbd", "hba",
                        "tpsa", "logp", "murcko_scaffold", "name", "source_formula", "source_metadata"]


def training_tautomer_caps():
    """The tautomer caps the TRAINING structure table used (`PreprocessingConfig` defaults)."""
    from casmi.config import PreprocessingConfig
    c = PreprocessingConfig()
    return c.tautomer_max_tautomers, c.tautomer_max_transforms


def molecular_formula(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None


def n_fragments(smiles):
    return len(str(smiles).split(".")) if isinstance(smiles, str) and smiles else 0


def standardize_records(records, n_jobs=1, show_progress=False, structure_table_fn=build_structure_table):
    """`records`: DataFrame with source, source_id, raw_smiles (+ optional name, source_formula,
    source_metadata). Returns `(standardized, rejected)`; every input row lands in exactly one of them.
    RDKit runs once per UNIQUE raw SMILES (the training function's own dedupe)."""
    rec = records.copy()
    for c in ("name", "source_formula", "source_metadata"):
        if c not in rec.columns:
            rec[c] = None
    rec["raw_smiles"] = rec["raw_smiles"].where(rec["raw_smiles"].notna(), None)
    blank = rec["raw_smiles"].isna() | (rec["raw_smiles"].astype(str).str.strip() == "")
    rejected = [rec.loc[blank, ["source", "source_id", "raw_smiles"]].assign(failure_reason="missing_smiles")]
    todo = rec[~blank]
    tmax, tran = training_tautomer_caps()
    table = structure_table_fn(todo["raw_smiles"], n_jobs=n_jobs, max_tautomers=tmax, max_transforms=tran, show_progress=show_progress)
    table = table.rename(columns={"smiles": "raw_smiles", "canonical_smiles": "normalized_smiles", "exact_mass": "neutral_monoisotopic_mass",
                                  "tautomer_hit_cap": "tautomer_hit_cap"})
    m = todo.merge(table, on="raw_smiles", how="left", validate="many_to_one")
    reason = np.where(~m["parse_ok"].fillna(False).astype(bool), "parse_failed: " + m["error"].fillna("").astype(str),
             np.where(m["connectivity_key"].isna(), "no_connectivity_key: " + m["error"].fillna("").astype(str),
             np.where(pd.to_numeric(m["neutral_monoisotopic_mass"], errors="coerce").isna(), "no_mass", "")))
    bad = reason != ""
    rejected.append(m.loc[bad, ["source", "source_id", "raw_smiles"]].assign(failure_reason=reason[bad]))
    ok = m[~bad].copy()
    uniq = pd.Series(ok["normalized_smiles"].unique())
    formula = dict(zip(uniq, uniq.map(molecular_formula)))
    ok["molecular_formula"] = ok["normalized_smiles"].map(formula)
    ok["representative_smiles"] = ok["normalized_smiles"]
    ok["n_fragments"] = ok["normalized_smiles"].map(n_fragments)
    ok["is_charged"] = pd.to_numeric(ok["formal_charge"], errors="coerce").fillna(0) != 0
    ok["neutral_monoisotopic_mass"] = ok["neutral_monoisotopic_mass"].astype(float)
    for c in STANDARDIZED_COLUMNS:
        if c not in ok.columns:
            ok[c] = None
    rej = pd.concat(rejected, ignore_index=True)[REJECT_COLUMNS]
    assert len(ok) + len(rej) == len(rec), "every input record must be standardized or rejected"
    return ok[STANDARDIZED_COLUMNS].reset_index(drop=True), rej.reset_index(drop=True)


def standardize_in_chunks(chunks, out_dir, n_jobs=1, prefix="part", show_progress=True):
    """Resumable chunked standardization: chunk i -> `<out_dir>/<prefix>-std-i.parquet` and
    `<prefix>-rej-i.parquet`; a chunk whose two files exist and whose record count matches is skipped.
    Returns `(standardized, rejected)` concatenated."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    std_parts, rej_parts = [], []
    for i, chunk in enumerate(chunks):
        fs, fr = out_dir / f"{prefix}-std-{i:05d}.parquet", out_dir / f"{prefix}-rej-{i:05d}.parquet"
        if fs.exists() and fr.exists():
            s, r = pd.read_parquet(fs), pd.read_parquet(fr)
            if len(s) + len(r) == len(chunk):
                std_parts.append(s)
                rej_parts.append(r)
                continue
        s, r = standardize_records(chunk, n_jobs=n_jobs, show_progress=False)
        s.to_parquet(fs, index=False)
        r.to_parquet(fr, index=False)
        std_parts.append(s)
        rej_parts.append(r)
        if show_progress:
            print(f"[standardize] chunk {i}: {len(s)} ok, {len(r)} rejected")
    std = pd.concat(std_parts, ignore_index=True) if std_parts else pd.DataFrame(columns=STANDARDIZED_COLUMNS)
    rej = pd.concat(rej_parts, ignore_index=True) if rej_parts else pd.DataFrame(columns=REJECT_COLUMNS)
    return std, rej
