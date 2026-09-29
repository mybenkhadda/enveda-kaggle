"""Molecule rankings -> the competition submission (`molecule_id,smiles` with up to 25
';'-joined SMILES, best first). Deduplication by connectivity happens BEFORE truncation; SMILES are
the exported representative SMILES of each connectivity.
"""
import numpy as np
import pandas as pd

SEPARATOR = ";"


def molecule_top_k(mol_ranking, conn_smiles, top_k=25, cand_col="conn_idx"):
    """`mol_ranking`: Aggregator.rank output. Returns {molecule_id: [smiles...]} (<= top_k, no
    duplicate connectivity, order = mol_rank)."""
    r = mol_ranking.sort_values(["molecule_id", "mol_rank"], kind="mergesort")
    r = r.drop_duplicates(["molecule_id", cand_col])
    out = {}
    for mid, g in r.groupby("molecule_id", sort=False):
        out[mid] = [conn_smiles[int(c)] if cand_col == "conn_idx" else conn_smiles[c] for c in g[cand_col].head(top_k)]
    return out


def pad_with_nearest(ranked_conn, extra_conn, top_k=25):
    """Append deterministic nearest-mass connectivities not already present, up to top_k (logged by
    the caller; extra guesses beyond the ranked pool can only add hits at ranks <= 25)."""
    seen, out = set(ranked_conn), list(ranked_conn)
    for c in extra_conn:
        if len(out) >= top_k:
            break
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_submission(sample_submission, smiles_by_molecule):
    """Exact sample-submission row order and columns; every molecule present."""
    sub = sample_submission[["molecule_id"]].copy()
    sub["smiles"] = [SEPARATOR.join(smiles_by_molecule.get(m, [])) for m in sub["molecule_id"]]
    return sub


class SubmissionBlocked(RuntimeError):
    pass


def write_submission(sub, path, selftest_report):
    """The ONLY writer of submission.csv: refuses unless the bundle self-test passed on the backend
    that produced these predictions."""
    if not selftest_report or not selftest_report.get("passed"):
        raise SubmissionBlocked("bundle self-test did not pass -- refusing to write a submission")
    sub.to_csv(path, index=False)
    return path
