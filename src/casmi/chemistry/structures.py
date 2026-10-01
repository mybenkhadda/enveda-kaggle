"""Higher-level structure processing: turn a column of (possibly millions of, heavily
repeated) SMILES into one row per *unique* SMILES with every descriptor/identity field this
project needs -- RDKit only ever runs on the unique set, never once per spectrum row.
"""
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from casmi.chemistry.connectivity import canonicalize_smiles, competition_connectivity_key_detailed
from casmi.chemistry.descriptors import DESCRIPTOR_FIELDS, compute_molecular_descriptors

STRUCTURE_TABLE_FIELDS = ["smiles", "parse_ok", "error", "canonical_smiles", "plain_inchikey14",
                          "connectivity_key", "tautomer_hit_cap"] + DESCRIPTOR_FIELDS


def _process_one(smiles, max_tautomers, max_transforms):
    """Worker function -- module-level (not a closure/lambda) so it's picklable for a
    multiprocessing (loky) backend."""
    row = {k: None for k in STRUCTURE_TABLE_FIELDS}
    row["smiles"] = smiles
    row["parse_ok"] = False
    try:
        canon = canonicalize_smiles(smiles)
        if canon is None:
            row["error"] = "parse_failed"
            return row
        row["parse_ok"] = True
        row["canonical_smiles"] = canon

        conn = competition_connectivity_key_detailed(smiles, max_tautomers=max_tautomers, max_transforms=max_transforms)
        row["plain_inchikey14"] = conn["plain_inchikey14"]
        row["connectivity_key"] = conn["conn_key"]
        row["tautomer_hit_cap"] = conn["hit_cap"]
        if conn["error"]:
            row["error"] = conn["error"]

        row.update(compute_molecular_descriptors(smiles))
    except Exception as exc:
        row["error"] = f"unexpected_error: {exc}"
    return row


def build_structure_table(smiles, n_jobs=1, max_tautomers=None, max_transforms=None, show_progress=True):
    """One row per unique value in `smiles` (a column/iterable, possibly with duplicates and
    nulls -- both handled). Failures never abort the run: a bad SMILES gets a row with
    `parse_ok=False` and its `error` message, everything else `None`.

    Each `loky` worker process builds and caches its own `TautomerEnumerator` the first time
    `_process_one` runs in it (RDKit objects aren't picklable, so this can't be warmed once in
    the parent and shared) -- but a worker is reused across many tasks, so that's once per
    worker process, not once per row.
    """
    unique_smiles = pd.unique(pd.Series(smiles, dtype="object").dropna())

    records = Parallel(n_jobs=n_jobs, backend="loky", batch_size=256)(
        delayed(_process_one)(s, max_tautomers, max_transforms)
        for s in tqdm(unique_smiles, disable=not show_progress, desc="build_structure_table")
    )
    table = pd.DataFrame.from_records(records, columns=STRUCTURE_TABLE_FIELDS)
    assert table["smiles"].is_unique, "build_structure_table should emit exactly one row per unique input SMILES"
    return table
