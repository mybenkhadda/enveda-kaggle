"""Local reproduction of the competition metric: molecule-level MRR@25 on the RDKit tautomer-canonical InChIKey14.

What the competition states (Kaggle overview / data page): up to 25 ranked SMILES per `molecule_id`, separated by
';'; a prediction matches when its RDKit tautomer-canonicalized InChIKey14 (connectivity block, stereo ignored)
equals the truth's; score = mean over molecules of 1/rank of the first match (0 if none in the first 25).

The canonicalizer is the project's single implementation, `casmi.chemistry.connectivity.competition_connectivity_key`
(MolFromSmiles -> rdMolStandardize.TautomerEnumerator().Canonicalize -> MolToInchiKey()[:14]) -- the same function
that built every training key, so local scores and training labels can never disagree.

Ambiguities the official text does not settle are resolved CONSERVATIVELY (never inflating the local score):
  * positions are LITERAL: an invalid SMILES or a repeated connectivity still occupies its slot (the writer
    `dedupe_by_connectivity` removes such waste before submission, so both readings then coincide);
  * entries beyond 25 are ignored;
  * an unparseable truth is an error (it would make the molecule unscorable).
`RDKIT_REFERENCE_VERSION` documents the version the keys were built with; `check_rdkit_version` warns on a mismatch
(tautomer rules changed between RDKit releases).
"""
import warnings

import numpy as np
import pandas as pd

SEPARATOR = ";"
TOP_K = 25
SUBMISSION_COLUMNS = ["molecule_id", "smiles"]
RDKIT_REFERENCE_VERSION = "2026.03"     # release line used to build the training connectivity keys (casmi2026 env)


def check_rdkit_version(reference=RDKIT_REFERENCE_VERSION):
    import rdkit
    v = rdkit.__version__
    if not v.startswith(reference):
        warnings.warn(f"RDKit {v} != reference {reference}.x: tautomer canonicalization may differ from the keys the project "
                      f"was built with -- re-verify a sample of keys (tests/test_official_metric.py) before trusting scores")
    return v


class KeyCache:
    """Memoized SMILES -> competition key (canonicalization costs ~ms per molecule)."""

    def __init__(self):
        from casmi.chemistry.connectivity import competition_connectivity_key, get_tautomer_enumerator
        self._fn = competition_connectivity_key
        self._enum = get_tautomer_enumerator()
        self._cache = {}

    def __call__(self, smiles):
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        s = smiles.strip()
        if s not in self._cache:
            self._cache[s] = self._fn(s, enumerator=self._enum)
        return self._cache[s]

    def __len__(self):
        return len(self._cache)


def split_predictions(field):
    if not isinstance(field, str) or not field:
        return []
    return [p.strip() for p in field.split(SEPARATOR)]


def reciprocal_rank(pred_keys, truth_key, k=TOP_K):
    """1/position of the first literal match within the first k entries (None entries occupy their slot)."""
    for i, key in enumerate(pred_keys[:k]):
        if key is not None and key == truth_key:
            return 1.0 / (i + 1)
    return 0.0


def score_submission(submission, truth, k=TOP_K, key_cache=None, truth_key_col=None):
    """`submission`: DataFrame[molecule_id, smiles(';'-joined)]; `truth`: DataFrame[molecule_id, smiles] (or with a
    precomputed `truth_key_col`). Every truth molecule is scored (a missing submission row scores 0).
    Returns `(mrr, per_molecule)`; per_molecule has rank (NaN = miss), rr, n_pred, n_invalid, n_dup_keys."""
    kc = key_cache or KeyCache()
    sub = submission.set_index("molecule_id")["smiles"] if len(submission) else pd.Series(dtype=object)
    if sub.index.duplicated().any():
        raise ValueError("submission has duplicate molecule_id rows")
    rows = []
    for mid, tsmi in zip(truth["molecule_id"], truth[truth_key_col] if truth_key_col else truth["smiles"]):
        tkey = tsmi if truth_key_col else kc(tsmi)
        if tkey is None:
            raise ValueError(f"truth SMILES for {mid} is not parseable: {tsmi!r}")
        preds = split_predictions(sub.get(mid))
        keys = [kc(p) for p in preds[:k]]
        rr = reciprocal_rank(keys, tkey, k)
        valid = [x for x in keys if x is not None]
        rows.append({"molecule_id": mid, "truth_key": tkey, "rank": (1.0 / rr) if rr > 0 else np.nan, "rr": rr, "n_pred": len(preds),
                     "n_invalid": sum(x is None for x in keys), "n_dup_keys": len(valid) - len(set(valid))})
    per = pd.DataFrame(rows)
    return (float(per["rr"].mean()) if len(per) else float("nan")), per


def dedupe_by_connectivity(smiles_list, k=TOP_K, key_cache=None):
    """Keep the first SMILES of each competition key, drop unparseable ones, truncate to k. Returns (smiles, keys)."""
    kc = key_cache or KeyCache()
    seen, out, keys = set(), [], []
    for s in smiles_list:
        key = kc(s)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(s)
        keys.append(key)
        if len(out) >= k:
            break
    return out, keys


def validate_submission_frame(sub, expected_ids=None, k=TOP_K, key_cache=None, check_chemistry=True):
    """Problems (list of str; empty = valid): exact columns, one row per molecule, no nulls, no duplicate ids,
    1..k entries, every SMILES parseable, no repeated connectivity within a row, coverage of `expected_ids`."""
    problems = []
    if list(sub.columns) != SUBMISSION_COLUMNS:
        return [f"columns must be exactly {SUBMISSION_COLUMNS}, got {list(sub.columns)}"]
    if sub["molecule_id"].isna().any():
        problems.append("null molecule_id")
    if sub["molecule_id"].duplicated().any():
        problems.append(f"{int(sub['molecule_id'].duplicated().sum())} duplicate molecule_id rows")
    if expected_ids is not None:
        exp, got = set(expected_ids), set(sub["molecule_id"])
        if exp - got:
            problems.append(f"{len(exp - got)} expected molecule_id missing")
        if got - exp:
            problems.append(f"{len(got - exp)} unexpected molecule_id")
    kc = key_cache or KeyCache() if check_chemistry else None
    for mid, field in zip(sub["molecule_id"], sub["smiles"]):
        preds = split_predictions(field)
        if not preds or any(not p for p in preds):
            problems.append(f"{mid}: empty / null prediction entry")
            continue
        if len(preds) > k:
            problems.append(f"{mid}: {len(preds)} predictions > {k}")
        if check_chemistry:
            keys = [kc(p) for p in preds]
            if any(x is None for x in keys):
                problems.append(f"{mid}: {sum(x is None for x in keys)} unparseable SMILES")
            valid = [x for x in keys if x is not None]
            if len(valid) != len(set(valid)):
                problems.append(f"{mid}: {len(valid) - len(set(valid))} repeated connectivity (wasted slots)")
    return problems
