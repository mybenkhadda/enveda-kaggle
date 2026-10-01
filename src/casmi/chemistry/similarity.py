"""Structural (Morgan/ECFP) fingerprints and Tanimoto similarity.

Scopes how much of the train structure library is close analogues vs. isolated singletons --
load-bearing for later hard-negative mining (`casmi.data.pairs`), never assumed instead of
measured. Pure functions over RDKit fingerprints: no file I/O.

Performance note: `nearest_structure_neighbors` is all-pairs (O(n^2) Tanimoto evaluations). At
this project's real scale (~hundreds of thousands of unique train structures) a full run is
expensive -- restrict to a subsample, or to a mass-windowed subset (see
`casmi.candidates.mass_index`), rather than running it over the whole structure table at once.
"""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_GENERATOR_CACHE = {}


def _get_generator(radius=2, n_bits=2048):
    key = (radius, n_bits)
    if key not in _GENERATOR_CACHE:
        _GENERATOR_CACHE[key] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _GENERATOR_CACHE[key]


def morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """RDKit `ExplicitBitVect` Morgan (ECFP-like) fingerprint for one SMILES, or `None` if
    unparseable -- never raises."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return _get_generator(radius, n_bits).GetFingerprint(mol)


def morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """`morgan_fingerprint` over a list of SMILES; output is the same length/order as the
    input (`None` where a SMILES failed to parse) so callers can index back into their own key
    list positionally."""
    return [morgan_fingerprint(s, radius, n_bits) for s in smiles_list]


def tanimoto_similarity(fp_a, fp_b):
    """Tanimoto similarity between two fingerprints; `nan` if either is `None`
    (unparseable)."""
    if fp_a is None or fp_b is None:
        return float("nan")
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def bulk_tanimoto(fp, fingerprints):
    """Tanimoto of `fp` against every entry in `fingerprints` (a list, possibly containing
    `None`s). Returns a float array, same length as `fingerprints`, `nan` wherever either side
    is `None`."""
    out = np.full(len(fingerprints), np.nan)
    if fp is None:
        return out
    valid_idx = [i for i, f in enumerate(fingerprints) if f is not None]
    if not valid_idx:
        return out
    sims = DataStructs.BulkTanimotoSimilarity(fp, [fingerprints[i] for i in valid_idx])
    for i, s in zip(valid_idx, sims):
        out[i] = s
    return out


def nearest_structure_neighbors(keys, fingerprints, top_k=5, similarity_threshold=0.4):
    """For every (key, fingerprint) pair, its similarity to every OTHER structure in the same
    set (self excluded), summarized as the nearest neighbor(s).

    `keys`, `fingerprints`: parallel sequences, same length. A `None` fingerprint (unparseable
    SMILES) gets a row of all-`nan`/zero fields rather than being silently dropped, so the
    output always has one row per input key.

    Returns a DataFrame: [key, nn_tanimoto, top_k_mean_tanimoto, n_neighbors_above_threshold,
    best_neighbor_key]. `nn_tanimoto` == 1.0 for exact structural duplicates (distinct
    `connectivity_key`s that nonetheless share a fingerprint, e.g. stereoisomers collapsed by a
    non-isomeric fingerprint) -- worth checking for, not assuming away.
    """
    keys = list(keys)
    n = len(keys)
    valid_idx = [i for i, f in enumerate(fingerprints) if f is not None]
    records = []
    for i in range(n):
        if fingerprints[i] is None or len(valid_idx) <= 1:
            records.append({
                "key": keys[i], "nn_tanimoto": float("nan"), "top_k_mean_tanimoto": float("nan"),
                "n_neighbors_above_threshold": 0, "best_neighbor_key": None,
            })
            continue
        others = [j for j in valid_idx if j != i]
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[i], [fingerprints[j] for j in others]))
        order = np.argsort(sims)[::-1]
        top = order[:top_k]
        best_j = others[order[0]]
        records.append({
            "key": keys[i],
            "nn_tanimoto": float(sims[order[0]]),
            "top_k_mean_tanimoto": float(sims[top].mean()),
            "n_neighbors_above_threshold": int((sims >= similarity_threshold).sum()),
            "best_neighbor_key": keys[best_j],
        })
    return pd.DataFrame.from_records(records)
