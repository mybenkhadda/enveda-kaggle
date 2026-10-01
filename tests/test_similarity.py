import math

from casmi.chemistry.similarity import (
    bulk_tanimoto,
    morgan_fingerprint,
    morgan_fingerprints,
    nearest_structure_neighbors,
    tanimoto_similarity,
)

ETHANOL = "CCO"
PROPANOL = "CCCO"
BENZENE = "c1ccccc1"


def test_identical_molecule_has_similarity_one():
    fp = morgan_fingerprint(ETHANOL)
    assert fp is not None
    assert tanimoto_similarity(fp, fp) == 1.0


def test_similar_molecules_score_higher_than_dissimilar():
    fp_ethanol = morgan_fingerprint(ETHANOL)
    fp_propanol = morgan_fingerprint(PROPANOL)
    fp_benzene = morgan_fingerprint(BENZENE)

    sim_close = tanimoto_similarity(fp_ethanol, fp_propanol)
    sim_far = tanimoto_similarity(fp_ethanol, fp_benzene)
    assert sim_close > sim_far


def test_unparseable_smiles_returns_none_and_nan():
    assert morgan_fingerprint("not a smiles") is None
    assert math.isnan(tanimoto_similarity(None, morgan_fingerprint(ETHANOL)))


def test_bulk_tanimoto_matches_pairwise():
    fps = morgan_fingerprints([ETHANOL, PROPANOL, BENZENE, "garbage"])
    sims = bulk_tanimoto(fps[0], fps)
    assert sims[0] == 1.0
    assert math.isnan(sims[3])
    assert sims[1] == tanimoto_similarity(fps[0], fps[1])


def test_nearest_structure_neighbors_excludes_self():
    keys = ["ethanol", "propanol", "benzene"]
    fps = morgan_fingerprints([ETHANOL, PROPANOL, BENZENE])
    out = nearest_structure_neighbors(keys, fps, top_k=2).set_index("key")

    assert out.loc["ethanol", "best_neighbor_key"] == "propanol"
    assert out.loc["ethanol", "nn_tanimoto"] < 1.0


def test_nearest_structure_neighbors_handles_unparseable():
    keys = ["ethanol", "bad"]
    fps = morgan_fingerprints([ETHANOL, "garbage"])
    out = nearest_structure_neighbors(keys, fps).set_index("key")

    assert math.isnan(out.loc["bad", "nn_tanimoto"])
    assert out.loc["bad", "best_neighbor_key"] is None
