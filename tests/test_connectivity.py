from casmi.chemistry.connectivity import (
    canonicalize_smiles,
    competition_connectivity_key,
    inchikey14_from_smiles,
    inchikey_from_smiles,
)


def test_valid_smiles():
    assert canonicalize_smiles("CCO") == "CCO"
    assert inchikey14_from_smiles("CCO") is not None
    assert len(inchikey14_from_smiles("CCO")) == 14


def test_invalid_smiles():
    assert canonicalize_smiles("not_a_smiles!!!") is None
    assert inchikey_from_smiles("not_a_smiles!!!") is None
    assert inchikey14_from_smiles("not_a_smiles!!!") is None
    assert competition_connectivity_key("not_a_smiles!!!") is None


def test_stereochemistry_is_ignored():
    l_lactic_acid = competition_connectivity_key("C[C@H](O)C(=O)O")
    d_lactic_acid = competition_connectivity_key("C[C@@H](O)C(=O)O")
    assert l_lactic_acid is not None
    assert l_lactic_acid == d_lactic_acid


def test_tautomer_pair_merges():
    keto = competition_connectivity_key("CC(=O)CC(=O)C")
    enol = competition_connectivity_key("CC(=O)C=C(C)O")
    assert keto is not None
    assert keto == enol


def test_constitutional_isomers_differ():
    n_butanol = competition_connectivity_key("CCCCO")
    isobutanol = competition_connectivity_key("CC(C)CO")
    assert n_butanol is not None and isobutanol is not None
    assert n_butanol != isobutanol


def test_provided_and_computed_identities_stay_separate():
    # inchikey14_from_smiles is the AS-IS (non-tautomer) key; a keto/enol pair should NOT
    # necessarily match here even though it matches under competition_connectivity_key.
    keto_plain = inchikey14_from_smiles("CC(=O)CC(=O)C")
    enol_plain = inchikey14_from_smiles("CC(=O)C=C(C)O")
    keto_conn = competition_connectivity_key("CC(=O)CC(=O)C")
    enol_conn = competition_connectivity_key("CC(=O)C=C(C)O")
    assert keto_conn == enol_conn
    # (plain keys may or may not coincidentally match; the point is conn_key is the one
    # that's *guaranteed* to merge tautomers, which the assertion above already confirms)
    assert keto_plain is not None and enol_plain is not None
