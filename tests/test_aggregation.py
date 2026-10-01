import pandas as pd

from casmi.data.aggregation import build_molecule_mass_variants


def _toy_train_metadata():
    return pd.DataFrame({
        "connectivity_key": ["A", "A", "B", "B", "B", "C"],
        # A: unlabeled (300.0) and deuterated (308.05) -- two real variants.
        # B: three rows, all the same tautomer with tiny float noise -- one variant.
        # C: a single, unambiguous molecule.
        "exact_mass": [300.000000, 308.050000, 150.123456, 150.123457, 150.123456, 400.5],
        "normalized_smiles": ["CCO", "CC[2H]", "c1ccccc1", "c1ccccc1", "c1ccccc1", "CCN"],
        "molecular_formula": ["C2H6O", "C2H6O", "C6H6", "C6H6", "C6H6", "C2H7N"],
        "inchikey": ["ik_a1", "ik_a2", "ik_b1", "ik_b1", "ik_b1", "ik_c"],
        "inchikey14": ["ika1234567890", "ika1234567890", "ikb1234567890", "ikb1234567890", "ikb1234567890", "ikc1234567890"],
    })


def test_collapses_float_noise_into_one_variant():
    variants = build_molecule_mass_variants(_toy_train_metadata())
    b_variants = variants[variants["connectivity_key"] == "B"]
    assert len(b_variants) == 1
    assert not b_variants["is_isotope_variant"].iloc[0]
    assert b_variants["n_train_spectra"].iloc[0] == 3


def test_keeps_real_isotope_variants_separate():
    variants = build_molecule_mass_variants(_toy_train_metadata())
    a_variants = variants[variants["connectivity_key"] == "A"].sort_values("exact_mass")
    assert len(a_variants) == 2
    assert a_variants["is_isotope_variant"].all()
    assert list(a_variants["exact_mass"]) == [300.0, 308.05]


def test_single_variant_key_flagged_false():
    variants = build_molecule_mass_variants(_toy_train_metadata())
    c_variant = variants[variants["connectivity_key"] == "C"]
    assert len(c_variant) == 1
    assert not c_variant["is_isotope_variant"].iloc[0]


def test_mass_variant_id_unique_and_stable_ordering():
    variants = build_molecule_mass_variants(_toy_train_metadata())
    assert variants["mass_variant_id"].is_unique
    a_variants = variants[variants["connectivity_key"] == "A"].sort_values("exact_mass")
    assert list(a_variants["mass_variant_id"]) == ["A__v0", "A__v1"]  # v0 = lower mass, ascending


def test_total_row_count_equals_unique_connectivity_mass_pairs():
    variants = build_molecule_mass_variants(_toy_train_metadata())
    assert len(variants) == 4  # A has 2, B has 1, C has 1
