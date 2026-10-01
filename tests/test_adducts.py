from casmi.chemistry.adducts import neutral_mass_from_precursor, parse_adduct, precursor_from_neutral_mass


def test_common_adduct_parsing():
    p = parse_adduct("[M+H]+")
    assert p["supported"]
    assert p["molecule_multiplier"] == 1
    assert p["charge"] == 1

    p = parse_adduct("[M-H]-")
    assert p["supported"]
    assert p["molecule_multiplier"] == 1
    assert p["charge"] == -1

    p = parse_adduct("[M+Na]+")
    assert p["supported"]
    assert p["charge"] == 1


def test_round_trip_neutral_mass():
    mz = 300.123456
    for adduct in ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+CH2O2-H]-"]:
        neutral = neutral_mass_from_precursor(mz, adduct)
        assert neutral is not None, f"{adduct} should parse"
        back = precursor_from_neutral_mass(neutral, adduct)
        assert abs(back - mz) < 1e-6, f"round-trip failed for {adduct}: {back} != {mz}"


def test_unsupported_adduct_behavior():
    p = parse_adduct("[Cat]2+")
    assert not p["supported"]
    assert p["molecule_multiplier"] is None
    assert p["charge"] is None
    assert p["mass_shift"] is None
    assert neutral_mass_from_precursor(100.0, "[Cat]2+") is None
    assert precursor_from_neutral_mass(100.0, "[Cat]2+") is None


def test_unparseable_and_missing_adduct():
    assert not parse_adduct(None)["supported"]
    assert not parse_adduct("")["supported"]
    assert not parse_adduct("garbage")["supported"]


def test_dimer_and_multimer_behavior():
    p_dimer = parse_adduct("[2M+H]+")
    assert p_dimer["supported"]
    assert p_dimer["molecule_multiplier"] == 2
    assert p_dimer["charge"] == 1

    mz = 401.0
    neutral = neutral_mass_from_precursor(mz, "[2M+H]+")
    single_neutral = neutral_mass_from_precursor(mz, "[M+H]+")
    # a dimer at the same observed m/z implies roughly half the neutral mass of a monomer
    assert neutral < single_neutral

    p_multicharge = parse_adduct("[M+2H]2+")
    assert p_multicharge["supported"]
    assert p_multicharge["charge"] == 2
