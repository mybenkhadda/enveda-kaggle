"""Unit tests for `casmi.spectra.reference_selection` -- includes the adversarial fixtures
B1/B2/B3/B4/B6 from the v4a reference-selection-correctness spec (B5, the serialization-mirror
case, lives in `test_deduplication.py` since it's purely an identity-tier test)."""
import pytest

from casmi.spectra.reference_selection import PROTOCOLS, compat_sort_key, is_eligible, walk_references


def _query(adduct="[M+H]+", ion_mode="positive", ce=30.0, ce_unit="eV", instrument="Orbitrap", source="lib_a"):
    return {"adduct": adduct, "ion_mode": ion_mode, "ce": ce, "ce_unit": ce_unit, "instrument": instrument, "source": source}


def _ref(spectrum_id, adduct="[M+H]+", ion_mode="positive", ce=30.0, ce_unit="eV", instrument="Orbitrap", source="lib_a"):
    return {"spectrum_id": spectrum_id, "adduct": adduct, "ion_mode": ion_mode, "ce": ce, "ce_unit": ce_unit,
            "instrument": instrument, "source": source}


def _classify_all_other(tier="T4"):
    """A classify_fn stub returning a fixed tier and placeholder similarity scores."""
    def fn(rid):
        return (tier, 0.5, 0.5, 0.5, 0.5)
    return fn


# --- compat_sort_key -------------------------------------------------------------------------

def test_compat_sort_key_prefers_same_adduct():
    q = _query(adduct="[M+H]+")
    same = _ref("r1", adduct="[M+H]+")
    diff = _ref("r2", adduct="[M-H]-")
    assert compat_sort_key(same, q) < compat_sort_key(diff, q)


def test_compat_sort_key_ce_incomparable_units_sort_after_comparable():
    q = _query(ce=30.0, ce_unit="eV")
    comparable = _ref("r1", ce=32.0, ce_unit="eV")     # comparable, diff=2
    incomparable = _ref("r2", ce=30.0, ce_unit="NCE")  # different unit -- never subtracted
    assert compat_sort_key(comparable, q) < compat_sort_key(incomparable, q)


def test_compat_sort_key_smaller_ce_diff_ranks_first_among_comparable():
    q = _query(ce=30.0, ce_unit="eV")
    close = _ref("r1", ce=31.0, ce_unit="eV")
    far = _ref("r2", ce=40.0, ce_unit="eV")
    assert compat_sort_key(close, q) < compat_sort_key(far, q)


def test_compat_sort_key_unknown_ce_is_incomparable_not_subtracted():
    q = _query(ce=30.0, ce_unit="eV")
    unknown = _ref("r1", ce=None, ce_unit=None)
    known = _ref("r2", ce=200.0, ce_unit="eV")  # huge diff but still comparable
    assert compat_sort_key(known, q) < compat_sort_key(unknown, q)


def test_compat_sort_key_stable_id_breaks_ties():
    q = _query()
    a = _ref("aaa")
    b = _ref("bbb")
    assert compat_sort_key(a, q) < compat_sort_key(b, q)


# --- is_eligible -------------------------------------------------------------------------------

def test_is_eligible_standard_allows_everything():
    q = _query(source="lib_a")
    r = _ref("r1", source="lib_a")
    assert is_eligible("standard", r, q, tier="T1") is True


def test_is_eligible_cross_library_excludes_same_source_only():
    q = _query(source="lib_a")
    same_source = _ref("r1", source="lib_a")
    cross_source = _ref("r2", source="lib_b")
    assert is_eligible("cross_library", same_source, q, tier="T4") is False
    assert is_eligible("cross_library", cross_source, q, tier="T1") is True  # mirror allowed under cross_library


def test_is_eligible_mirror_aware_excludes_mirrors_not_near_dup():
    q = _query(source="lib_a")
    r = _ref("r1", source="lib_b")
    assert is_eligible("mirror_aware", r, q, tier="T1") is False
    assert is_eligible("mirror_aware", r, q, tier="T2") is False
    assert is_eligible("mirror_aware", r, q, tier="T3") is True
    assert is_eligible("mirror_aware", r, q, tier="T4") is True


def test_is_eligible_near_dup_strict_excludes_mirrors_and_near_dup():
    q = _query(source="lib_a")
    r = _ref("r1", source="lib_b")
    assert is_eligible("near_dup_strict", r, q, tier="T1") is False
    assert is_eligible("near_dup_strict", r, q, tier="T2") is False
    assert is_eligible("near_dup_strict", r, q, tier="T3") is False
    assert is_eligible("near_dup_strict", r, q, tier="T4") is True


def test_is_eligible_unknown_protocol_raises():
    with pytest.raises(ValueError):
        is_eligible("bogus", _ref("r1"), _query(), tier="T4")


# --- walk_references: basic behavior ------------------------------------------------------------

def test_walk_stops_once_all_protocols_reach_five():
    query_meta = _query(source="lib_a")
    refs = [_ref(f"r{i}", source="lib_b") for i in range(50)]  # all cross-source, all T4
    lookup = {r["spectrum_id"]: r for r in refs}
    ranked = [r["spectrum_id"] for r in refs]
    result = walk_references(ranked, query_meta, lookup, _classify_all_other("T4"))
    assert result.n_references_walked == 5  # all 4 protocols satisfied by the same first 5 refs
    assert all(result.reached5[p] for p in PROTOCOLS)
    assert result.exhausted is False


def test_walk_exhausts_with_fewer_than_five_available():
    # Fixture B3: only 3 eligible references total -- accept all 3, no padding, no error.
    query_meta = _query(source="lib_a")
    refs = [_ref(f"r{i}", source="lib_b") for i in range(3)]
    lookup = {r["spectrum_id"]: r for r in refs}
    ranked = [r["spectrum_id"] for r in refs]
    result = walk_references(ranked, query_meta, lookup, _classify_all_other("T4"))
    assert result.exhausted is True
    for p in PROTOCOLS:
        assert len(result.accepted[p]) == 3
        assert result.reached5[p] is False
        assert result.has_at_least(p, 3) is True
        assert result.has_at_least(p, 5) is False


def test_walk_empty_reference_list_never_raises():
    result = walk_references([], _query(), {}, _classify_all_other())
    assert result.exhausted is True
    assert all(len(result.accepted[p]) == 0 for p in PROTOCOLS)


# --- Fixture B1: no hidden rank-15 (or any) cap ------------------------------------------------

def test_fixture_b1_cross_library_selects_references_beyond_rank_fifteen():
    query_meta = _query(source="lib_a")
    same_source_refs = [_ref(f"same{i}", source="lib_a") for i in range(20)]     # ranks 1-20
    cross_source_refs = [_ref(f"cross{i}", source="lib_b") for i in range(5)]     # ranks 21-25
    all_refs = same_source_refs + cross_source_refs
    lookup = {r["spectrum_id"]: r for r in all_refs}
    ranked = [r["spectrum_id"] for r in all_refs]  # already in the "ranked" order for this fixture

    result = walk_references(ranked, query_meta, lookup, _classify_all_other("T4"))
    assert result.accepted["cross_library"] == [f"cross{i}" for i in range(5)]
    assert result.walk_depth["cross_library"] == 25  # had to walk to rank 25 to find them
    assert result.n_references_walked == 25  # proves it did NOT stop at any earlier fixed cap


# --- Fixture B2: mirrors before usable references ----------------------------------------------

def test_fixture_b2_mirror_aware_skips_leading_mirrors():
    query_meta = _query(source="lib_a")
    mirror_refs = [_ref(f"mirror{i}", source="lib_b") for i in range(10)]   # ranks 1-10, all T1
    clean_refs = [_ref(f"clean{i}", source="lib_b") for i in range(5)]       # ranks 11-15, T4
    all_refs = mirror_refs + clean_refs
    lookup = {r["spectrum_id"]: r for r in all_refs}
    ranked = [r["spectrum_id"] for r in all_refs]

    def classify(rid):
        tier = "T1" if rid.startswith("mirror") else "T4"
        return (tier, 0.99, 0.99, 0.99, 0.99)

    result = walk_references(ranked, query_meta, lookup, classify)
    assert result.accepted["mirror_aware"] == [f"clean{i}" for i in range(5)]
    assert result.accepted["cross_library"][:5] == [f"mirror{i}" for i in range(5)]  # cross_library allows mirrors


# --- Fixture B4: true/decoy symmetry ------------------------------------------------------------

def test_fixture_b4_true_and_decoy_candidate_rows_use_identical_walk_logic():
    # walk_references never takes an is_true flag at all -- there is no code path for it to
    # branch on. Calling it twice with identical inputs (as would happen for a true-candidate
    # row and a decoy row that happen to share the same reference pool) must be identical.
    query_meta = _query(source="lib_a")
    refs = [_ref(f"r{i}", source="lib_b") for i in range(8)]
    lookup = {r["spectrum_id"]: r for r in refs}
    ranked = [r["spectrum_id"] for r in refs]

    result_true = walk_references(ranked, query_meta, lookup, _classify_all_other("T4"))
    result_decoy = walk_references(ranked, query_meta, lookup, _classify_all_other("T4"))
    assert result_true.accepted == result_decoy.accepted
    assert result_true.walk_depth == result_decoy.walk_depth


# --- Fixture B6: near_dup_strict reaches well beyond rank 30 ------------------------------------

def test_fixture_b6_near_dup_strict_reaches_beyond_rank_thirty():
    query_meta = _query(source="lib_a")
    same_source = [_ref(f"same{i}", source="lib_a") for i in range(10)]        # ranks 1-10
    cross_mirrors = [_ref(f"mirror{i}", source="lib_b") for i in range(10)]     # ranks 11-20, T1
    cross_t3 = [_ref(f"t3_{i}", source="lib_b") for i in range(10)]             # ranks 21-30, T3
    cross_clean = [_ref(f"clean{i}", source="lib_b") for i in range(5)]         # ranks 31-35, T4
    all_refs = same_source + cross_mirrors + cross_t3 + cross_clean
    lookup = {r["spectrum_id"]: r for r in all_refs}
    ranked = [r["spectrum_id"] for r in all_refs]

    def classify(rid):
        if rid.startswith("mirror"):
            return ("T1", 0.99, 0.99, 0.99, 0.99)
        if rid.startswith("t3_"):
            return ("T3", 0.96, 0.96, 0.96, 0.96)
        return ("T4", 0.3, 0.3, 0.3, 0.3)

    result = walk_references(ranked, query_meta, lookup, classify)
    assert result.accepted["near_dup_strict"] == [f"clean{i}" for i in range(5)]
    assert result.walk_depth["near_dup_strict"] == 35
    assert result.n_references_walked == 35  # proves no hidden 15/30 cap
