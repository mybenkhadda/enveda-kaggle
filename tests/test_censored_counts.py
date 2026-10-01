from casmi.qcr.censored import censored_count_from_walk_result, censored_count_summary
from casmi.spectra.reference_selection import walk_references


def _query():
    return {"adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV", "instrument": "Orbitrap", "source": "lib_a"}


def _ref(spectrum_id):
    return {"spectrum_id": spectrum_id, "adduct": "[M+H]+", "ion_mode": "positive", "ce": 30.0, "ce_unit": "eV",
            "instrument": "Orbitrap", "source": "lib_a"}


def _walk(n_refs):
    """n_refs references, every one eligible under 'standard' (same-source is fine for
    'standard'; see is_eligible) with a fixed T4 tier -- so acceptance is governed purely by
    the reference count, letting each scenario below control it exactly via n_refs."""
    refs = [_ref(f"r{i:03d}") for i in range(n_refs)]
    lookup = {r["spectrum_id"]: r for r in refs}
    ranked = [r["spectrum_id"] for r in refs]

    def classify(rid):
        return ("T4", 0.5, 0.5, 0.5, 0.5)

    return walk_references(ranked, _query(), lookup, classify, protocols=("standard",), max_refs=5)


def test_exhausted_with_1_ref_is_exact():
    wr = _walk(1)
    assert wr.exhausted
    c = censored_count_from_walk_result(wr, "standard", max_refs=5)
    assert c.is_exact
    assert c.exact_value == 1
    assert c.lower_bound == 1


def test_exhausted_with_4_refs_is_exact():
    wr = _walk(4)
    assert wr.exhausted
    c = censored_count_from_walk_result(wr, "standard", max_refs=5)
    assert c.is_exact
    assert c.exact_value == 4


def test_exhausted_with_exactly_5_refs_is_exact_not_censored():
    wr = _walk(5)
    assert wr.exhausted  # coincidence of exactly 5 refs: quota reached exactly as the list ends
    c = censored_count_from_walk_result(wr, "standard", max_refs=5)
    assert c.is_exact
    assert c.exact_value == 5


def test_non_exhausted_after_5_accepted_is_censored():
    wr = _walk(10)
    assert not wr.exhausted  # stops after the 5th acceptance, refs 6-10 never walked
    c = censored_count_from_walk_result(wr, "standard", max_refs=5)
    assert not c.is_exact
    assert c.exact_value is None
    assert c.lower_bound == 5


def test_repr_distinguishes_exact_from_censored():
    exact = censored_count_from_walk_result(_walk(3), "standard", max_refs=5)
    censored = censored_count_from_walk_result(_walk(10), "standard", max_refs=5)
    assert repr(exact) == "3"
    assert repr(censored) == ">=5"


def test_summary_buckets_are_mutually_exclusive_and_exhaustive():
    counts = [
        censored_count_from_walk_result(_walk(1), "standard", max_refs=5),   # exact <=5
        censored_count_from_walk_result(_walk(4), "standard", max_refs=5),   # exact <=5
        censored_count_from_walk_result(_walk(10), "standard", max_refs=5),  # censored >=5
    ]
    summary = censored_count_summary(counts)
    assert summary["n"] == 3
    assert summary["share_exact_le5"] == 2 / 3
    assert summary["share_censored_ge5"] == 1 / 3
    assert summary["share_exact_gt5"] == 0.0


def test_summary_of_empty_is_nan_not_zero_division_error():
    summary = censored_count_summary([])
    assert summary["n"] == 0
    import math
    assert math.isnan(summary["share_exact_le5"])
