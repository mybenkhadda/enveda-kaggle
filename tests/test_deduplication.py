import numpy as np

from casmi.spectra.deduplication import (
    classify_identity_tier, classify_reference_relationship, compute_peak_hash, find_near_duplicate_references,
    is_mirror_reference, is_near_duplicate_reference, is_tolerant_mirror_reference,
)


def _peaks(mzs, intensities, precursor_mz):
    return {"mzs": np.array(mzs, dtype=float), "intensities": np.array(intensities, dtype=float), "precursor_mz": precursor_mz}


def test_identical_spectrum_is_a_near_duplicate():
    query = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.1)
    reference = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.1005)
    assert is_near_duplicate_reference(query, reference) is True


def test_different_spectrum_same_precursor_is_not_a_near_duplicate():
    query = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.1)
    reference = _peaks([110.0, 160.0, 210.0], [0.9, 0.6, 0.3], precursor_mz=250.1)
    assert is_near_duplicate_reference(query, reference) is False


def test_precursor_mismatch_short_circuits_before_expensive_cosine():
    # identical peak arrays, but precursor differs by more than the threshold -- must NOT be
    # flagged, even though spectral similarity would be 1.0.
    query = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.10)
    reference = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.20)
    assert is_near_duplicate_reference(query, reference, precursor_diff_da=0.01) is False


def test_thresholds_are_configurable():
    query = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    reference = _peaks([100.0, 150.0, 201.0], [1.0, 0.5, 0.15], precursor_mz=250.1002)
    # loose threshold flags it, strict threshold doesn't -- same pair, different config
    assert is_near_duplicate_reference(query, reference, cosine_threshold=0.5) is True
    assert is_near_duplicate_reference(query, reference, cosine_threshold=0.9999) is False


def test_find_near_duplicate_references_returns_flagged_ids_only():
    query = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    candidates = {
        "ref_dup": _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.1003),
        "ref_distinct": _peaks([120.0, 180.0], [0.8, 0.4], precursor_mz=250.10),
        "ref_wrong_mass": _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=260.10),
    }
    flagged = find_near_duplicate_references(query, candidates)
    assert flagged == {"ref_dup"}


def test_find_near_duplicate_references_empty_input_never_raises():
    query = _peaks([100.0], [1.0], precursor_mz=250.10)
    assert find_near_duplicate_references(query, {}) == set()


def test_peak_hash_identical_for_exact_same_spectrum():
    a = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.1)
    b = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.1)
    assert compute_peak_hash(a["mzs"], a["intensities"], a["precursor_mz"]) == compute_peak_hash(b["mzs"], b["intensities"], b["precursor_mz"])


def test_peak_hash_invariant_to_peak_order():
    # canonicalization sorts by m/z -- ingestion order must not matter
    a = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.1)
    b = _peaks([200.0, 100.0, 150.0], [0.2, 1.0, 0.5], precursor_mz=250.1)
    assert compute_peak_hash(a["mzs"], a["intensities"], a["precursor_mz"]) == compute_peak_hash(b["mzs"], b["intensities"], b["precursor_mz"])


def test_peak_hash_invariant_to_intensity_scale():
    # max-normalization means an absolute intensity SCALE difference (same relative shape)
    # must hash identically -- e.g. two exports of the same raw spectrum at different units.
    a = _peaks([100.0, 150.0], [1000.0, 500.0], precursor_mz=250.1)
    b = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.1)
    assert compute_peak_hash(a["mzs"], a["intensities"], a["precursor_mz"]) == compute_peak_hash(b["mzs"], b["intensities"], b["precursor_mz"])


def test_peak_hash_differs_for_different_spectra():
    a = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.1)
    b = _peaks([100.0, 160.0], [1.0, 0.5], precursor_mz=250.1)
    assert compute_peak_hash(a["mzs"], a["intensities"], a["precursor_mz"]) != compute_peak_hash(b["mzs"], b["intensities"], b["precursor_mz"])


def test_is_mirror_reference_true_for_exact_match():
    a = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.1)
    b = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.1)
    assert is_mirror_reference(a, b) is True


def test_is_mirror_reference_false_for_near_but_not_exact_match():
    # spectrally near-identical (would pass is_near_duplicate_reference) but NOT byte-identical
    # -- must NOT be classified as a mirror.
    a = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    b = _peaks([100.0, 150.0, 200.1], [1.0, 0.5, 0.2], precursor_mz=250.1003)
    assert is_mirror_reference(a, b) is False
    assert is_near_duplicate_reference(a, b) is True


def test_classify_reference_relationship_three_way():
    query = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    mirror = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    near_dup = _peaks([100.0, 150.0, 200.1], [1.0, 0.5, 0.2], precursor_mz=250.1003)
    unrelated = _peaks([300.0, 350.0], [1.0, 0.5], precursor_mz=400.0)

    assert classify_reference_relationship(query, mirror) == "mirror"
    assert classify_reference_relationship(query, near_dup) == "near_duplicate_non_mirror"
    assert classify_reference_relationship(query, unrelated) == "other"


def test_is_mirror_reference_accepts_precomputed_hashes():
    a = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.1)
    b = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.1)
    c = _peaks([110.0, 160.0], [1.0, 0.5], precursor_mz=260.1)
    h_a = compute_peak_hash(a["mzs"], a["intensities"], a["precursor_mz"])
    h_b = compute_peak_hash(b["mzs"], b["intensities"], b["precursor_mz"])
    h_c = compute_peak_hash(c["mzs"], c["intensities"], c["precursor_mz"])
    assert is_mirror_reference(a, b, query_peak_hash=h_a, reference_peak_hash=h_b) is True
    assert is_mirror_reference(a, c, query_peak_hash=h_a, reference_peak_hash=h_c) is False
    # precomputed hash must be authoritative even if it doesn't match what peaks would produce
    # (caller's responsibility -- this proves the shortcut is actually taken, not silently ignored)
    assert is_mirror_reference(a, c, query_peak_hash=h_a, reference_peak_hash=h_a) is True


def test_classify_reference_relationship_is_symmetric_regardless_of_which_argument_is_query():
    # spec 6.2's symmetry requirement: the classification must not depend on which spectrum is
    # arbitrarily labeled "query" vs "reference" -- a true candidate and a decoy candidate must
    # be classified through the exact same code path.
    a = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    b = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    assert classify_reference_relationship(a, b) == classify_reference_relationship(b, a) == "mirror"


def test_tolerant_mirror_true_for_rounded_reserialization():
    # same underlying spectrum, reference m/z rounded to 3 decimals (rounding/serialization
    # noise) -- not byte-identical (peak_hash differs at its own 4-decimal rounding), but T2
    # should still recognize it as the same measurement.
    query = _peaks([100.12340, 150.56780, 200.99990], [1.0, 0.5, 0.25], precursor_mz=250.1000)
    reference = _peaks([100.123, 150.568, 201.000], [1.0, 0.5, 0.25], precursor_mz=250.1002)
    assert is_mirror_reference(query, reference) is False  # peak_hash differs -- not T1
    assert is_tolerant_mirror_reference(query, reference) is True


def test_tolerant_mirror_false_when_precursor_too_far():
    query = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.100)
    reference = _peaks([100.0, 150.0], [1.0, 0.5], precursor_mz=250.110)  # 0.01 Da > 0.005 threshold
    assert is_tolerant_mirror_reference(query, reference) is False


def test_tolerant_mirror_false_when_match_fraction_too_low():
    # query has an extra strong peak with no counterpart -- matched fraction(query) < 0.95
    query = _peaks([100.0, 150.0, 175.0, 300.0], [1.0, 0.9, 0.8, 0.7], precursor_mz=250.100)
    reference = _peaks([100.0, 150.0], [1.0, 0.9], precursor_mz=250.1005)
    assert is_tolerant_mirror_reference(query, reference) is False


def test_tolerant_mirror_false_when_intensities_uncorrelated():
    # peaks align in m/z within tolerance, but relative intensities are swapped/uncorrelated
    query = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.1], precursor_mz=250.100)
    reference = _peaks([100.0, 150.0, 200.0], [0.1, 0.5, 1.0], precursor_mz=250.1005)
    assert is_tolerant_mirror_reference(query, reference) is False


def test_tolerant_mirror_false_with_fewer_than_two_matched_peaks():
    query = _peaks([100.0], [1.0], precursor_mz=250.100)
    reference = _peaks([100.0], [1.0], precursor_mz=250.1005)
    assert is_tolerant_mirror_reference(query, reference) is False


def test_classify_identity_tier_four_way_priority():
    query = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    t1_mirror = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    t2_mirror = _peaks([100.001, 150.001, 200.001], [1.0, 0.5, 0.2], precursor_mz=250.1005)
    t3_near_dup = _peaks([100.0, 150.0, 200.1], [1.0, 0.5, 0.2], precursor_mz=250.1003)
    t4_other = _peaks([300.0, 350.0], [1.0, 0.5], precursor_mz=400.0)

    assert classify_identity_tier(query, t1_mirror) == "T1"
    assert classify_identity_tier(query, t2_mirror) == "T2"
    assert classify_identity_tier(query, t3_near_dup) == "T3"
    assert classify_identity_tier(query, t4_other) == "T4"


def test_classify_identity_tier_mirror_and_near_dup_flags_are_mutually_exclusive():
    query = _peaks([100.0, 150.0, 200.0], [1.0, 0.5, 0.2], precursor_mz=250.10)
    t3_near_dup = _peaks([100.0, 150.0, 200.1], [1.0, 0.5, 0.2], precursor_mz=250.1003)
    tier = classify_identity_tier(query, t3_near_dup)
    mirror = tier in ("T1", "T2")
    near_dup_non_mirror = tier == "T3"
    assert mirror is False
    assert near_dup_non_mirror is True


def test_match_peaks_greedy_is_deterministic_and_one_to_one():
    from casmi.spectra.deduplication import _match_peaks_greedy

    # two query peaks could both plausibly match the same reference peak -- must resolve to a
    # one-to-one assignment, not double-match, and the result must be identical across calls.
    q_mzs = np.array([100.000, 100.0015])
    q_int = np.array([1.0, 0.9])
    r_mzs = np.array([100.0005])
    r_int = np.array([1.0])
    mq1, mr1 = _match_peaks_greedy(q_mzs, q_int, r_mzs, r_int, ppm_tol=5.0, abs_tol_da=0.002)
    mq2, mr2 = _match_peaks_greedy(q_mzs, q_int, r_mzs, r_int, ppm_tol=5.0, abs_tol_da=0.002)
    assert len(mq1) == 1  # only one reference peak available -- one-to-one, not both matched
    assert list(mq1) == list(mq2) and list(mr1) == list(mr2)
