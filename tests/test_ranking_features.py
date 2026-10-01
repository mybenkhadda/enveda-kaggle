import math

from casmi.ranking.features import compute_pair_spectral_scores


def _peaks(mzs, ints, precursor):
    return {"mzs": mzs, "intensities": ints, "precursor_mz": precursor}


def test_no_reference_spectra_returns_nan_fields():
    out = compute_pair_spectral_scores(_peaks([100.0], [1.0], 200.0), [])
    assert out["n_reference_spectra"] == 0
    assert not out["has_reference_spectrum"]
    assert math.isnan(out["cosine_max"])
    assert math.isnan(out["modified_cosine_max"])


def test_identical_single_reference_scores_near_perfect():
    query = _peaks([100.0, 150.0, 190.0], [1.0, 2.0, 3.0], 200.0)
    reference = [_peaks([100.0, 150.0, 190.0], [1.0, 2.0, 3.0], 200.0)]
    out = compute_pair_spectral_scores(query, reference)
    assert out["has_reference_spectrum"]
    assert out["n_reference_spectra"] == 1
    assert math.isclose(out["cosine_max"], 1.0, rel_tol=1e-6)
    assert math.isclose(out["modified_cosine_max"], 1.0, rel_tol=1e-6)
    assert math.isclose(out["peak_overlap_frac_max"], 1.0)


def test_multiple_references_aggregated_max_picks_the_best():
    query = _peaks([100.0, 150.0, 190.0], [1.0, 2.0, 3.0], 200.0)
    good_ref = _peaks([100.0, 150.0, 190.0], [1.0, 2.0, 3.0], 200.0)
    bad_ref = _peaks([500.0, 600.0], [1.0, 1.0], 700.0)
    out = compute_pair_spectral_scores(query, [bad_ref, good_ref])
    assert out["n_reference_spectra"] == 2
    assert math.isclose(out["cosine_max"], 1.0, rel_tol=1e-6)


def test_completely_dissimilar_spectra_score_near_zero():
    query = _peaks([100.0], [1.0], 150.0)
    reference = [_peaks([500.0], [1.0], 550.0)]
    out = compute_pair_spectral_scores(query, reference)
    assert out["cosine_max"] == 0.0
    assert out["peak_overlap_frac_max"] == 0.0


def test_single_pair_scores_match_compute_pair_spectral_scores_for_one_reference():
    from casmi.ranking.features import compute_single_pair_scores

    query = _peaks([100.0, 150.0, 190.0], [1.0, 2.0, 3.0], 200.0)
    reference = _peaks([100.0, 150.0, 190.0], [1.0, 2.0, 3.0], 200.0)
    cosine, modified_cosine, overlap, nl_cosine = compute_single_pair_scores(query, reference)
    aggregated = compute_pair_spectral_scores(query, [reference])
    assert math.isclose(cosine, aggregated["cosine_max"], rel_tol=1e-9)
    assert math.isclose(modified_cosine, aggregated["modified_cosine_max"], rel_tol=1e-9)
    assert math.isclose(overlap, aggregated["peak_overlap_frac_max"], rel_tol=1e-9)
    assert math.isclose(nl_cosine, aggregated["neutral_loss_cosine_max"], rel_tol=1e-9)


def test_aggregate_pair_scores_from_values_matches_raw_aggregation():
    from casmi.ranking.features import aggregate_pair_scores_from_values, compute_single_pair_scores

    query = _peaks([100.0, 150.0, 190.0], [1.0, 2.0, 3.0], 200.0)
    ref_a = _peaks([100.0, 150.0, 190.0], [1.0, 2.0, 3.0], 200.0)
    ref_b = _peaks([500.0, 600.0], [1.0, 1.0], 700.0)

    scores_a = compute_single_pair_scores(query, ref_a)
    scores_b = compute_single_pair_scores(query, ref_b)
    from_values = aggregate_pair_scores_from_values(
        [scores_a[0], scores_b[0]], [scores_a[1], scores_b[1]], [scores_a[2], scores_b[2]], [scores_a[3], scores_b[3]],
    )
    from_raw = compute_pair_spectral_scores(query, [ref_a, ref_b])
    for key in ("cosine_max", "modified_cosine_max", "peak_overlap_frac_max", "neutral_loss_cosine_max"):
        assert math.isclose(from_values[key], from_raw[key], rel_tol=1e-9)


def test_aggregate_pair_scores_from_values_empty_matches_empty_contract():
    from casmi.ranking.features import aggregate_pair_scores_from_values

    out = aggregate_pair_scores_from_values([], [], [], [])
    assert out["n_reference_spectra"] == 0
    assert not out["has_reference_spectrum"]
    assert math.isnan(out["cosine_max"])
