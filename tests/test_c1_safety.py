"""C1 strict reproducibility safety tests (spec section 61): duplicate-key, missing-key, and
feature-mismatch fixtures must each FAIL reproducibility, never overwrite the canonical
artifact, and write a mismatch diagnostic artifact instead.
"""
import pandas as pd
import pytest

from casmi.validation.strict_reproducibility import (
    ReproducibilityKeyError, persist_if_pass, strict_reproducibility_check,
)

KEYS = ["query_id", "candidate_connectivity_key"]
FEATURE_COLS = ["cosine_max", "peak_overlap_frac_max"]


def _table(rows):
    return pd.DataFrame(rows)


def _good_pair(query_id="q1", cand="c1", cosine=0.5, overlap=0.4):
    return {"query_id": query_id, "candidate_connectivity_key": cand, "cosine_max": cosine, "peak_overlap_frac_max": overlap}


def test_duplicate_key_in_rebuilt_raises_not_silently_reported():
    rebuilt = _table([_good_pair(), _good_pair()])  # same key twice
    existing = _table([_good_pair()])
    with pytest.raises(ReproducibilityKeyError):
        strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)


def test_duplicate_key_in_existing_raises_not_silently_reported():
    rebuilt = _table([_good_pair()])
    existing = _table([_good_pair(), _good_pair()])
    with pytest.raises(ReproducibilityKeyError):
        strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)


def test_missing_key_fails_not_silently_passes_on_intersection():
    rebuilt = _table([_good_pair(query_id="q1"), _good_pair(query_id="q2")])
    existing = _table([_good_pair(query_id="q1")])  # q2 missing from existing entirely
    result = strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)
    assert result.passed is False
    assert "not present in both" in result.reason or "not a full one-to-one cover" in result.reason


def test_feature_mismatch_fails():
    rebuilt = _table([_good_pair(cosine=0.5)])
    existing = _table([_good_pair(cosine=0.9)])  # same key, different cosine
    result = strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)
    assert result.passed is False
    assert any(m["feature"] == "cosine_max" for m in result.mismatches)


def test_exact_match_passes():
    rebuilt = _table([_good_pair()])
    existing = _table([_good_pair()])
    result = strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)
    assert result.passed is True
    assert result.mismatches == []


def test_persist_if_pass_writes_canonical_on_success(tmp_path):
    rebuilt = _table([_good_pair()])
    existing = _table([_good_pair()])
    result = strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)
    canonical = tmp_path / "canonical.parquet"
    mismatch = tmp_path / "canonical__MISMATCH.parquet"
    persist_if_pass(result, rebuilt, canonical, mismatch)
    assert canonical.exists()
    assert not mismatch.exists()


def test_persist_if_pass_never_overwrites_canonical_on_failure_and_writes_mismatch(tmp_path):
    canonical = tmp_path / "canonical.parquet"
    mismatch = tmp_path / "canonical__MISMATCH.parquet"
    original = _table([_good_pair(cosine=0.123)])
    original.to_parquet(canonical, index=False)  # pre-existing canonical artifact from a prior good run

    rebuilt = _table([_good_pair(cosine=0.999)])  # this run's rebuild disagrees
    existing = _table([_good_pair(cosine=0.5)])
    result = strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)
    assert result.passed is False

    with pytest.raises(AssertionError):
        persist_if_pass(result, rebuilt, canonical, mismatch)

    # canonical must be untouched (still the ORIGINAL pre-existing content, not overwritten)
    reread = pd.read_parquet(canonical)
    assert reread["cosine_max"].iloc[0] == pytest.approx(0.123)
    # mismatch diagnostic artifact must exist and contain the failed rebuild
    assert mismatch.exists()
    mismatch_df = pd.read_parquet(mismatch)
    assert mismatch_df["cosine_max"].iloc[0] == pytest.approx(0.999)


@pytest.mark.parametrize("scenario", ["duplicate_key", "missing_key", "feature_mismatch"])
def test_each_defect_scenario_never_overwrites_canonical(tmp_path, scenario):
    canonical = tmp_path / "canonical.parquet"
    mismatch = tmp_path / "canonical__MISMATCH.parquet"
    original = _table([_good_pair(cosine=0.111)])
    original.to_parquet(canonical, index=False)

    if scenario == "duplicate_key":
        rebuilt = _table([_good_pair(), _good_pair()])
        existing = _table([_good_pair()])
        with pytest.raises(ReproducibilityKeyError):
            strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)
        # a key-structure error must not even attempt to persist
        assert pd.read_parquet(canonical)["cosine_max"].iloc[0] == pytest.approx(0.111)
        return

    if scenario == "missing_key":
        rebuilt = _table([_good_pair(query_id="q1"), _good_pair(query_id="q2")])
        existing = _table([_good_pair(query_id="q1")])
    else:
        rebuilt = _table([_good_pair(cosine=0.5)])
        existing = _table([_good_pair(cosine=0.9)])

    result = strict_reproducibility_check(rebuilt, existing, KEYS, FEATURE_COLS)
    assert result.passed is False
    with pytest.raises(AssertionError):
        persist_if_pass(result, rebuilt, canonical, mismatch)
    assert pd.read_parquet(canonical)["cosine_max"].iloc[0] == pytest.approx(0.111)
    assert mismatch.exists()
