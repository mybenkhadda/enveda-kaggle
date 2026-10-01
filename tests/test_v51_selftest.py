"""Kaggle self-test machinery on synthetic fixtures: comparison detects feature / score / rank /
candidate-set changes; numba failure -> numpy fallback; both fail -> raises; the submission writer is
blocked after a failed self-test; P4 local preflight never proves the Kaggle offline environment."""
import numpy as np
import pandas as pd
import pytest

import casmi_infer.selftest as st
from casmi_infer.features import SPECTRAL_FEATURES
from casmi_infer.submission import SubmissionBlocked, write_submission
from casmi_infer.validation import kaggle_offline_environment_proven, p4_status


def _expected(n=6):
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"spectrum_id": ["s1"] * 3 + ["s2"] * 3, "conn_idx": [1, 2, 3, 4, 5, 6][:n], "abs_mass_error_ppm": rng.uniform(0, 5, n),
                         **{c: rng.random(n) for c in SPECTRAL_FEATURES}, "score": rng.normal(size=n), "rank": [1, 2, 3, 1, 2, 3]})
    base.loc[5, SPECTRAL_FEATURES[0]] = np.nan          # NaN == NaN must count as equal
    key = ["spectrum_id", "conn_idx"]
    return base, base[key + ["abs_mass_error_ppm", *SPECTRAL_FEATURES]], base[key + ["score"]], base[key + ["rank"]]


def test_identical_outputs_pass():
    actual, ef, es, er = _expected()
    r = st.compare_to_expected(actual, ef, es, er)
    assert r == {"n_candidate_rows": 6, "candidate_set_mismatches": 0, "feature_mismatches": 0, "score_mismatches": 0, "rank_mismatches": 0}


def test_tiny_feature_noise_within_tolerance_passes():
    actual, ef, es, er = _expected()
    actual = actual.copy()
    actual["cosine_max"] += 1e-12
    assert st.compare_to_expected(actual, ef, es, er)["feature_mismatches"] == 0


@pytest.mark.parametrize("col, delta, check", [("cosine_max", 1e-6, "feature_mismatches"), ("abs_mass_error_ppm", 1e-6, "feature_mismatches"),
                                               ("score", 1e-6, "score_mismatches")])
def test_changes_detected(col, delta, check):
    actual, ef, es, er = _expected()
    actual = actual.copy()
    actual.loc[2, col] += delta
    assert st.compare_to_expected(actual, ef, es, er)[check] == 1


def test_rank_change_detected():
    actual, ef, es, er = _expected()
    actual = actual.copy()
    actual.loc[[0, 1], "rank"] = [2, 1]
    assert st.compare_to_expected(actual, ef, es, er)["rank_mismatches"] == 2


def test_missing_candidate_detected():
    actual, ef, es, er = _expected()
    r = st.compare_to_expected(actual.iloc[1:], ef, es, er)
    assert r["candidate_set_mismatches"] == 1 and r["feature_mismatches"] >= 1 and r["rank_mismatches"] >= 1


def _result(backend, passed):
    return {"backend": backend, "passed": passed, "n_fixture_queries": 2, "n_candidate_rows": 6, "feature_mismatches": 0 if passed else 3,
            "score_mismatches": 0, "rank_mismatches": 0, "candidate_set_mismatches": 0}


@pytest.fixture
def patched(monkeypatch):
    state = {"set": []}
    monkeypatch.setattr(st, "set_backend", lambda b: state["set"].append(b))
    return state


def test_numba_pass(monkeypatch, patched):
    monkeypatch.setattr(st, "active_backend", lambda: "numba")
    monkeypatch.setattr(st, "numba_available", lambda: True)
    rep = st.run_selftest_with_fallback(None, runner=lambda b, be: _result(be, True), log=lambda *a: None)
    assert rep["self_test_backend_final"] == "numba" and not rep["numba_fallback_used"] and patched["set"] == ["numba"]


def test_numba_fail_numpy_fallback(monkeypatch, patched):
    monkeypatch.setattr(st, "active_backend", lambda: "numba")
    monkeypatch.setattr(st, "numba_available", lambda: True)
    logs = []
    rep = st.run_selftest_with_fallback(None, runner=lambda b, be: _result(be, be == "numpy"), log=logs.append)
    assert rep["self_test_backend_final"] == "NUMPY_FALLBACK" and rep["numba_fallback_used"] and patched["set"] == ["numpy"]
    assert any("NUMBA_PARITY_FAILED" in l for l in logs) and any("USING_NUMPY_FALLBACK" in l for l in logs)


def test_both_fail_raises(monkeypatch, patched):
    monkeypatch.setattr(st, "active_backend", lambda: "numba")
    monkeypatch.setattr(st, "numba_available", lambda: True)
    with pytest.raises(st.SelfTestFailed):
        st.run_selftest_with_fallback(None, runner=lambda b, be: _result(be, False), log=lambda *a: None)


def test_numpy_only_fail_raises_without_retry(monkeypatch, patched):
    monkeypatch.setattr(st, "active_backend", lambda: "numpy")
    monkeypatch.setattr(st, "numba_available", lambda: False)
    calls = []
    with pytest.raises(st.SelfTestFailed):
        st.run_selftest_with_fallback(None, runner=lambda b, be: calls.append(be) or _result(be, False), log=lambda *a: None)
    assert calls == ["numpy"]


def test_submission_writer_blocked_after_failed_selftest(tmp_path):
    sub = pd.DataFrame({"molecule_id": ["m"], "smiles": ["C"]})
    with pytest.raises(SubmissionBlocked):
        write_submission(sub, tmp_path / "submission.csv", {"passed": False})
    with pytest.raises(SubmissionBlocked):
        write_submission(sub, tmp_path / "submission.csv", None)
    assert not (tmp_path / "submission.csv").exists()
    write_submission(sub, tmp_path / "submission.csv", {"passed": True})
    assert (tmp_path / "submission.csv").exists()


def test_p4_local_never_proves_kaggle_offline():
    assert p4_status(True, False) == {"P4_LOCAL_PREFLIGHT": "PASS", "P4_KAGGLE_OFFLINE": "NOT_PROVEN"}
    assert not kaggle_offline_environment_proven(env={}, input_root="/kaggle/input")
    assert not kaggle_offline_environment_proven(env={"KAGGLE_KERNEL_RUN_TYPE": "Batch", "CASMI_KAGGLE_INPUT": "C:/x"}, input_root="C:/x")
    assert kaggle_offline_environment_proven(env={"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, input_root="/kaggle/input")


def test_fixture_selection_deterministic_and_covers_hard_cases():
    rng = np.random.default_rng(1)
    ids = [f"train_{i}" for i in range(400)]
    meta = pd.DataFrame({"query_id": ids, "adduct": rng.choice(["[M+H]+", "[M-H]-", "[M+Na]+"], 400), "ce_known": rng.random(400) < 0.8})
    pool = pd.Series(rng.integers(5, 900, 400), index=ids)
    refs = pd.Series(rng.integers(0, 8, 400), index=ids)
    a = st.select_fixture_queries(meta, pool, refs, n=50, n_hard=10, seed=42)
    assert a == st.select_fixture_queries(meta, pool, refs, n=50, n_hard=10, seed=42) and len(a) == 50 and len(set(a)) == 50
    hard = set(pool[(pool >= pool.quantile(0.9)) & (refs >= 5)].index)
    assert len(set(a) & hard) >= min(10, len(hard))


def test_numba_kernels_match_numpy_when_available():
    pytest.importorskip("numba")
    from casmi.spectra.binning import bin_spectrum
    from casmi.spectra.similarity import binned_cosine_similarity, peak_overlap
    from casmi_infer.backend import numba_kernels
    k = numba_kernels()
    rng = np.random.default_rng(3)
    for _ in range(50):
        a, b = np.sort(rng.uniform(50, 500, rng.integers(1, 80))), np.sort(rng.uniform(50, 500, rng.integers(1, 80)))
        assert k["overlap_count"](a, b, 0.02) == peak_overlap(a, b, 0.02)["n_matched"]
        qa, va = bin_spectrum(a, rng.random(len(a)))
        qb, vb = bin_spectrum(b, rng.random(len(b)))
        assert abs(k["sorted_bin_dot"](qa, va, qb, vb) - binned_cosine_similarity(qa, va, qb, vb)) <= 1e-12


def test_submission_log_schema(tmp_path):
    from casmi.submissions import FIELDS, load_submission_log, log_submission
    log_submission(tmp_path / "log.jsonl", "sub-1", "cfg-V1", "V1_TL_1K", "A6_rrf", self_test_status="PASS", backend="numba",
                   numba_fallback_used=False, runtime_seconds=123.0, peak_ram_gb=4.2, public_score=None, notes="anchor")
    e = load_submission_log(tmp_path / "log.jsonl")[0]
    assert set(FIELDS) <= set(e) and e["public_score"] is None and e["backend"] == "numba"
