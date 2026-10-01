"""RRF formula / absent-contributes-zero / tie rule, the calibrated aggregators, dedup before top-25,
deterministic ties, temperature fit, and submission-validator negative cases."""
import numpy as np
import pandas as pd
import pytest

from casmi.ranking.calibration import calibration_report, fit_temperature
from casmi.ranking.molecule_eval import molecule_rr, multiplicity_resamples, select_aggregator_dev_only
from casmi_infer.aggregation import (FirstSpectrum, MaxProb, MaxScore, MeanProb, MostConfidentSpectrum, RRFAggregator, SumLogProb,
                                     make_aggregator, softmax_by_group)
from casmi_infer.submission import build_submission, molecule_top_k, pad_with_nearest
from casmi_infer.validation import validate_submission


def _spec(rows):
    """rows: (molecule, spectrum, cand, rank, score, ppm)"""
    return pd.DataFrame(rows, columns=["molecule_id", "spectrum_id", "conn_idx", "rank", "score", "abs_mass_error_ppm"])


def test_rrf_formula_and_absent_contributes_zero():
    df = _spec([("m", "s1", 10, 1, 5.0, 1.0), ("m", "s1", 20, 2, 4.0, 1.0),
                ("m", "s2", 20, 1, 3.0, 1.0), ("m", "s2", 30, 2, 2.0, 1.0)])
    r = RRFAggregator(k=60).rank(df).set_index("conn_idx")
    assert r.loc[20, "agg_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert r.loc[10, "agg_score"] == pytest.approx(1 / 61)         # absent from s2 -> contributes 0
    assert r.loc[30, "agg_score"] == pytest.approx(1 / 62)
    assert r.loc[20, "mol_rank"] == 1


def test_rrf_tie_rule_best_rank_then_key():
    df = _spec([("m", "s1", 7, 1, 1.0, 1.0), ("m", "s1", 3, 2, 0.5, 1.0),
                ("m", "s2", 3, 1, 1.0, 1.0), ("m", "s2", 7, 2, 0.5, 1.0),
                ("m", "s3", 9, 1, 1.0, 1.0), ("m", "s3", 8, 2, 0.5, 1.0), ("m", "s4", 8, 1, 1.0, 1.0), ("m", "s4", 9, 2, 0.5, 1.0)])
    r = RRFAggregator(k=60).rank(df)
    # every candidate has RRF 1/61 + 1/62 and best rank 1 -> connectivity ASC
    assert r["conn_idx"].tolist() == [3, 7, 8, 9]


def test_rrf_deterministic_under_row_shuffle():
    rng = np.random.default_rng(0)
    rows = [("m", f"s{s}", int(c), int(rk), 0.0, 1.0) for s in range(3) for rk, c in enumerate(rng.permutation(30)[:20], start=1)]
    df = _spec(rows)
    a = RRFAggregator().rank(df)
    b = RRFAggregator().rank(df.sample(frac=1.0, random_state=5))
    assert a[["conn_idx", "mol_rank"]].reset_index(drop=True).equals(b[["conn_idx", "mol_rank"]].reset_index(drop=True))


def test_calibrated_aggregators():
    df = _spec([("m", "s1", 1, 1, 2.0, 1.0), ("m", "s1", 2, 2, 1.0, 3.0), ("m", "s2", 2, 1, 5.0, 3.0), ("m", "s2", 3, 2, 0.0, 2.0)])
    df["prob"] = softmax_by_group(df["score"], df["spectrum_id"], 1.0)
    p = df.set_index(["spectrum_id", "conn_idx"])["prob"]
    mean = MeanProb().rank(df).set_index("conn_idx")["agg_score"]
    assert mean.loc[1] == pytest.approx(p[("s1", 1)] / 2) and mean.loc[2] == pytest.approx((p[("s1", 2)] + p[("s2", 2)]) / 2)
    slp = SumLogProb(eps=1e-6).rank(df).set_index("conn_idx")["agg_score"]
    assert slp.loc[1] == pytest.approx(np.log(p[("s1", 1)]) + np.log(1e-6))       # absent spectrum -> log eps
    assert MaxScore().rank(df).iloc[0]["conn_idx"] == 2
    assert MaxProb().rank(df).set_index("conn_idx").loc[2, "agg_score"] == pytest.approx(max(p[("s1", 2)], p[("s2", 2)]))
    mc = MostConfidentSpectrum().rank(df)
    assert set(mc["conn_idx"]) == {2, 3}                                            # s2 is the more confident spectrum
    fs = FirstSpectrum().rank(df)
    assert fs["conn_idx"].tolist() == [1, 2]


def test_make_aggregator_from_config():
    assert isinstance(make_aggregator({"name": "RRF", "k": 60}), RRFAggregator)
    a = make_aggregator({"name": "A4_sum_log_prob", "eps": 1e-3})
    assert isinstance(a, SumLogProb) and a.eps == 1e-3


def test_dedupe_before_top25_truncation():
    mol = pd.DataFrame({"molecule_id": ["m"] * 30, "conn_idx": list(range(30)), "mol_rank": list(range(1, 31))})
    mol = pd.concat([mol.iloc[:3], mol.iloc[[1]], mol.iloc[3:]], ignore_index=True)      # duplicated connectivity 1
    smiles = {i: f"C{i}" for i in range(30)}
    top = molecule_top_k(mol, smiles, top_k=25)["m"]
    assert len(top) == 25 and len(set(top)) == 25 and top[:3] == ["C0", "C1", "C2"]
    assert pad_with_nearest([1, 2], [2, 3, 4, 5], top_k=4) == [1, 2, 3, 4]


def test_temperature_fit_bounded_and_better_than_T1():
    rng = np.random.default_rng(1)
    rows = []
    for q in range(200):
        s = rng.normal(size=10) * 0.2
        s[0] += 0.5
        rows += [{"query_id": q, "score": v, "is_true_candidate": i == 0} for i, v in enumerate(s)]
    df = pd.DataFrame(rows)
    fit = fit_temperature(df)
    assert np.exp(-3) - 1e-9 <= fit["temperature"] <= np.exp(3) + 1e-9 and fit["nll"] <= fit["nll_at_T1"] + 1e-12
    rep, rel = calibration_report(df, fit["temperature"])
    assert 0 <= rep["ece_top1"] <= 1 and rel["n"].sum() == 200


def test_multiplicity_resamples_and_molecule_rr():
    ps = _spec([(m, f"{m}_s{i}", 1, 1, 0.0, 1.0) for m in ("A", "B") for i in range(5)])
    for _, sub in multiplicity_resamples(ps, test_n_spectra=[2], n_resamples=3, seed=0):
        assert (sub.groupby("molecule_id")["spectrum_id"].nunique() == 2).all()
    ranking = pd.DataFrame({"molecule_id": ["A", "A", "B"], "conn_idx": [9, 1, 9], "mol_rank": [1, 2, 1]})
    rr = molecule_rr(ranking, {"A": 1, "B": 5, "C": 1}, "conn_idx")
    assert rr.to_dict() == {"A": 0.5, "B": 0.0, "C": 0.0}


def test_aggregator_selection_prefers_simplest_within_ci():
    s = pd.DataFrame({"aggregator": ["A6_rrf", "A3_mean_prob", "A8_first_spectrum"], "dev_mrr_mean": [0.60, 0.61, 0.40],
                      "delta_ci_low": [-0.02, 0.0, -0.25], "delta_ci_high": [0.01, 0.0, -0.15]})
    sel, best, _ = select_aggregator_dev_only(s, {"A6_rrf": 2, "A3_mean_prob": 4, "A8_first_spectrum": 0})
    assert best == "A3_mean_prob" and sel == "A6_rrf"


# ---- submission validator -----------------------------------------------------------------------

SAMPLE = pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["CCO;CCO", "CCO"]})


def test_validator_accepts_good_submission():
    sub = build_submission(SAMPLE, {"m1": ["CCO", "CCN"], "m2": ["C"]})
    assert validate_submission(sub, SAMPLE) == []


@pytest.mark.parametrize("bad, fragment", [
    (pd.DataFrame({"molecule_id": ["m1"], "smiles": ["C"]}), "coverage"),
    (pd.DataFrame({"molecule_id": ["m2", "m1"], "smiles": ["C", "C"]}), "order"),
    (pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["", "C"]}), "empty"),
    (pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": [None, "C"]}), "empty"),
    (pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["C;C", "C"]}), "duplicate SMILES"),
    (pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["C;nan", "C"]}), "null SMILES"),
    (pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": [";".join(f"C{i}" for i in range(26)), "C"]}), "predictions"),
    (pd.DataFrame({"molecule_id": ["m1", "m2"], "pred": ["C", "C"]}), "columns"),
    (pd.DataFrame({"molecule_id": ["m1", "m1", "m2"], "smiles": ["C", "C", "C"]}), "duplicate molecule_id"),
])
def test_validator_negative_cases(bad, fragment):
    problems = validate_submission(bad, SAMPLE)
    assert problems and any(fragment in p for p in problems), problems
