import numpy as np
import pandas as pd
import pytest

from casmi.ranking.molecule_fusion import FUSION_METHODS, evaluate_fusions, fuse, molecule_truth_ranks


def _features():
    # molecule M (two spectra q1, q2) with candidates 1 (truth), 2, 3; molecule N (one spectrum q3), truth 5 absent
    return pd.DataFrame({"query_id": ["q1", "q1", "q1", "q2", "q2", "q3", "q3"],
                         "candidate_id": [1, 2, 3, 1, 2, 4, 6],
                         "s": [0.9, 1.0, 0.1, 0.8, 0.2, 0.5, 0.4],
                         "is_true_candidate": [True, False, False, True, False, False, False]})


def _queries():
    return pd.DataFrame({"query_id": ["q1", "q2", "q3"], "fold": [0, 0, 1], "regime": ["C2", "C2", "C1"], "true_connectivity_key": ["M", "M", "N"]})


def test_mean_and_max_fusion():
    q2m = {"q1": "M", "q2": "M", "q3": "N"}
    f = fuse(_features(), "s", q2m, "mean").set_index(["molecule_id", "candidate_id"])
    assert f.loc[("M", 1), "fused_score"] == pytest.approx(0.85) and f.loc[("M", 2), "fused_score"] == pytest.approx(0.6)
    assert f.loc[("M", 3), "fused_score"] == pytest.approx(0.1) and f.loc[("M", 3), "n_spectra"] == 1   # absent -> no fake zero
    g = fuse(_features(), "s", q2m, "max").set_index(["molecule_id", "candidate_id"])
    assert g.loc[("M", 2), "fused_score"] == pytest.approx(1.0)


def test_lse_interpolates_between_mean_and_max():
    q2m = {"q1": "M", "q2": "M", "q3": "N"}
    lo = fuse(_features(), "s", q2m, "lse", temperature=1e-3).set_index(["molecule_id", "candidate_id"])["fused_score"]
    hi = fuse(_features(), "s", q2m, "lse", temperature=1e3).set_index(["molecule_id", "candidate_id"])["fused_score"]
    assert lo[("M", 2)] == pytest.approx(1.0, abs=1e-2) and hi[("M", 2)] == pytest.approx(0.6, abs=1e-2)


def test_molecule_ranks_and_missing_truth():
    out = evaluate_fusions(_features(), "s", _queries(), methods=("mean", "max"))
    m = out[out.fusion == "mean"].set_index("true_connectivity_key")
    assert m.loc["M", "truth_rank"] == 1 and m.loc["M", "n_spectra"] == 2
    assert np.isnan(m.loc["N", "truth_rank"]) and m.loc["N", "pool_size"] == 2
    mx = out[out.fusion == "max"].set_index("true_connectivity_key")
    assert mx.loc["M", "truth_rank"] == 2                       # max favours candidate 2 (1.0 in q1)


def test_every_method_runs_and_is_deterministic():
    a = evaluate_fusions(_features(), "s", _queries(), methods=FUSION_METHODS)
    b = evaluate_fusions(_features().sample(frac=1, random_state=0), "s", _queries(), methods=FUSION_METHODS)
    cols = ["fusion", "molecule_id", "truth_rank"]
    pd.testing.assert_frame_equal(a[cols].sort_values(cols[:2]).reset_index(drop=True), b[cols].sort_values(cols[:2]).reset_index(drop=True))
    with pytest.raises(ValueError):
        fuse(_features(), "s", {"q1": "M"}, "mean")             # unmapped query rows are an error, not silently dropped
