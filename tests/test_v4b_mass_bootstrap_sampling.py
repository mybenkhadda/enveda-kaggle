"""v4b mass likelihood (back-off, HOST refusal, shape), connectivity-level bootstrap, and
test-like nested training-set sampling."""
import numpy as np
import pandas as pd
import pytest

from casmi.ranking.mass_likelihood import MassLikelihoodModel, signed_truth_ppm
from casmi.ranking.training_sets import nested_scale_sets, test_like_weights, v2_filter, v3_filter
from casmi.validation.cluster_bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap, paired_comparison_row


# ---- mass likelihood -----------------------------------------------------------------------------

def _fit_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for inst, add, n, mu, sd in [("QTOF", "[M+H]+", 400, 2.0, 1.0), ("QTOF", "[M+Na]+", 50, -3.0, 2.0), ("Orbitrap", "[M+H]+", 300, 0.0, 0.5)]:
        for e in rng.normal(mu, sd, n):
            rows.append({"truth_ppm": e, "instrument_type": inst, "adduct": add, "source": "libA"})
    return pd.DataFrame(rows)


def test_backoff_hierarchy():
    m = MassLikelihoodModel(min_n=100).fit(_fit_frame())
    assert m.resolve("QTOF", "[M+H]+")[1] == "instrument_adduct"
    assert m.resolve("QTOF", "[M+Na]+")[1] == "instrument"       # only 50 samples -> backs off
    assert m.resolve("timsTOF", "[M+H]+")[1] == "adduct"
    assert m.resolve("timsTOF", "[M+K]+")[1] == "global"


def test_refuses_host_rows():
    df = _fit_frame().assign(source=lambda d: np.where(d.index % 50 == 0, "enveda-np-examples", "libA"))
    with pytest.raises(ValueError):
        MassLikelihoodModel().fit(df, forbidden_sources=("enveda-np-examples",))


def test_loglik_decreases_away_from_mode_and_is_finite():
    m = MassLikelihoodModel(min_n=100).fit(_fit_frame())
    cand = pd.DataFrame({"abs_mass_error_ppm": [0.0, 1.0, 5.0, 20.0, 49.0], "instrument_type": "Orbitrap", "adduct": "[M+H]+"})
    ll, lvl = m.loglik(cand)
    assert np.all(np.isfinite(ll)) and np.all(np.diff(ll) <= 1e-12)
    assert set(lvl) == {"instrument_adduct"}


def test_folded_density_invariant_to_sign_convention():
    df = _fit_frame()
    a = MassLikelihoodModel(min_n=100).fit(df)
    b = MassLikelihoodModel(min_n=100).fit(df.assign(truth_ppm=-df["truth_ppm"]))
    cand = pd.DataFrame({"abs_mass_error_ppm": [0.3, 2.0, 7.0], "instrument_type": "QTOF", "adduct": "[M+H]+"})
    assert np.allclose(a.loglik(cand)[0], b.loglik(cand)[0])


def test_signed_truth_ppm_matches_generator_convention():
    # generator: mass_error_ppm = 1e6 * (candidate_exact_mass - query_neutral_mass) / query_neutral_mass
    assert signed_truth_ppm([100.0], [100.001])[0] == pytest.approx(10.0)


# ---- connectivity bootstrap ---------------------------------------------------------------------

def test_identical_models_zero_delta_ci():
    v = np.random.default_rng(0).random(50)
    r = paired_cluster_bootstrap(v, v, clusters=np.repeat(np.arange(10), 5), n_boot=200)
    assert r["delta"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0


def test_cluster_bootstrap_keeps_clusters_intact():
    # two clusters with constant within-cluster values: every replicate mean is a mix of 0 and 1 by CLUSTER count
    values = np.array([0.0] * 5 + [1.0] * 5)
    clusters = np.array(["a"] * 5 + ["b"] * 5)
    r = cluster_bootstrap_mean(values, clusters, n_boot=300, seed=1)
    assert r["n_clusters"] == 2 and r["ci_low"] >= 0.0 and r["ci_high"] <= 1.0
    # a row-level bootstrap of 10 rows would almost never produce an all-0 or all-1 replicate;
    # with 2 intact clusters each replicate is all-0 / half / all-1 with prob 1/4, 1/2, 1/4
    assert r["ci_low"] == pytest.approx(0.0) and r["ci_high"] == pytest.approx(1.0)


def test_cluster_ci_wider_than_row_ci_for_correlated_rows():
    rng = np.random.default_rng(2)
    cl = np.repeat(np.arange(20), 10)
    vals = np.repeat(rng.random(20), 10)  # perfectly correlated within cluster
    wide = cluster_bootstrap_mean(vals, cl, n_boot=500)
    narrow = cluster_bootstrap_mean(vals, np.arange(len(vals)), n_boot=500)
    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


def test_bootstrap_deterministic_under_seed():
    v = np.random.default_rng(3).random(40)
    w = np.random.default_rng(4).random(40)
    cl = np.repeat(np.arange(8), 5)
    assert paired_cluster_bootstrap(v, w, cl, n_boot=100, seed=9) == paired_cluster_bootstrap(v, w, cl, n_boot=100, seed=9)


def test_paired_row_requires_same_queries():
    a = pd.DataFrame({"query_id": ["x", "y"], "rr": [1.0, 0.5]})
    b = pd.DataFrame({"query_id": ["x", "z"], "rr": [1.0, 0.5]})
    with pytest.raises(ValueError):
        paired_comparison_row("A", a, "B", b, {"x": "k1", "y": "k2", "z": "k3"})


# ---- training sets -------------------------------------------------------------------------------

def _pool_queries(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "query_id": [f"q{i}" for i in range(n)], "true_connectivity_key": [f"k{i % 150}" for i in range(n)],
        "source": rng.choice(["libA", "libB", "enveda-np-examples"], n, p=[0.6, 0.35, 0.05]),
        "instrument_type": rng.choice(["QTOF", "Orbitrap", "timsTOF"], n, p=[0.5, 0.4, 0.1]),
        "adduct": rng.choice(["[M+H]+", "[M+Na]+", "[M-H]-", "[M+K]+"], n),
        "ionization_mode": rng.choice(["positive", "negative"], n), "neutral_mass": rng.uniform(100, 900, n)})


def _test_queries(n=100, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"instrument_type": rng.choice(["timsTOF", "QTOF"], n, p=[0.9, 0.1]),
                         "adduct": rng.choice(["[M+H]+", "[M-H]-"], n), "ionization_mode": rng.choice(["positive", "negative"], n),
                         "neutral_mass": rng.uniform(200, 600, n)})


def test_nested_sets_include_base_and_are_supersets():
    pool = _pool_queries()
    base = pool["query_id"].iloc[:50].tolist()
    w, _ = test_like_weights(pool, _test_queries())
    sets = nested_scale_sets(pool, base, sizes=(100, 200), strategy="test_like", weights=w, seed=0)
    assert list(sets) == ["50", "100", "200"]
    ordered = [sets["50"], sets["100"], sets["200"]]
    assert ordered[0] == base
    assert set(ordered[0]) <= set(ordered[1]) <= set(ordered[2])
    assert [len(s) for s in ordered] == [50, 100, 200] and len(set(ordered[2])) == 200


def test_nested_sets_deterministic_and_named():
    pool = _pool_queries()
    base = pool["query_id"].iloc[:50].tolist()
    a = nested_scale_sets(pool, base, sizes=(100,), strategy="random", seed=5)
    b = nested_scale_sets(pool, base, sizes=(100,), strategy="random", seed=5)
    assert a == b and list(a) == ["50", "100"]
    from casmi.ranking.training_sets import size_name
    assert [size_name(n) for n in (1000, 3000, 10000, 250)] == ["1k", "3k", "10k", "250"]


def test_test_like_weights_favor_test_distribution_and_drop_unseen_adducts():
    pool = _pool_queries(2000)
    w, rep = test_like_weights(pool, _test_queries(500))
    assert (w[pool["adduct"].isin(["[M+Na]+", "[M+K]+"]).to_numpy()] == 0).all()
    ts = w[(pool["instrument_type"] == "timsTOF").to_numpy()].mean()
    orb = w[(pool["instrument_type"] == "Orbitrap").to_numpy()].mean()
    assert ts > orb and set(rep["dim"]) == {"instrument_type", "adduct", "ionization_mode", "mass_bin"}


def test_v2_v3_filters():
    pool = _pool_queries()
    v2 = v2_filter(pool, "enveda-np-examples")
    assert "enveda-np-examples" not in set(v2["source"])
    v3 = v3_filter(pool, "enveda-np-examples", {"k1", "k2"})
    assert not v3["true_connectivity_key"].isin({"k1", "k2"}).any() and set(v3["query_id"]) <= set(v2["query_id"])
