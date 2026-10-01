"""v5 manifests: TL_EVAL exclusions, nesting TL_1K ⊂ TL_3K ⊂ TL_10K, connectivity disjointness from
TL_EVAL / MOL_DEV / HOST, HOST-source exclusion, deterministic regeneration under seed 42."""
import numpy as np
import pandas as pd
import pytest

from casmi.ranking.manifests import (
    manifest_checks, manifest_fingerprint, select_mol_dev, select_tl_eval, select_training_manifests, tl_eval_eligibility,
    training_pool, weighted_order,
)

HOST_SRC = "enveda-np-examples"


def _universe(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    u = pd.DataFrame({
        "query_id": [f"train_{i}" for i in range(n)],
        "connectivity_key": [f"K{rng.integers(0, 2500):05d}" for _ in range(n)],
        "source": rng.choice(["enveda-180", "mona", "gnps", HOST_SRC], n, p=[0.5, 0.25, 0.2, 0.05]),
        "instrument": rng.choice(["timsTOF", "Orbitrap", "QTOF"], n, p=[0.55, 0.3, 0.15]),
        "adduct": rng.choice(["[M+H]+", "[M-H]-", "[M+Na]+", "[2M+H]+"], n, p=[0.5, 0.2, 0.15, 0.15]),
        "ionization_mode": rng.choice(["positive", "negative"], n),
        "neutral_mass": rng.uniform(100, 1400, n), "precursor_mz": rng.uniform(100, 1400, n),
    })
    u["fold"] = u["connectivity_key"].map(lambda k: int(k[1:]) % 5)
    u["n_candidates_window"] = rng.integers(0, 400, n)
    return u.sort_values("query_id").reset_index(drop=True)


PROFILE = {"instruments": ["timsTOF"], "adducts": ["[M+H]+", "[M-H]-", "[M+Na]+"], "neutral_mass_min": 157.0, "neutral_mass_max": 1159.0}
TEST = pd.DataFrame({"instrument_type": ["timsTOF"] * 50, "adduct": ["[M+H]+"] * 40 + ["[M-H]-"] * 10,
                     "ionization_mode": ["positive"] * 40 + ["negative"] * 10, "neutral_mass": np.linspace(160, 1150, 50)})


@pytest.fixture(scope="module")
def built():
    u = _universe()
    host_keys = set(u.loc[u["source"] == HOST_SRC, "connectivity_key"].head(40))
    elig, _ = tl_eval_eligibility(u, PROFILE, HOST_SRC, host_keys)
    tl_eval = select_tl_eval(elig, TEST, n=200, seed=42)
    elig_md, _ = tl_eval_eligibility(u, PROFILE, HOST_SRC, host_keys, excluded_keys=set(tl_eval["connectivity_key"]))
    mol_dev = select_mol_dev(elig_md, TEST, n_connectivities=30, min_spectra=3, seed=42)
    pool = training_pool(u, HOST_SRC, host_keys, set(tl_eval["connectivity_key"]) | set(mol_dev["connectivity_key"]))
    train, counts, _ = select_training_manifests(pool, TEST, sizes=(("TL_1K", 100), ("TL_3K", 300), ("TL_10K", 100000)), rnd_size=100, seed=42)
    return u, host_keys, elig, pool, {"TL_EVAL": tl_eval, "MOL_DEV": mol_dev, **train}, counts


def test_tl_eval_excludes_host_source_and_structures(built):
    _, host_keys, _, _, m, _ = built
    assert HOST_SRC not in set(m["TL_EVAL"]["source"])
    assert not (set(m["TL_EVAL"]["connectivity_key"]) & host_keys)


def test_tl_eval_criteria(built):
    _, _, _, _, m, _ = built
    t = m["TL_EVAL"]
    assert set(t["instrument"]) <= {"timsTOF"} and set(t["adduct"]) <= set(PROFILE["adducts"])
    assert t["neutral_mass"].between(157.0, 1159.0).all() and (t["n_candidates_window"] >= 1).all()
    assert not t["connectivity_key"].duplicated().any()  # one spectrum per connectivity


def test_nesting_and_disjointness(built):
    _, host_keys, _, _, m, _ = built
    checks = manifest_checks(m, host_keys, HOST_SRC)
    assert checks["passed"].all(), checks[~checks["passed"]]
    assert set(m["TL_1K"]["query_id"]) <= set(m["TL_3K"]["query_id"]) <= set(m["TL_10K"]["query_id"])
    for n in ("TL_1K", "TL_3K", "TL_10K", "RND_1K"):
        assert not (set(m[n]["connectivity_key"]) & set(m["TL_EVAL"]["connectivity_key"]))
        assert not (set(m[n]["connectivity_key"]) & host_keys)


def test_undersized_set_takes_all_eligible_and_records_count(built):
    _, _, _, pool, m, counts = built
    assert counts["TL_10K"]["actual"] < counts["TL_10K"]["requested"]
    assert "ALL" in m["TL_10K"]["selection_reason"].iloc[0]


def test_tl_sets_only_test_adducts(built):
    _, _, _, _, m, _ = built
    assert "[2M+H]+" not in set(m["TL_10K"]["adduct"])      # zero test weight -> never drawn


def test_checks_detect_violations(built):
    _, host_keys, _, _, m, _ = built
    broken = dict(m)
    broken["TL_1K"] = pd.concat([m["TL_1K"], m["TL_EVAL"].head(1)], ignore_index=True)
    assert not manifest_checks(broken, host_keys, HOST_SRC)["passed"].all()


def test_deterministic_regeneration_seed_42(built):
    u, host_keys, elig, pool, m, _ = built
    again = select_tl_eval(elig, TEST, n=200, seed=42)
    assert manifest_fingerprint(again) == manifest_fingerprint(m["TL_EVAL"])
    t2, _, _ = select_training_manifests(pool, TEST, sizes=(("TL_1K", 100), ("TL_3K", 300), ("TL_10K", 100000)), rnd_size=100, seed=42)
    for n in ("TL_1K", "TL_3K", "TL_10K", "RND_1K"):
        assert manifest_fingerprint(t2[n]) == manifest_fingerprint(m[n])
    other = select_tl_eval(elig, TEST, n=200, seed=7)
    assert manifest_fingerprint(other) != manifest_fingerprint(m["TL_EVAL"])


def test_weighted_order_prefixes_and_zero_weights():
    ids = np.array([f"q{i:03d}" for i in range(100)])
    w = np.where(np.arange(100) % 4 == 0, 0.0, 1.0 + np.arange(100) % 3)
    o = weighted_order(ids, w, 42)
    assert len(o) == int((w > 0).sum()) and not np.isin(o, np.where(w == 0)[0]).any()
    assert np.array_equal(o, weighted_order(ids, w, 42))
