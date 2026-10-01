"""Out-of-fold mass-error calibration utilities (nothing assumed: offsets are estimated)."""
import numpy as np
import pandas as pd
import pytest

from casmi.validation.mass_calibration import (apply_offset, calibration_effect, cross_fold_offsets, fit_offsets, lookup_offsets, robust_summary,
                                               signed_ppm)


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(5):
        for inst, adduct, off, n in (("timsTOF", "[M+H]+", 2.0, 200), ("timsTOF", "[M-H]-", 0.5, 120), ("Orbitrap", "[M+H]+", 0.0, 150),
                                     ("rare", "[M+K]+", 5.0, 3)):
            true = rng.uniform(200, 800, n)
            obs = true * (1 + (off + rng.normal(0, 0.3, n)) * 1e-6)
            rows.append(pd.DataFrame({"fold": fold, "instrument_type": inst, "adduct": adduct, "exact_mass_true": true, "neutral_mass": obs}))
    d = pd.concat(rows, ignore_index=True)
    d["signed_ppm"] = signed_ppm(d["neutral_mass"], d["exact_mass_true"])
    return d


def test_signed_ppm_and_apply_offset_roundtrip():
    true = np.array([300.0, 500.0])
    obs = true * (1 + 2e-6)
    assert signed_ppm(obs, true) == pytest.approx([2.0, 2.0])
    assert apply_offset(obs, 2.0) == pytest.approx(true)


def test_fit_recovers_group_offsets_with_hierarchical_fallback():
    d = _frame()
    t = fit_offsets(d, min_group_size=50, n_bootstrap=50)
    fine = t[t["level"] == "instrument_type|adduct"].set_index("key")
    assert fine.loc["timsTOF|[M+H]+", "median_ppm"] == pytest.approx(2.0, abs=0.1)
    assert fine.loc["timsTOF|[M+H]+", "ci_low"] <= 2.0 <= fine.loc["timsTOF|[M+H]+", "ci_high"] + 0.1
    assert not fine.loc["rare|[M+K]+", "usable"]                         # too few rows -> parent / global fallback
    off = lookup_offsets(d[d["instrument_type"] == "rare"].head(1), t)
    glob = t[t["level"] == "global"]["median_ppm"].iloc[0]
    assert off.iloc[0] == pytest.approx(glob)                            # 'rare' instrument level also unusable -> global


def test_gross_errors_are_excluded_from_the_fit():
    d = _frame()
    d.loc[d.index[:20], "signed_ppm"] = 5000.0                           # adduct-label errors
    t = fit_offsets(d, n_bootstrap=0)
    assert t.attrs["n_excluded_gross_errors"] == 20
    assert t[t["level"] == "global"]["median_ppm"].iloc[0] < 50


def test_cross_fold_offsets_never_use_the_held_out_fold():
    d = _frame()
    d.loc[d["fold"] == 0, "signed_ppm"] += 100.0                         # fold 0 alone shifted: its own offset must not leak
    off, table = cross_fold_offsets(d, "fold", ("instrument_type", "adduct"), min_group_size=50, n_bootstrap=0, max_abs_ppm=500.0)
    assert off[d["fold"] == 0].abs().max() < 10                          # fitted on folds 1-4 only
    assert set(table["held_out_fold"]) == {0, 1, 2, 3, 4}
    assert off.notna().all()


def test_robust_summary_handles_empty():
    s = robust_summary([])
    assert s["n"] == 0 and np.isnan(s["median_ppm"])


def test_calibration_effect_before_after():
    from casmi.candidates.mass_index import CandidateMassIndex
    masses = np.array([300.0, 300.0009, 300.003, 500.0])
    idx = CandidateMassIndex(masses, np.arange(4))
    before = np.array([300.0 * (1 + 3e-6)])                             # truth 0 at +3 ppm
    after = apply_offset(before, 3.0)
    t = calibration_effect(idx, before, after, np.array([0]), [2, 5])
    b = t.set_index(["variant", "ppm"])
    assert b.loc[("before", 2), "recall_all"] == 0.0 and b.loc[("after", 2), "recall_all"] == 1.0
