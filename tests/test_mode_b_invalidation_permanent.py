"""Permanent regression control (spec section 1.7): the ACTUAL project baseline registry, not a
synthetic fixture, must keep Mode B marked invalid forever -- a future notebook run that
"accidentally" re-validates it (e.g. by registering a flat, status-free entry again) fails this
test, not just a generic unit test of the registry mechanism (see test_baseline_registry.py for
that). Skips (doesn't fail) when the registry hasn't been generated yet -- e.g. a fresh clone
before any notebook has run -- since there's nothing to regress against yet.
"""
import pytest

from casmi.paths import get_project_paths
from casmi.validation.baseline_registry import InvalidBaselineError, load_baseline, load_promoted_baseline


def _registry_path():
    return get_project_paths(create=False).processed / "baseline_registry.json"


def test_mode_b_lambdamart_permanently_marked_invalid():
    registry_path = _registry_path()
    entry = load_baseline("notebook_09_lambdamart", registry_path)
    if entry is None:
        pytest.skip("data/processed/baseline_registry.json has no notebook_09_lambdamart entry yet -- run notebook 09 first")

    mode_b = entry.get("mode_b")
    if mode_b is None:
        pytest.skip("notebook_09_lambdamart entry predates the nested mode_a/mode_b schema -- re-run notebook 09")

    assert mode_b["status"] == "INVALID"
    assert mode_b["use_for_model_selection"] is False


def test_mode_a_lambdamart_still_promoted():
    registry_path = _registry_path()
    entry = load_baseline("notebook_09_lambdamart", registry_path)
    if entry is None or "mode_a" not in entry:
        pytest.skip("data/processed/baseline_registry.json has no notebook_09_lambdamart[mode_a] entry yet")

    mode_a = entry["mode_a"]
    assert mode_a["status"] == "VALID_INTERNAL_BASELINE"
    assert mode_a["use_for_model_selection"] is True


def test_load_promoted_baseline_refuses_the_real_mode_b_entry():
    registry_path = _registry_path()
    if load_baseline("notebook_09_lambdamart", registry_path) is None:
        pytest.skip("registry not generated yet")

    with pytest.raises(InvalidBaselineError):
        load_promoted_baseline("notebook_09_lambdamart", registry_path, regime="mode_b")


def test_mode_b_audit_verdict_recorded():
    registry_path = _registry_path()
    audit_entry = load_baseline("notebook_09_mode_b_audit", registry_path)
    if audit_entry is None:
        pytest.skip("09b_mode_b_shortcut_audit.ipynb hasn't been run yet")

    assert audit_entry["status"] == "INVALID_FOLD_MISSINGNESS_SHORTCUT"
    assert audit_entry["shortcut_confirmed"] is True
