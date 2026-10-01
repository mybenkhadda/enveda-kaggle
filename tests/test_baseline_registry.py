import pytest

from casmi.validation.baseline_registry import (
    InvalidBaselineError, list_baselines, load_baseline, load_promoted_baseline, register_baseline,
)


def test_register_and_load(tmp_path):
    path = tmp_path / "registry.json"
    register_baseline("mass_only", {"mrr_at_25": 0.29}, path, source_notebook="04")
    assert load_baseline("mass_only", path) == {"mrr_at_25": 0.29}


def test_identical_reregistration_is_noop(tmp_path):
    path = tmp_path / "registry.json"
    register_baseline("mass_only", {"mrr_at_25": 0.29}, path)
    register_baseline("mass_only", {"mrr_at_25": 0.29}, path)  # should not raise
    assert load_baseline("mass_only", path) == {"mrr_at_25": 0.29}


def test_changed_metrics_refused_without_overwrite(tmp_path):
    path = tmp_path / "registry.json"
    register_baseline("mass_only", {"mrr_at_25": 0.29}, path)
    with pytest.raises(ValueError):
        register_baseline("mass_only", {"mrr_at_25": 0.31}, path)
    assert load_baseline("mass_only", path) == {"mrr_at_25": 0.29}  # unchanged


def test_explicit_overwrite_allowed(tmp_path):
    path = tmp_path / "registry.json"
    register_baseline("mass_only", {"mrr_at_25": 0.29}, path)
    register_baseline("mass_only", {"mrr_at_25": 0.31}, path, overwrite=True)
    assert load_baseline("mass_only", path) == {"mrr_at_25": 0.31}


def test_load_missing_returns_none(tmp_path):
    path = tmp_path / "registry.json"
    assert load_baseline("nonexistent", path) is None


def test_list_baselines(tmp_path):
    path = tmp_path / "registry.json"
    register_baseline("a", {"x": 1}, path)
    register_baseline("b", {"x": 2}, path)
    assert set(list_baselines(path)) == {"a", "b"}


def test_load_promoted_baseline_flat_entry_defaults_to_promoted(tmp_path):
    # no status/use_for_model_selection field at all -- the older, flat convention -- must not
    # be treated as invalid just because the new convention exists.
    path = tmp_path / "registry.json"
    register_baseline("mass_only", {"mrr_at_25": 0.29}, path)
    assert load_promoted_baseline("mass_only", path) == {"mrr_at_25": 0.29}


def test_load_promoted_baseline_refuses_invalid_regime(tmp_path):
    path = tmp_path / "registry.json"
    register_baseline("notebook_09_lambdamart", {
        "mode_a": {"mrr_at_25": 0.7531, "status": "VALID_INTERNAL_BASELINE", "use_for_model_selection": True},
        "mode_b": {"mrr_at_25": 0.4999, "status": "INVALID", "use_for_model_selection": False},
    }, path)
    assert load_promoted_baseline("notebook_09_lambdamart", path, regime="mode_a")["mrr_at_25"] == 0.7531
    with pytest.raises(InvalidBaselineError):
        load_promoted_baseline("notebook_09_lambdamart", path, regime="mode_b")


def test_load_promoted_baseline_allow_invalid_bypasses_refusal(tmp_path):
    path = tmp_path / "registry.json"
    register_baseline("notebook_09_lambdamart", {
        "mode_b": {"mrr_at_25": 0.4999, "status": "INVALID", "use_for_model_selection": False},
    }, path)
    result = load_promoted_baseline("notebook_09_lambdamart", path, regime="mode_b", allow_invalid=True)
    assert result["mrr_at_25"] == 0.4999


def test_load_promoted_baseline_missing_name_returns_none(tmp_path):
    path = tmp_path / "registry.json"
    assert load_promoted_baseline("nonexistent", path) is None
