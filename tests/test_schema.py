import pytest

from casmi.data.schema import TRAIN_REQUIRED_COLUMNS, validate_test_schema, validate_train_schema
from casmi.paths import get_project_paths
from casmi.validation.checks import check_required_columns

_paths = get_project_paths(create=False)


@pytest.mark.skipif(not _paths.train.exists(), reason="raw train.parquet not available in this environment")
def test_train_schema_validation():
    result = validate_train_schema(_paths.train)
    assert result.passed, result.detail


@pytest.mark.skipif(not _paths.test.exists(), reason="raw test.parquet not available in this environment")
def test_test_schema_validation():
    result = validate_test_schema(_paths.test)
    assert result.passed, result.detail


def test_missing_required_field_failure():
    columns_missing_adduct = [c for c in TRAIN_REQUIRED_COLUMNS if c != "adduct"]
    result = check_required_columns(columns_missing_adduct, TRAIN_REQUIRED_COLUMNS, name="train schema")
    assert not result.passed
    assert "adduct" in result.detail
