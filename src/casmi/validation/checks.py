"""Reusable, structured pipeline checks.

Every check returns a `CheckResult` (never just a bare bool or a print) so a pipeline stage
can collect them into `PreprocessingResult.checks` and a notebook can inspect *why* something
failed without re-running the check.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def __bool__(self):
        return self.passed


def check_required_columns(df_or_columns, required, name="required columns"):
    have = set(df_or_columns.columns) if hasattr(df_or_columns, "columns") else set(df_or_columns)
    missing = [c for c in required if c not in have]
    return CheckResult(
        name=name,
        passed=not missing,
        detail="all present" if not missing else f"missing: {missing}",
    )


def check_unique(series, name=None):
    name = name or f"unique: {getattr(series, 'name', 'series')}"
    n = len(series)
    n_unique = series.nunique(dropna=False)
    return CheckResult(
        name=name,
        passed=n_unique == n,
        detail=f"{n_unique}/{n} unique" if n_unique != n else f"{n}/{n} unique",
    )


def check_no_row_loss(before_n, after_n, name="no row loss"):
    return CheckResult(
        name=name,
        passed=after_n >= before_n,
        detail=f"{before_n} -> {after_n} rows" + ("" if after_n >= before_n else " (LOST ROWS)"),
    )


def check_structure_mapping(spectrum_df, structure_key_col, name="every spectrum maps to a structure"):
    n_total = len(spectrum_df)
    n_mapped = spectrum_df[structure_key_col].notna().sum()
    return CheckResult(
        name=name,
        passed=n_mapped == n_total,
        detail=f"{n_mapped}/{n_total} spectra have a {structure_key_col}",
    )


def check_test_ids(test_df, molecule_id_col="molecule_id", spectrum_id_col="spectrum_id",
                    name="test molecule_id/spectrum_id integrity"):
    n_spectrum_dupes = test_df[spectrum_id_col].duplicated().sum()
    n_molecules = test_df[molecule_id_col].nunique()
    passed = n_spectrum_dupes == 0 and test_df[spectrum_id_col].notna().all() and test_df[molecule_id_col].notna().all()
    return CheckResult(
        name=name,
        passed=passed,
        detail=f"{len(test_df)} spectra, {n_molecules} molecules, {n_spectrum_dupes} duplicate spectrum_id",
    )


def check_artifact_exists(path, name=None):
    path = Path(path)
    name = name or f"artifact exists: {path.name}"
    return CheckResult(name=name, passed=path.exists(), detail=str(path))


def summarize_checks(checks):
    """checks: iterable of CheckResult. Returns (all_passed, {name: passed})."""
    by_name = {c.name: c.passed for c in checks}
    return all(by_name.values()) if by_name else True, by_name
