"""Project path management.

No user-specific or absolute Windows paths are hard-coded anywhere in this module. The repo
root is found by walking upward from the current working directory for a `pyproject.toml`
(or, failing that, a `data` folder) -- this is what makes `get_project_paths()` work
identically whether it's called from the repo root, from `notebooks/`, or from a test runner.

Environment variable overrides (highest priority, for headless/Kaggle runs):
    CASMI_PROJECT_ROOT   overrides the auto-detected repo root
    CASMI_INPUT_ROOT     overrides where raw competition files are read from
    CASMI_OUTPUT_ROOT    overrides where all outputs/artifacts are written

Raw-data layout: this project currently keeps `train.parquet` / `test.parquet` /
`sample_submission.csv` directly under `data/`, with a `data/raw/` layout as the intended
future home (see the project's refactor plan). `raw` below prefers `data/raw` when it
exists and transparently falls back to `data` itself otherwise, so callers never need to
know which layout is in use.
"""
import os
from dataclasses import dataclass
from pathlib import Path

_ROOT_MARKERS = ("pyproject.toml", "data")


def _find_repo_root(start=None):
    d = Path(start or Path.cwd()).resolve()
    for _ in range(6):
        if any((d / marker).exists() for marker in _ROOT_MARKERS):
            return d
        if d.parent == d:
            break
        d = d.parent
    return Path(start or Path.cwd()).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    raw: Path
    interim: Path
    processed: Path
    outputs: Path
    figures: Path
    reports: Path
    logs: Path

    train: Path
    test: Path
    sample_submission: Path

    def as_dict(self):
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def get_project_paths(project_root=None, create=True):
    """Build a `ProjectPaths` for this repo, honoring `CASMI_PROJECT_ROOT` /
    `CASMI_INPUT_ROOT` / `CASMI_OUTPUT_ROOT` env vars. Creates the interim/processed/output
    directories (never the raw data directory) unless `create=False`.
    """
    root = Path(project_root) if project_root is not None else Path(os.environ.get("CASMI_PROJECT_ROOT", "")) if os.environ.get("CASMI_PROJECT_ROOT") else _find_repo_root()
    root = root.resolve()

    data_root = root / "data"
    default_raw = data_root / "raw" if (data_root / "raw").exists() else data_root
    raw = Path(os.environ["CASMI_INPUT_ROOT"]).resolve() if os.environ.get("CASMI_INPUT_ROOT") else default_raw

    outputs = Path(os.environ["CASMI_OUTPUT_ROOT"]).resolve() if os.environ.get("CASMI_OUTPUT_ROOT") else root / "outputs"

    paths = ProjectPaths(
        root=root,
        raw=raw,
        interim=data_root / "interim",
        processed=data_root / "processed",
        outputs=outputs,
        figures=outputs / "figures",
        reports=outputs / "reports",
        logs=outputs / "logs",
        train=raw / "train.parquet",
        test=raw / "test.parquet",
        sample_submission=raw / "sample_submission.csv",
    )

    if create:
        for d in (paths.interim, paths.processed, paths.outputs, paths.figures, paths.reports, paths.logs):
            d.mkdir(parents=True, exist_ok=True)

    return paths
