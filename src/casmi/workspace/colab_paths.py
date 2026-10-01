"""Canonical Colab layout constants: SOURCE from GitHub, PERSISTENCE on Drive, SCRATCH on the Colab SSD.

    /content/Enveda                         git clone of github.com/mybenkhadda/enveda-kaggle (source, notebooks,
                                            configs, tests) -- never stored on Drive
    /content/drive/MyDrive/EnvedaCASMI      persistent data + artifacts (DRIVE_TREE below)
    /content/enveda_work                    disposable local scratch (staged copies, temporaries)

There is NO path vocabulary here: physical roots are `casmi.workspace.config.V2Paths`, semantic locations are
`casmi.workspace.artifact_registry.ArtifactRegistry`. This module only holds the directory skeleton (which must
equal the `directories` list of configs/drive_asset_map.yaml -- a test checks it; the Windows scripts create the
same tree on the Drive-for-desktop side), the required-input list and the clone helper.
"""
import subprocess
from pathlib import Path

DRIVE_ROOT_DEFAULT = "/content/drive/MyDrive/EnvedaCASMI"
REPO_ROOT_DEFAULT = "/content/Enveda"
SCRATCH_DIR_DEFAULT = "/content/enveda_work"
GITHUB_REPO = "mybenkhadda/enveda-kaggle"
DEFAULT_BRANCH = "main"

DRIVE_TREE = (
    "data/raw/competition", "data/raw/spectra", "data/raw/metadata", "data/processed", "data/interim", "data/external/coconut",
    "data/external/pubchem", "bundle", "candidates/standardized", "candidates/formula_index", "candidates/manifests",
    "validation/splits", "validation/regimes", "validation/metrics", "features/direct", "features/analog", "features/formula",
    "features/fingerprint", "features/contrastive", "features/fragmentation", "embeddings/spectra", "embeddings/molecules",
    "checkpoints/fingerprint", "checkpoints/contrastive", "checkpoints/ranker", "predictions/validation", "predictions/leaderboard",
    "reports/environment", "reports/validation", "reports/candidate_recall", "reports/analog", "reports/fingerprint",
    "reports/contrastive", "reports/ranking", "reports/error_analysis", "cache/analog_neighbors", "cache/fingerprints",
    "cache/fragments", "cache/staging", "runs/records", "exports/kaggle", "exports/models", "exports/manifests",
)
REQUIRED_PROCESSED_FILES = ("train_spectrum_metadata.parquet", "connectivity_folds.parquet", "structure_table.parquet",
                            "dev_queries.parquet", "molecule_mass_variants.parquet")
REQUIRED_BUNDLE_FILES = ("config.json",)


def missing_required(paths):
    """Required Drive inputs that are absent for `paths` (a V2Paths): the processed inputs and the bundle identity
    file. Existence only -- the frozen bundle is identified by its config labels, never by hashing files."""
    miss = [str(Path(paths.processed_dir) / f) for f in REQUIRED_PROCESSED_FILES if not (Path(paths.processed_dir) / f).is_file()]
    miss += [str(Path(paths.bundle_dir) / f) for f in REQUIRED_BUNDLE_FILES if not (Path(paths.bundle_dir) / f).is_file()]
    return miss


def repo_url(token=None):
    """HTTPS clone URL; with a token (e.g. Colab secret GITHUB_TOKEN) for a private repository."""
    return f"https://{token}@github.com/{GITHUB_REPO}.git" if token else f"https://github.com/{GITHUB_REPO}.git"


def clone_or_update_repo(repo_root=REPO_ROOT_DEFAULT, branch=DEFAULT_BRANCH, token=None, run=subprocess.run):
    """git clone into `repo_root` if absent, else fetch + checkout + pull --ff-only. Never copies source from Drive.
    Returns the list of git commands run (the token is never echoed). Notebooks use the generated bootstrap cell
    (`casmi.workspace.notebook_cells.BOOTSTRAP_CELL`), which performs the same sequence before casmi is importable."""
    repo_root = Path(repo_root)
    if not (repo_root / ".git").exists():
        cmds = [["git", "clone", "--branch", branch, repo_url(token), str(repo_root)]]
    else:
        cmds = [["git", "-C", str(repo_root), "fetch", "origin"], ["git", "-C", str(repo_root), "checkout", branch],
                ["git", "-C", str(repo_root), "pull", "--ff-only", "origin", branch]]
    for c in cmds:
        run(c, check=True)
    return [[("<token-url>" if token and token in part else part) for part in c] for c in cmds]
