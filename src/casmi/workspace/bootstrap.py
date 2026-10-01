"""Notebook bootstrap (everything AFTER the repository is cloned / fast-forwarded and `src` is on sys.path).

Every v2 notebook starts with two cells:

  CELL 0  the generated bootstrap cell (`casmi.workspace.notebook_cells.BOOTSTRAP_CELL`, identical in every notebook):
          mount Drive -> ENVEDA_DRIVE_ROOT / ENVEDA_REPO_ROOT -> clone if missing, else fetch + checkout + pull --ff-only
          -> drop already-imported `casmi` modules -> `src` on sys.path -> export ENVEDA_BOOTSTRAP_CELL (its version).
          It must be inline: nothing from the repository can be imported before the repository exists.
  CELL 1  CTX = bootstrap(NOTEBOOK, notebook_api=..., uses_paths=..., uses_artifacts=..., requires=..., ...)
          CFG, P, ARTIFACTS = CTX.cfg, CTX.paths, CTX.artifacts

`bootstrap()`: version check (notebook API + bootstrap cell) -> missing packages -> config (schema-checked) -> V2Paths
-> P.ensure() -> ArtifactRegistry -> preflight (repo freshness, path fields, artifact names, upstream artifacts,
signatures, external sources, bundle identity) -> seeds -> environment summary. Any problem raises BEFORE work starts.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from casmi.workspace.config import ENV_REPO_ROOT, load_v2_config
from casmi.workspace.versions import VERSIONS

ENV_BOOTSTRAP_CELL = "ENVEDA_BOOTSTRAP_CELL"

# importable name -> pip requirement; only MISSING ones are installed
BASE_PACKAGES = {"yaml": "pyyaml", "pyarrow": "pyarrow", "psutil": "psutil", "tqdm": "tqdm"}
CHEM_PACKAGES = {"rdkit": "rdkit"}
RANK_PACKAGES = {"lightgbm": "lightgbm"}


@dataclass
class NotebookContext:
    notebook: str
    cfg: dict
    paths: object
    artifacts: object
    monitor: object
    seed: int
    report: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    @property
    def git_commit(self):
        return (self.report.get("repo") or {}).get("short_sha")

    def metadata(self, **extra):
        """Provenance block for metadata.json / reports / experiment records."""
        repo = self.report.get("repo") or {}
        return {"notebook": self.notebook, "git_commit": self.git_commit, "git_sha": repo.get("sha"), "dirty": bool(repo.get("dirty_files")),
                "versions": dict(VERSIONS), "config_path": self.cfg.get("_config_path"), "seed": self.seed, **extra}


def bootstrap(notebook, *, notebook_api, uses_paths=(), uses_artifacts=(), requires=(), optional=(), signatures=None, packages=None,
              require_external_sources=False, check_bundle=False, config_path=None, overrides=None, branch=None, fetch=False,
              allow_dirty=True, allow_behind=False, ensure_dirs=True, log=print):
    """See module docstring. Raises `casmi.workspace.preflight.PreflightError` (or a subclass) / `ConfigSchemaError`
    before any work. Never falls back to an incompatible API."""
    from casmi.workspace.artifact_registry import ArtifactRegistry
    from casmi.workspace.environment import ensure_packages, gpu_report, set_seeds, system_report
    from casmi.workspace.preflight import CompatibilityError, check_versions, preflight, problem
    from casmi.workspace.resources import ResourceMonitor

    cell_version = os.environ.get(ENV_BOOTSTRAP_CELL)
    if cell_version is None:
        raise CompatibilityError(problem("the generated bootstrap cell has not run in this kernel (ENVEDA_BOOTSTRAP_CELL unset)",
                                         "run the first code cell of the notebook (the CASMI BOOTSTRAP cell) before this one"))
    skew = check_versions(notebook_api, cell_version)            # cheapest check first: skew is the first message the user sees
    if skew:
        raise CompatibilityError("PREFLIGHT FAILED for " + notebook + ":\n  " + "\n  ".join(skew))
    installed = ensure_packages({**BASE_PACKAGES, **(packages or {})})
    if installed:
        log(f"[bootstrap] installed missing packages: {installed}")
    repo_root = Path(os.environ.get(ENV_REPO_ROOT, "."))
    cfg, P = load_v2_config(config_path or repo_root / "configs" / "casmi_v2_colab.yaml", overrides=overrides)
    if ensure_dirs:
        P.ensure()
    ARTIFACTS = ArtifactRegistry(cfg, P)
    report = preflight(cfg, P, ARTIFACTS, notebook, notebook_api, uses_paths=uses_paths, uses_artifacts=uses_artifacts, requires=requires,
                       optional=optional, signatures=signatures, bootstrap_cell_version=cell_version,
                       require_external_sources=require_external_sources, check_bundle=check_bundle,
                       branch=branch or os.environ.get("ENVEDA_BRANCH", "main"), fetch=fetch, allow_dirty=allow_dirty,
                       allow_behind=allow_behind, log=log)
    seed = set_seeds(cfg["runtime"]["seed"], deterministic=cfg["runtime"].get("deterministic", True))
    env = {"system": system_report(), "gpu": gpu_report() if "torch" in (packages or {}) or _torch_loaded() else None}
    sysr = env["system"]
    log(f"[bootstrap] python {sysr['python']} | cpus {sysr['cpu_count']} | RAM {sysr.get('ram_available_gb')}/{sysr.get('ram_total_gb')} GB free"
        f" | drive {P.drive_root} | scratch {P.scratch_dir} | seed {seed}")
    if env["gpu"]:
        g = env["gpu"]
        log(f"[bootstrap] gpu {g['gpu_name'] or 'none'} | VRAM {g['vram_gb']} GB | bf16 {g['bf16_supported']}")
    return NotebookContext(notebook=notebook, cfg=cfg, paths=P, artifacts=ARTIFACTS, monitor=ResourceMonitor(log=log), seed=seed,
                           report=report, environment=env)


def _torch_loaded():
    import sys
    return "torch" in sys.modules


def write_json(path, obj):
    """Atomic small-JSON writer for reports (tmp + replace: a Drive sync never sees half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path
