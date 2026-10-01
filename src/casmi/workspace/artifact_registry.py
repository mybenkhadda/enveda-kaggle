"""ArtifactRegistry: the ONE place where semantic artifact locations are derived from the physical `V2Paths` roots.

    ARTIFACTS = ArtifactRegistry(CFG, P)
    ARTIFACTS.validation_regimes      -> <drive>/validation/regimes/validation_regimes.parquet
    ARTIFACTS['universe_manifest']    -> <drive>/candidates/universe_manifest.json

Notebooks never build artifact paths by hand: they use `P.<root>` for roots and `ARTIFACTS.<name>` for everything
else, and declare upstream dependencies BY NAME (`bootstrap(..., requires=[...])`) so a missing artifact fails in
seconds with "produced by notebook 11" instead of 30 minutes later. Unknown names raise (with suggestions).

Canonical universe layout (the universe root IS `P.candidate_db_dir`; downstream loaders rely on it):

    candidates/candidate_keys.npy  bucket_offsets.json  universe_manifest.json  index/  formula_index/  buckets/  variants/
    candidates/standardized/ rejected/ filtered_out/ _done/       (stage A of notebook 12)
    candidates/manifests/                                         (reports: source summary, overlap, rejected rows)

`validation/`, `features/` and `exports/` are registry-owned subtrees of `drive_root` (they have no V2Paths root).
"""
import difflib
import json
from dataclasses import dataclass
from pathlib import Path

from casmi.workspace.config import INPUT_KEYS, V2Paths, input_path
from casmi.workspace.versions import ARTIFACT_REGISTRY_API_VERSION

FROZEN_BUNDLE_IDENTITY = {"bundle_version": "v2-A7", "CONFIG_HASH": "60174e39a2a3c6b4", "model_id": "V1_TL_1K_TESTSIM_STRICT"}


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    base: str             # a V2Paths field, or "inputs" for the five processed inputs
    rel: str              # relative path below the base ('' = the base itself; for inputs: the input key)
    kind: str             # "file" | "dir"
    producer: str
    description: str = ""


N10, N11, N12, N13, N14 = ("10_colab_v2_setup", "11_colab_hidden_like_validation", "12_colab_candidate_universe",
                           "13_colab_candidate_recall", "14_colab_analog_baseline")
COPY = "Windows copy (scripts/copy_colab_assets_to_drive.ps1)"
USER = "user upload (never downloaded by code)"

_SPECS = [ArtifactSpec(k, "inputs", k, "file", COPY, "canonical processed input (data/processed/)") for k in INPUT_KEYS] + [
    # ---- external sources / bundle --------------------------------------------------------------------------------
    ArtifactSpec("external_coconut_dir", "external_dir", "coconut", "dir", USER, "COCONUT export (coconut.csv)"),
    ArtifactSpec("external_pubchem_dir", "external_dir", "pubchem", "dir", USER, "curated PubChem subset"),
    ArtifactSpec("bundle_config", "bundle_dir", "config.json", "file", COPY, "frozen bundle identity labels"),
    ArtifactSpec("bundle_ref_meta", "bundle_dir", "ref_meta.parquet", "file", COPY, "reference-library spectrum metadata"),
    ArtifactSpec("bundle_connectivities", "bundle_dir", "connectivities.parquet", "file", COPY, "bundle connectivity table"),
    ArtifactSpec("copy_manifest", "drive_root", "exports/manifests/local_to_drive_copy_manifest.json", "file", COPY),
    ArtifactSpec("exports_dir", "drive_root", "exports", "dir", "any notebook"),
    # ---- notebook 10 --------------------------------------------------------------------------------------------------
    ArtifactSpec("environment_reports_dir", "reports_dir", "environment", "dir", N10),
    # ---- notebook 11: regimes -------------------------------------------------------------------------------------
    ArtifactSpec("validation_dir", "drive_root", "validation", "dir", N11),
    ArtifactSpec("regimes_dir", "drive_root", "validation/regimes", "dir", N11),
    ArtifactSpec("validation_regimes", "drive_root", "validation/regimes/validation_regimes.parquet", "file", N11),
    ArtifactSpec("validation_regimes_meta", "drive_root", "validation/regimes/validation_regimes.metadata.json", "file", N11),
    ArtifactSpec("validation_reports_dir", "reports_dir", "validation", "dir", N11),
    ArtifactSpec("validation_summary", "reports_dir", "validation/validation_summary.json", "file", N11),
    ArtifactSpec("regime_checks", "reports_dir", "validation/regime_checks.parquet", "file", N11),
    # ---- notebook 12: candidate universe ----------------------------------------------------------------------------
    ArtifactSpec("universe_root", "candidate_db_dir", "", "dir", N12, "universe root (canonical layout, see module docstring)"),
    ArtifactSpec("candidate_staging_dir", "candidate_db_dir", "", "dir", N12, "stage A output root (standardized/ rejected/ ...)"),
    ArtifactSpec("universe_manifest", "candidate_db_dir", "universe_manifest.json", "file", N12),
    ArtifactSpec("candidate_keys", "candidate_db_dir", "candidate_keys.npy", "file", N12),
    ArtifactSpec("bucket_offsets", "candidate_db_dir", "bucket_offsets.json", "file", N12),
    ArtifactSpec("candidate_mass_index", "candidate_db_dir", "index", "dir", N12),
    ArtifactSpec("candidate_formula_index", "candidate_db_dir", "formula_index", "dir", N12),
    ArtifactSpec("universe_buckets", "candidate_db_dir", "buckets", "dir", N12),
    ArtifactSpec("universe_variants", "candidate_db_dir", "variants", "dir", N12),
    ArtifactSpec("candidate_manifests_dir", "candidate_db_dir", "manifests", "dir", N12),
    ArtifactSpec("universe_source_summary", "candidate_db_dir", "manifests/universe_source_summary.json", "file", N12),
    # ---- notebook 13: Gate A ------------------------------------------------------------------------------------------
    ArtifactSpec("candidate_recall_reports_dir", "reports_dir", "candidate_recall", "dir", N13),
    ArtifactSpec("gate_a_decision", "reports_dir", "candidate_recall/gate_a_decision.json", "file", N13),
    ArtifactSpec("c2_recall_per_query", "reports_dir", "candidate_recall/c2_candidate_recall_per_query.parquet", "file", N13),
    ArtifactSpec("mass_calibration_dir", "reports_dir", "candidate_recall/mass_calibration", "dir", N13),
    # ---- notebook 14: analog baseline ---------------------------------------------------------------------------------
    ArtifactSpec("features_dir", "drive_root", "features", "dir", N14),
    ArtifactSpec("analog_features_dir", "drive_root", "features/analog", "dir", N14, "identity-namespaced feature shards"),
    ArtifactSpec("analog_neighbors_cache_dir", "cache_dir", "analog_neighbors", "dir", N14, "identity-namespaced neighbor shards"),
    ArtifactSpec("analog_library_cache_dir", "cache_dir", "analog_library", "dir", N14, "binned reference library (per bundle identity)"),
    ArtifactSpec("fingerprint_cache_dir", "cache_dir", "fingerprints", "dir", N14, "buffered packed Morgan fingerprints"),
    ArtifactSpec("analog_reports_dir", "reports_dir", "analog", "dir", N14),
    ArtifactSpec("analog_ranker_checkpoints_dir", "checkpoints_dir", "ranker/analog", "dir", N14),
    ArtifactSpec("validation_predictions_dir", "predictions_dir", "validation", "dir", N14),
    # ---- experiment tracking (any notebook) ---------------------------------------------------------------------------
    ArtifactSpec("experiment_records_dir", "experiments_dir", "records", "dir", "any notebook", "append-only run records"),
    ArtifactSpec("experiment_index", "experiments_dir", "experiments.parquet", "file", "any notebook", "derived view (rebuildable)"),
    ArtifactSpec("experiment_leaderboard", "experiments_dir", "leaderboard.parquet", "file", "any notebook", "derived view (rebuildable)"),
]
ARTIFACTS_SPECS = {s.name: s for s in _SPECS}
assert len(ARTIFACTS_SPECS) == len(_SPECS), "duplicate artifact names"
_bad = [s.name for s in _SPECS if s.base != "inputs" and s.base not in V2Paths.field_names()]
assert not _bad, f"artifact specs reference unknown V2Paths fields: {_bad}"      # import-time self-check


class ArtifactRegistry:
    """Resolved artifact paths (attribute or item access). Unknown names raise with suggestions."""

    API_VERSION = ARTIFACT_REGISTRY_API_VERSION

    def __init__(self, cfg, paths):
        self._cfg, self._paths = cfg, paths

    @staticmethod
    def names():
        return sorted(ARTIFACTS_SPECS)

    def spec(self, name):
        if name not in ARTIFACTS_SPECS:
            hint = difflib.get_close_matches(name, ARTIFACTS_SPECS, n=3)
            raise KeyError(f"unknown artifact {name!r}; did you mean {hint}? (registry: casmi/workspace/artifact_registry.py, "
                           f"API {ARTIFACT_REGISTRY_API_VERSION})")
        return ARTIFACTS_SPECS[name]

    def path(self, name):
        s = self.spec(name)
        if s.base == "inputs":
            return input_path(self._cfg, self._paths, s.rel)
        base = Path(getattr(self._paths, s.base))
        return base / s.rel if s.rel else base

    def __getitem__(self, name):
        return self.path(name)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.path(name)
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def table(self, names=None):
        import pandas as pd
        rows = []
        for n in (names or self.names()):
            s, p = self.spec(n), self.path(n)
            rows.append({"name": n, "exists": Path(p).exists(), "kind": s.kind, "producer": s.producer, "path": str(p)})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# frozen bundle identity (labels only -- the no-hash policy: CONFIG_HASH is an identity LABEL, nothing is hashed)
# ---------------------------------------------------------------------------------------------------------------

def read_json(path):
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None


def bundle_identity(artifacts):
    """{'bundle_version', 'CONFIG_HASH', 'model_id'} read from bundle/config.json (None values if absent)."""
    cfg = read_json(artifacts.bundle_config) or {}
    return {k: cfg.get(k) for k in FROZEN_BUNDLE_IDENTITY}


def check_bundle_identity(artifacts, expected=FROZEN_BUNDLE_IDENTITY):
    """Problems (list of str). Compares identity LABELS from config.json with the frozen ones; never hashes files."""
    if not Path(artifacts.bundle_config).is_file():
        return [f"frozen bundle missing: {artifacts.bundle_config} -- copy it with scripts/copy_colab_assets_to_drive.ps1"]
    got = bundle_identity(artifacts)
    return [] if got == expected else [f"bundle identity {got} != frozen {expected} -- wrong bundle folder on Drive"]


# ---------------------------------------------------------------------------------------------------------------
# pipeline state (read-only): which stage is done / stale / blocked, and what to run next
# ---------------------------------------------------------------------------------------------------------------

def pipeline_status(cfg, paths):
    """One row per stage of the v2 protocol with status DONE / STALE / BLOCKED / TODO and the reason, plus the
    next notebook to run. Reads only small JSON manifests (never a parquet table)."""
    import pandas as pd
    from casmi.validation.c2_protocol import assess_universe, gate_a_protocol_status

    A = ArtifactRegistry(cfg, paths)
    rows = []

    def add(stage, notebook, status, reason):
        rows.append({"stage": stage, "notebook": notebook, "status": status, "reason": reason})

    missing_inputs = [k for k in INPUT_KEYS if not Path(A.path(k)).exists()]
    bundle_problems = check_bundle_identity(A)
    add("inputs + frozen bundle", "Windows copy", "DONE" if not missing_inputs and not bundle_problems else "TODO",
        "processed inputs + bundle present" if not missing_inputs and not bundle_problems else f"missing inputs {missing_inputs}; {bundle_problems}")
    reg_meta = read_json(A.validation_regimes_meta)
    add("regimes (preliminary)", N11, "DONE" if reg_meta else "TODO",
        f"c2_mode={reg_meta.get('c2_mode')}" if reg_meta else "validation_regimes.parquet absent")
    uni = assess_universe(A, cfg)
    if not uni.universe_present:
        add("candidate universe", N12, "TODO", uni.reason)
    else:
        add("candidate universe", N12, "DONE" if uni.c2_protocol_valid else "BLOCKED", uni.reason)
    if reg_meta and uni.universe_present:
        sig = reg_meta.get("signature") or {}
        same = sig.get("universe_identity") == uni.identity
        add("regimes (REBUILD after 12)", f"{N11} (REBUILD=True)", "DONE" if same else "STALE",
            "regime table matches the current universe" if same else "regimes were built for a different / older universe -- REBUILD=True")
    gate = read_json(A.gate_a_decision)
    if gate is None:
        add("Gate A", N13, "TODO", "gate_a_decision.json absent")
    else:
        valid = gate.get("protocol_valid") is True
        add("Gate A", N13, "DONE" if valid else "BLOCKED", f"{gate.get('decision')}: {str(gate.get('reason') or '')[:160]}")
    gate_ok = gate_a_protocol_status(gate)
    analog = Path(A.analog_reports_dir)
    has_analog = analog.exists() and any(analog.glob("*/analog_baseline_metrics.json"))
    add("analog baseline", N14, "DONE" if has_analog else ("TODO" if gate_ok else "BLOCKED"),
        "metrics present" if has_analog else ("ready" if gate_ok else "waits for a protocol-valid Gate A"))
    t = pd.DataFrame(rows)
    todo = t[t["status"] != "DONE"]
    nxt = todo.iloc[0]["notebook"] if len(todo) else "all v2 protocol stages done"
    return t, nxt
