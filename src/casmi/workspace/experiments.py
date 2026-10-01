"""Append-only experiment tracker.

Storage (under `P.experiments_dir`; locations via ArtifactRegistry `experiment_records_dir` / `experiment_index` /
`experiment_leaderboard`):

    runs/records/<timestamp>-<uuid>-<experiment>.json   ONE immutable file per logged run (source of truth; never
                                                       rewritten -- writing to an existing name raises)
    runs/experiments.parquet                           DERIVED index of every record (rebuildable; safe to delete)
    runs/leaderboard.parquet                           DERIVED leaderboard (`write_leaderboard`; rebuildable)

`experiment_id` names WHAT was evaluated (e.g. 'analog_v2:lgbm_mass+analog'); `run_id` names one execution.
Re-running an experiment ADDS a run -- nothing is overwritten. `latest_runs` / `leaderboard` are views.

Every record carries: run_id, timestamp, notebook, git commit, config identity (config_hash), regime identity,
candidate-universe identity, metric summary, artifact paths, status, decision, notes. A metric that was not
computed is stored as NaN -- never a placeholder number. Unknown metric names raise.
"""
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TRACKER_VERSION = "casmi-v2-tracker-3"

# competition-facing metrics first (MRR@25 is the primary target), then diagnostics / cost
METRICS = ("MRR25", "C1_MRR25", "C2_MRR25", "C3_MRR25", "composite_MRR25",
           "Top1", "Top5", "Top10", "Top25", "C1_Top1", "C2_Top1", "C3_Top1",
           "mean_rank", "median_rank", "candidate_recall", "n_queries",
           "runtime", "ram_gb", "gpu_mem_gb", "disk_gb")
LEGACY_METRIC_ALIASES = {"Hit1": "Top1"}          # notebooks written before tracker v2
NUMERIC = METRICS                                  # backward-compatible name
TEXT_FIELDS = ("experiment_id", "run_id", "timestamp", "git_commit", "notebook", "config_hash", "config_path", "regime_identity",
               "universe_identity", "description", "channels", "fold", "model", "features", "candidate_generator", "ranker", "seed",
               "gpu", "status", "decision", "notes")
EXPERIMENT_COLUMNS = list(TEXT_FIELDS) + list(METRICS) + ["artifact_paths_json", "extra_json", "tracker_version"]
DECISIONS = (None, "KEEP", "REJECT", "NEEDS_MORE_EVIDENCE", "BASELINE", "DIAGNOSTIC")
STATUSES = (None, "COMPLETED", "PRELIMINARY", "PROTOCOL_INVALID", "FAILED")


def records_dir(experiments_dir):
    return Path(experiments_dir) / "records"


def experiments_path(experiments_dir):
    return Path(experiments_dir) / "experiments.parquet"


def leaderboard_path(experiments_dir):
    return Path(experiments_dir) / "leaderboard.parquet"


def config_hash(config):
    """Stable 16-hex identity of a JSON-serialisable config (sorted keys)."""
    import hashlib
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _as_text(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, default=str)
    if isinstance(v, (list, tuple, set)):
        return ",".join(map(str, v))
    return str(v)


def _as_float(v):
    if v is None:
        return math.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise TypeError(f"metric value {v!r} is not numeric")
    return f


def _safe_name(s, n=60):
    return re.sub(r"[^A-Za-z0-9_.+=-]+", "_", str(s))[:n]


def log_experiment(experiments_dir, experiment_id, description, channels, config_path=None, git_commit=None, notes="", *,
                   notebook=None, config=None, fold=None, model=None, features=None, candidate_generator=None, ranker=None,
                   seed=None, gpu=None, decision=None, status=None, regime_identity=None, universe_identity=None,
                   artifact_paths=None, extra=None, **metrics):
    """Append ONE immutable run record. Returns the record (dict). Unknown metric names raise KeyError.

    `fold=None` means "all folds pooled"; log per-fold rows with fold=<int> in addition when available.
    `decision` in DECISIONS (research decision framework); `status` in STATUSES (e.g. PROTOCOL_INVALID when the
    evaluation protocol was not valid -- such runs never enter the leaderboard). `regime_identity` /
    `universe_identity`: small dicts or strings identifying the regime table / candidate universe evaluated.
    `artifact_paths`: {name: path} of the artifacts this run wrote."""
    metrics = {LEGACY_METRIC_ALIASES.get(k, k): v for k, v in metrics.items()}
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise KeyError(f"unknown experiment metrics {sorted(unknown)}; allowed {METRICS} (+ legacy {sorted(LEGACY_METRIC_ALIASES)})")
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    now = datetime.now(timezone.utc)
    stamp, uid = now.strftime("%Y%m%dT%H%M%S%fZ"), uuid.uuid4().hex[:8]
    run_id = f"{experiment_id}@{stamp}-{uid}"
    rec = {"experiment_id": str(experiment_id), "run_id": run_id, "timestamp": now.isoformat(), "git_commit": git_commit,
           "notebook": notebook, "config_hash": config_hash(config) if config is not None else None,
           "config_path": str(config_path) if config_path else None, "regime_identity": _as_text(regime_identity),
           "universe_identity": _as_text(universe_identity), "description": description, "channels": _as_text(channels),
           "fold": _as_text(fold), "model": model, "features": _as_text(features), "candidate_generator": candidate_generator,
           "ranker": ranker, "seed": _as_text(seed), "gpu": gpu, "status": status, "decision": decision, "notes": notes,
           **{m: _as_float(metrics.get(m)) for m in METRICS},
           "artifact_paths_json": json.dumps({k: str(v) for k, v in (artifact_paths or {}).items()}, sort_keys=True) if artifact_paths else None,
           "extra_json": json.dumps(extra, default=str, sort_keys=True) if extra is not None else None,
           "tracker_version": TRACKER_VERSION}
    rd = records_dir(experiments_dir)
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / f"{stamp}-{uid}-{_safe_name(experiment_id)}.json"      # unique part FIRST: truncation can never collide
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment record {path}")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)
    rebuild_index(experiments_dir)
    return rec


def _legacy_rows(experiments_dir):
    """Rows of a tracker-1 `experiments.parquet` (written before records/ existed) -- kept, never dropped."""
    p = experiments_path(experiments_dir)
    if not p.exists():
        return []
    try:
        old = pd.read_parquet(p)
    except Exception:
        return []
    if "tracker_version" in old.columns:
        old = old[old["tracker_version"].isna()]
    rows = []
    for r in old.to_dict(orient="records"):
        rows.append({"experiment_id": r.get("experiment_id"), "run_id": r.get("run_id") or f"{r.get('experiment_id')}@legacy-{r.get('date')}",
                     "timestamp": r.get("date") or r.get("timestamp"), "git_commit": r.get("git_commit"), "description": r.get("description"),
                     "channels": r.get("channels"), "config_path": r.get("config_path"), "notes": r.get("notes"),
                     **{m: r.get(m) for m in METRICS if m in r}, "Top1": r.get("Top1", r.get("Hit1")), "tracker_version": None})
    return rows


def _record_rows(experiments_dir):
    rd = records_dir(experiments_dir)
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(rd.glob("*.json"))] if rd.is_dir() else []


def _frame(rows):
    df = pd.DataFrame(rows, columns=EXPERIMENT_COLUMNS)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce").astype(np.float64)
    for c in TEXT_FIELDS + ("artifact_paths_json", "extra_json", "tracker_version"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df.sort_values("timestamp", kind="mergesort", na_position="first").reset_index(drop=True)


def load_experiments(experiments_dir):
    """Every run ever logged (records + legacy rows), oldest first."""
    return _frame(_legacy_rows(experiments_dir) + _record_rows(experiments_dir))


def rebuild_index(experiments_dir):
    """Rebuild the derived `experiments.parquet` from every record (+ legacy tracker-1 rows). Returns the table."""
    df = load_experiments(experiments_dir)
    p = experiments_path(experiments_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    return df


def latest_runs(experiments_dir, pooled_only=True):
    """Most recent run per experiment_id (pooled rows, i.e. fold=None, unless `pooled_only=False`)."""
    df = load_experiments(experiments_dir)
    if pooled_only:
        df = df[df["fold"].isna()]
    return df.groupby("experiment_id", sort=False).tail(1).reset_index(drop=True)


LEADERBOARD_COLUMNS = ["experiment_id", "notebook", "model", "features", "candidate_generator", "ranker", "C1_MRR25", "C2_MRR25",
                       "C3_MRR25", "composite_MRR25", "MRR25", "Top1", "Top25", "candidate_recall", "runtime", "status", "decision",
                       "universe_identity", "git_commit", "timestamp"]


def leaderboard(experiments_dir, sort_by="composite_MRR25", columns=None, include_invalid=False):
    """Derived leaderboard: latest pooled run per experiment, sorted by `sort_by` (NaN last). Runs whose status is
    PROTOCOL_INVALID (e.g. C2 on a TRAIN-only universe) are excluded unless `include_invalid=True`."""
    df = latest_runs(experiments_dir)
    if not include_invalid:
        df = df[df["status"].astype(object).ne("PROTOCOL_INVALID")]
    return df.sort_values(sort_by, ascending=False, na_position="last")[columns or LEADERBOARD_COLUMNS].reset_index(drop=True)


def write_leaderboard(experiments_dir, **kw):
    """Regenerate `leaderboard.parquet` from the append-only records. Returns the table."""
    lb = leaderboard(experiments_dir, **kw)
    p = leaderboard_path(experiments_dir)
    tmp = p.with_suffix(".parquet.tmp")
    lb.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    return lb


def fold_stability(experiments_dir, experiment_id, baseline_id, metric="MRR25"):
    """Per-fold delta of `experiment_id` vs `baseline_id` (latest per-fold rows of each)."""
    df = load_experiments(experiments_dir)
    df = df[df["fold"].notna()]
    a = df[df["experiment_id"] == experiment_id].groupby("fold").tail(1).set_index("fold")[metric]
    b = df[df["experiment_id"] == baseline_id].groupby("fold").tail(1).set_index("fold")[metric]
    out = pd.DataFrame({"experiment": a, "baseline": b}).dropna()
    out["delta"] = out["experiment"] - out["baseline"]
    return out.reset_index()
