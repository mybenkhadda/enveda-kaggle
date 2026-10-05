"""Lightweight experiment tracker: one parquet table, one row per USER-executed experiment.

Values are whatever the calling notebook computed; a metric that was not computed is stored as
NaN -- never a placeholder number.

Protocol v2 BINDING: every new row is bound to the code and to the validation protocol it was measured under
(`log_bound_experiment`): timestamp, notebook, git_commit (empty refused), universe_build_id, regime_signature,
panel_manifest_identity, policy_identity, model_name, status; ranking rows also need feature_contract and
checkpoint_identity. Two rows are COMPARABLE only when universe_build_id, regime_signature, panel_manifest_identity and
policy_identity are all equal and present. Rows written before binding existed are kept and labelled
`legacy_unbound=True` (never deleted, never comparable)."""
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPERIMENT_COLUMNS = ["experiment_id", "date", "git_commit", "description", "channels", "config_path", "C1_MRR25", "C2_MRR25",
                      "C3_MRR25", "composite_MRR25", "Hit1", "candidate_recall", "runtime", "notes"]
NUMERIC = ("C1_MRR25", "C2_MRR25", "C3_MRR25", "composite_MRR25", "Hit1", "candidate_recall", "runtime")
BINDING_COLUMNS = ["timestamp", "notebook", "universe_build_id", "regime_signature", "panel_manifest_identity", "policy_identity",
                   "model_name", "status", "kind", "feature_contract", "checkpoint_identity", "legacy_unbound"]
ALL_COLUMNS = EXPERIMENT_COLUMNS + BINDING_COLUMNS
REQUIRED_BINDING = ("notebook", "git_commit", "universe_build_id", "regime_signature", "panel_manifest_identity", "policy_identity",
                    "model_name", "status")
RANKING_REQUIRED = ("feature_contract", "checkpoint_identity")
COMPARABILITY_KEYS = ("universe_build_id", "regime_signature", "panel_manifest_identity", "policy_identity")
KINDS = ("coverage", "ranking", "diagnostic")


class ExperimentBindingError(ValueError):
    """A bound experiment row lacks a required identity field (e.g. an empty git_commit)."""


def experiments_path(experiments_dir):
    return Path(experiments_dir) / "experiments.parquet"


def _blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def load_experiments(experiments_dir):
    """The table with every v2 column; rows without the binding fields are labelled legacy_unbound=True."""
    p = experiments_path(experiments_dir)
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=ALL_COLUMNS)
    for c in ALL_COLUMNS:
        if c not in df.columns:
            df[c] = None
    unbound = df[["universe_build_id", "regime_signature"]].map(_blank).any(axis=1) if len(df) else pd.Series(dtype=bool)
    df["legacy_unbound"] = unbound.astype(bool) if len(df) else df["legacy_unbound"]
    return df[ALL_COLUMNS]


def _append(experiments_dir, row):
    df = load_experiments(experiments_dir)
    new = pd.DataFrame([{c: row.get(c) for c in ALL_COLUMNS}])
    df = pd.concat([df[df["experiment_id"] != row["experiment_id"]], new], ignore_index=True)[ALL_COLUMNS]
    p = experiments_path(experiments_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def _metrics_row(metrics):
    unknown = set(metrics) - set(NUMERIC)
    if unknown:
        raise KeyError(f"unknown experiment metrics {sorted(unknown)}; allowed {NUMERIC}")
    row = {c: math.nan for c in NUMERIC}
    row.update({k: (float(v) if v is not None else math.nan) for k, v in metrics.items()})
    return row


def log_experiment(experiments_dir, experiment_id, description, channels, config_path=None, git_commit=None, notes="", **metrics):
    """LEGACY (unbound) logger, kept for old notebooks: the row is stored with legacy_unbound=True and is never comparable.
    New notebooks use `log_bound_experiment`. Unknown metric names raise."""
    row = _metrics_row(metrics)
    row.update(experiment_id=str(experiment_id), date=datetime.now(timezone.utc).isoformat(), git_commit=git_commit,
               description=description, channels=",".join(channels) if isinstance(channels, (list, tuple)) else str(channels),
               config_path=str(config_path) if config_path else None, notes=notes, legacy_unbound=True)
    return _append(experiments_dir, row)


def log_bound_experiment(experiments_dir, experiment_id, *, notebook, git_commit, universe_build_id, regime_signature,
                         panel_manifest_identity, policy_identity, model_name, status, description="", channels=(), kind="coverage",
                         feature_contract=None, checkpoint_identity=None, config_path=None, notes="", **metrics):
    """Append (or replace, same experiment_id) one BOUND row. Raises ExperimentBindingError on a missing identity field
    (an empty git_commit is refused: a row that cannot be traced to code is not an experiment)."""
    if kind not in KINDS:
        raise ExperimentBindingError(f"kind must be one of {KINDS}, got {kind!r}")
    vals = {"notebook": notebook, "git_commit": git_commit, "universe_build_id": universe_build_id, "regime_signature": regime_signature,
            "panel_manifest_identity": panel_manifest_identity, "policy_identity": policy_identity, "model_name": model_name,
            "status": status, "feature_contract": feature_contract, "checkpoint_identity": checkpoint_identity}
    missing = [k for k in REQUIRED_BINDING if _blank(vals[k])]
    if kind == "ranking":
        missing += [k for k in RANKING_REQUIRED if _blank(vals[k])]
    if missing:
        raise ExperimentBindingError(f"experiment {experiment_id!r}: missing binding fields {missing}")
    now = datetime.now(timezone.utc).isoformat()
    row = _metrics_row(metrics)
    row.update(vals, experiment_id=str(experiment_id), date=now, timestamp=now, description=description, kind=kind,
               channels=",".join(channels) if isinstance(channels, (list, tuple)) else str(channels),
               config_path=str(config_path) if config_path else None, notes=notes, legacy_unbound=False)
    return _append(experiments_dir, row)


def comparable_experiments(df, reference_row):
    """Rows comparable with `reference_row`: same, non-empty COMPARABILITY_KEYS and not legacy_unbound."""
    ref = {k: reference_row.get(k) for k in COMPARABILITY_KEYS}
    if any(_blank(v) for v in ref.values()):
        return df.iloc[0:0]
    m = ~df["legacy_unbound"].astype(bool)
    for k, v in ref.items():
        m &= df[k].astype(object).eq(v)
    return df[m]
