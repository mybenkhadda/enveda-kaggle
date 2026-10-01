"""Lightweight experiment tracker: one parquet table, one row per USER-executed experiment.

Values are whatever the calling notebook computed; a metric that was not computed is stored as
NaN -- never a placeholder number."""
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPERIMENT_COLUMNS = ["experiment_id", "date", "git_commit", "description", "channels", "config_path", "C1_MRR25", "C2_MRR25",
                      "C3_MRR25", "composite_MRR25", "Hit1", "candidate_recall", "runtime", "notes"]
NUMERIC = ("C1_MRR25", "C2_MRR25", "C3_MRR25", "composite_MRR25", "Hit1", "candidate_recall", "runtime")


def experiments_path(experiments_dir):
    return Path(experiments_dir) / "experiments.parquet"


def load_experiments(experiments_dir):
    p = experiments_path(experiments_dir)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=EXPERIMENT_COLUMNS)


def log_experiment(experiments_dir, experiment_id, description, channels, config_path=None, git_commit=None, notes="", **metrics):
    """Append (or replace, same `experiment_id`) one row. Unknown metric names raise."""
    unknown = set(metrics) - set(NUMERIC)
    if unknown:
        raise KeyError(f"unknown experiment metrics {sorted(unknown)}; allowed {NUMERIC}")
    row = {c: math.nan for c in NUMERIC}
    row.update({k: (float(v) if v is not None else math.nan) for k, v in metrics.items()})
    row.update(experiment_id=str(experiment_id), date=datetime.now(timezone.utc).isoformat(), git_commit=git_commit,
               description=description, channels=",".join(channels) if isinstance(channels, (list, tuple)) else str(channels),
               config_path=str(config_path) if config_path else None, notes=notes)
    df = load_experiments(experiments_dir)
    df = pd.concat([df[df["experiment_id"] != row["experiment_id"]], pd.DataFrame([row])], ignore_index=True)[EXPERIMENT_COLUMNS]
    p = experiments_path(experiments_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df
