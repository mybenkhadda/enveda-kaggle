"""Append-only decision log: every choice influenced by host-holdout metrics gets one recorded
entry, so "did we tune on the host?" is answerable by reading a file instead of reconstructing
notebook history. The experimental-governance rule this exists to enforce: DEV selects, HOST
confirms -- a host benchmark that quietly turns into a tuning set is not a benchmark anymore.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = ("bug_fix", "pre_registered_choice", "post_host_change")


def log_decision(decision, evidence_used, host_visible, category, log_path):
    """Append one entry to the `.jsonl` decision log at `log_path`. `category` must be one of
    `CATEGORIES` -- `post_host_change` is the one that should be rare and always justified
    (a decision made, or changed, AFTER seeing a host-metric result). Returns the entry written."""
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "evidence_used": evidence_used,
        "host_visible": bool(host_visible),
        "category": category,
    }
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def load_decision_log(log_path):
    """All entries as a list of dicts, oldest first. Empty list if the file doesn't exist yet
    (never raises -- a fresh project has no decisions logged, that's not an error)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def post_host_change_count(log_path):
    """How many logged decisions were made (or changed) AFTER seeing a host result -- the
    number that should stay at or near 0 for the host benchmark to remain a genuine holdout."""
    return sum(1 for e in load_decision_log(log_path) if e.get("category") == "post_host_change")
