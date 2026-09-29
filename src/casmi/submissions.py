"""Manual Kaggle submission log (`outputs/submissions/submission_log.jsonl`). Append-only. Scores are
typed in by the user after a manual Kaggle submission -- nothing here talks to Kaggle (no API client,
no network, no subprocess), and nothing here submits.

v6 adds `submission_name`, `aggregator`, `validation_resemblance_status` and
`candidate_universe_version`; older entries stay readable (missing keys load as None). Leaderboard
use is governed by docs/LEADERBOARD_GOVERNANCE.md: the public score may flag a major validation
mismatch / pipeline direction / a major candidate-universe change, never tune anything.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("timestamp", "submission_name", "submission_id", "bundle_version", "CONFIG_HASH", "model_id", "validation_resemblance_status", "aggregator",
          "aggregator_id", "candidate_universe", "candidate_universe_version", "self_test_status", "backend", "numba_fallback_used",
          "runtime_seconds", "peak_ram_gb", "submission_sha256", "public_score", "private_score", "notes")
CLOSED_WORLD_V1 = "closed_world_train_library_v1"


def log_submission(log_path, submission_id=None, bundle_version=None, model_id=None, aggregator_id=None, candidate_universe="closed_world_class1",
                   self_test_status=None, backend=None, numba_fallback_used=None, runtime_seconds=None, peak_ram_gb=None,
                   public_score=None, private_score=None, notes="", submission_name=None, aggregator=None,
                   validation_resemblance_status=None, candidate_universe_version=CLOSED_WORLD_V1, submission_sha256=None, config_hash=None):
    """Append one entry. `public_score` is typed in manually after the user submits on Kaggle; nothing here submits."""
    if submission_name is None and submission_id is None:
        raise ValueError("give submission_name (v6) or submission_id")
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "submission_name": submission_name or submission_id,
             "submission_id": submission_id or submission_name, "bundle_version": bundle_version, "CONFIG_HASH": config_hash, "model_id": model_id,
             "validation_resemblance_status": validation_resemblance_status, "aggregator": aggregator or aggregator_id,
             "aggregator_id": aggregator_id or aggregator, "candidate_universe": candidate_universe,
             "candidate_universe_version": candidate_universe_version, "self_test_status": self_test_status, "backend": backend,
             "numba_fallback_used": numba_fallback_used, "runtime_seconds": runtime_seconds, "peak_ram_gb": peak_ram_gb,
             "submission_sha256": submission_sha256, "public_score": public_score, "private_score": private_score, "notes": notes}
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def fields_from_run_report(run_report_path):
    """Pre-fill the engineering fields from a downloaded Kaggle `run_report.json` (scores stay manual)."""
    r = json.loads(Path(run_report_path).read_text(encoding="utf-8"))
    agg = r.get("aggregator") or {}
    if isinstance(agg, str):                      # v6.3 report: aggregator = production name, aggregator_config = the block
        agg = {**(r.get("aggregator_config") or {}), "name": agg}
    return {"bundle_version": r.get("bundle_version"), "config_hash": r.get("CONFIG_HASH"), "model_id": r.get("model_id"),
            "candidate_universe_version": r.get("candidate_universe_version") or CLOSED_WORLD_V1,
            "aggregator_id": agg.get("aggregator_id") or agg.get("name"), "aggregator": agg.get("name"),
            "self_test_status": r.get("self_test_status") or ("PASS" if r.get("self_test_passed") else "FAIL"),
            "backend": r.get("backend") or r.get("self_test_backend_final"),
            "numba_fallback_used": r.get("numpy_fallback_used", r.get("numba_fallback_used")),
            "runtime_seconds": r.get("runtime_seconds", r.get("total_seconds")),
            "peak_ram_gb": r.get("peak_ram_gb", r.get("peak_ram", r.get("peak_rss_gb"))),
            "validation_resemblance_status": r.get("validation_realism_status"), "submission_sha256": r.get("submission_sha256")}


def resemblance_status_from(validation_resemblance_json):
    """The v6 test-match status (or PENDING when the diagnostic has not been run)."""
    p = Path(validation_resemblance_json)
    return json.loads(p.read_text(encoding="utf-8")).get("status") if p.exists() else "PENDING_TEST_MATCH_DIAGNOSTIC"


def load_submission_log(log_path):
    p = Path(log_path)
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{k: r.get(k) for k in FIELDS} | {k: v for k, v in r.items() if k not in FIELDS} for r in rows]
