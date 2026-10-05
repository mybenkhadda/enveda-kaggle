"""USER-RUN (Colab or any machine with the Drive universe): verify the legacy TRAIN ∪ COCONUT universe and write the
adoption sidecar `candidates/universe/universe_manifest_v3_adopted.json`. Read-only for every scientific file: buckets,
indexes and the legacy `universe_manifest.json` are never rewritten.

    python scripts/adopt_legacy_universe_manifest.py \
        --universe /content/drive/MyDrive/EnvedaCASMI/candidates/universe \
        --work /content/drive/MyDrive/EnvedaCASMI/candidates/universe_work \
        --expect-n-candidates 696447 --expect-train 274195 --expect-coconut 446794 --expect-train-and-coconut 24542

Exit codes: 0 adopted (or already adopted / modern manifest, nothing to do), 2 refused (an invariant failed, TRAIN-only
universe, or a different sidecar exists -> pass --force to archive it and re-adopt), 3 universe absent.
The expected counts are optional; when given they must match EXACTLY (the counts reported by the earlier Colab run).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", required=True, help="finalized universe directory (candidates/universe)")
    ap.add_argument("--work", default=None, help="legacy Stage-A work directory (candidates/universe_work) -- audit only")
    ap.add_argument("--expect-n-candidates", type=int, default=None)
    ap.add_argument("--expect-train", type=int, default=None)
    ap.add_argument("--expect-coconut", type=int, default=None)
    ap.add_argument("--expect-train-and-coconut", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="verify and print; do not write the sidecar")
    ap.add_argument("--force", action="store_true", help="archive a DIFFERENT existing sidecar and re-adopt")
    ap.add_argument("--report", default=None, help="optional path for the full verification report (json)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    from casmi.candidates.legacy_adoption import UNIVERSE_ABSENT, LegacyAdoptionRefused, adopt_legacy_universe, verify_legacy_universe
    expected = {k: v for k, v in (("n_candidates", a.expect_n_candidates), ("train", a.expect_train), ("coconut", a.expect_coconut),
                                  ("train_and_coconut", a.expect_train_and_coconut)) if v is not None}
    try:
        res = adopt_legacy_universe(a.universe, expected_counts=expected, work_root=a.work, write=not a.dry_run, force=a.force)
        v = res["verification"]
    except LegacyAdoptionRefused as e:
        v = verify_legacy_universe(a.universe, expected_counts=expected, work_root=a.work)
        print(json.dumps({"status": "ADOPTION_REFUSED", "verification_status": v["status"], "reason": str(e),
                          "failed_checks": [c for c in v["checks"] if not c["passed"]]}, indent=2, default=str))
        _report(a.report, v)
        return 2 if v["status"] != UNIVERSE_ABSENT else 3
    if v["status"] == UNIVERSE_ABSENT:
        print(json.dumps({"status": UNIVERSE_ABSENT, "universe": a.universe}, indent=2))
        return 3
    side = res["adopted_manifest"] or {}
    print(json.dumps({"status": res["status"], "written": res["written"], "path": res["path"], "dry_run": a.dry_run,
                      "adopted_build_id": side.get("adopted_build_id"), "counts": v.get("counts"), "per_source": v.get("per_source"),
                      "contents_verified": v.get("contents_verified"), "reproducible_build_id_available": False if side else None,
                      "legacy_stage_a": v.get("legacy_stage_a")}, indent=2, default=str))
    _report(a.report, v)
    return 0


def _report(path, verification):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({k: x for k, x in verification.items() if k != "legacy_manifest"}, indent=2, default=str),
                              encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
