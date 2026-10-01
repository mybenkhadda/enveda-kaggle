"""Upload the built Kaggle Model payload as a new version of the PRIVATE model variation
    <USERNAME>/enveda-casmi-v2-a7/sklearn/full-e2e
with kagglehub.model_upload. Credentials come ONLY from your normal Kaggle / KaggleHub configuration
(~/.kaggle/kaggle.json, KAGGLE_USERNAME / KAGGLE_KEY, or `kagglehub.login()`); nothing is embedded here.

    python scripts/upload_kaggle_model.py --handle USERNAME/enveda-casmi-v2-a7/sklearn/full-e2e [--version-notes "..."]
    python scripts/upload_kaggle_model.py --handle USERNAME/enveda-casmi-v2-a7/sklearn/full-e2e --dry-run   # checks only

Before anything is sent (all static, filename-only -- no hashing):
  * the handle is <user>/enveda-casmi-v2-a7/<framework>/full-e2e and not the template placeholder;
  * the payload (default kaggle_model_build/payload) holds every path in its kaggle_model_inventory.json, the critical
    files spelled out below, and every file the bundle manifest lists; no competition file, no runtime casmi_infer copy;
  * PACKAGE_INFO.json says bundle_version v2-A7, model_id V1_TL_1K_TESTSIM_STRICT, CONFIG_HASH 60174e39a2a3c6b4.
Create the private parent model ONCE first:  kaggle models create -p kaggle_model_build/metadata
Nested directories are preserved; __pycache__/, *.pyc and .ipynb_checkpoints/ are ignored.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = REPO / "kaggle_model_build" / "payload"
MODEL_SLUG = "enveda-casmi-v2-a7"
VARIATION = "full-e2e"
PLACEHOLDER = "YOUR_KAGGLE_USERNAME"
IDENTITY = {"bundle_version": "v2-A7", "model_id": "V1_TL_1K_TESTSIM_STRICT", "CONFIG_HASH": "60174e39a2a3c6b4"}
IGNORE_PATTERNS = ["__pycache__/", "*.pyc", ".ipynb_checkpoints/"]
CRITICAL = (["PACKAGE_INFO.json", "kaggle_model_inventory.json", "runtime/src/casmi_runtime/named_inference.py",
             "runtime/src/casmi_runtime/frozen.py", "runtime/src/casmi_runtime/backends.py", "runtime/notebooks/02_kaggle_inference.ipynb",
             "bundle/config.json", "bundle/manifest.json", "bundle/structures.parquet", "bundle/connectivities.parquet",
             "bundle/ref_peaks_mz.npy", "bundle/ref_peaks_int.npy", "bundle/aggregation/calibration.json",
             "bundle/aggregation/selected_aggregator.json", "bundle/code/casmi_infer/__init__.py", "bundle/code/casmi_infer/pipeline.py"]
            + [f"bundle/models/v1_fold{k}.txt" for k in range(5)]
            + [f"bundle/selftest/{f}" for f in ("fixture_manifest.json", "queries.parquet", "expected_features.parquet", "expected_scores.parquet",
                                                "expected_ranks.parquet", "expected_probs.parquet", "expected_molecules.parquet",
                                                "expected_molecule_ranking.parquet")])
FORBIDDEN_NAMES = ("test.parquet", "sample_submission.csv", "train.parquet")


def check_handle(handle):
    m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_-]*)/([^/]+)/([^/]+)/([^/]+)", handle or "")
    if not m:
        return [f"--handle {handle!r} is not <username>/{MODEL_SLUG}/<framework>/{VARIATION}"]
    user, slug, _framework, variation = m.groups()
    problems = []
    if user == PLACEHOLDER:
        problems.append(f"--handle still uses the placeholder {PLACEHOLDER}")
    if slug != MODEL_SLUG:
        problems.append(f"model slug {slug!r} != {MODEL_SLUG!r}")
    if variation != VARIATION:
        problems.append(f"variation {variation!r} != {VARIATION!r}")
    return problems


def _ignored(rel):
    parts = Path(rel).parts
    return "__pycache__" in parts or ".ipynb_checkpoints" in parts or rel.endswith(".pyc")


def check_payload(payload):
    """(problems, summary) -- filename presence + PACKAGE_INFO identity; nothing is hashed or imported."""
    p = Path(payload)
    if not p.is_dir():
        return [f"payload {p} does not exist -- run scripts/prepare_kaggle_model.ps1 first"], {}
    problems = []
    inv_path = p / "kaggle_model_inventory.json"
    inv = json.loads(inv_path.read_text(encoding="utf-8")) if inv_path.is_file() else None
    required = list(CRITICAL)
    if inv:
        required += list(inv["payload_top_level"]["required"]) + list(inv["runtime_source"]["required"]) + list(inv["frozen_bundle"]["required"])
        problems += [f"missing directory {d}" for d in inv["frozen_bundle"].get("required_directories", []) if not (p / d).is_dir()]
        problems += [f"forbidden in payload: {f}" for f in inv["runtime_source"].get("forbidden", []) if (p / f).exists()]
    else:
        problems.append("payload has no kaggle_model_inventory.json")
    man_path = p / "bundle" / "manifest.json"
    if man_path.is_file():
        required += [f"bundle/{f}" for f in (json.loads(man_path.read_text(encoding="utf-8")).get("files") or {})]
    problems += [f"missing {r}" for r in dict.fromkeys(required) if not (p / r).is_file()]
    if (p / "runtime" / "src" / "casmi_infer").exists():
        problems.append("runtime/src/casmi_infer must not be shipped (bundle/code/casmi_infer is the only inference code)")
    files = [f for f in p.rglob("*") if f.is_file()]
    problems += [f"competition file in the payload: {f.relative_to(p).as_posix()}" for f in files if f.name in FORBIDDEN_NAMES]
    info_path = p / "PACKAGE_INFO.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        problems += [f"PACKAGE_INFO.json {k} {info.get(k)!r} != {v!r}" for k, v in IDENTITY.items() if info.get(k) != v]
    kept = [f for f in files if not _ignored(f.relative_to(p).as_posix())]
    summary = {"payload": str(p.resolve()), "files": len(kept), "ignored": len(files) - len(kept),
               "gb": round(sum(f.stat().st_size for f in kept) / 1024 ** 3, 3)}
    return problems, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handle", required=True, help=f"<username>/{MODEL_SLUG}/sklearn/{VARIATION}")
    ap.add_argument("--payload", default=str(DEFAULT_PAYLOAD))
    ap.add_argument("--version-notes", default="v2-A7 full E2E payload (FILENAMES_ONLY integrity; functional self-test gate)")
    ap.add_argument("--license-name", default=None, help="only used when kagglehub creates the variation; e.g. 'Apache 2.0'")
    ap.add_argument("--dry-run", action="store_true", help="run every check, upload nothing")
    args = ap.parse_args(argv)

    problems = check_handle(args.handle)
    payload_problems, summary = check_payload(args.payload)
    problems += payload_problems
    for pr in problems:
        print("PROBLEM:", pr)
    if problems:
        print(f"UPLOAD BLOCKED ({len(problems)} problems) -- nothing was sent")
        return 1
    print(f"payload OK: {summary}")
    print(f"target: {args.handle} (PRIVATE parent model must already exist: kaggle models create -p kaggle_model_build/metadata)")
    if args.dry_run:
        print("DRY RUN -- nothing was uploaded")
        return 0

    import kagglehub                                   # uses the caller's own Kaggle credentials
    kwargs = {"handle": args.handle, "local_model_dir": str(Path(args.payload).resolve()), "version_notes": args.version_notes}
    if args.license_name:
        kwargs["license_name"] = args.license_name
    try:
        kagglehub.model_upload(**kwargs, ignore_patterns=IGNORE_PATTERNS)
    except TypeError as e:                             # an older kagglehub without ignore_patterns
        if summary["ignored"]:
            print(f"this kagglehub cannot ignore {summary['ignored']} cache files ({e}); rebuild the payload or upgrade kagglehub")
            return 1
        kagglehub.model_upload(**kwargs)
    print(f"upload finished for {args.handle} -- check the model page, then attach it to 02_kaggle_inference.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
