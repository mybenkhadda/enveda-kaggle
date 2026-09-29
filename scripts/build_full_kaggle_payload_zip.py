#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

EXCLUDE_PARTS = {".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc"}

RUNTIME_FILES = {
    "notebooks/02_kaggle_inference.ipynb": "runtime/notebooks/02_kaggle_inference.ipynb",
    "requirements-kaggle.txt": "runtime/requirements-kaggle.txt",
    "KAGGLE_RUN_GUIDE.md": "runtime/KAGGLE_RUN_GUIDE.md",
}

PACKAGE_INFO = {
    "package": "enveda-casmi-v2-a7",
    "framework": "sklearn",
    "variation": "full-e2e",
    "privacy": "private",
    "bundle_version": "v2-A7",
    "CONFIG_HASH": "60174e39a2a3c6b4",
    "model_id": "V1_TL_1K_TESTSIM_STRICT",
    "aggregator": "MOST_CONFIDENT_SPECTRUM",
    "calibration_temperature": 1.1947045372735254,
    "input_mode": "FILENAMES_ONLY",
    "bundle_hash_verification": False,
    "fixture_hash_verification": False,
    "competition_is_separate": True,
}

def allowed(p: Path) -> bool:
    return not any(part in EXCLUDE_PARTS for part in p.parts) and p.suffix.lower() not in EXCLUDE_SUFFIXES

def add_file(zf: zipfile.ZipFile, src: Path, arcname: str):
    if not src.is_file():
        raise FileNotFoundError(src)
    zf.write(src, arcname)

def main():
    ap = argparse.ArgumentParser(description="Build the complete ENVEDA CASMI Kaggle Model ZIP.")
    ap.add_argument("--repo-root", default=".", help="Repository root")
    ap.add_argument("--bundle-root", default=None, help="Frozen bundle directory; default <repo-root>/bundle")
    ap.add_argument("--output", default=None, help="Output ZIP")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    bundle = Path(args.bundle_root).resolve() if args.bundle_root else (repo / "bundle").resolve()
    output = Path(args.output).resolve() if args.output else (repo / "enveda-casmi-v2-a7-full-e2e.zip").resolve()

    if not repo.is_dir():
        raise SystemExit(f"Repository not found: {repo}")
    if not bundle.is_dir():
        raise SystemExit(f"Bundle not found: {bundle}")

    required = [
        bundle / "config.json",
        bundle / "manifest.json",
        bundle / "models" / "model_info.json",
        *(bundle / "models" / f"v1_fold{i}.txt" for i in range(5)),
        bundle / "aggregation" / "calibration.json",
        bundle / "aggregation" / "selected_aggregator.json",
        bundle / "selftest" / "fixture_manifest.json",
        bundle / "selftest" / "queries.parquet",
        bundle / "code" / "casmi_infer" / "__init__.py",
        repo / "src" / "casmi_runtime" / "named_inference.py",
        repo / "notebooks" / "02_kaggle_inference.ipynb",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))

    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    model_info = json.loads((bundle / "models" / "model_info.json").read_text(encoding="utf-8"))
    agg_name = cfg.get("aggregator_name") or (cfg.get("aggregator") or {}).get("name")

    if cfg.get("bundle_version") != "v2-A7":
        raise SystemExit(f"Unexpected bundle_version: {cfg.get('bundle_version')!r}")
    if cfg.get("CONFIG_HASH") != "60174e39a2a3c6b4":
        raise SystemExit(f"Unexpected CONFIG_HASH: {cfg.get('CONFIG_HASH')!r}")
    if cfg.get("model_id") != "V1_TL_1K_TESTSIM_STRICT":
        raise SystemExit(f"Unexpected model_id: {cfg.get('model_id')!r}")
    if model_info.get("model_id") != "V1_TL_1K_TESTSIM_STRICT":
        raise SystemExit(f"Unexpected model_info.model_id: {model_info.get('model_id')!r}")
    if model_info.get("freeze_status") != "FROZEN":
        raise SystemExit(f"Unexpected freeze_status: {model_info.get('freeze_status')!r}")
    if agg_name != "MOST_CONFIDENT_SPECTRUM":
        raise SystemExit(f"Unexpected aggregator: {agg_name!r}")
    if float(cfg.get("calibration_temperature")) != 1.1947045372735254:
        raise SystemExit(f"Unexpected calibration temperature: {cfg.get('calibration_temperature')!r}")

    runtime_root = repo / "src" / "casmi_runtime"
    runtime_files = [p for p in runtime_root.rglob("*") if p.is_file() and allowed(p)]
    bundle_files = [p for p in bundle.rglob("*") if p.is_file() and allowed(p)]

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in sorted(bundle_files):
            add_file(zf, p, f"bundle/{p.relative_to(bundle).as_posix()}")

        for p in sorted(runtime_files):
            add_file(zf, p, f"runtime/src/casmi_runtime/{p.relative_to(runtime_root).as_posix()}")

        for src_rel, arcname in RUNTIME_FILES.items():
            add_file(zf, repo / src_rel, arcname)

        zf.writestr("PACKAGE_INFO.json", json.dumps(PACKAGE_INFO, indent=2))

        inv = repo / "kaggle_model_inventory.json"
        if inv.exists():
            add_file(zf, inv, "kaggle_model_inventory.json")

    with zipfile.ZipFile(output, "r") as zf:
        names = set(zf.namelist())

    required_in_zip = {
        "bundle/_ref_peaks_int.f8",
        "bundle/_ref_peaks_mz.f8",
        "bundle/ref_peaks_int.npy",
        "bundle/ref_peaks_mz.npy",
        "bundle/ref_meta.parquet",
        "bundle/connectivities.parquet",
        "bundle/structures.parquet",
        *(f"bundle/models/v1_fold{i}.txt" for i in range(5)),
        "bundle/aggregation/calibration.json",
        "bundle/aggregation/selected_aggregator.json",
        "bundle/selftest/queries.parquet",
        "bundle/selftest/expected_features.parquet",
        "bundle/selftest/expected_scores.parquet",
        "bundle/selftest/expected_ranks.parquet",
        "bundle/selftest/expected_probs.parquet",
        "bundle/selftest/expected_molecules.parquet",
        "bundle/selftest/expected_molecule_ranking.parquet",
        "bundle/code/casmi_infer/pipeline.py",
        "runtime/src/casmi_runtime/named_inference.py",
        "runtime/notebooks/02_kaggle_inference.ipynb",
        "PACKAGE_INFO.json",
    }

    missing_zip = sorted(required_in_zip - names)
    if missing_zip:
        output.unlink(missing_ok=True)
        raise SystemExit("ZIP incomplete:\n  " + "\n  ".join(missing_zip))

    print("KAGGLE MODEL ZIP READY")
    print("ZIP:", output)
    print("FILES:", len(names))
    print("SIZE_GB:", round(output.stat().st_size / (1024**3), 3))
    print("Competition files stay separate: test.parquet + sample_submission.csv")

if __name__ == "__main__":
    main()
