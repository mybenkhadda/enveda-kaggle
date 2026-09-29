#!/usr/bin/env python
"""Upload the prepared ENVEDA CASMI payload as a Kaggle Model variation.

This script does NOT build the payload. Run scripts/prepare_kaggle_model.ps1 first.

The parent Kaggle Model must already exist. Create it once either in the Kaggle UI
or with:
    kaggle models create -p kaggle_model_build/metadata

Then:
    python scripts/upload_kaggle_model.py \
        --handle YOUR_KAGGLE_USERNAME/enveda-casmi-v2-a7/sklearn/full-e2e
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "bundle/config.json",
    "bundle/models/model_info.json",
    "bundle/models/v1_fold0.txt",
    "bundle/models/v1_fold1.txt",
    "bundle/models/v1_fold2.txt",
    "bundle/models/v1_fold3.txt",
    "bundle/models/v1_fold4.txt",
    "bundle/aggregation/calibration.json",
    "bundle/aggregation/selected_aggregator.json",
    "bundle/selftest/fixture_manifest.json",
    "bundle/code/casmi_infer/pipeline.py",
    "runtime/src/casmi_runtime/named_inference.py",
    "runtime/notebooks/02_kaggle_inference.ipynb",
    "PACKAGE_INFO.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--handle",
        required=True,
        help="Kaggle model variation handle: owner/model/framework/variation",
    )
    ap.add_argument(
        "--payload",
        default="kaggle_model_build/payload",
        help="Prepared payload directory",
    )
    ap.add_argument(
        "--version-notes",
        default="CASMI 2026 v2-A7 complete offline E2E filename-only runtime",
    )
    args = ap.parse_args()

    payload = Path(args.payload).resolve()
    if not payload.is_dir():
        raise SystemExit(f"Payload directory not found: {payload}")

    missing = [p for p in sorted(EXPECTED) if not (payload / p).exists()]
    if missing:
        raise SystemExit("Payload incomplete:\n  " + "\n  ".join(missing))

    info = json.loads((payload / "PACKAGE_INFO.json").read_text(encoding="utf-8-sig"))
    if info.get("bundle_version") != "v2-A7":
        raise SystemExit(f"Unexpected bundle_version: {info.get('bundle_version')!r}")
    if info.get("model_id") != "V1_TL_1K_TESTSIM_STRICT":
        raise SystemExit(f"Unexpected model_id: {info.get('model_id')!r}")

    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "kagglehub is not installed. Install it in the LOCAL upload environment first: "
            "python -m pip install kagglehub"
        ) from e

    print("Uploading Kaggle Model variation:")
    print("  handle :", args.handle)
    print("  payload:", payload)
    print("  mode   : FILENAMES_ONLY")
    print("  bundle : v2-A7")
    print("  model  : V1_TL_1K_TESTSIM_STRICT")

    kagglehub.model_upload(
        args.handle,
        str(payload),
        version_notes=args.version_notes,
        ignore_patterns=[
            "__pycache__/",
            "*.pyc",
            ".ipynb_checkpoints/",
        ],
    )

    print("Upload request completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
