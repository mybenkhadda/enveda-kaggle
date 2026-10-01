"""Derive protocol features / counts (+ training manifests) from EXISTING QCR -- filter + aggregate only.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python scripts/v5_build_protocol_features.py --manifest TL_3K
    PYTHONPATH=src PYTHONNOUSERSITE=1 python scripts/v5_build_protocol_features.py --manifest TL_10K
    PYTHONPATH=src PYTHONNOUSERSITE=1 python scripts/v5_build_protocol_features.py --manifest TL_EVAL HOST MOL_DEV

Default protocol: test_simulated_strict. Training manifests (TL_1K / TL_3K / TL_10K / RND_1K) also get
their derived `<name>_TESTSIM_STRICT` manifest. The QCR itself must already exist
(scripts/v5_build_scale_features.py --manifest <name>); this script never builds QCR.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from casmi.paths import get_project_paths  # noqa: E402
from casmi.qcr.context import v4b_paths, v5_paths  # noqa: E402
from casmi.qcr.protocol_features import build_host_protocol_artifacts, build_protocol_artifacts, qcr_status  # noqa: E402
from casmi.qcr.protocols import PRIMARY_PROTOCOL, PROTOCOL_DEFS  # noqa: E402
from casmi.qcr.settlement import load_settlement_or_refuse  # noqa: E402

TRAINING = {"TL_1K", "TL_3K", "TL_10K", "RND_1K"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True, choices=["TL_1K", "TL_3K", "TL_10K", "RND_1K", "TL_EVAL", "MOL_DEV", "HOST"])
    ap.add_argument("--protocol", default=PRIMARY_PROTOCOL, choices=sorted(PROTOCOL_DEFS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    paths = get_project_paths()
    vp, D = v4b_paths(paths), v5_paths(paths)
    st = load_settlement_or_refuse(vp.settlement_json)
    results = []
    for name in args.manifest:
        if name == "HOST":
            host_q = pd.read_parquet(vp.host_processed / "host_holdout_spectra.parquet")
            results.append(build_host_protocol_artifacts(D, st["evidence_artifacts"], host_q, args.protocol, st["evidence_fingerprint"], force=args.force))
        elif qcr_status(D, name) != "READY":
            print(f"{name}: QCR MISSING -- first run: python scripts/v5_build_scale_features.py --manifest {name}")
            continue
        else:
            results.append(build_protocol_artifacts(D, name, args.protocol, st["evidence_fingerprint"], derive_training_manifest=name in TRAINING, force=args.force))
        print(results[-1])
    return results


if __name__ == "__main__":
    main()
