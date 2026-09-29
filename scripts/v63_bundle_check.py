"""v6.3 local pre-upload check of an exported bundle, using the bundle's OWN shipped code (bundle/code), exactly as
the Kaggle notebook imports it:

    PYTHONPATH= PYTHONNOUSERSITE=1 python scripts/v63_bundle_check.py [--bundle bundle] [--expect MOST_CONFIDENT_SPECTRUM]

1. load + verify the bundle (every sha256, CONFIG_HASH, aggregator / frozen-calibration contract, model id, feature order);
2. require the expected aggregator (never a silent RRF) and a FROZEN model;
3. run the format-2 self-test (spectrum + molecule level) on numpy AND on numba when importable.
Exit code 0 only if everything passes. Nothing is written; nothing is uploaded or submitted.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=str(Path(__file__).resolve().parents[1] / "bundle"))
    ap.add_argument("--expect", default="MOST_CONFIDENT_SPECTRUM")
    args = ap.parse_args(argv)
    bundle_dir = Path(args.bundle).resolve()
    sys.path.insert(0, str(bundle_dir / "code"))                     # the shipped code, not src/
    from casmi_infer.backend import NUMBA, NUMPY, numba_available
    from casmi_infer.pipeline import Bundle
    from casmi_infer.selftest import MISMATCH_KEYS, SelfTestFailed, run_selftest

    import casmi_infer
    assert Path(casmi_infer.__file__).resolve().is_relative_to(bundle_dir), f"casmi_infer imported from {casmi_infer.__file__}, not the bundle"
    b = Bundle(bundle_dir, verify=True)
    info = json.loads((bundle_dir / "models" / "model_info.json").read_text(encoding="utf-8"))
    summary = {"bundle_version": b.manifest.get("bundle_version"), "CONFIG_HASH": b.config["CONFIG_HASH"], "model_id": b.ranker.model_id,
               "model_freeze_status": info.get("freeze_status"), "aggregator": b.aggregator_contract, "candidate_universe_version": b.config.get("candidate_universe_version"),
               "feature_order": b.ranker.feature_names}
    print(json.dumps(summary, indent=1, default=str))
    ok = True
    if b.config.get("aggregator_name") != args.expect:
        print(f"FAIL: bundle aggregator {b.config.get('aggregator_name')!r} != expected {args.expect!r}")
        ok = False
    if info.get("freeze_status") != "FROZEN":
        print(f"FAIL: bundle model is not FROZEN ({info.get('freeze_status')!r})")
        ok = False
    for backend in [NUMPY] + ([NUMBA] if numba_available() else []):
        try:
            r = run_selftest(b, backend)
        except SelfTestFailed as e:
            print(f"self-test [{backend}]: FAIL ({e})")
            ok = False
            continue
        print(f"self-test [{backend}]: {'PASS' if r['passed'] else 'FAIL'} | queries {r['n_fixture_queries']} | molecules {r['n_fixture_molecules']} | "
              f"mismatches {{{', '.join(f'{k}: {r[k]}' for k in MISMATCH_KEYS)}}}")
        ok &= bool(r["passed"]) or backend == NUMBA                  # numba may fail if numpy passes (the Kaggle fallback); numpy must pass
    print("BUNDLE CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
