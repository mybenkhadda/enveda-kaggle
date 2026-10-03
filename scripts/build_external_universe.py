"""Build the external candidate universe (Stage A / B / C) from the command line -- the SAME functions notebook 12 calls,
so it runs unchanged on Colab, a local Windows / Linux workstation or a high-CPU cloud VM. Nothing is downloaded.

    # Stage A for COCONUT with all CPUs, then Stage B + C
    python scripts/build_external_universe.py --stage A --source COCONUT --n-jobs auto
    python scripts/build_external_universe.py --stage BC

    # a 16-core workstation with the Drive tree mirrored locally
    python scripts/build_external_universe.py --drive-root D:/EnvedaCASMI --scratch-root D:/scratch --stage all --n-jobs 16

Resumable: re-running skips chunks / buckets whose markers match the CURRENT build identity. Changing the source
file, column mapping, chunk size, filters, standardization profile or the RDKit version starts a new Stage-A build
(old builds are kept, never mixed in). See docs/STAGE_A_OPTIMIZATION.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "casmi_v2_colab.yaml"))
    ap.add_argument("--drive-root", help="persistent root (sets ENVEDA_DRIVE_ROOT)")
    ap.add_argument("--repo-root", default=None, help="repository root for source templates (default: this checkout)")
    ap.add_argument("--scratch-root", help="local SSD scratch (default: paths.scratch_dir of the config)")
    ap.add_argument("--stage", default="all", help="A, B, C, BC or all")
    ap.add_argument("--source", action="append", help="limit Stage A to these sources (e.g. COCONUT); Stage B always uses every configured source")
    ap.add_argument("--n-jobs", default=None, help="worker processes: auto or an integer (default: universe.n_jobs)")
    ap.add_argument("--chunk-records", type=int, default=None, help="read / checkpoint chunk size (changes the build identity)")
    ap.add_argument("--batch-records", type=int, default=None, help="unique SMILES per worker task (never changes outputs)")
    ap.add_argument("--profile", choices=("universe_minimal", "full"), default=None)
    ap.add_argument("--no-cache", action="store_true", help="disable the optional SQLite canonicalization cache")
    ap.add_argument("--no-stage-input", action="store_true", help="read the source directly from its persistent location")
    ap.add_argument("--progress-every", type=int, default=None, help="structures between progress lines (0 = quiet)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.drive_root:
        os.environ["ENVEDA_DRIVE_ROOT"] = a.drive_root
    os.environ.setdefault("ENVEDA_REPO_ROOT", a.repo_root or str(REPO))
    sys.path.insert(0, str(Path(os.environ["ENVEDA_REPO_ROOT"]) / "src"))
    import pandas as pd

    from casmi.candidates.stage_a import StageAPerformance, configured_sources, resolve_stage_a_inputs, run_stage_a, stage_a_plan
    from casmi.candidates.universe import UniverseBuildConfig, finalize_universe, run_stage_b, universe_source_status
    from casmi.workspace.config import input_path, load_v2_config

    cfg, P = load_v2_config(a.config)
    ucfg = UniverseBuildConfig.from_dict(cfg["universe"])
    perf = StageAPerformance.from_dict(cfg["universe"].get("performance"))
    if a.chunk_records:
        ucfg.chunk_records = a.chunk_records
    if a.n_jobs is not None:
        ucfg.n_jobs = a.n_jobs
    if a.batch_records:
        perf.standardize_batch_records = a.batch_records
    if a.profile:
        perf.standardization_profile = a.profile
    if a.no_cache:
        perf.cache_enabled = False
    if a.no_stage_input:
        perf.stage_input_to_scratch = False
    if a.progress_every is not None:
        perf.progress_every_records = a.progress_every
    scratch = Path(a.scratch_root) if a.scratch_root else P.scratch_dir
    out_root, work_root = P.candidate_db_dir / "universe", P.candidate_db_dir / "universe_work"
    stage = a.stage.upper()
    run_a, run_b, run_c = stage in ("A", "ALL"), stage in ("B", "BC", "ALL"), stage in ("C", "BC", "ALL")

    specs = configured_sources(cfg, P.drive_root, os.environ["ENVEDA_REPO_ROOT"], scratch, perf, stage=run_a)
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(stage_a_plan(specs, ucfg, perf).T.to_string())
    if run_a:
        for s in specs:
            if not s.exists or (a.source and s.source not in {x.upper() for x in a.source}):
                continue
            res = run_stage_a(s.cfg, work_root, scratch, ucfg, perf, source_identity_path=s.drive_path)
            print(json.dumps({k: res.manifest[k] for k in ("source", "build_id", "n_chunks", "n_input", "n_kept")}, indent=2))
    if run_b or run_c:
        mv = pd.read_parquet(input_path(cfg, P, "molecule_mass_variants"))
        st = pd.read_parquet(input_path(cfg, P, "structure_table"), columns=["smiles", "connectivity_key", "molecular_weight", "formal_charge"])
        st = st.dropna(subset=["connectivity_key"])
        inputs = resolve_stage_a_inputs(work_root, specs, ucfg, perf)
        sb = run_stage_b(inputs, mv, st, out_root, ucfg)
        if run_c:
            man = finalize_universe(out_root, sb["buckets"], stage_b_id=sb["stage_b_id"],
                                    build_info={"stage_a": {k: v.get("build_id") for k, v in inputs["sources"].items()}})
            print(json.dumps({k: man[k] for k in ("n_candidates", "external_source_present", "n_train_candidates", "n_external_only",
                                                  "n_train_and_external", "sources", "build_id")}, indent=2))
            print("universe status:", universe_source_status(out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
