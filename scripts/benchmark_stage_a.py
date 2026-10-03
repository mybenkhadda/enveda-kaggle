"""Stage-A benchmark: LEGACY standardizer vs the optimized engine on the SAME deterministic subset (the first N records
of the source file -- never random). Scientific identity is verified BEFORE any speedup is printed; a mismatch makes
the script exit with status 1 and no speedup is reported for that size.

    python scripts/benchmark_stage_a.py --source-file /content/enveda_work/external/coconut.csv \
        --template configs/v6/coconut_source.json --sizes 1000,10000,25000 --n-jobs auto

Compared per size:
    legacy            casmi.candidates.standardize.standardize_records (full profile, joblib/loky)
    optimized/<prof>  casmi.candidates.stage_a.standardize_records_fast with a persistent pool, for each --profiles entry
Reported: input records, unique raw SMILES, canonicalizer calls, elapsed, records/s, unique structures/s, peak RSS,
identity mismatches (standardized + rejected + filter outcome). Formula prefilter is applied first, as in Stage A.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _peak_rss_gb():
    try:
        import resource
        self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return round(self_kb / 1024 ** 2, 2), round(child_kb / 1024 ** 2, 2)       # Linux: KiB
    except ImportError:
        try:
            import psutil
            return round(psutil.Process(os.getpid()).memory_info().peak_wset / 1024 ** 3, 2), None   # Windows
        except (ImportError, AttributeError):
            return None, None


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-file", required=True)
    ap.add_argument("--template", default=str(REPO / "configs" / "v6" / "coconut_source.json"))
    ap.add_argument("--sizes", default="1000,10000,25000")
    ap.add_argument("--n-jobs", default="auto")
    ap.add_argument("--batch-records", type=int, default=2000)
    ap.add_argument("--profiles", default="full,universe_minimal")
    ap.add_argument("--skip-hit-cap", action="store_true", help="also benchmark universe_minimal with compute_tautomer_hit_cap=False")
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)

    import pandas as pd

    from casmi.candidates.filters import UniverseFilterConfig, apply_filters
    from casmi.candidates.sources import iter_source_records, load_source_config
    from casmi.candidates.stage_a import IDENTITY_COLUMNS, StructureEngine, identity_mismatches, standardize_records_fast
    from casmi.candidates.standardize import standardize_records
    from casmi.candidates.universe import UniverseBuildConfig, formula_prefilter
    from casmi.workspace.parallel import resolve_n_jobs

    n_jobs = resolve_n_jobs(a.n_jobs)
    ucfg = UniverseBuildConfig(filters=UniverseFilterConfig())
    variants = [(p, True) for p in a.profiles.split(",") if p]
    if a.skip_hit_cap:
        variants.append(("universe_minimal", False))
    report, exit_code = [], 0
    for n in [int(x) for x in a.sizes.split(",")]:
        cfg = load_source_config(a.template)
        cfg.path, cfg.max_records, cfg.chunksize = a.source_file, n, n
        records = next(iter_source_records(cfg))                         # FIRST n records: deterministic
        keep, _ = formula_prefilter(records, ucfg)
        n_unique = int(pd.Series(keep["raw_smiles"], dtype="object").dropna().nunique())
        standardize_records(keep.head(5), n_jobs=n_jobs)                  # warm up the loky executor (not timed)
        t0 = time.time()
        std_l, rej_l = standardize_records(keep, n_jobs=n_jobs)
        t_legacy = time.time() - t0
        kept_l, filt_l = apply_filters(std_l, ucfg.filters)
        row = {"n_input": int(len(records)), "n_after_prefilter": int(len(keep)), "n_unique_raw_smiles": n_unique,
               "legacy_seconds": round(t_legacy, 2), "legacy_records_per_s": round(len(keep) / t_legacy, 1) if t_legacy else None,
               "variants": {}}
        for profile, hit_cap in variants:
            name = f"{profile}{'' if hit_cap else '+no_hit_cap'}"
            with StructureEngine(profile, n_jobs, a.batch_records, hit_cap) as engine:   # pool start-up excluded from the timing
                t0 = time.time()
                std_f, rej_f, stats = standardize_records_fast(keep, engine)
                t_fast = time.time() - t0
            kept_f, filt_f = apply_filters(std_f, ucfg.filters)
            cols = [c for c in IDENTITY_COLUMNS + (["tautomer_hit_cap"] if hit_cap else []) if c in std_f.columns]
            mm = pd.concat([identity_mismatches(std_l, std_f, cols).assign(table="standardized"),
                            identity_mismatches(rej_l, rej_f, ["source", "source_id", "raw_smiles", "failure_reason"]).assign(table="rejected"),
                            identity_mismatches(kept_l, kept_f, ["source_id", "connectivity_key"]).assign(table="kept"),
                            identity_mismatches(filt_l, filt_f, ["source_id", "filter_reason"]).assign(table="filtered")], ignore_index=True)
            v = {"seconds": round(t_fast, 2), "records_per_s": round(len(keep) / t_fast, 1) if t_fast else None,
                 "unique_structures_per_s": round(stats["n_unique_raw_smiles"] / t_fast, 1) if t_fast else None,
                 "canonicalizer_calls": stats["n_standardizer_calls"], "n_identity_mismatches": int(len(mm))}
            if len(mm):
                exit_code = 1
                v["first_mismatches"] = mm.head(10).astype(str).to_dict(orient="records")
                print(f"[n={n}] {name}: IDENTITY MISMATCH -- no speedup reported\n{mm.head(10).to_string()}")
            else:
                v["speedup_vs_legacy"] = round(t_legacy / t_fast, 2) if t_fast else None
                print(f"[n={n}] {name}: identical to legacy on {len(cols)} identity columns | legacy {t_legacy:.1f}s | "
                      f"optimized {t_fast:.1f}s | speedup x{v['speedup_vs_legacy']}")
            row["variants"][name] = v
        row["peak_rss_gb_self_children"] = _peak_rss_gb()
        report.append(row)
    print(json.dumps(report, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
