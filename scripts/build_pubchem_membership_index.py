"""Build the PubChem KEY-MEMBERSHIP index (PUBCHEM_KEY_MEMBERSHIP_ESTIMATE) from a LOCAL CID -> InChIKey file.

No download, no network: the file must already be on disk (e.g. Drive data/external/pubchem/). Chunked (never the whole
file in RAM) and resumable (per-chunk spool markers). Spool to local scratch, final shards to Drive:

    python scripts/build_pubchem_membership_index.py \
        --input /content/drive/MyDrive/EnvedaCASMI/data/external/pubchem/CID-InChI-Key.gz \
        --out-dir /content/drive/MyDrive/EnvedaCASMI/candidates/pubchem_membership \
        --work-dir /content/enveda_work/pubchem_membership_spool

Column layout comes from configs/v6/pubchem_membership_source.json (override with --config).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="local CID -> InChIKey file (tsv / csv, optionally .gz)")
    ap.add_argument("--out-dir", required=True, help="final shard directory (persistent, e.g. on Drive)")
    ap.add_argument("--work-dir", default=None, help="spool directory (local scratch recommended); default <out-dir>/_spool")
    ap.add_argument("--config", default=str(REPO / "configs" / "v6" / "pubchem_membership_source.json"))
    ap.add_argument("--chunksize", type=int, default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    from casmi.candidates.pubchem_membership import build_membership_index, load_membership_config
    cfg = load_membership_config(a.config, path=a.input)
    if a.chunksize:
        cfg.chunksize = a.chunksize
    man = build_membership_index(cfg, a.out_dir, a.work_dir)
    print(json.dumps({k: man[k] for k in ("label", "n_input_rows", "n_invalid_inchikeys", "n_unique_connectivity_keys", "caveat")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
