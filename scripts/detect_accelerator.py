"""Print the host / accelerator description used by the runtime (never fails; no GPU context is created).

    python scripts/detect_accelerator.py            # human-readable
    python scripts/detect_accelerator.py --json     # machine-readable
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from casmi_runtime.accelerator import detect_accelerator  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    info = detect_accelerator()
    if args.json:
        print(json.dumps(info, indent=1, default=str))
    else:
        for k, v in info.items():
            print(f"{k:28s} {v}")
        print(f"\nDEVICE = {info['device']!r}  (hardware only -- a GPU BACKEND is used only if the frozen bundle implements one "
              "that is declared parity-validated and passes the bundle self-test)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
