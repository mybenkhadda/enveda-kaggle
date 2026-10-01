"""Keep every v2 notebook's first code cell identical to `casmi.workspace.notebook_cells.BOOTSTRAP_CELL`.

    python scripts/stamp_notebooks.py                         # re-stamp notebooks/1[0-9]_*.ipynb, 2[0-9]_*.ipynb
    python scripts/stamp_notebooks.py notebooks/15_*.ipynb
    python scripts/stamp_notebooks.py --from-percent src.py out.ipynb   # build a notebook from percent format

Percent format: cells separated by lines starting with `# %%`; `# %% [markdown]` cells hold markdown as `# ` comment
lines; a cell whose header is `# %% BOOTSTRAP` becomes the canonical bootstrap cell. Outputs are never written
(notebooks ship clean; Colab executes them).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from casmi.workspace.notebook_cells import BOOTSTRAP_CELL, BOOTSTRAP_MARKER_PREFIX  # noqa: E402

DEFAULT_GLOBS = ("notebooks/1[0-9]_*.ipynb", "notebooks/2[0-9]_*.ipynb")
METADATA = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}}


def _lines(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(text.strip("\n"))}


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip("\n"))}


def stamp(nb):
    """Replace (any version of) the bootstrap cell, or insert it as the first code cell. Returns True if changed."""
    cells = nb["cells"]
    for c in cells:
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if c["cell_type"] == "code" and src.lstrip().startswith(BOOTSTRAP_MARKER_PREFIX):
            new = _lines(BOOTSTRAP_CELL)
            changed = c["source"] != new
            c["source"] = new
            return changed
    first_code = next((i for i, c in enumerate(cells) if c["cell_type"] == "code"), len(cells))
    cells.insert(first_code, code_cell(BOOTSTRAP_CELL))
    return True


def from_percent(text):
    cells, header, buf = [], None, []

    def flush():
        if header is None:
            return
        body = "\n".join(buf).strip("\n")
        if header.startswith("# %% [markdown]"):
            md = "\n".join(ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln for ln in body.split("\n"))
            cells.append(markdown_cell(md))
        elif header.startswith("# %% BOOTSTRAP"):
            cells.append(code_cell(BOOTSTRAP_CELL))
        elif body:
            cells.append(code_cell(body))

    for ln in text.split("\n"):
        if ln.startswith("# %%"):
            flush()
            header, buf = ln, []
        else:
            buf.append(ln)
    flush()
    return {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}


def write_nb(nb, path):
    for i, c in enumerate(nb["cells"]):                  # nbformat >= 4.5 requires unique cell ids (deterministic here)
        c["id"] = f"cell-{i:03d}"
    nb["nbformat"], nb["nbformat_minor"] = 4, max(int(nb.get("nbformat_minor", 5)), 5)
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--from-percent", nargs=2, metavar=("SRC_PY", "OUT_IPYNB"))
    a = ap.parse_args(argv)
    if a.from_percent:
        src, out = a.from_percent
        write_nb(from_percent(Path(src).read_text(encoding="utf-8")), out)
        print("wrote", out)
        return 0
    files = [Path(p) for p in a.paths] if a.paths else sorted({p for g in DEFAULT_GLOBS for p in REPO.glob(g)})
    for f in files:
        nb = json.loads(f.read_text(encoding="utf-8"))
        if stamp(nb):
            write_nb(nb, f)
            print("stamped", f.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
