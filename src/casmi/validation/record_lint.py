"""Static lint: scan a notebook's code cells for `casmi.results.record(...)` calls whose
`value` argument is a bare literal constant (`True`/`False`/an int/a float/a string) rather
than a runtime-computed expression. The entire point of `record()` is that every gate-evidence
value is derived from execution, never hand-typed to make a gate pass (spec section 11) --
`record("s1", "c1.pass", True)` defeats that even though it "looks like" a normal call.
"""
import ast
import json


def _iter_code_cell_sources(notebook_path):
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            yield "".join(source) if isinstance(source, list) else source


def find_literal_record_violations(notebook_path):
    """Returns a list of `{"lineno", "source_snippet"}` for every `record(...)` call whose
    `value` argument (3rd positional argument, or the `value=` keyword) is a bare
    `ast.Constant`. A cell with a syntax error is reported as its own violation (never silently
    skipped), so a lint failure can't be masked by an unrelated parse error."""
    violations = []
    for cell_source in _iter_code_cell_sources(notebook_path):
        if not cell_source.strip():
            continue
        try:
            tree = ast.parse(cell_source)
        except SyntaxError as e:
            violations.append({"lineno": getattr(e, "lineno", -1), "source_snippet": f"SYNTAX ERROR: {e}"})
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "record"):
                continue
            value_node = node.args[2] if len(node.args) >= 3 else None
            for kw in node.keywords:
                if kw.arg == "value":
                    value_node = kw.value
            if value_node is not None and isinstance(value_node, ast.Constant):
                snippet = ast.unparse(node) if hasattr(ast, "unparse") else "<record(...) call with a literal value>"
                violations.append({"lineno": getattr(node, "lineno", -1), "source_snippet": snippet})
    return violations
