"""Collects the hand-encoded programs in test/test_*.py, keyed `<module>.<scope>[.<k>]`."""
import ast
import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_WORD = re.compile(r"^(0b[01_]+|encode\()")


def hand_encoded():
    progs = {}
    for path in sorted((ROOT / "test").glob("test_*.py")):
        mod = importlib.import_module(f"test.{path.stem}")
        src = path.read_text()
        found = {}

        def visit(node, scope):
            for child in ast.iter_child_nodes(node):
                s = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
                if isinstance(child, ast.List) and child.elts and all(
                        _WORD.match(ast.get_source_segment(src, e)) for e in child.elts):
                    found.setdefault(s, []).append(eval(compile(ast.Expression(child), str(path), "eval"), vars(mod)))
                visit(child, s)

        visit(ast.parse(src), "module")
        for name in dir(mod):
            if name.startswith("build_"):
                found.setdefault(name, []).append(getattr(mod, name)())
        for scope, lists in found.items():
            for k, words in enumerate(lists):
                progs[f"{path.stem}.{scope}" + (f".{k}" if len(lists) > 1 else "")] = words
    return progs
