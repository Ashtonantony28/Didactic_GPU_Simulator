"""Every `bench/<file>.json#<key>` cited in docs/final-report.md names an existing bench row (SG23)."""
import importlib.util

import pytest

from xf.tests.legacy import ROOT

_spec = importlib.util.spec_from_file_location("bench_run", ROOT / "bench" / "run.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_final_report_citations_resolve():
    assert bench.check_citations() == []


@pytest.mark.parametrize("cite, ok", [
    ("phase-11-all.json#kernels/{default,all}/matmul/L{0,10}", True),
    ("phase-11-all.json#kernels/default/matmul/L{0,7}", False),
    ("phase-5-coalesce.json#matmul", True),
    ("phase-5-coalesce.json#tiled_matmul", False),
    ("xf-matmul.json#4x8x16/tiled/COALESCE=1+DCACHE_LINES=16", True),
    ("phase-99.json#matmul", False),
])
def test_resolve(cite, ok):
    file, key = cite.split("#")
    assert all(bench.resolve(file, k) for k in bench.expand(key)) == ok
