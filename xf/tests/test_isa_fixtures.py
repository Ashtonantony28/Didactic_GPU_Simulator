"""Contract 11: xf.isasim still reproduces every committed ISA fixture (test/fixtures/isa)."""
import pytest

from xf.isa_fixtures import FIXTURE_DIR, WIDTHS, load, straight_line
from xf.isasim import run

FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_model_reproduces(path):
    fx = load(path)
    r = run(fx["program"], fx["data"], fx["threads"], **fx["params"])
    assert not r.divergent
    assert r.memory == fx["memory"] + [0] * (len(r.memory) - len(fx["memory"]))


def test_every_width_covered():
    assert {load(p)["params"]["word_bits"] for p in FIXTURES} == set(WIDTHS)


@pytest.mark.parametrize("w", WIDTHS)
def test_straight_line_shape(w):
    prog, data, threads = straight_line(0, w)
    assert len(prog) == 92 and threads == 8 and len(data) == 88
