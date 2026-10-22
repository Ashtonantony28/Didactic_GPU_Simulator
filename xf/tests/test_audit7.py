"""AUDIT-7 H2 / SG16: every golden segment of both blocks keeps its held-out gain and RMS error (contract 8)."""

import functools

import pytest

from xf import export, fixture


@functools.cache
def _stats(nanogpt_dir, block):
    return export.heldout(nanogpt_dir, block, fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")["image"])[2]


@pytest.mark.parametrize("block", export.SEGMENTS)
def test_every_segment_has_bounds(nanogpt_dir, block):
    assert set(_stats(nanogpt_dir, block)) == set(export.SEGMENTS[block])


@pytest.mark.parametrize("block, seg", [(b, s) for b, segs in export.SEGMENTS.items() for s in segs])
def test_segment_gain_and_rms_on_heldout(nanogpt_dir, block, seg):
    (g, r), (g0, r0) = _stats(nanogpt_dir, block)[seg], export.SEGMENTS[block][seg]
    assert abs(g - g0) <= export.GAIN_TOL, f"{block} {seg}: gain {g:.4f}, HEAD {g0} ± {export.GAIN_TOL}"
    assert r <= r0 * export.RMS_TOL, f"{block} {seg}: RMS {r:.3f} LSB, HEAD {r0} × {export.RMS_TOL}"
