"""test_isa_ext at the xf widths (WORD_BITS=32, 4096-word data memory)."""
import cocotb

from .helpers.runner import data_bus
from .test_isa_ext import test_cmp_unchanged, test_sra_max  # noqa: F401


@cocotb.test()
async def test_xf_bus_width(dut):
    assert data_bus(dut) == (32, 12)
