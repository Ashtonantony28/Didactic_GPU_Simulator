"""Deliberately failing module for `make harness_selftest`."""
import cocotb


@cocotb.test()
async def deliberately_fails(dut):
    assert False, "this failure must fail the make target"
