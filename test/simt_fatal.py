"""Kernels that must end in simt_stack's $fatal (`make test_divergence`); SIMT_FATAL picks one. Each test passes if
the simulation survives, so the make target, which expects a failure, then fails."""
import os

import cocotb

from xf.asm import assemble
from .helpers.runner import run_kernel
from .test_divergence import CAPPED_LOOP, M1

PROGRAMS = {
    "overflow": (CAPPED_LOOP.format(cap=5), 8),   # 5 divergent exits at 8 lanes: 10 entries
    "nosync": (M1.format(sync=""), 4),            # a divergent branch with no pending SYNC
}


@cocotb.test()
async def simt_fatal(dut):
    src, threads = PROGRAMS[os.environ["SIMT_FATAL"]]
    await run_kernel(dut, assemble(src), [0] * 8, threads)
