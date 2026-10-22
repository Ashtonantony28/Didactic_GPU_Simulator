"""Phase-11 bench sweep (SG23): run named kernels at several memory latencies on one build.

bench/run.py --phase 11 builds each configuration and runs this module with BENCH_KERNELS (comma-separated names;
default all), BENCH_LATENCIES (comma-separated cycles, default 0; both memories, or data memory only with
BENCH_DATA_ONLY=1) and BENCH_OUT (JSON path, optional). Every run's memory is checked against xf.isasim at the DUT's
parameters; metrics, warp issues and barrier stalls go to BENCH_OUT.
"""
import json
import os

import cocotb
from cocotb.clock import Clock

from xf.asm import assemble
from xf.isasim import run
from .helpers.memory import Memory
from .helpers.runner import _run, data_bus
from .helpers.setup import launch
from . import test_matadd, test_matmul, test_vector_dot, test_conv1d, test_transpose, test_divergent_relu, \
    test_tiled_matmul

# test_relu's bench run uses this copy kernel, not its module-level PROGRAM.
RELU = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #0
    CONST R2, #8
    ADD R4, R1, R0
    LDR R4, R4
    ADD R7, R2, R0
    STR R7, R4
    RET
"""
# Multi-warp kernels: 32 threads, thread i reads a[i + 32j] for j < 4 and stores to 128 + i.
# STREAM adds each load straight into the sum; POLY does 8 ALU instructions per load.
STREAM = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #32
    CONST R2, #4
    CONST R3, #1
    CONST R4, #0
    CONST R6, #0
    ADD R5, R0, R6
loop:
    LDR R7, R5
    ADD R4, R4, R7
    ADD R5, R5, R1
    SUB R2, R2, R3
    CMP R2, R6
    BRn loop
    CONST R8, #128
    ADD R8, R8, R0
    STR R8, R4
    RET
"""
POLY = STREAM.replace("    ADD R4, R4, R7\n", """    MUL R9, R7, R7
    ADD R9, R9, R3
    MUL R9, R9, R7
    ADD R9, R9, R1
    MUL R9, R9, R7
    ADD R9, R9, R3
    ADD R4, R4, R9
""")
# SIMT kernels, 8 threads: thread i writes in[i] + 1 (i even) or in[i] - 1 (i odd) with an if/else
# (divergent_relu does the same with a mask blend), and thread i sums i % 4 + 1 trips (LANE_LOOP) or 4 (UNIFORM_LOOP).
BRANCH_RELU = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    LDR R5, R0
    CONST R3, #1
    CONST R4, #2
    DIV R6, R0, R4
    MUL R6, R6, R4
    SUB R6, R0, R6
    CONST R7, #0
    SYNC join
    CMP R6, R7
    BRz even
    SUB R5, R5, R3
    BRnzp join
even:
    ADD R5, R5, R3
join:
    CONST R8, #8
    ADD R8, R8, R0
    STR R8, R5
    RET
"""
LOOP = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #1
    CONST R6, #0
    CONST R4, #4
    {trips}
    CONST R3, #0
loop:
    SYNC done
    ADD R3, R3, R2
    SUB R2, R2, R1
    CMP R2, R6
    BRn loop
done:
    CONST R8, #8
    ADD R8, R8, R0
    STR R8, R3
    RET
"""
LANE_LOOP = LOOP.format(trips="DIV R2, %threadIdx, R4\n    MUL R2, R2, R4\n    SUB R2, %threadIdx, R2\n    ADD R2, R2, R1")
UNIFORM_LOOP = LOOP.format(trips="ADD R2, R4, R6")

PARITY = [10, 20, 30, 40, 50, 60, 70, 80]
INPUT = [(7 * a + 3) & 0xFF for a in range(128)]


def _legacy(mod):
    return mod.PROGRAM, mod.DATA, mod.THREADS


KERNELS = {
    "matadd": _legacy(test_matadd),
    "matmul": _legacy(test_matmul),
    "vector_dot": _legacy(test_vector_dot),
    "relu": (assemble(RELU), [3, 0, 7, 1, 9, 2, 4, 5], 8),
    "conv1d": _legacy(test_conv1d),
    "transpose": _legacy(test_transpose),
    "divergent_relu": _legacy(test_divergent_relu),
    "tiled_matmul": _legacy(test_tiled_matmul),
    "stream": (assemble(STREAM), INPUT, 32),
    "poly": (assemble(POLY), INPUT, 32),
    "branch_relu": (assemble(BRANCH_RELU), PARITY, 8),
    "lane_loop": (assemble(LANE_LOOP), [], 8),
    "uniform_loop": (assemble(UNIFORM_LOOP), [], 8),
}


@cocotb.test()
async def bench_sweep(dut):
    names = os.environ.get("BENCH_KERNELS", ",".join(KERNELS)).split(",")
    latencies = [int(x) for x in os.environ.get("BENCH_LATENCIES", "0").split(",")]
    bits, addr = data_bus(dut)
    tpb, warps = int(dut.THREADS_PER_BLOCK.value), int(dut.WARPS_PER_CORE.value)
    simt = bool(int(dut.SIMT_STACK.value))
    data_only = os.environ.get("BENCH_DATA_ONLY") == "1"
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rows = {}
    for name in names:
        program, data, threads = KERNELS[name]
        want = run(program, data, threads, word_bits=bits, addr_bits=addr, threads_per_block=tpb,
                   warps_per_core=warps, simt=simt)
        for lat in latencies:
            pm = Memory(dut, 8, 16, 1, "program", latency=0 if data_only else lat)
            dm = Memory(dut, addr, bits, len(dut.data_mem_read_ready), "data", latency=lat)
            await launch(dut, pm, program, dm, data, threads)
            metrics = await _run(dut, pm, dm, threads, 200_000)
            assert dm.memory == want.memory, f"{name} L{lat}: memory differs from xf.isasim"
            issues = int(dut.warp_issues_total.value)
            assert issues == want.steps, f"{name} L{lat}: warp_issues_total {issues} != model steps {want.steps}"
            rows[f"{name}/L{lat}"] = {**metrics, "warp_issues": issues,
                                      "bar_stall_cycles": int(dut.bar_stall_cycles_total.value)}
    dut.start.value = 0
    if os.environ.get("BENCH_OUT"):
        with open(os.environ["BENCH_OUT"], "w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)
