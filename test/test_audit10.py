"""AUDIT-10 (performance under load) RTL probes. Each asserting probe states the behaviour the design should have;
the ones that fail today are AUDIT-10 findings, wrapped as strict xfails by xf/tests/test_audit10.py.

Run one probe on a build without the Makefile (AUDIT-10.md, "Running the probes"):
    COCOTB_TEST_MODULES=test.test_audit10 COCOTB_TEST_FILTER=<probe> \
        vvp -M $(cocotb-config --lib-dir) -m libcocotbvpi_icarus <build>/sim.vvp
`l1d_stale_fill` needs a build whose top level is `l1d_cache` (NUM_CONSUMERS=2, LINES=4). Probes return early
(pass) on builds whose parameters they do not target. AUDIT10_LATENCY sets the memory latency (default MEM_LATENCY).
"""
import logging
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

from xf.asm import assemble
from xf.isasim import run
from .helpers.runner import MEM_LATENCY, run_kernel

logger = logging.getLogger("cocotb.audit10")
LATENCY = int(os.environ.get("AUDIT10_LATENCY", MEM_LATENCY))


def is_gpu(dut):
    return hasattr(dut, "per_core_warp_issues")


def p(dut, name):
    return int(getattr(dut, name).value)


def iv(sig):
    try:
        return int(sig.value)
    except ValueError:   # X before reset
        return 0


def per_core(dut, sig):
    v, n = iv(sig), p(dut, "NUM_CORES")
    return [(v >> (32 * c)) & 0xFFFFFFFF for c in range(n)]


async def sample(dut, log, stop):
    """Every cycle: per-core warp issues and the core_start/core_done bits."""
    while not stop:
        await ReadOnly()
        log.append((per_core(dut, dut.per_core_warp_issues), iv(dut.core_start), iv(dut.core_done)))
        await RisingEdge(dut.clk)


# Compute-only: every thread runs N trips of a 3-instruction loop, then stores its trip count.
SPIN = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #1
    CONST R2, #{n}
    CONST R3, #0
loop:
    ADD R3, R3, R1
    CMP R3, R2
    BRn loop
    STR R0, R3
    RET
"""

# Load-heavy: trip i loads a[threadIdx + i] and accumulates it; the sum goes to 128 + global thread id.
LOADS = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #1
    CONST R2, #{n}
    CONST R3, #0
    CONST R5, #0
loop:
    ADD R6, %threadIdx, R3
    LDR R4, R6
    ADD R5, R5, R4
    ADD R3, R3, R1
    CMP R3, R2
    BRn loop
    CONST R7, #128
    ADD R7, R7, R0
    STR R7, R5
    RET
"""

# Long-running: N trips of a load and 4 ALU instructions; every warp issues 5N + 9 instructions.
LONG = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #1
    CONST R2, #{n}
    CONST R3, #0
    CONST R5, #0
loop:
    LDR R4, R0
    ADD R5, R5, R4
    ADD R3, R3, R1
    CMP R3, R2
    BRn loop
    CONST R7, #128
    ADD R7, R7, R0
    STR R7, R3
    RET
"""

# Loads, SMEM and a BAR every trip (the contract-12 counters under load).
BARLOOP = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #1
    CONST R2, #{n}
    CONST R3, #0
    CONST R5, #0
loop:
    ADD R6, %threadIdx, R3
    LDR R4, R6
    ADD R5, R5, R4
    STS %threadIdx, R5
    BAR
    ADD R3, R3, R1
    CMP R3, R2
    BRn loop
    CONST R7, #128
    ADD R7, R7, R0
    STR R7, R5
    RET
"""


@cocotb.test()
async def dispatch_spread(dut):
    """16 blocks of a compute loop on NUM_CORES >= 3: every core should run a fair share of the blocks.
    The program controller has one channel and claims fetchers in fixed index order, so the low cores win."""
    if not is_gpu(dut) or p(dut, "NUM_CORES") < 3:
        return
    cores, tpb = p(dut, "NUM_CORES"), p(dut, "THREADS_PER_BLOCK")
    blocks = min(255 // tpb, 4 * cores)
    log, stop = [], []
    cocotb.start_soon(sample(dut, log, stop))
    metrics, mem = await run_kernel(dut, assemble(SPIN.format(n=6)), [], blocks * tpb, timeout_cycles=200_000,
                                    mem_latency=LATENCY)
    stop.append(1)
    assert mem.memory[:blocks * tpb] == [6] * (blocks * tpb)
    starts, prev = [0] * cores, 0
    for _, s, _ in log:
        for c in range(cores):
            starts[c] += bool(s >> c & 1 and not prev >> c & 1)
        prev = s
    logger.info(f"dispatch_spread: L{LATENCY} {blocks} blocks, cycles {metrics['cycles']}, blocks per core {starts}, "
                f"issues per core {per_core(dut, dut.per_core_warp_issues)}")
    assert sum(starts) == blocks
    assert min(starts) >= blocks // cores // 2, f"blocks per core {starts}: the high cores are starved"


@cocotb.test()
async def data_fairness(dut):
    """Report: one block per core of a load loop; each core's finish cycle (fixed-priority data channels)."""
    if not is_gpu(dut):
        return
    cores, tpb = p(dut, "NUM_CORES"), p(dut, "THREADS_PER_BLOCK")
    n, data = 8, list(range(64))
    log, stop = [], []
    cocotb.start_soon(sample(dut, log, stop))
    metrics, mem = await run_kernel(dut, assemble(LOADS.format(n=n)), data, cores * tpb, timeout_cycles=200_000,
                                    mem_latency=LATENCY)
    stop.append(1)
    mask = (1 << p(dut, "WORD_BITS")) - 1
    assert mem.memory[128:128 + cores * tpb] == [sum(data[t % tpb + i] for i in range(n)) & mask
                                                 for t in range(cores * tpb)]
    done_at = [next(i for i, (_, _, d) in enumerate(log) if d >> c & 1) for c in range(cores)]
    logger.info(f"data_fairness: L{LATENCY} cycles {metrics['cycles']}, done per core {done_at}, requests "
                f"{metrics['data_mem_requests']}")


@cocotb.test()
async def scaling(dut):
    """Report: cycles of 16 blocks of the compute loop and of the load loop, to compare NUM_CORES builds."""
    if not is_gpu(dut):
        return
    tpb = p(dut, "THREADS_PER_BLOCK")
    blocks = min(16, 255 // tpb)
    busy, stop = [0, 0, 0], []

    async def util():
        while not stop:
            await ReadOnly()
            busy[0] += 1
            busy[1] += iv(dut.program_mem_read_valid) & 1
            busy[2] += bin(iv(dut.data_mem_read_valid) | iv(dut.data_mem_write_valid)).count("1")
            await RisingEdge(dut.clk)
    cocotb.start_soon(util())
    m1, _ = await run_kernel(dut, assemble(SPIN.format(n=6)), [], blocks * tpb, timeout_cycles=200_000,
                             mem_latency=LATENCY)
    m2, _ = await run_kernel(dut, assemble(LOADS.format(n=8)), list(range(64)), blocks * tpb,
                             timeout_cycles=200_000, mem_latency=LATENCY)
    stop.append(1)
    ch = len(dut.data_mem_read_ready)
    logger.info(f"scaling: NUM_CORES {p(dut, 'NUM_CORES')} W {p(dut, 'WARPS_PER_CORE')} L{LATENCY}: "
                f"spin {m1['cycles']} cycles, loads {m2['cycles']} cycles; over both, program channel busy "
                f"{busy[1] / busy[0]:.0%}, data channels busy {busy[2] / busy[0] / ch:.0%}")


@cocotb.test()
async def gto_long_run_fairness(dut):
    """WARPS_PER_CORE >= 4, one block, a load loop of 5N + 9 > 254 instructions per warp. ADR-014: GTO's oldest is
    the smallest age and ages saturate at 254, after which every tie goes to the lowest warp. Asserts the last warp
    finishes within 1.1x of the first (LRR, GTLRR, and GTO below 254 issues per warp, finish within 1%)."""
    if not is_gpu(dut) or p(dut, "WARPS_PER_CORE") < 4:
        return
    w, tpb = p(dut, "WARPS_PER_CORE"), p(dut, "THREADS_PER_BLOCK")
    n = int(os.environ.get("AUDIT10_TRIPS", 80))
    sched = dut.cores[0].core_instance.scheduler_instance
    fin, stop = {}, []

    async def watch():
        cyc = 0
        while not stop:
            await ReadOnly()
            v = iv(sched.warp_done)
            for i in range(w):
                if v >> i & 1 and i not in fin:
                    fin[i] = cyc
            await RisingEdge(dut.clk)
            cyc += 1
    cocotb.start_soon(watch())
    metrics, mem = await run_kernel(dut, assemble(LONG.format(n=n)), list(range(tpb)), tpb,
                                    timeout_cycles=400_000, mem_latency=LATENCY)
    stop.append(1)
    assert mem.memory[128:128 + tpb] == [n] * tpb
    order = [fin[i] for i in range(w)]
    logger.info(f"gto_long_run_fairness: policy {p(dut, 'WARP_POLICY')} L{LATENCY} trips {n}, cycles "
                f"{metrics['cycles']}, warp finish cycles {order}")
    assert max(order) <= 1.1 * min(order), f"warp finish cycles {order}: some warps ran only after others finished"


@cocotb.test()
async def counters_under_load(dut):
    """Two blocks per core with loads, SMEM and a BAR per trip: memory equals xf.isasim, and warp_issues_total
    equals the model's steps (contract 12); check_counters runs inside run_kernel."""
    if not is_gpu(dut):
        return
    cores, tpb, w = p(dut, "NUM_CORES"), p(dut, "THREADS_PER_BLOCK"), p(dut, "WARPS_PER_CORE")
    n, data = 6, list(range(64))
    prog = assemble(BARLOOP.format(n=n))
    threads = min(255 // tpb * tpb, 2 * cores * tpb)
    model = run(prog, data, threads, word_bits=p(dut, "WORD_BITS"), addr_bits=p(dut, "DATA_MEM_ADDR_BITS"),
                threads_per_block=tpb, warps_per_core=w)
    metrics, mem = await run_kernel(dut, prog, data, threads, timeout_cycles=400_000, mem_latency=LATENCY)
    issues, stalls = p(dut, "warp_issues_total"), p(dut, "bar_stall_cycles_total")
    logger.info(f"counters_under_load: W {w} policy {p(dut, 'WARP_POLICY')} L{LATENCY}: cycles {metrics['cycles']}, "
                f"issues {issues} (model {model.steps}), bar stalls {stalls}, requests {metrics['data_mem_requests']}")
    assert mem.memory[128:128 + threads] == model.memory[128:128 + threads]
    assert issues == model.steps
    assert w > 1 or stalls == 0


@cocotb.test()
async def l1d_stale_fill(dut):
    """Unit probe on an `l1d_cache` top (NUM_CONSUMERS=2): lane 0 misses on X while lane 1 writes X through, and
    the downstream read and write complete in the same cycle (the read returning the pre-write value). Memory now
    holds the write, so lane 0's next read of X must return it, not the line filled with the old value."""
    if is_gpu(dut) or not hasattr(dut, "consumer_read_valid"):
        return
    X, NEW = 5, 7
    ab, db = len(dut.consumer_read_address) // 2, len(dut.consumer_read_data) // 2
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for s in (dut.consumer_read_valid, dut.consumer_write_valid, dut.mem_read_ready, dut.mem_write_ready,
              dut.consumer_read_address, dut.consumer_write_address, dut.consumer_write_data, dut.mem_read_data):
        s.value = 0
    dut.telemetry_reset.value = 0
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    dut.consumer_read_valid.value = 0b01                     # lane 0 reads X
    dut.consumer_read_address.value = X
    dut.consumer_write_valid.value = 0b10                    # lane 1 writes NEW to X
    dut.consumer_write_address.value = X << ab
    dut.consumer_write_data.value = NEW << db
    await RisingEdge(dut.clk)                                # lane 0 misses, lane 1 starts write-through
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert iv(dut.mem_read_valid) & 1 and iv(dut.mem_write_valid) & 2
    await RisingEdge(dut.clk)
    dut.mem_read_ready.value = 0b01                          # memory serves the read (old value 0)
    dut.mem_read_data.value = 0
    dut.mem_write_ready.value = 0b10                         # and the write, in the same cycle
    await RisingEdge(dut.clk)
    dut.mem_read_ready.value = 0
    dut.mem_write_ready.value = 0
    await RisingEdge(dut.clk)
    dut.consumer_read_valid.value = 0
    dut.consumer_write_valid.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.consumer_read_valid.value = 0b01                     # lane 0 reads X again, long after both completed
    got = None
    for _ in range(6):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if iv(dut.consumer_read_ready) & 1:
            got = iv(dut.consumer_read_data) & ((1 << db) - 1)
            break
    hit = iv(dut.cache_hits_total)
    logger.info(f"l1d_stale_fill: second read of X returned {got} (hits {hit}, memory holds {NEW})")
    assert got == NEW, f"L1-D returned {got} for X after the write of {NEW} completed (stale fill, hits={hit})"
