"""Data memory controller (SG18, ADR-013): idle channels claim distinct consumers, and the telemetry
counters are exact (contract 12). runner.check_counters compares them with the memory model on every run."""
import cocotb

from xf.asm import assemble
from .helpers.logger import logger
from .helpers.memory import Memory
from .helpers.runner import MEM_LATENCY, _run, data_bus
from .helpers.setup import setup

# Thread i loads a[i] and stores it to n + i.
COPY = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    LDR R1, R0
    CONST R2, #{n}
    ADD R2, R2, R0
    STR R2, R1
    RET
"""
# Every thread loads a[0] and stores it to n + i.
BROADCAST = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #0
    LDR R1, R1
    CONST R2, #{n}
    ADD R2, R2, R0
    STR R2, R1
    RET
"""
N = 8


async def probe(dut, program, data, threads):
    """Run a kernel, recording the addresses on the data channels every cycle."""
    bits, addr = data_bus(dut)
    pm = Memory(dut, 8, 16, 1, "program", latency=MEM_LATENCY)
    dm = Memory(dut, addr, bits, len(dut.data_mem_read_ready), "data", latency=MEM_LATENCY)
    trace, run = [], dm.run

    def traced():
        n = run()
        trace.append((dm.read_in_flight, dm.write_in_flight))
        return n

    dm.run = traced
    await setup(dut, pm, assemble(program), dm, data, threads)
    metrics = await _run(dut, pm, dm, threads, 20_000)
    return metrics, dm, trace


def in_flight(trace, kind):
    """Per cycle, the addresses with valid high on the read (0) or write (1) channels."""
    return [[a for a in cycle[kind] if a is not None] for cycle in trace]


@cocotb.test()
async def test_distinct_channels(dut):
    """8 threads, distinct addresses: every channel busy with a different consumer, nothing issued twice."""
    channels = len(dut.data_mem_read_ready)
    warps = int(dut.WARPS_PER_CORE.value)
    lanes = int(dut.THREADS_PER_BLOCK.value) // warps
    data = [(7 * i + 3) & 0xFF for i in range(N)]
    metrics, dm, trace = await probe(dut, COPY.format(n=N), data, N)
    logger.info(f"distinct-address probe: {metrics['cycles']} cycles")
    for i in range(N):
        assert dm.memory[N + i] == data[i], f"thread {i}: stored {dm.memory[N + i]}, expected {data[i]}"
    for kind, name, want in ((0, "read", range(N)), (1, "write", range(N, 2 * N))):
        cycles = in_flight(trace, kind)
        dup = [c for c, a in enumerate(cycles) if len(a) != len(set(a))]
        assert not dup, f"{len(dup)} cycles with duplicate {name} transactions, first {dup[0]}: {cycles[dup[0]]}"
        most = max(len(set(a)) for a in cycles)
        if warps == 1:
            assert most == channels, f"at most {most} distinct {name} addresses in flight, expected {channels}"
        else:   # a warp issues only its lanes' loads at once (ADR-014)
            assert most >= min(channels, lanes), f"at most {most} distinct {name} addresses in flight, {lanes} lanes"
        served = sorted(a for _, a in (dm.reads if kind == 0 else dm.writes))
        assert served == list(want), f"{name} transactions {served}, expected each of {list(want)} once"
    assert metrics["data_mem_requests"] == 2 * N, metrics
    assert metrics["data_mem_requests_coalesced"] == 0, metrics


@cocotb.test()
async def test_same_address(dut):
    """8 threads load one address: one transaction per consumer, or fewer with COALESCE, never more."""
    metrics, dm, _ = await probe(dut, BROADCAST.format(n=N), [42], N)
    for i in range(N):
        assert dm.memory[N + i] == 42, f"thread {i}: stored {dm.memory[N + i]}"
    reads, coalesced = len(dm.reads), metrics["data_mem_requests_coalesced"]
    logger.info(f"same-address probe: {metrics['cycles']} cycles, {reads} reads, {coalesced} coalesced")
    if int(dut.DCACHE_LINES.value) == 0:
        assert reads + coalesced == N, f"{reads} read transactions + {coalesced} coalesced != {N} loads"
        assert (coalesced > 0) == bool(int(dut.COALESCE.value)), f"coalesced = {coalesced}"
    assert len(dm.writes) == N and metrics["data_mem_requests"] == reads + N, metrics


@cocotb.test()
async def test_icache_totals(dut):
    """One block per core: every core fetches, and the totals are the sum over all NUM_CORES."""
    cores, tpb = int(dut.NUM_CORES.value), int(dut.THREADS_PER_BLOCK.value)
    threads = cores * tpb
    metrics, dm, _ = await probe(dut, COPY.format(n=threads), list(range(threads)), threads)
    v = int(dut.per_core_icache_misses.value), int(dut.per_core_icache_hits.value)
    per_core = [((v[0] >> 32 * c) & 0xFFFFFFFF) + ((v[1] >> 32 * c) & 0xFFFFFFFF) for c in range(cores)]
    assert all(per_core), f"per-core fetches {per_core}: every core should run a block"
    assert metrics["icache_hits"] + metrics["icache_misses"] == sum(per_core), (metrics, per_core)
