"""Common driver loop + bench stats for kernel tests.

Returns a dict of metrics; bench/run.py sets BENCH_RESULTS=1 to persist them into bench/results/.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

from xf.kernels import check_block
from .memory import Memory
from .setup import setup, launch
from .logger import logger

# Memory latency in cycles for every test; `make matrix` sets it for one configuration.
MEM_LATENCY = int(os.environ.get("MEM_LATENCY", 0))


def _safe_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0


async def run_kernel(dut, program: List[int], data: List[int], threads: int,
                     timeout_cycles: int = 20_000,
                     program_addr_bits: int = 8, program_data_bits: int = 16,
                     program_channels: int = 1,
                     data_addr_bits: Optional[int] = None, data_data_bits: Optional[int] = None,
                     data_channels: int = 4,
                     mem_latency: int = MEM_LATENCY, trace: Optional[dict] = None) -> dict:
    bits, addr = data_bus(dut)
    data_data_bits = data_data_bits or bits
    data_addr_bits = data_addr_bits or addr
    program_memory = Memory(dut, program_addr_bits, program_data_bits,
                            program_channels, "program", latency=mem_latency)
    data_memory = Memory(dut, data_addr_bits, data_data_bits,
                         data_channels, "data", latency=mem_latency)

    await setup(dut, program_memory, program, data_memory, data, threads)
    return await _run(dut, program_memory, data_memory, threads, timeout_cycles, trace), data_memory


def data_bus(dut):
    """(word bits, address bits) of the DUT's data memory bus."""
    channels = len(dut.data_mem_read_ready)
    return len(dut.data_mem_read_data) // channels, len(dut.data_mem_read_address) // channels


async def _run(dut, program_memory: Memory, data_memory: Memory, threads: int,
               timeout_cycles: int, trace: Optional[dict] = None) -> dict:
    """trace, if given, collects each block's warp issue order ({block: [warp, ...]}, xf.isasim.run_trace)."""
    cycles = 0
    reads0, writes0 = len(data_memory.reads), len(data_memory.writes)
    ready = dut.data_memory_controller.consumer_read_ready
    prev_ready = consumer_reads = 0
    cores = [dut.cores[c].core_instance for c in range(int(dut.NUM_CORES.value))] if trace is not None else []
    while _safe_int(dut.done.value) != 1:
        data_memory.run()
        program_memory.run()
        await ReadOnly()
        for c in cores:
            s = c.scheduler_instance
            if _safe_int(s.issue.value):
                trace.setdefault(_safe_int(c.block_id.value), []).append(_safe_int(s.selected_warp.value))
        r = _safe_int(ready.value)
        consumer_reads += bin(r & ~prev_ready).count("1")
        prev_ready = r
        await RisingEdge(dut.clk)
        cycles += 1
        if cycles >= timeout_cycles:
            raise TimeoutError(f"Kernel did not complete in {timeout_cycles} cycles")

    metrics = {
        "cycles": cycles,
        "threads": threads,
        "data_mem_requests": _safe_int(dut.data_mem_requests.value),
        "data_mem_requests_coalesced": _safe_int(dut.data_mem_requests_coalesced.value),
        "icache_hits": _safe_int(dut.icache_hits_total.value),
        "icache_misses": _safe_int(dut.icache_misses_total.value),
        "dcache_hits": _safe_int(dut.dcache_hits_total.value),
        "dcache_misses": _safe_int(dut.dcache_misses_total.value),
    }
    logger.info(f"Kernel finished: {metrics}")
    check_counters(dut, metrics, len(data_memory.reads) - reads0, len(data_memory.writes) - writes0, consumer_reads)
    return metrics


def check_counters(dut, metrics: dict, reads: int, writes: int, consumer_reads: int):
    """Contract 12: telemetry equals what the memory model and the controller's consumer handshakes saw."""
    assert metrics["data_mem_requests"] == reads + writes, \
        f"data_mem_requests {metrics['data_mem_requests']} != {reads} reads + {writes} writes serviced"
    assert metrics["data_mem_requests_coalesced"] == consumer_reads - reads, \
        f"data_mem_requests_coalesced {metrics['data_mem_requests_coalesced']} != " \
        f"{consumer_reads} consumer reads - {reads} read transactions"
    cores = len(dut.per_core_icache_hits) // 32
    for key, sig in (("icache_hits", dut.per_core_icache_hits), ("icache_misses", dut.per_core_icache_misses)):
        v = _safe_int(sig.value)
        per_core = [(v >> (32 * c)) & 0xFFFFFFFF for c in range(cores)]
        assert metrics[key] == sum(per_core), f"{key} total {metrics[key]} != per-core sum {per_core}"
    fetches, issues = metrics["icache_hits"] + metrics["icache_misses"], _safe_int(dut.warp_issues_total.value)
    if _safe_int(dut.WARPS_PER_CORE.value) == 1:   # a yielded warp re-fetches (ADR-014)
        assert fetches == issues, f"{fetches} I-cache hits + misses != {issues} warp issues"
    assert _safe_int(dut.ICACHE_LINES.value) or not metrics["icache_hits"], f"{metrics['icache_hits']} hits, no I-cache"


@dataclass
class Launch:
    program: List[int]
    threads: int
    name: str = ""
    timeout_cycles: int = 20_000
    max_block: Optional[int] = None   # largest supported blockDim; defaults to program.max_block (xf.kernels)

    def __post_init__(self):
        if self.max_block is None:
            self.max_block = getattr(self.program, "max_block", None)


async def run_launches(dut, image, launches: List[Launch], mem_latency: int = MEM_LATENCY):
    """Run kernels back to back on one data memory loaded from an xf.image.Image.

    The GPU is reset before every launch; data memory persists. Returns
    (per-launch metrics, data memory). Raises ValueError, before any launch runs, when the DUT's
    THREADS_PER_BLOCK is one a launch's kernel does not support (xf.kernels.check_block).
    """
    channels = len(dut.data_mem_read_ready)
    bus = data_bus(dut)
    if bus != (image.word_bits, image.addr_bits):
        raise ValueError(f"image is {image.word_bits}-bit x 2**{image.addr_bits}, "
                         f"DUT data memory is {bus[0]}-bit x 2**{bus[1]}")
    tpb = int(dut.THREADS_PER_BLOCK.value)
    for ln in launches:
        check_block(ln.max_block, tpb, ln.name)
    data_memory = Memory(dut, image.addr_bits, image.word_bits, channels, "data", latency=mem_latency)
    data_memory.load(image.words)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    results = []
    for i, ln in enumerate(launches):
        program_memory = Memory(dut, 8, 16, 1, "program", latency=mem_latency)
        data_memory._pending_read = [0] * data_memory.channels
        data_memory._pending_write = [0] * data_memory.channels
        data_memory._served_read = [False] * data_memory.channels
        data_memory._served_write = [False] * data_memory.channels
        await launch(dut, program_memory, ln.program, data_memory, None, ln.threads)
        metrics = await _run(dut, program_memory, data_memory, ln.threads, ln.timeout_cycles)
        results.append({"launch": i, "name": ln.name, **metrics})
    dut.start.value = 0
    return results, data_memory


def write_bench_result(name: str, metrics: dict):
    if not os.environ.get("BENCH_RESULTS"):
        return
    os.makedirs("bench/results", exist_ok=True)
    path = f"bench/results/{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info(f"Wrote {path}")
