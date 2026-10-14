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

from .memory import Memory
from .setup import setup, launch
from .logger import logger


def _safe_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0


async def run_kernel(dut, program: List[int], data: List[int], threads: int,
                     timeout_cycles: int = 20_000,
                     program_addr_bits: int = 8, program_data_bits: int = 16,
                     program_channels: int = 1,
                     data_addr_bits: int = 8, data_data_bits: int = 8,
                     data_channels: int = 4,
                     mem_latency: int = 0) -> dict:
    program_memory = Memory(dut, program_addr_bits, program_data_bits,
                            program_channels, "program", latency=mem_latency)
    data_memory = Memory(dut, data_addr_bits, data_data_bits,
                         data_channels, "data", latency=mem_latency)

    await setup(dut, program_memory, program, data_memory, data, threads)
    return await _run(dut, program_memory, data_memory, threads, timeout_cycles), data_memory


async def _run(dut, program_memory: Memory, data_memory: Memory, threads: int,
               timeout_cycles: int) -> dict:
    cycles = 0
    serviced_total = 0
    while _safe_int(dut.done.value) != 1:
        data_memory.run()
        program_memory.run()
        await ReadOnly()
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
    return metrics


@dataclass
class Launch:
    program: List[int]
    threads: int
    name: str = ""
    timeout_cycles: int = 20_000


async def run_launches(dut, image, launches: List[Launch], mem_latency: int = 0):
    """Run kernels back to back on one data memory loaded from an xf.image.Image.

    The GPU is reset before every launch; data memory persists. Returns
    (per-launch metrics, data memory).
    """
    channels = len(dut.data_mem_read_ready)
    bus = (len(dut.data_mem_read_data) // channels, len(dut.data_mem_read_address) // channels)
    if bus != (image.word_bits, image.addr_bits):
        raise ValueError(f"image is {image.word_bits}-bit x 2**{image.addr_bits}, "
                         f"DUT data memory is {bus[0]}-bit x 2**{bus[1]}")
    data_memory = Memory(dut, image.addr_bits, image.word_bits, channels, "data", latency=mem_latency)
    data_memory.load(image.words)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    results = []
    for i, ln in enumerate(launches):
        program_memory = Memory(dut, 8, 16, 1, "program", latency=mem_latency)
        data_memory._pending_read = [0] * data_memory.channels
        data_memory._pending_write = [0] * data_memory.channels
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
