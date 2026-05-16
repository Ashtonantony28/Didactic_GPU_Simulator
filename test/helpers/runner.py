"""Common driver loop + bench stats for kernel tests.

Returns a dict of metrics so each test can persist them into bench/.
"""
import json
import os
from typing import List, Optional
import cocotb
from cocotb.triggers import RisingEdge, ReadOnly

from .memory import Memory
from .setup import setup
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
    }
    logger.info(f"Kernel finished: {metrics}")
    return metrics, data_memory


def write_bench_result(name: str, metrics: dict):
    os.makedirs("bench/results", exist_ok=True)
    path = f"bench/results/{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info(f"Wrote {path}")
