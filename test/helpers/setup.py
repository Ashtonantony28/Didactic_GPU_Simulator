from typing import List
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from .memory import Memory


async def setup(dut, program_memory: Memory, program: List[int],
                data_memory: Memory, data: List[int], threads: int):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await launch(dut, program_memory, program, data_memory, data, threads)


async def launch(dut, program_memory: Memory, program: List[int],
                 data_memory: Memory, data: List[int], threads: int):
    """Reset the GPU, load program (and data, unless None), set the thread count, start."""
    dut.reset.value = 1
    dut.start.value = 0
    dut.device_control_write_enable.value = 0
    dut.device_control_data.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0

    program_memory.load(program)
    if data is not None:
        data_memory.load(data)

    dut.device_control_write_enable.value = 1
    dut.device_control_data.value = threads
    await RisingEdge(dut.clk)
    dut.device_control_write_enable.value = 0

    dut.start.value = 1
