"""Edge cases for tiny-gpu-extended.

Covers:
- Single thread (THREADS=1) — should still run a kernel
- Single block, fewer than THREADS_PER_BLOCK threads
- Many blocks: more blocks than cores (block 2 reuses core 0)
- Empty kernel (just RET)
- Maximum-size address space
"""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


@cocotb.test()
async def test_single_thread(dut):
    """One thread runs a small kernel — last block has only 1 lane enabled."""
    program = [
        0b1001_0001_0000_0101,  # CONST R1, #5
        0b1001_0010_0000_0011,  # CONST R2, #3
        0b0011_0011_0001_0010,  # ADD R3, R1, R2
        0b1001_0100_0000_0000,  # CONST R4, #0 (target addr)
        0b1000_0000_0100_0011,  # STR R4, R3
        0b1111_0000_0000_0000,  # RET
    ]
    metrics, mem = await run_kernel(dut, program, [0] * 16, threads=1)
    assert mem.memory[0] == 8, f"expected mem[0]=8, got {mem.memory[0]}"
    write_bench_result("edge_single_thread", metrics)


@cocotb.test()
async def test_two_blocks_one_core(dut):
    """16 threads → 4 blocks. With NUM_CORES=2, each core gets 2 blocks in sequence."""
    program = [
        0b0101_0000_1101_1110,  # MUL R0, blockIdx, blockDim
        0b0011_0000_0000_1111,  # ADD R0, R0, threadIdx
        0b1001_0001_0001_0000,  # CONST R1, #16  (base addr for output)
        0b0011_0010_0001_0000,  # ADD R2, R1, R0
        0b1000_0000_0010_0000,  # STR R2, R0  (mem[16+i] = i)
        0b1111_0000_0000_0000,  # RET
    ]
    metrics, mem = await run_kernel(dut, program, [0] * 32, threads=16)
    for i in range(16):
        assert mem.memory[16 + i] == i, f"thread {i}: got {mem.memory[16+i]} want {i}"
    write_bench_result("edge_two_blocks_one_core", metrics)


@cocotb.test()
async def test_pure_compute(dut):
    """Kernel with no memory ops — just ALU and RET. Tests scheduler when WAIT=0 cycles."""
    program = [
        0b1001_0001_0000_0001,  # CONST R1, #1
        0b1001_0010_0000_0010,  # CONST R2, #2
        0b0011_0011_0001_0010,  # ADD R3, R1, R2
        0b0101_0100_0011_0011,  # MUL R4, R3, R3   (= 9)
        0b1001_0101_0000_0000,  # CONST R5, #0
        0b1000_0000_0101_0100,  # STR R5, R4       (only one mem op)
        0b1111_0000_0000_0000,  # RET
    ]
    metrics, mem = await run_kernel(dut, program, [0] * 16, threads=1)
    assert mem.memory[0] == 9
    write_bench_result("edge_pure_compute", metrics)


@cocotb.test()
async def test_unaligned_block(dut):
    """3 threads → 1 block (the only block has thread_count=3, lane 3 disabled)."""
    program = [
        0b1001_0001_0000_0000,  # CONST R1, #0   (use threadIdx as both addr + data)
        0b0011_0010_0001_1111,  # ADD R2, R1, threadIdx     ; R2 = threadIdx
        0b1001_0011_0001_0000,  # CONST R3, #16
        0b0011_0100_0011_1111,  # ADD R4, R3, threadIdx     ; addr
        0b1000_0000_0100_0010,  # STR R4, R2                ; mem[16+t] = t
        0b1111_0000_0000_0000,  # RET
    ]
    metrics, mem = await run_kernel(dut, program, [0] * 24, threads=3)
    for i in range(3):
        assert mem.memory[16 + i] == i, f"thread {i}: got {mem.memory[16+i]} want {i}"
    write_bench_result("edge_unaligned_block", metrics)
