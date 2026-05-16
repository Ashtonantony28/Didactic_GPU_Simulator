"""vector_dot: per-thread products into mem[17..24]."""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


PROGRAM = [
    0b0101_0000_1101_1110,  # MUL R0, %blockIdx, %blockDim
    0b0011_0000_0000_1111,  # ADD R0, R0, %threadIdx
    0b1001_0001_0000_0000,  # CONST R1, #0  baseA
    0b1001_0010_0000_1000,  # CONST R2, #8  baseB
    0b1001_0011_0001_0001,  # CONST R3, #17 baseProducts
    0b0011_0100_0001_0000,  # ADD R4, R1, R0
    0b0111_0100_0100_0000,  # LDR R4, R4
    0b0011_0101_0010_0000,  # ADD R5, R2, R0
    0b0111_0101_0101_0000,  # LDR R5, R5
    0b0101_0110_0100_0101,  # MUL R6, R4, R5
    0b0011_0111_0011_0000,  # ADD R7, R3, R0
    0b1000_0000_0111_0110,  # STR R7, R6
    0b1111_0000_0000_0000,  # RET
]
A = [1, 2, 3, 4, 5, 6, 7, 8]
B = [8, 7, 6, 5, 4, 3, 2, 1]
DATA = A + B
THREADS = 8


@cocotb.test()
async def test_vector_dot(dut):
    metrics, mem = await run_kernel(dut, PROGRAM, DATA, THREADS)
    products = [a * b for a, b in zip(A, B)]
    for i, p in enumerate(products):
        assert mem.memory[17 + i] == p, f"product[{i}]: got {mem.memory[17+i]} want {p}"
    assert sum(products) == 120
    write_bench_result("vector_dot", metrics)
