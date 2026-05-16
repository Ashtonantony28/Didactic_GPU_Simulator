"""matadd kernel — 1-D vector addition, mirrors upstream."""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


PROGRAM = [
    0b0101000011011110,  # MUL R0, %blockIdx, %blockDim
    0b0011000000001111,  # ADD R0, R0, %threadIdx
    0b1001000100000000,  # CONST R1, #0   ; baseA
    0b1001001000001000,  # CONST R2, #8   ; baseB
    0b1001001100010000,  # CONST R3, #16  ; baseC
    0b0011010000010000,  # ADD R4, R1, R0
    0b0111010001000000,  # LDR R4, R4
    0b0011010100100000,  # ADD R5, R2, R0
    0b0111010101010000,  # LDR R5, R5
    0b0011011001000101,  # ADD R6, R4, R5
    0b0011011100110000,  # ADD R7, R3, R0
    0b1000000001110110,  # STR R7, R6
    0b1111000000000000,  # RET
]
DATA = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
THREADS = 8


@cocotb.test()
async def test_matadd(dut):
    metrics, mem = await run_kernel(dut, PROGRAM, DATA, THREADS)
    expected = [a + b for a, b in zip(DATA[0:8], DATA[8:16])]
    for i, e in enumerate(expected):
        assert mem.memory[i + 16] == e, f"matadd[{i}]: got {mem.memory[i+16]} want {e}"
    write_bench_result("matadd", metrics)
