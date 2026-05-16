"""1-D convolution with a fixed 3-tap kernel [1, 1, 1] over an 8-element signal.

Output o[i] = in[i] + in[i+1] + in[i+2]. We emit 6 outputs (threads 0..5).
"""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


PROGRAM = [
    0b0101_0000_1101_1110,  # MUL R0
    0b0011_0000_0000_1111,  # ADD R0,R0,tid                    i
    0b1001_0001_0000_0000,  # CONST R1,#0   baseIn
    0b1001_0010_0001_0000,  # CONST R2,#16  baseOut
    0b1001_0011_0000_0001,  # CONST R3,#1
    0b1001_0100_0000_0010,  # CONST R4,#2
    0b0011_0101_0001_0000,  # ADD R5, R1, R0       &in[i]
    0b0111_0101_0101_0000,  # LDR R5, R5            a = in[i]
    0b0011_0110_0101_0011,  # ADD R6, R5, R3        (placeholder, redo below for &in[i+1])
    0b0011_0110_0001_0000,  # ADD R6, R1, R0
    0b0011_0110_0110_0011,  # ADD R6, R6, R3        &in[i+1]
    0b0111_0110_0110_0000,  # LDR R6, R6
    0b0011_0111_0001_0000,  # ADD R7, R1, R0
    0b0011_0111_0111_0100,  # ADD R7, R7, R4        &in[i+2]
    0b0111_0111_0111_0000,  # LDR R7, R7
    0b0011_1000_0101_0110,  # ADD R8, R5, R6        a+b
    0b0011_1000_1000_0111,  # ADD R8, R8, R7        +c
    0b0011_1001_0010_0000,  # ADD R9, R2, R0        &out[i]
    0b1000_0000_1001_1000,  # STR R9, R8
    0b1111_0000_0000_0000,  # RET
]
SIGNAL = [1, 2, 3, 4, 5, 6, 7, 8]
DATA = SIGNAL
THREADS = 6  # only valid for i in 0..5


@cocotb.test()
async def test_conv1d(dut):
    metrics, mem = await run_kernel(dut, PROGRAM, DATA, THREADS)
    expected = [SIGNAL[i] + SIGNAL[i+1] + SIGNAL[i+2] for i in range(6)]
    for i, e in enumerate(expected):
        assert mem.memory[16 + i] == e, f"conv1d[{i}]: got {mem.memory[16+i]} want {e}"
    write_bench_result("conv1d", metrics)
