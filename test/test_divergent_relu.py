"""Divergent ReLU stand-in.

Half the threads (i even) write in[i]+1, half (i odd) write in[i]-1.
Without a SIMT stack we still execute both paths under a single PC — this
test ensures correctness when the *data path* diverges, which is the
common case in a no-divergence SIMT and surfaces any issues with masking.
"""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


# Strategy: every thread computes both candidates, then selects via mul-by-mask.
# mask = i & 1   (least significant bit of threadIdx)
# r_even = in[i] + 1
# r_odd  = in[i] - 1
# out[i] = (1 - mask) * r_even + mask * r_odd
# All threads run identical instructions; no per-thread PC needed.
PROGRAM = [
    0b0101_0000_1101_1110,  # 0  MUL  R0, blockIdx, blockDim
    0b0011_0000_0000_1111,  # 1  ADD  R0, R0, threadIdx
    0b1001_0001_0000_0000,  # 2  CONST R1, #0    baseIn
    0b1001_0010_0000_1000,  # 3  CONST R2, #8    baseOut
    0b1001_0011_0000_0001,  # 4  CONST R3, #1
    0b1001_0100_0000_0010,  # 5  CONST R4, #2
    0b0011_0101_0001_0000,  # 6  ADD  R5, R1, R0   &in[i]
    0b0111_0101_0101_0000,  # 7  LDR  R5, R5       x
    # parity = i mod 2:  i / 2 *2 then subtract from i.
    0b0110_0110_0000_0100,  # 8  DIV  R6, R0, R4   q = i/2
    0b0101_0110_0110_0100,  # 9  MUL  R6, R6, R4   q*2
    0b0100_0110_0000_0110,  # 10 SUB  R6, R0, R6   parity = i - 2*q
    # one_minus_parity:
    0b0100_0111_0011_0110,  # 11 SUB  R7, R3, R6   1 - parity
    # r_even = x + 1, r_odd = x - 1
    0b0011_1000_0101_0011,  # 12 ADD  R8, R5, R3
    0b0100_1001_0101_0011,  # 13 SUB  R9, R5, R3
    # out = (1-parity)*r_even + parity*r_odd
    0b0101_1010_0111_1000,  # 14 MUL  R10, R7, R8
    0b0101_1011_0110_1001,  # 15 MUL  R11, R6, R9
    0b0011_1100_1010_1011,  # 16 ADD  R12, R10, R11
    0b0011_0111_0010_0000,  # 17 ADD  R7, R2, R0   &out[i]   (R7 reused)
    0b1000_0000_0111_1100,  # 18 STR  R7, R12
    0b1111_0000_0000_0000,  # 19 RET
]
DATA = [10, 20, 30, 40, 50, 60, 70, 80]
THREADS = 8


@cocotb.test()
async def test_divergent_relu(dut):
    metrics, mem = await run_kernel(dut, PROGRAM, DATA, THREADS)
    expected = [DATA[i] + 1 if i % 2 == 0 else DATA[i] - 1 for i in range(8)]
    for i, e in enumerate(expected):
        got = mem.memory[8 + i]
        assert got == e, f"divergent_relu[{i}]: got {got} want {e}"
    write_bench_result("divergent_relu", metrics)
