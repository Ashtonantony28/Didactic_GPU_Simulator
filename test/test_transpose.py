"""Transpose a 2x4 matrix into a 4x2 (in memory order)."""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


# input[r][c] at addr r*4+c  for r in 0..1, c in 0..3
# output[c][r] at addr 8 + c*2 + r
# Each thread handles one input cell. threadIdx i -> r=i//4, c=i%4.
# Easier: dispatch 8 threads, each computes (r,c) from i, reads in[i], writes out[8+c*2+r].
PROGRAM = [
    0b0101_0000_1101_1110,  # 0 MUL R0
    0b0011_0000_0000_1111,  # 1 ADD R0
    0b1001_0001_0000_0000,  # 2 CONST R1,#0  baseIn
    0b1001_0010_0000_1000,  # 3 CONST R2,#8  baseOut
    0b1001_0011_0000_0100,  # 4 CONST R3,#4  ncols
    0b1001_0100_0000_0010,  # 5 CONST R4,#2  nrows
    0b0110_0101_0000_0011,  # 6 DIV R5, R0, R3   ; r = i / 4
    0b0101_0110_0101_0011,  # 7 MUL R6, R5, R3   ; r*4
    0b0100_0111_0000_0110,  # 8 SUB R7, R0, R6   ; c = i - r*4
    0b0011_1000_0001_0000,  # 9 ADD R8, R1, R0   ; &in[i]
    0b0111_1000_1000_0000,  # 10 LDR R8, R8      ; val
    0b0101_1001_0111_0100,  # 11 MUL R9, R7, R4  ; c*2
    0b0011_1001_1001_0101,  # 12 ADD R9, R9, R5  ; c*2+r
    0b0011_1001_1001_0010,  # 13 ADD R9, R9, R2  ; +baseOut
    0b1000_0000_1001_1000,  # 14 STR R9, R8
    0b1111_0000_0000_0000,  # 15 RET
]
INPUT = [1, 2, 3, 4, 5, 6, 7, 8]   # 2x4
DATA = INPUT
THREADS = 8


@cocotb.test()
async def test_transpose(dut):
    metrics, mem = await run_kernel(dut, PROGRAM, DATA, THREADS)
    # in[r,c] -> out[c,r], output 4x2
    M = [INPUT[0:4], INPUT[4:8]]
    expected = []
    for c in range(4):
        for r in range(2):
            expected.append(M[r][c])
    for i, e in enumerate(expected):
        assert mem.memory[8 + i] == e, f"transpose[{i}]: got {mem.memory[8+i]} want {e}"
    write_bench_result("transpose", metrics)
