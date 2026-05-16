"""matmul kernel — 2x2 matrix product, mirrors upstream."""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


PROGRAM = [
    0b0101000011011110,
    0b0011000000001111,
    0b1001000100000001,
    0b1001001000000010,
    0b1001001100000000,
    0b1001010000000100,
    0b1001010100001000,
    0b0110011000000010,
    0b0101011101100010,
    0b0100011100000111,
    0b1001100000000000,
    0b1001100100000000,
    0b0101101001100010,
    0b0011101010101001,
    0b0011101010100011,
    0b0111101010100000,
    0b0101101110010010,
    0b0011101110110111,
    0b0011101110110100,
    0b0111101110110000,
    0b0101110010101011,
    0b0011100010001100,
    0b0011100110010001,
    0b0010000010010010,
    0b0001100000001100,
    0b0011100101010000,
    0b1000000010011000,
    0b1111000000000000,
]
DATA = [1, 2, 3, 4, 1, 2, 3, 4]
THREADS = 4


@cocotb.test()
async def test_matmul(dut):
    metrics, mem = await run_kernel(dut, PROGRAM, DATA, THREADS)
    A = [DATA[0:2], DATA[2:4]]
    B = [DATA[4:6], DATA[6:8]]
    expected = [
        A[0][0]*B[0][0] + A[0][1]*B[1][0],
        A[0][0]*B[0][1] + A[0][1]*B[1][1],
        A[1][0]*B[0][0] + A[1][1]*B[1][0],
        A[1][0]*B[0][1] + A[1][1]*B[1][1],
    ]
    for i, e in enumerate(expected):
        assert mem.memory[i + 8] == e, f"matmul[{i}]: got {mem.memory[i+8]} want {e}"
    write_bench_result("matmul", metrics)
