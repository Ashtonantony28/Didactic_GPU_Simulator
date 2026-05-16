"""Tiled 2x2 matmul using per-block shared memory.

Each block of 4 threads computes one row of C. Threads cooperate:
1. Load A[r][k], B[k][c] into shared memory.
2. BAR.
3. Read back from SMEM, multiply-accumulate.
4. Store the result.

Verifies the LDS / STS / BAR opcodes wired in Phase 7.
"""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


# Layout (8-bit data memory):
#   A at 0..3   (2x2)
#   B at 4..7   (2x2)
#   C at 8..11  (2x2)
#
# Each thread t in a block computes C[r][c] where r = blockIdx, c = t/2 mixed.
# Use 1 block (THREADS_PER_BLOCK=4) per matrix row pair.
#
# Simpler kernel (single block, 4 threads):
#   thread t computes C[t//2][t%2]
#   stage A: each thread loads A[t//2][t%2] into SMEM[t] from address t
#   stage B: each thread loads B[t//2][t%2] into SMEM[4+t] from address 4+t
#   BAR
#   thread t computes: row = t//2, col = t%2
#     acc = SMEM[row*2 + 0] * SMEM[4 + 0*2 + col] + SMEM[row*2 + 1] * SMEM[4 + 1*2 + col]
#   STR C[t] = acc
#
# We assemble this in tiny-gpu ISA. Register usage:
#   R0 = threadIdx (since one block, blockIdx=0)
#   R1 = 2          (matrix dim)
#   R2 = 4          (B base in DMEM, also SMEM B offset)
#   R3 = 8          (C base in DMEM)
#   R4 = row (t/2)
#   R5 = col (t - row*2)
#   R6 = address in A = t
#   R7 = A[t]
#   R8 = address in B = 4 + t
#   R9 = B[t]
#   R10 = base+t   (smem address)
#   R11/R12 = work
PROGRAM = [
    # 0  R0 = threadIdx
    0b0011_0000_1111_1111,  # ADD R0, %threadIdx, %threadIdx -> 2*tid; oops
]
# Easier: encode by hand explicitly without that trick. Let me build instructions cleanly.


def encode(opcode, rd=0, rs=0, rt=0, imm=None, nzp=None):
    if imm is not None:
        return (opcode << 12) | (rd << 8) | (imm & 0xFF)
    if nzp is not None:
        return (opcode << 12) | (nzp << 9) | (imm & 0xFF if imm else 0)
    return (opcode << 12) | (rd << 8) | (rs << 4) | rt


# Opcodes
ADD = 0b0011
SUB = 0b0100
MUL = 0b0101
DIV = 0b0110
LDR = 0b0111
STR = 0b1000
CONST = 0b1001
LDS = 0b1010
STS = 0b1011
BAR = 0b1100
RET = 0b1111

BLOCKIDX = 13
BLOCKDIM = 14
THREADIDX = 15


def build_program():
    p = []
    # R0 = threadIdx + 0
    p.append(encode(ADD, rd=0, rs=THREADIDX, rt=THREADIDX) ^ 0)  # placeholder
    # Easier strategy: copy threadIdx via CONST 0 then ADD.
    # We'll fully rebuild using clean ops:
    p = []
    # R0 = threadIdx
    # tiny-gpu has no MOV; use ADD R0, threadIdx, 0-register? No "zero" register.
    # Trick: CONST R0, 0 then ADD R0, R0, threadIdx
    p.append(encode(CONST, rd=0, imm=0))                          # 0  R0 = 0
    p.append(encode(ADD,   rd=0, rs=0, rt=THREADIDX))             # 1  R0 = threadIdx
    p.append(encode(CONST, rd=1, imm=2))                          # 2  R1 = 2  (matrix dim)
    p.append(encode(CONST, rd=2, imm=4))                          # 3  R2 = 4  (B base)
    p.append(encode(CONST, rd=3, imm=8))                          # 4  R3 = 8  (C base)
    # Stage A: load A[t] from DMEM address t into SMEM[t]
    p.append(encode(LDR,   rd=7, rs=0, rt=0))                     # 5  R7 = MEM[R0]
    p.append(encode(STS,   rs=0, rt=7))                           # 6  SMEM[R0] = R7
    # Stage B: load B[t] from DMEM address (4+t) into SMEM[4+t]
    p.append(encode(ADD,   rd=8, rs=2, rt=0))                     # 7  R8 = 4 + t
    p.append(encode(LDR,   rd=9, rs=8, rt=0))                     # 8  R9 = MEM[R8]
    p.append(encode(STS,   rs=8, rt=9))                           # 9  SMEM[R8] = R9
    # BAR — single-warp it's a no-op
    p.append(encode(BAR))                                         # 10
    # Compute row = t/2, col = t mod 2
    p.append(encode(DIV,   rd=4, rs=0, rt=1))                     # 11 R4 = t/2 (row)
    p.append(encode(MUL,   rd=11, rs=4, rt=1))                    # 12 R11 = row*2
    p.append(encode(SUB,   rd=5, rs=0, rt=11))                    # 13 R5 = t - row*2 (col)
    # Load A[row][0] from SMEM[row*2 + 0]  -> R6 = row*2  -> LDS R7 = SMEM[R6]
    p.append(encode(CONST, rd=10, imm=0))                         # 14 R10 = 0
    p.append(encode(ADD,   rd=6, rs=11, rt=10))                   # 15 R6 = row*2 + 0
    p.append(encode(LDS,   rd=7, rs=6))                           # 16 R7 = SMEM[R6]
    # Load B[0][col] from SMEM[4 + 0*2 + col] = SMEM[4 + col]
    p.append(encode(ADD,   rd=6, rs=2, rt=5))                     # 17 R6 = 4 + col
    p.append(encode(LDS,   rd=9, rs=6))                           # 18 R9 = SMEM[R6]
    # acc = R7 * R9
    p.append(encode(MUL,   rd=12, rs=7, rt=9))                    # 19 R12 = R7 * R9
    # Load A[row][1] = SMEM[row*2 + 1], B[1][col] = SMEM[4 + 2 + col]
    p.append(encode(CONST, rd=10, imm=1))                         # 20 R10 = 1
    p.append(encode(ADD,   rd=6, rs=11, rt=10))                   # 21 R6 = row*2 + 1
    p.append(encode(LDS,   rd=7, rs=6))                           # 22 R7 = SMEM[R6]
    p.append(encode(CONST, rd=10, imm=6))                         # 23 R10 = 6  (=4+2)
    p.append(encode(ADD,   rd=6, rs=10, rt=5))                    # 24 R6 = 6 + col
    p.append(encode(LDS,   rd=9, rs=6))                           # 25 R9 = SMEM[R6]
    p.append(encode(MUL,   rd=11, rs=7, rt=9))                    # 26 R11 = R7 * R9
    p.append(encode(ADD,   rd=12, rs=12, rt=11))                  # 27 R12 = R12 + R11
    # Store C[t] = R12
    p.append(encode(ADD,   rd=6, rs=3, rt=0))                     # 28 R6 = 8 + t
    p.append(encode(STR,   rs=6, rt=12))                          # 29 MEM[R6] = R12
    p.append(encode(RET))                                         # 30
    return p


PROGRAM = build_program()
DATA = [1, 2, 3, 4, 5, 6, 7, 8]  # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
THREADS = 4  # one block of 4 threads computing one 2x2 matmul


@cocotb.test()
async def test_tiled_matmul(dut):
    metrics, mem = await run_kernel(dut, PROGRAM, DATA, THREADS, timeout_cycles=8000)
    # Expected C = A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    expected = [19, 22, 43, 50]
    for i, e in enumerate(expected):
        assert mem.memory[8 + i] == e, f"tiled_matmul[{i}]: got {mem.memory[8+i]} want {e}"
    write_bench_result("tiled_matmul", metrics)
