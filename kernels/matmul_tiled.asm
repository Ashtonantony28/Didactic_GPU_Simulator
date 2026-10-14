; matmul_tiled — C = A @ B, int8 operands, int32 accumulators, A tiles staged in shared memory.
; Each block of 4 threads computes a 4x4 tile of C: thread t owns column c = c0+t, rows r0..r0+3.
; K is walked in chunks of 4: the block stages A[r0:r0+4, k0:k0+4] into SMEM[4i+t], BAR, then each
; thread streams B[k, c] from global memory into four accumulators. SMEM[16+t], SMEM[20+t] hold
; &C[r0, c] and the tile's valid row count. Rows >= M and columns >= K stage don't-care words that
; are never used.
; Needs blockDim = 4 and N % 4 == 0. Launch ceil(M/4) * N threads.
;
; Args block (i32, see xf/kernels.py): a, b, c, M, K, N, mq, s, rnd.
; mq = 0 stores the int32 accumulator; otherwise C = clamp((acc*mq + rnd) >> s, -128, 127).
    CONST R12, #0               ; ARGS prologue: R12 = hi*256 + lo, patched by xf.kernels.program
    CONST R0, #16
    MUL  R12, R12, R0
    MUL  R12, R12, R0
    CONST R0, #0
    ADD  R12, R12, R0
    CONST R10, #0
    CONST R11, #5
    ADD  R11, R12, R11
    LDR  R6, R11                ; N
    DIV  R0, R6, %blockDim      ; tiles per row of C
    DIV  R1, %blockIdx, R0
    MUL  R2, R1, R0
    SUB  R2, %blockIdx, R2
    MUL  R1, R1, %blockDim      ; r0
    MUL  R2, R2, %blockDim
    ADD  R2, R2, %threadIdx     ; c
    MUL  R3, R1, R6
    ADD  R3, R3, R2
    CONST R11, #2
    ADD  R11, R12, R11
    LDR  R4, R11
    ADD  R3, R3, R4
    CONST R11, #16
    ADD  R11, R11, %threadIdx
    STS  R11, R3                ; SMEM[16+t] = &C[r0, c]
    CONST R4, #3
    ADD  R4, R12, R4
    LDR  R4, R4
    SUB  R4, R4, R1
    SUB  R4, R10, R4
    SUB  R3, R10, %blockDim
    MAX  R4, R4, R3
    SUB  R4, R10, R4
    ADD  R11, R11, %blockDim
    STS  R11, R4                ; SMEM[20+t] = min(4, M - r0)
    CONST R7, #4
    ADD  R7, R12, R7
    LDR  R7, R7                 ; K remaining
    LDR  R4, R12
    MUL  R3, R1, R7
    ADD  R4, R4, R3
    ADD  R4, R4, %threadIdx     ; &A[r0, k0+t]
    CONST R5, #1
    ADD  R5, R12, R5
    LDR  R5, R5
    ADD  R5, R5, R2             ; &B[k0, c]
    CONST R0, #0
    CONST R1, #0
    CONST R2, #0
    CONST R3, #0
chunk:
    CONST R8, #4
    ADD  R8, R12, R8
    LDR  R8, R8                 ; K, the row stride of A
    CONST R11, #0
    ADD  R11, R11, %threadIdx
    LDR  R9, R4
    STS  R11, R9
    ADD  R9, R4, R8
    ADD  R11, R11, %blockDim
    LDR  R10, R9
    STS  R11, R10
    ADD  R9, R9, R8
    ADD  R11, R11, %blockDim
    LDR  R10, R9
    STS  R11, R10
    ADD  R9, R9, R8
    ADD  R11, R11, %blockDim
    LDR  R10, R9
    STS  R11, R10
    BAR
    CONST R10, #0
    SUB  R8, R10, R7
    SUB  R9, R10, %blockDim
    MAX  R8, R8, R9
    SUB  R8, R10, R8            ; n = min(4, K remaining)
    CONST R11, #0               ; kk
inner:
    LDR  R9, R5                 ; B[k0+kk, c]
    LDS  R10, R11
    MUL  R10, R10, R9
    ADD  R0, R0, R10
    ADD  R11, R11, %blockDim
    LDS  R10, R11
    MUL  R10, R10, R9
    ADD  R1, R1, R10
    ADD  R11, R11, %blockDim
    LDS  R10, R11
    MUL  R10, R10, R9
    ADD  R2, R2, R10
    ADD  R11, R11, %blockDim
    LDS  R10, R11
    MUL  R10, R10, R9
    ADD  R3, R3, R10
    CONST R10, #11
    SUB  R11, R11, R10          ; kk + 1
    ADD  R5, R5, R6
    CMP  R11, R8
    BRn  inner
    BAR
    SUB  R7, R7, R8
    ADD  R4, R4, %blockDim
    CONST R10, #0
    CMP  R7, R10
    BRn  chunk
    ADD  R11, R10, %threadIdx
    STS  R11, R0
    ADD  R11, R11, %blockDim
    STS  R11, R1
    ADD  R11, R11, %blockDim
    STS  R11, R2
    ADD  R11, R11, %blockDim
    STS  R11, R3                ; SMEM[4i+t] = acc_i
    CONST R11, #16
    ADD  R11, R11, %threadIdx
    LDS  R2, R11                ; &C[r0, c]
    ADD  R11, R11, %blockDim
    LDS  R0, R11                ; valid rows
    CONST R9, #1
    CONST R11, #6
    ADD  R11, R12, R11
    LDR  R4, R11                ; mq
    ADD  R11, R11, R9
    LDR  R5, R11                ; s
    ADD  R11, R11, R9
    LDR  R7, R11                ; rnd
    CONST R1, #0                ; i
    ADD  R3, R10, %threadIdx
epi:
    LDS  R8, R3
    CMP  R4, R10
    BRz  store
    MUL  R8, R8, R4
    ADD  R8, R8, R7
    SRA  R8, R8, R5
    CONST R11, #128
    SUB  R11, R10, R11
    MAX  R8, R8, R11            ; max(x, -128)
    SUB  R8, R10, R8
    ADD  R11, R11, R9
    MAX  R8, R8, R11
    SUB  R8, R10, R8            ; min(x, 127) = -max(-x, -127)
store:
    STR  R2, R8
    ADD  R2, R2, R6
    ADD  R3, R3, %blockDim
    ADD  R1, R1, R9
    CMP  R1, R0
    BRn  epi
    RET
