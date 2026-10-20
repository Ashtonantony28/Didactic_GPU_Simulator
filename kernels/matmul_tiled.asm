; matmul_tiled — C = A @ B, int8 operands, int32 accumulators, A tiles staged in shared memory.
; Threads form groups of G = min(4, blockDim) consecutive lanes; group g computes a 4xG tile of C:
; lane l owns column c = c0+l, rows r0..r0+3. K is walked in chunks of G: the group stages
; A[r0:r0+4, k0:k0+G] into SMEM[i*blockDim + t] (t = threadIdx), BAR, then each thread streams B[k, c]
; from global memory into four accumulators. Rows >= M and columns >= K stage don't-care words that
; are never used. Every branch is uniform across the block: with blockDim > 4 the epilogue stores rows
; min(i, valid-1) four times, since a block's groups may straddle a tile row.
; Needs N % 4 == 0 and blockDim in {1, 2, 4, 8, 12, 16} (4*blockDim words of SMEM; xf.kernels.check_block). Launch ceil(M/4) * N threads.
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
    MUL  R0, %blockIdx, %blockDim
    ADD  R0, R0, %threadIdx     ; g = r0/4 * N + c: G divides N, so a group never straddles a row of tiles
    CONST R11, #5
    ADD  R11, R12, R11
    LDR  R6, R11                ; N
    DIV  R2, R0, R6
    MUL  R3, R2, R6
    SUB  R3, R0, R3             ; c
    CONST R8, #4
    MUL  R2, R2, R8             ; r0
    DIV  R7, %threadIdx, R8
    MUL  R7, R7, R8
    SUB  R7, %threadIdx, R7     ; l
    CONST R11, #4
    ADD  R11, R12, R11
    LDR  R8, R11                ; K
    LDR  R4, R12
    MUL  R9, R2, R8
    ADD  R4, R4, R9
    ADD  R4, R4, R7             ; &A[r0, k0+l]
    ADD  R7, R8, R10            ; K remaining
    CONST R5, #1
    ADD  R5, R12, R5
    LDR  R5, R5
    ADD  R5, R5, R3             ; &B[k0, c]
    CONST R0, #0
    CONST R1, #0
    CONST R2, #0
    CONST R3, #0
chunk:
    CONST R8, #4
    ADD  R8, R12, R8
    LDR  R8, R8                 ; K, the row stride of A
    ADD  R11, R10, %threadIdx   ; R10 = 0
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
    CONST R8, #4
    DIV  R9, %threadIdx, R8
    MUL  R11, R9, R8            ; SMEM[t - l], row 0 of the group's chunk
    SUB  R8, R10, R8
    SUB  R9, R10, %blockDim
    MAX  R8, R8, R9
    SUB  R9, R10, R7
    MAX  R8, R8, R9
    SUB  R8, R10, R8            ; n = min(4, blockDim, K remaining)
    SUB  R7, R7, R8
    ADD  R4, R4, R8
    MUL  R8, R8, R6
    ADD  R8, R8, R5             ; &B[k0+n, c]
inner:
    LDR  R9, R5                 ; B[k0+kk, c], kk even: rows 0..3
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
    CONST R10, #1
    ADD  R11, R11, R10          ; kk + 1, row 3
    ADD  R5, R5, R6
    CMP  R5, R8
    BRz  done
    LDR  R9, R5                 ; kk odd: rows 3..0
    LDS  R10, R11
    MUL  R10, R10, R9
    ADD  R3, R3, R10
    SUB  R11, R11, %blockDim
    LDS  R10, R11
    MUL  R10, R10, R9
    ADD  R2, R2, R10
    SUB  R11, R11, %blockDim
    LDS  R10, R11
    MUL  R10, R10, R9
    ADD  R1, R1, R10
    SUB  R11, R11, %blockDim
    LDS  R10, R11
    MUL  R10, R10, R9
    ADD  R0, R0, R10
    CONST R10, #1
    ADD  R11, R11, R10          ; kk + 1, row 0
    ADD  R5, R5, R6
    CMP  R5, R8
    BRn  inner
done:
    BAR
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
    STS  R11, R3                ; SMEM[i*blockDim + t] = acc_i
    MUL  R0, %blockIdx, %blockDim
    ADD  R0, R0, %threadIdx
    DIV  R2, R0, R6
    MUL  R3, R2, R6
    SUB  R3, R0, R3             ; c
    CONST R8, #4
    MUL  R2, R2, R8             ; r0
    CONST R9, #1
    CONST R11, #3
    ADD  R11, R12, R11
    LDR  R0, R11
    SUB  R0, R2, R0             ; r0 - M
    MUL  R2, R2, R6
    ADD  R2, R2, R3
    CONST R11, #2
    ADD  R11, R12, R11
    LDR  R3, R11
    ADD  R2, R2, R3             ; &C[r0, c]
    CONST R11, #6
    ADD  R11, R12, R11
    LDR  R4, R11                ; mq
    ADD  R11, R11, R9
    LDR  R5, R11                ; s
    ADD  R11, R11, R9
    LDR  R7, R11                ; rnd
    SUB  R12, R10, R8           ; -4
    CONST R1, #0                ; -i
    MAX  R11, %blockDim, R8
    CMP  R11, R8
    BRz  small                  ; blockDim <= 4: one group per block
    ADD  R0, R0, R9
    CONST R11, #3
    SUB  R11, R10, R11
    MAX  R0, R0, R11            ; -(valid rows - 1) = max(r0 + 1 - M, -3)
epi:
    MAX  R11, R1, R0
    SUB  R11, R10, R11          ; j = min(i, valid rows - 1)
    MUL  R3, R11, %blockDim
    ADD  R3, R3, %threadIdx
    LDS  R8, R3                 ; acc_j
    MUL  R3, R11, R6
    ADD  R3, R3, R2             ; &C[r0+j, c]
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
    STR  R3, R8
    SUB  R1, R1, R9
    CMP  R1, R12
    BRn  epi
    RET
small:
    MAX  R0, R0, R12            ; -(valid rows) = max(r0 - M, -4)
    ADD  R3, R10, %threadIdx
epi4:
    LDS  R8, R3
    CMP  R4, R10
    BRz  store4
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
store4:
    STR  R2, R8
    ADD  R2, R2, R6
    ADD  R3, R3, %blockDim
    SUB  R1, R1, R9
    CMP  R1, R0
    BRn  epi4
    RET
