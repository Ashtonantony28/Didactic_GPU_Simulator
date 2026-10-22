; tiled_matmul — 2x2 matrix product through shared memory; the program of test/test_tiled_matmul.py.
; The int8/int32 tiled matmul of the transformer track is matmul_tiled.asm.
;
; One block of 4 threads; thread t computes C[t/2][t%2].
; Global memory: A at 0..3, B at 4..7, C at 8..11 (row-major 2x2).
; Each thread stages A[t] into SMEM[t] and B[t] into SMEM[4+t], BAR, then reads its row of A and
; column of B back with LDS.
    CONST R0, #0
    ADD  R0, R0, %threadIdx     ; t
    CONST R1, #2                ; matrix dim
    CONST R2, #4                ; B base (global and SMEM)
    CONST R3, #8                ; C base
    LDR  R7, R0
    STS  R0, R7                 ; SMEM[t] = A[t]
    ADD  R8, R2, R0
    LDR  R9, R8
    STS  R8, R9                 ; SMEM[4+t] = B[t]
    BAR
    DIV  R4, R0, R1             ; row
    MUL  R11, R4, R1
    SUB  R5, R0, R11            ; col
    CONST R10, #0
    ADD  R6, R11, R10
    LDS  R7, R6                 ; A[row][0]
    ADD  R6, R2, R5
    LDS  R9, R6                 ; B[0][col]
    MUL  R12, R7, R9
    CONST R10, #1
    ADD  R6, R11, R10
    LDS  R7, R6                 ; A[row][1]
    CONST R10, #6
    ADD  R6, R10, R5
    LDS  R9, R6                 ; B[1][col]
    MUL  R11, R7, R9
    ADD  R12, R12, R11
    ADD  R6, R3, R0
    STR  R6, R12                ; C[row][col]
    RET
