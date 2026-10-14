; matmul_naive — C = A @ B, int8 operands, int32 accumulator; one thread per element of C.
; Launch M*N threads. Operands are read straight from global memory.
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
    CONST R11, #1
    MUL  R0, %blockIdx, %blockDim
    ADD  R0, R0, %threadIdx     ; i = r*N + c
    CONST R1, #5
    ADD  R1, R12, R1
    LDR  R1, R1                 ; N
    DIV  R2, R0, R1             ; r
    MUL  R3, R2, R1
    SUB  R3, R0, R3             ; c
    CONST R4, #4
    ADD  R4, R12, R4
    LDR  R4, R4                 ; K, counts down
    LDR  R5, R12
    MUL  R2, R2, R4
    ADD  R5, R5, R2             ; &A[r, 0]
    ADD  R6, R12, R11
    LDR  R6, R6
    ADD  R6, R6, R3             ; &B[0, c]
    CONST R7, #0                ; acc
loop:
    LDR  R8, R5
    LDR  R9, R6
    MUL  R8, R8, R9
    ADD  R7, R7, R8
    ADD  R5, R5, R11
    ADD  R6, R6, R1
    SUB  R4, R4, R11
    CMP  R4, R10
    BRn  loop
    CONST R1, #6
    ADD  R1, R12, R1
    LDR  R2, R1                 ; mq
    CMP  R2, R10
    BRz  store
    ADD  R1, R1, R11
    LDR  R3, R1                 ; s
    ADD  R1, R1, R11
    LDR  R4, R1                 ; rnd
    MUL  R7, R7, R2
    ADD  R7, R7, R4
    SRA  R7, R7, R3
    CONST R8, #128
    SUB  R8, R10, R8
    MAX  R7, R7, R8             ; max(x, -128)
    SUB  R7, R10, R7
    ADD  R8, R8, R11
    MAX  R7, R7, R8
    SUB  R7, R10, R7            ; min(x, 127) = -max(-x, -127)
store:
    ADD  R1, R12, R11
    ADD  R1, R1, R11
    LDR  R1, R1
    ADD  R1, R1, R0             ; &C[r, c]
    STR  R1, R7
    RET
