; requant — out[i] = clamp((in[i]*M + rnd) >> s, -128, 127), int32 in, int8 out; one thread per element.
;
; Args block (i32, see xf/elementwise.py): in, out, M, s, rnd.
    CONST R12, #0               ; ARGS prologue: R12 = hi*256 + lo, patched by xf.kernels.program
    CONST R0, #16
    MUL  R12, R12, R0
    MUL  R12, R12, R0
    CONST R0, #0
    ADD  R12, R12, R0
    CONST R10, #0
    CONST R11, #1
    MUL  R0, %blockIdx, %blockDim
    ADD  R0, R0, %threadIdx     ; i
    LDR  R1, R12
    ADD  R1, R1, R0
    LDR  R1, R1                 ; in[i]
    ADD  R2, R12, R11
    ADD  R3, R2, R11
    LDR  R4, R3                 ; M
    ADD  R3, R3, R11
    LDR  R5, R3                 ; s
    ADD  R3, R3, R11
    LDR  R6, R3                 ; rnd
    MUL  R1, R1, R4
    ADD  R1, R1, R6
    SRA  R1, R1, R5
    CONST R8, #128
    SUB  R8, R10, R8
    MAX  R1, R1, R8             ; max(x, -128)
    SUB  R1, R10, R1
    ADD  R8, R8, R11
    MAX  R1, R1, R8
    SUB  R1, R10, R1            ; min(x, 127) = -max(-x, -127)
    LDR  R2, R2
    ADD  R2, R2, R0
    STR  R2, R1
    RET
