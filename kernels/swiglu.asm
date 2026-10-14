; swiglu — out[i] = clamp((silu_lut[gate[i] + 128] * up[i] * M + rnd) >> s, -128, 127); one thread per element.
;
; Args block (i32, see xf/elementwise.py): gate, up, lut, out, M, s, rnd.
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
    LDR  R1, R1                 ; gate[i]
    CONST R2, #128
    ADD  R1, R1, R2
    ADD  R3, R12, R11
    LDR  R2, R3
    ADD  R2, R2, R0
    LDR  R2, R2                 ; up[i]
    ADD  R3, R3, R11
    LDR  R4, R3
    ADD  R1, R1, R4
    LDR  R1, R1                 ; silu_lut[gate[i] + 128]
    MUL  R1, R1, R2
    ADD  R3, R3, R11
    LDR  R9, R3                 ; &out
    ADD  R3, R3, R11
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
    ADD  R9, R9, R0
    STR  R9, R1
    RET
