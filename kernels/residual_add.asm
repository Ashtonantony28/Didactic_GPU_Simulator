; residual_add — out[i] = clamp((a[i]*Ma + b[i]*Mb + rnd) >> s, -128, 127); one thread per element.
; Ma, Mb align the two input scales to the output scale.
;
; Args block (i32, see xf/elementwise.py): a, b, out, Ma, Mb, s, rnd.
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
    LDR  R1, R1                 ; a[i]
    ADD  R3, R12, R11
    LDR  R2, R3
    ADD  R2, R2, R0
    LDR  R2, R2                 ; b[i]
    ADD  R3, R3, R11
    LDR  R9, R3                 ; &out
    ADD  R3, R3, R11
    LDR  R4, R3
    MUL  R1, R1, R4             ; a*Ma
    ADD  R3, R3, R11
    LDR  R4, R3
    MUL  R2, R2, R4             ; b*Mb
    ADD  R1, R1, R2
    ADD  R3, R3, R11
    LDR  R5, R3                 ; s
    ADD  R3, R3, R11
    LDR  R6, R3                 ; rnd
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
