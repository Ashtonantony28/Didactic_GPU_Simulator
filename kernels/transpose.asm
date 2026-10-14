; transpose — out[c, r] = x[r, c] for x (R, C); one thread per element.
;
; Args block (i32, see xf/elementwise.py): x, out, R, C.
    CONST R12, #0               ; ARGS prologue: R12 = hi*256 + lo, patched by xf.kernels.program
    CONST R0, #16
    MUL  R12, R12, R0
    MUL  R12, R12, R0
    CONST R0, #0
    ADD  R12, R12, R0
    CONST R11, #1
    MUL  R0, %blockIdx, %blockDim
    ADD  R0, R0, %threadIdx     ; i = r*C + c
    CONST R1, #3
    ADD  R1, R12, R1
    LDR  R1, R1                 ; C
    DIV  R2, R0, R1             ; r
    MUL  R3, R2, R1
    SUB  R3, R0, R3             ; c
    LDR  R4, R12
    ADD  R4, R4, R0
    LDR  R4, R4                 ; x[r, c]
    CONST R5, #2
    ADD  R5, R12, R5
    LDR  R5, R5                 ; R
    MUL  R3, R3, R5
    ADD  R3, R3, R2             ; c*R + r
    ADD  R5, R12, R11
    LDR  R5, R5
    ADD  R5, R5, R3
    STR  R5, R4
    RET
