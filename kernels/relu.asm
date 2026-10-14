; relu — out[i] = max(in[i], 0); one thread per element.
;
; Args block (i32, see xf/elementwise.py): in, out.
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
    LDR  R1, R1
    MAX  R1, R1, R10
    ADD  R2, R12, R11
    LDR  R2, R2
    ADD  R2, R2, R0
    STR  R2, R1
    RET
