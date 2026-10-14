; rmsnorm — per row of x (T, d): r = lut[min(sum(x^2) >> shift, lutmax)],
; out = clamp((x * w * r * M + rnd) >> s, -128, 127). One thread per element; each thread
; sums its own row (the d-trip loop is uniform across the block).
;
; Args block (i32, see xf/elementwise.py): x, w, lut, out, d, shift, lutmax, M, s, rnd.
    CONST R12, #0               ; ARGS prologue: R12 = hi*256 + lo, patched by xf.kernels.program
    CONST R0, #16
    MUL  R12, R12, R0
    MUL  R12, R12, R0
    CONST R0, #0
    ADD  R12, R12, R0
    CONST R10, #0
    CONST R11, #1
    MUL  R0, %blockIdx, %blockDim
    ADD  R0, R0, %threadIdx     ; i = row*d + c
    CONST R1, #4
    ADD  R1, R12, R1
    LDR  R1, R1                 ; d, counts down
    DIV  R2, R0, R1
    MUL  R2, R2, R1
    SUB  R3, R0, R2             ; c
    LDR  R4, R12
    ADD  R2, R4, R2             ; &x[row, 0]
    ADD  R4, R4, R0
    LDR  R4, R4                 ; x[i]
    CONST R5, #0                ; sum of squares
loop:
    LDR  R6, R2
    MUL  R6, R6, R6
    ADD  R5, R5, R6
    ADD  R2, R2, R11
    SUB  R1, R1, R11
    CMP  R1, R10
    BRn  loop
    CONST R1, #5
    ADD  R1, R12, R1
    LDR  R6, R1
    SRA  R5, R5, R6             ; sum >> shift
    ADD  R1, R1, R11
    LDR  R6, R1
    SUB  R6, R10, R6
    SUB  R5, R10, R5
    MAX  R5, R5, R6
    SUB  R5, R10, R5            ; idx = min(sum >> shift, lutmax)
    ADD  R6, R12, R11
    LDR  R7, R6
    ADD  R7, R7, R3
    LDR  R7, R7                 ; w[c]
    ADD  R6, R6, R11
    LDR  R8, R6
    ADD  R8, R8, R5
    LDR  R8, R8                 ; r = lut[idx]
    ADD  R6, R6, R11
    LDR  R9, R6                 ; &out
    MUL  R4, R4, R7
    MUL  R4, R4, R8
    ADD  R1, R1, R11
    LDR  R6, R1
    MUL  R4, R4, R6             ; x*w*r*M
    ADD  R1, R1, R11
    LDR  R5, R1                 ; s
    ADD  R1, R1, R11
    LDR  R6, R1                 ; rnd
    ADD  R4, R4, R6
    SRA  R4, R4, R5
    CONST R8, #128
    SUB  R8, R10, R8
    MAX  R4, R4, R8             ; max(x, -128)
    SUB  R4, R10, R4
    ADD  R8, R8, R11
    MAX  R4, R4, R8
    SUB  R4, R10, R4            ; min(x, 127) = -max(-x, -127)
    ADD  R9, R9, R0
    STR  R9, R4
    RET
