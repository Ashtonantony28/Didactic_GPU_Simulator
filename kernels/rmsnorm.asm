; rmsnorm — per row of x (T, d): ss = sum(x^2), shifted right by 2 while ss > lutmax (k times),
; out = clamp((x * w * lut[ss] * M + 2^(s+k-1)) >> (s+k), -128, 127). One thread per element; each
; thread sums its own row. Both loops are uniform across the block (d and kmax trips); the
; normalization step is branch-free, so rows of one block may take different k.
;
; Args block (i32, see xf/elementwise.py): x, w, lut, out, d, kmax, lutmax, M, s, 2^s.
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
    LDR  R2, R1                 ; kmax, counts down
    ADD  R1, R1, R11
    LDR  R6, R1                 ; lutmax
    CONST R1, #8
    ADD  R1, R12, R1
    LDR  R9, R1                 ; shift = s, + k below
    CONST R7, #1                ; 2^k
norm:
    SUB  R8, R5, R6
    MAX  R8, R8, R10            ; t = max(ss - lutmax, 0)
    SUB  R1, R8, R11
    MAX  R1, R1, R10
    SUB  R8, R8, R1             ; f = min(t, 1)
    ADD  R9, R9, R8
    MUL  R1, R7, R8
    ADD  R7, R7, R1
    ADD  R1, R8, R8
    SRA  R5, R5, R1             ; ss >>= 2f
    SUB  R2, R2, R11
    CMP  R2, R10
    BRn  norm
    ADD  R6, R12, R11
    LDR  R8, R6
    ADD  R8, R8, R3
    LDR  R8, R8                 ; w[c]
    MUL  R4, R4, R8
    ADD  R6, R6, R11
    LDR  R8, R6
    ADD  R8, R8, R5
    LDR  R8, R8                 ; r = lut[ss]
    MUL  R4, R4, R8
    ADD  R6, R6, R11
    LDR  R2, R6                 ; &out
    CONST R1, #7
    ADD  R1, R12, R1
    LDR  R8, R1
    MUL  R4, R4, R8             ; x*w*r*M
    CONST R1, #9
    ADD  R1, R12, R1
    LDR  R8, R1
    MUL  R8, R8, R7
    SRA  R8, R8, R11            ; rnd = 2^(s+k) >> 1
    ADD  R4, R4, R8
    SRA  R4, R4, R9
    CONST R8, #128
    SUB  R8, R10, R8
    MAX  R4, R4, R8             ; max(x, -128)
    SUB  R4, R10, R4
    ADD  R8, R8, R11
    MAX  R4, R4, R8
    SUB  R4, R10, R4            ; min(x, 127) = -max(-x, -127)
    ADD  R2, R2, R0
    STR  R2, R4
    RET
