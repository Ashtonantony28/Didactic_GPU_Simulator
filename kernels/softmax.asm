; softmax — causal row softmax of int8 scores (T, T): for j <= i, e = lut[s[i,j] - rowmax + 255],
; p[i,j] = (e*one + S/2) / S with S the row's sum of e; masked p = 0. One thread per element;
; each thread scans its own row twice (max, then exp sum) in T-trip loops, uniform across the block.
; Column j of row r is masked when j - r > 0: its max operand is s - 255*(j - r) <= -128 (never
; above rowmax, since column 0 is kept), and keep = 1 - min(max(j - r, 0), 1) zeroes its exp.
;
; Args block (i32, see xf/elementwise.py): scores, lut, out, T, one.
    CONST R12, #0               ; ARGS prologue: R12 = hi*256 + lo, patched by xf.kernels.program
    CONST R0, #16
    MUL  R12, R12, R0
    MUL  R12, R12, R0
    CONST R0, #0
    ADD  R12, R12, R0
    CONST R10, #0
    CONST R11, #1
    MUL  R0, %blockIdx, %blockDim
    ADD  R0, R0, %threadIdx     ; i = r*T + c
    CONST R1, #3
    ADD  R1, R12, R1
    LDR  R1, R1                 ; T
    DIV  R2, R0, R1             ; r
    MUL  R3, R2, R1
    LDR  R4, R12
    ADD  R4, R4, R3             ; &s[r, 0]
    SUB  R6, R10, R2            ; j - r, j = 0
    ADD  R7, R1, R10            ; T, counts down
    CONST R5, #128
    SUB  R5, R10, R5            ; rowmax = -128
    CONST R2, #255
rowmax:
    LDR  R8, R4
    MAX  R9, R6, R10
    MUL  R9, R9, R2
    SUB  R8, R8, R9
    MAX  R5, R5, R8
    ADD  R4, R4, R11
    ADD  R6, R6, R11
    SUB  R7, R7, R11
    CMP  R7, R10
    BRn  rowmax
    SUB  R4, R4, R1
    SUB  R6, R6, R1
    ADD  R7, R1, R10
    ADD  R1, R12, R11
    LDR  R1, R1
    ADD  R1, R1, R2             ; &lut[255]
    SUB  R3, R10, R11           ; -1
    CONST R2, #0                ; S
expsum:
    MAX  R9, R6, R10
    SUB  R9, R10, R9
    MAX  R9, R9, R3
    ADD  R9, R9, R11            ; keep
    LDR  R8, R4
    SUB  R8, R8, R5
    MUL  R8, R8, R9
    ADD  R8, R1, R8
    LDR  R8, R8
    MUL  R8, R8, R9             ; e = keep * lut[keep*(s - rowmax) + 255]
    ADD  R2, R2, R8
    ADD  R4, R4, R11
    ADD  R6, R6, R11
    SUB  R7, R7, R11
    CMP  R7, R10
    BRn  expsum
    MUL  R0, %blockIdx, %blockDim
    ADD  R0, R0, %threadIdx     ; i
    CONST R4, #3
    ADD  R4, R12, R4
    LDR  R4, R4                 ; T
    DIV  R6, R0, R4             ; r
    MUL  R7, R6, R4
    SUB  R7, R0, R7             ; c
    SUB  R6, R7, R6             ; c - r
    MAX  R9, R6, R10
    SUB  R9, R10, R9
    MAX  R9, R9, R3
    ADD  R9, R9, R11            ; keep
    LDR  R8, R12
    ADD  R8, R8, R0
    LDR  R8, R8                 ; s[r, c]
    SUB  R8, R8, R5
    MUL  R8, R8, R9
    ADD  R8, R1, R8
    LDR  R8, R8
    MUL  R8, R8, R9             ; e
    CONST R4, #4
    ADD  R4, R12, R4
    LDR  R4, R4                 ; one
    MUL  R8, R8, R4
    SRA  R4, R2, R11            ; S / 2
    ADD  R8, R8, R4
    DIV  R8, R8, R2             ; p = (e*one + S/2) / S
    CONST R4, #2
    ADD  R4, R12, R4
    LDR  R4, R4
    ADD  R4, R4, R0             ; &out[r, c]
    STR  R4, R8
    RET
