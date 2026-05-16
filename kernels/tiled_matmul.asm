; tiled_matmul — 4x4 matrix multiply using per-block shared memory
;
; Reserved opcodes: LDS (1010), STS (1011), BAR (1100). Currently unwired
; in src/ — this kernel is documentation for the Phase 7 design.
;
; Layout (global memory):
;   A at addr 0..15     (4x4)
;   B at addr 16..31    (4x4)
;   C at addr 32..47    (4x4)  result
;
; Each block of 4 threads computes one row of C.
; threadIdx maps to column c. blockIdx maps to row r.
;
;   row r, col c:
;     accum = 0
;     for k in 0..3:
;         load A[r][k] into SMEM[k]     by the corresponding thread
;         load B[k][c] into SMEM[4+k]   by the corresponding thread
;         BAR
;         LDS x, k
;         LDS y, 4+k
;         accum += x * y
;         BAR
;     STR C[r][c], accum

MUL  R0, %blockIdx, %blockDim       ; not directly used; blockIdx = r
ADD  R0, R0, %threadIdx             ; column index
; ... (rest of the kernel left as documentation; ISA wiring pending)
RET
