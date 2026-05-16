; vector_dot — eight-element dot product.
; Each thread loads A[i] and B[i], multiplies, then summed into mem[16].
; Layout: A at addr 0..7, B at addr 8..15, scalar result at addr 16.
; Threads cooperate via global memory (no atomics: we accumulate sequentially through threads)
; — for tiny-gpu without atomics, this kernel writes per-thread products at addr 17..24
;   and the host sums them.
MUL  R0, %blockIdx, %blockDim
ADD  R0, R0, %threadIdx        ; i = blockIdx * blockDim + threadIdx
CONST R1, #0                   ; baseA
CONST R2, #8                   ; baseB
CONST R3, #17                  ; baseProducts (per-thread products)
ADD  R4, R1, R0                ; &A[i]
LDR  R4, R4
ADD  R5, R2, R0                ; &B[i]
LDR  R5, R5
MUL  R6, R4, R5                ; A[i] * B[i]
ADD  R7, R3, R0                ; &products[i]
STR  R7, R6
RET
