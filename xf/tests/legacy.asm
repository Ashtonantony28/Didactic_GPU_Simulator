;; == test_conv1d.module ==
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #16
CONST R3, #1
CONST R4, #2
ADD R5, R1, R0
LDR R5, R5
ADD R6, R5, R3
ADD R6, R1, R0
ADD R6, R6, R3
LDR R6, R6
ADD R7, R1, R0
ADD R7, R7, R4
LDR R7, R7
ADD R8, R5, R6
ADD R8, R8, R7
ADD R9, R2, R0
STR R9, R8
RET

;; == test_divergent_relu.module ==
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #8
CONST R3, #1
CONST R4, #2
ADD R5, R1, R0
LDR R5, R5
DIV R6, R0, R4
MUL R6, R6, R4
SUB R6, R0, R6
SUB R7, R3, R6
ADD R8, R5, R3
SUB R9, R5, R3
MUL R10, R7, R8
MUL R11, R6, R9
ADD R12, R10, R11
ADD R7, R2, R0
STR R7, R12
RET

;; == test_edge_cases.test_single_thread ==
CONST R1, #5
CONST R2, #3
ADD R3, R1, R2
CONST R4, #0
STR R4, R3
RET

;; == test_edge_cases.test_two_blocks_one_core ==
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #16
ADD R2, R1, R0
STR R2, R0
RET

;; == test_edge_cases.test_pure_compute ==
CONST R1, #1
CONST R2, #2
ADD R3, R1, R2
MUL R4, R3, R3
CONST R5, #0
STR R5, R4
RET

;; == test_edge_cases.test_unaligned_block ==
CONST R1, #0
ADD R2, R1, %threadIdx
CONST R3, #16
ADD R4, R3, %threadIdx
STR R4, R2
RET

;; == test_l1d_indepth.test_hot_address_hits ==
CONST R0, #0
LDR R1, R0
LDR R2, R0
LDR R3, R0
ADD R4, R1, R2
ADD R4, R4, R3
CONST R5, #16
ADD R6, R5, %threadIdx
STR R6, R4
RET

;; == test_l1d_indepth.test_read_after_write_same_thread ==
LDR R1, %threadIdx
LDR R1, %threadIdx
CONST R2, #1
ADD R1, R1, R2
STR %threadIdx, R1
LDR R3, %threadIdx
CONST R4, #8
ADD R4, R4, %threadIdx
STR R4, R3
RET

;; == test_l1d_indepth.test_write_read_by_later_block ==
LDR R1, %threadIdx
MUL R2, %blockIdx, %blockDim
ADD R2, R2, %threadIdx
CONST R3, #8
ADD R2, R2, R3
STR R2, R1
CONST R4, #7
LDR R4, R4
STR %threadIdx, R4
RET

;; == test_l1d_indepth.test_l1d_disable_equivalence ==
CONST R0, #0
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #8
ADD R3, R1, R0
LDR R4, R3
ADD R5, R2, R0
STR R5, R4
RET

;; == test_matadd.module ==
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #8
CONST R3, #16
ADD R4, R1, R0
LDR R4, R4
ADD R5, R2, R0
LDR R5, R5
ADD R6, R4, R5
ADD R7, R3, R0
STR R7, R6
RET

;; == test_matmul.module ==
; 2x2 matmul (upstream); A at 0, B at 4, C at 8
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #1
CONST R2, #2
CONST R3, #0
CONST R4, #4
CONST R5, #8
DIV R6, R0, R2
MUL R7, R6, R2
SUB R7, R0, R7
CONST R8, #0
CONST R9, #0
LOOP:
MUL R10, R6, R2
ADD R10, R10, R9
ADD R10, R10, R3
LDR R10, R10
MUL R11, R9, R2
ADD R11, R11, R7
ADD R11, R11, R4
LDR R11, R11
MUL R12, R10, R11
ADD R8, R8, R12
ADD R9, R9, R1
CMP R9, R2
BRn LOOP
ADD R9, R5, R0
STR R9, R8
RET

;; == test_param_sweep.module ==
CONST R0, #0
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #8
ADD R3, R1, R0
LDR R4, R3
ADD R5, R2, R0
STR R5, R4
RET

;; == test_relu.module ==
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #16
CONST R3, #128
ADD R4, R1, R0
LDR R4, R4
CONST R6, #0
CMP R4, R3
SUB R5, R4, R3
CMP R4, R3
BRzp 14
ADD R7, R2, R0
STR R7, R6
RET
ADD R7, R2, R0
STR R7, R4
RET

;; == test_relu.test_relu ==
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #8
ADD R4, R1, R0
LDR R4, R4
ADD R7, R2, R0
STR R7, R4
RET

;; == test_shared_mem_indepth.test_smem_roundtrip ==
CONST R0, #0
ADD R0, R0, %threadIdx
CONST R1, #1
ADD R2, R0, R1
STS R0, R2
BAR
LDS R3, R0
STR R0, R3
RET

;; == test_shared_mem_indepth.test_smem_cross_thread_read ==
CONST R0, #0
ADD R0, R0, %threadIdx
CONST R1, #10
ADD R2, R0, R1
STS R0, R2
BAR
CONST R10, #1
CONST R4, #4
ADD R5, R0, R10
DIV R6, R5, R4
MUL R6, R6, R4
SUB R5, R5, R6
LDS R7, R5
CONST R8, #16
ADD R9, R8, R0
STR R9, R7
RET

;; == test_shared_mem_indepth.test_smem_persistence ==
CONST R0, #42
CONST R1, #0
STS R1, R0
LDS R2, R1
LDS R3, R1
ADD R4, R2, R3
STR R1, R4
RET

;; == test_stress.test_long_loop ==
CONST R0, #0
ADD R0, R0, %threadIdx
CONST R1, #8
CONST R2, #0
CONST R3, #0
CONST R4, #1
loop: ADD R3, R3, R2
ADD R2, R2, R4
CMP R2, R1
BRn loop
CONST R5, #16
ADD R6, R5, %threadIdx
STR R6, R3
RET

;; == test_stress.test_many_threads ==
CONST R0, #0
ADD R0, R0, %threadIdx
MUL R1, %blockIdx, %blockDim
ADD R1, R1, R0
CONST R2, #64
ADD R3, R2, R1
ADD R4, R1, R1
STR R3, R4
RET

;; == test_tiled_matmul.module ==
ADD R0, %threadIdx, %threadIdx

;; == test_tiled_matmul.build_program ==
CONST R0, #0
ADD R0, R0, %threadIdx
CONST R1, #2
CONST R2, #4
CONST R3, #8
LDR R7, R0
STS R0, R7
ADD R8, R2, R0
LDR R9, R8
STS R8, R9
BAR
DIV R4, R0, R1
MUL R11, R4, R1
SUB R5, R0, R11
CONST R10, #0
ADD R6, R11, R10
LDS R7, R6
ADD R6, R2, R5
LDS R9, R6
MUL R12, R7, R9
CONST R10, #1
ADD R6, R11, R10
LDS R7, R6
CONST R10, #6
ADD R6, R10, R5
LDS R9, R6
MUL R11, R7, R9
ADD R12, R12, R11
ADD R6, R3, R0
STR R6, R12
RET

;; == test_transpose.module ==
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #8
CONST R3, #4
CONST R4, #2
DIV R5, R0, R3
MUL R6, R5, R3
SUB R7, R0, R6
ADD R8, R1, R0
LDR R8, R8
MUL R9, R7, R4
ADD R9, R9, R5
ADD R9, R9, R2
STR R9, R8
RET

;; == test_vector_dot.module ==
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx
CONST R1, #0
CONST R2, #8
CONST R3, #17
ADD R4, R1, R0
LDR R4, R4
ADD R5, R2, R0
LDR R5, R5
MUL R6, R4, R5
ADD R7, R3, R0
STR R7, R6
RET

;; == test_medic.program.0 ==
CONST R12, #2
DIV R0, %threadIdx, R12
CONST R7, #4
MUL R7, R7, %blockIdx
MUL R7, R7, R0
SUB R2, %threadIdx, R12
ADD R2, R2, R7
CONST R11, #16
MUL R1, %blockIdx, R12
ADD R1, R1, R11
ADD R1, R1, %threadIdx
SUB R5, R2, R1
MUL R5, R5, R0
ADD R5, R1, R5
CONST R3, #32
MUL R6, R3, R0
SUB R3, R3, R6
CONST R4, #0
CONST R10, #240
MUL R9, %blockIdx, %blockDim
ADD R10, R10, R9
ADD R10, R10, %threadIdx

;; == test_medic.program.1 ==
LDR R6, R5
ADD R4, R4, R6
STR R10, R4
ADD R5, R5, R3

;; == test_medic.program.2 ==
RET
