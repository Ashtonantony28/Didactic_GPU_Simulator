# tiny-gpu-extended Makefile
.PHONY: gpu_v compile compile_xf test_% test_xf_% regression regression_xf test_all require_nanogpt matrix matrix_run clean bench bench_clean harness_selftest test_multiwarp multiwarp_% test_divergence divergence_run simt_fatal_run

SRC := $(wildcard src/*.sv)

# Pass parameters via DEFINES, e.g. `make test_matadd PARAMS="-PICACHE_LINES=16 -PCOALESCE=1"`.
PARAMS ?=
# Build directory; give each configuration its own so they can run in parallel.
BUILD ?= build

export PYGPI_PYTHON_BIN := $(shell which python3)
export LIBPYTHON_LOC    := $(shell cocotb-config --libpython)

# $(call cocotb,<test module>,<vvp>): run cocotb, fail on any test failure or zero tests run.
cocotb = rm -f $(BUILD)/results.xml && COCOTB_TEST_MODULES=$(1) COCOTB_RESULTS_FILE=$(BUILD)/results.xml \
	vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus $(2) && python3 test/helpers/check_results.py $(BUILD)/results.xml

$(BUILD):
	mkdir -p $(BUILD) test/logs bench/results

GPU_SRC := $(filter-out src/warp_scheduler_tb.sv,$(SRC))
WS_SRC  := src/warp_scheduler.sv src/warp_scheduler_tb.sv

gpu_v: $(BUILD)
	sv2v -w $(BUILD)/gpu.v $(GPU_SRC) 2>/dev/null
	printf '`timescale 1ns/1ns\n' > $(BUILD)/_ts.v && cat $(BUILD)/_ts.v $(BUILD)/gpu.v > $(BUILD)/gpu_ts.v && mv $(BUILD)/gpu_ts.v $(BUILD)/gpu.v

compile: gpu_v
	iverilog -o $(BUILD)/sim.vvp -s gpu -g2012 $(PARAMS) $(BUILD)/gpu.v

# Transformer (xf) build: 32-bit words, 4096-word data memory; PARAMS are appended.
XF_PARAMS := -Pgpu.WORD_BITS=32 -Pgpu.DATA_MEM_DATA_BITS=32 -Pgpu.DATA_MEM_ADDR_BITS=12

compile_xf: gpu_v
	iverilog -o $(BUILD)/sim_xf.vvp -s gpu -g2012 $(XF_PARAMS) $(PARAMS) $(BUILD)/gpu.v

compile_ws: $(BUILD)
	sv2v -w $(BUILD)/ws.v $(WS_SRC)
	printf '`timescale 1ns/1ns\n' > $(BUILD)/_ts.v && cat $(BUILD)/_ts.v $(BUILD)/ws.v > $(BUILD)/ws_ts.v && mv $(BUILD)/ws_ts.v $(BUILD)/ws.v
	iverilog -o $(BUILD)/ws.vvp -s warp_scheduler_tb -g2012 $(WS_PARAMS) $(BUILD)/ws.v

test_warp_scheduler_lrr: WS_PARAMS=-Pwarp_scheduler_tb.POLICY=0
test_warp_scheduler_lrr: WS_TEST=lrr
test_warp_scheduler_lrr: compile_ws
	$(call cocotb,test.test_warp_scheduler_$(WS_TEST),$(BUILD)/ws.vvp)

test_warp_scheduler_gto: WS_PARAMS=-Pwarp_scheduler_tb.POLICY=1
test_warp_scheduler_gto: WS_TEST=gto
test_warp_scheduler_gto: compile_ws
	$(call cocotb,test.test_warp_scheduler_$(WS_TEST),$(BUILD)/ws.vvp)

test_warp_scheduler_gtlrr: WS_PARAMS=-Pwarp_scheduler_tb.POLICY=2
test_warp_scheduler_gtlrr: WS_TEST=gtlrr
test_warp_scheduler_gtlrr: compile_ws
	$(call cocotb,test.test_warp_scheduler_$(WS_TEST),$(BUILD)/ws.vvp)

# MeDiC mixed workload: the same configuration at MEDIC=0 (built in $(BUILD)/medic0) sets the hot-lane baseline.
test_medic: PARAMS = -Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1
test_medic: compile
	$(MAKE) --no-print-directory BUILD=$(BUILD)/medic0 PARAMS="$(subst MEDIC=1,MEDIC=0,$(PARAMS))" compile
	export MEDIC_BASELINE=$(BUILD)/medic0/medic.json; $(call cocotb,test.test_medic,$(BUILD)/medic0/sim.vvp)
	export MEDIC_BASELINE=$(BUILD)/medic0/medic.json; $(call cocotb,test.test_medic,$(BUILD)/sim.vvp)

# Multi-warp core (SG20): WARPS_PER_CORE=4 under each WARP_POLICY, and WARPS_PER_CORE=2, each in $(BUILD)/<cfg>.
MW_w4_lrr   := -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=0
MW_w4_gto   := -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=1
MW_w4_gtlrr := -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=2
MW_w2_lrr   := -Pgpu.WARPS_PER_CORE=2 -Pgpu.WARP_POLICY=0

test_multiwarp: $(addprefix multiwarp_,w4_lrr w4_gto w4_gtlrr w2_lrr)

multiwarp_%:
	@$(MAKE) --no-print-directory BUILD=$(BUILD)/$* PARAMS="$(MW_$*) $(PARAMS)" compile >/dev/null
	$(MAKE) --no-print-directory BUILD=$(BUILD)/$* multiwarp_run

multiwarp_run:
	$(call cocotb,test.test_multiwarp,$(BUILD)/sim.vvp)

# SIMT stack (SG21): test_divergence with the stack at WARPS_PER_CORE 1, 2, 4, NUM_CORES 1, 2, 4 and 8-lane warps, and
# with it off, each in $(BUILD)/<cfg>; then the runs of test/simt_fatal.py must end in simt_stack's $fatal.
DV_w1      := -Pgpu.SIMT_STACK=1
DV_w1c1    := -Pgpu.SIMT_STACK=1 -Pgpu.NUM_CORES=1
DV_w2c4    := -Pgpu.SIMT_STACK=1 -Pgpu.WARPS_PER_CORE=2 -Pgpu.NUM_CORES=4
DV_w4c1gto := -Pgpu.SIMT_STACK=1 -Pgpu.WARPS_PER_CORE=4 -Pgpu.NUM_CORES=1 -Pgpu.WARP_POLICY=1
DV_w4c4    := -Pgpu.SIMT_STACK=1 -Pgpu.WARPS_PER_CORE=4 -Pgpu.NUM_CORES=4
DV_t8      := -Pgpu.SIMT_STACK=1 -Pgpu.THREADS_PER_BLOCK=8
DV_t8w2    := -Pgpu.SIMT_STACK=1 -Pgpu.THREADS_PER_BLOCK=8 -Pgpu.WARPS_PER_CORE=2
DV_off     :=
DV_offw2   := -Pgpu.WARPS_PER_CORE=2
DV_CFGS    := w1 w1c1 w2c4 w4c1gto w4c4 t8 t8w2 off offw2

test_divergence: $(addprefix divergence_,$(DV_CFGS)) simt_fatal_overflow simt_fatal_nosync

divergence_%:
	@$(MAKE) --no-print-directory BUILD=$(BUILD)/$* PARAMS="$(DV_$*) $(PARAMS)" compile >/dev/null
	$(MAKE) --no-print-directory BUILD=$(BUILD)/$* divergence_run

divergence_run:
	$(call cocotb,test.test_divergence,$(BUILD)/sim.vvp)

simt_fatal_overflow: divergence_t8
	@$(MAKE) --no-print-directory BUILD=$(BUILD)/t8 SIMT_FATAL=overflow FATAL_MSG="deeper than 8" simt_fatal_run

simt_fatal_nosync: divergence_w1
	@$(MAKE) --no-print-directory BUILD=$(BUILD)/w1 SIMT_FATAL=nosync FATAL_MSG="without SYNC" simt_fatal_run

simt_fatal_run:
	@export SIMT_FATAL=$(SIMT_FATAL); ! ($(call cocotb,test.simt_fatal,$(BUILD)/sim.vvp)) > $(BUILD)/fatal_$(SIMT_FATAL).log 2>&1 \
	  && grep -q "msg: simt_stack: .*$(FATAL_MSG)" $(BUILD)/fatal_$(SIMT_FATAL).log && echo "simt_fatal $(SIMT_FATAL): OK" \
	  || { tail -n 20 $(BUILD)/fatal_$(SIMT_FATAL).log; echo "simt_fatal $(SIMT_FATAL): FAILED"; exit 1; }

test_xf_%: compile_xf
	$(call cocotb,test.test_xf_$*,$(BUILD)/sim_xf.vvp)

test_%: compile
	$(call cocotb,test.test_$*,$(BUILD)/sim.vvp)

REGRESSION := matadd matmul vector_dot relu conv1d transpose divergent_relu tiled_matmul
TESTS      := $(sort $(patsubst test/%.py,%,$(wildcard test/test_*.py)))
XF_TESTS   := $(filter test_xf_%,$(TESTS))
WS_TESTS   := $(filter test_warp_scheduler_%,$(TESTS))

# $(call suite,<vvp>,<test modules>): run every module, then fail if any failed.
suite = fail=; for t in $(2); do echo "=== $$t ==="; $(call cocotb,test.$$t,$(1)) || fail="$$fail $$t"; done; \
	test -z "$$fail" || { echo "FAILED:$$fail"; exit 1; }

regression: compile
	@$(call suite,$(BUILD)/sim.vvp,$(addprefix test_,$(REGRESSION)))

# The kernel regression at the xf widths (WORD_BITS=32, 4096-word data memory).
regression_xf: compile_xf
	@$(call suite,$(BUILD)/sim_xf.vvp,$(addprefix test_,$(REGRESSION)))

# Every test module once at default parameters, plus harness_selftest, the xf pytest suite, export --check and
# isa_fixtures --check.
# The held-out tolerance checks need torch and NANOGPT_DIR, so test_all fails rather than skips them.
test_all: require_nanogpt compile compile_xf
	@$(call suite,$(BUILD)/sim.vvp,$(filter-out $(XF_TESTS) $(WS_TESTS) test_medic,$(TESTS)))
	@$(call suite,$(BUILD)/sim_xf.vvp,$(XF_TESTS))
	@for t in $(WS_TESTS) test_medic harness_selftest; do $(MAKE) --no-print-directory $$t || exit 1; done
	XF_REQUIRE_NANOGPT=1 python3 -m pytest xf/tests -q
	python3 -m xf.export --check
	python3 -m xf.isa_fixtures --check

require_nanogpt:
	@python3 -c "import torch" 2>/dev/null && test -n "$$NANOGPT_DIR" \
	  || { echo "test_all needs torch and NANOGPT_DIR (a nanoGPT Extended checkout @ 03dbab8)"; exit 1; }

# regression, regression_xf, test_l1d_indepth, test_isa_diff, test_controller, test_multiwarp (test_divergence for simt) and every test_xf_* per
# configuration, each in build/matrix/<cfg>;
# test_medic too where MEDIC=1.
# `make -j4 matrix` runs configurations in parallel.
MATRIX := default icache pipelined coalesce dcache medic all cores1 cores4 latency warps2 warps4gto simt
CFG_icache    := -Pgpu.ICACHE_LINES=16
CFG_pipelined := -Pgpu.PIPELINED=1
CFG_coalesce  := -Pgpu.COALESCE=1
CFG_dcache    := -Pgpu.DCACHE_LINES=16
CFG_medic     := -Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1
CFG_all       := $(CFG_icache) $(CFG_pipelined) $(CFG_coalesce) $(CFG_medic)
CFG_cores1    := -Pgpu.NUM_CORES=1
CFG_cores4    := -Pgpu.NUM_CORES=4
ENV_latency   := MEM_LATENCY=3
CFG_warps2    := -Pgpu.WARPS_PER_CORE=2
CFG_warps4gto := -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=1
# W=4 runs about 4x the cycles; skip the long attention/MLP chains to keep `make -j4 matrix` within budget.
XF_warps4gto  := $(filter-out test_xf_attention test_xf_mlp_block test_xf_bench,$(XF_TESTS))
# The stack on: test_multiwarp's kernels branch divergently without SYNC, so test_divergence replaces it; xf kernels are
# block-uniform (same cycles as default), so the long chains are skipped too.
CFG_simt      := -Pgpu.SIMT_STACK=1
T8_simt       := divergence
XF_simt       := $(XF_warps4gto)

matrix: $(addprefix matrix_,$(MATRIX))
	@echo "matrix: OK"

matrix_%:
	@mkdir -p build/matrix/$*
	@$(ENV_$*) $(MAKE) --no-print-directory BUILD=build/matrix/$* PARAMS="$(CFG_$*)" MATRIX_XF="$(XF_$*)" MATRIX_T8="$(T8_$*)" matrix_run > build/matrix/$*/log.txt 2>&1 \
	  && echo "matrix $*: OK" || { tail -n 30 build/matrix/$*/log.txt; echo "matrix $*: FAILED (build/matrix/$*/log.txt)"; exit 1; }

matrix_run: compile compile_xf
	@$(call suite,$(BUILD)/sim.vvp,$(addprefix test_,$(REGRESSION) l1d_indepth isa_diff controller $(or $(MATRIX_T8),multiwarp)))
	@$(call suite,$(BUILD)/sim_xf.vvp,$(addprefix test_,$(REGRESSION)) $(or $(MATRIX_XF),$(XF_TESTS)))
	@$(if $(findstring MEDIC=1,$(PARAMS)),$(MAKE) --no-print-directory test_medic)

# A failing test, a module with no tests and a missing module must each fail the target.
harness_selftest: compile
	! ($(call cocotb,test.harness_fail,$(BUILD)/sim.vvp))
	! ($(call cocotb,test.harness_empty,$(BUILD)/sim.vvp))
	! ($(call cocotb,test.harness_missing,$(BUILD)/sim.vvp))
	@echo "harness_selftest: OK"

clean:
	rm -rf build test/logs/*.txt

bench_clean:
	rm -f bench/results/*.json
