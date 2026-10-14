# tiny-gpu-extended Makefile
.PHONY: gpu_v compile compile_xf test_% test_xf_% regression regression_xf test_all matrix matrix_run clean bench bench_clean harness_selftest

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

# Every test module once at default parameters, plus harness_selftest and the xf pytest suite.
test_all: compile compile_xf
	@$(call suite,$(BUILD)/sim.vvp,$(filter-out $(XF_TESTS) $(WS_TESTS),$(TESTS)))
	@$(call suite,$(BUILD)/sim_xf.vvp,$(XF_TESTS))
	@for t in $(WS_TESTS) harness_selftest; do $(MAKE) --no-print-directory $$t || exit 1; done
	python3 -m pytest xf/tests -q

# regression, regression_xf, test_l1d_indepth and every test_xf_* per configuration, each in build/matrix/<cfg>.
# `make -j4 matrix` runs configurations in parallel.
MATRIX := default icache pipelined coalesce dcache medic all cores1 cores4 latency
CFG_icache    := -Pgpu.ICACHE_LINES=16
CFG_pipelined := -Pgpu.PIPELINED=1
CFG_coalesce  := -Pgpu.COALESCE=1
CFG_dcache    := -Pgpu.DCACHE_LINES=16
CFG_medic     := -Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1
CFG_all       := $(CFG_icache) $(CFG_pipelined) $(CFG_coalesce) $(CFG_medic)
CFG_cores1    := -Pgpu.NUM_CORES=1
CFG_cores4    := -Pgpu.NUM_CORES=4
ENV_latency   := MEM_LATENCY=3

matrix: $(addprefix matrix_,$(MATRIX))
	@echo "matrix: OK"

matrix_%:
	@mkdir -p build/matrix/$*
	@$(ENV_$*) $(MAKE) --no-print-directory BUILD=build/matrix/$* PARAMS="$(CFG_$*)" matrix_run > build/matrix/$*/log.txt 2>&1 \
	  && echo "matrix $*: OK" || { tail -n 30 build/matrix/$*/log.txt; echo "matrix $*: FAILED (build/matrix/$*/log.txt)"; exit 1; }

matrix_run: compile compile_xf
	@$(call suite,$(BUILD)/sim.vvp,$(addprefix test_,$(REGRESSION) l1d_indepth))
	@$(call suite,$(BUILD)/sim_xf.vvp,$(addprefix test_,$(REGRESSION)) $(XF_TESTS))

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
