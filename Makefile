# tiny-gpu-extended Makefile
.PHONY: compile test_% clean bench bench_clean harness_selftest

SRC := $(wildcard src/*.sv)

# Pass parameters via DEFINES, e.g. `make test_matadd PARAMS="-PICACHE_LINES=16 -PCOALESCE=1"`.
PARAMS ?=

export PYGPI_PYTHON_BIN := $(shell which python3)
export LIBPYTHON_LOC    := $(shell cocotb-config --libpython)

# $(call cocotb,<test module>,<vvp>): run cocotb, fail on any test failure or zero tests run.
cocotb = rm -f build/results.xml && COCOTB_TEST_MODULES=$(1) COCOTB_RESULTS_FILE=build/results.xml \
	vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus $(2) && python3 test/helpers/check_results.py build/results.xml

build:
	mkdir -p build test/logs bench/results

GPU_SRC := $(filter-out src/warp_scheduler_tb.sv,$(SRC))
WS_SRC  := src/warp_scheduler.sv src/warp_scheduler_tb.sv

compile: build
	sv2v -w build/gpu.v $(GPU_SRC) 2>/dev/null
	printf '`timescale 1ns/1ns\n' > build/_ts.v && cat build/_ts.v build/gpu.v > build/gpu_ts.v && mv build/gpu_ts.v build/gpu.v
	iverilog -o build/sim.vvp -s gpu -g2012 $(PARAMS) build/gpu.v

compile_ws: build
	sv2v -w build/ws.v $(WS_SRC)
	printf '`timescale 1ns/1ns\n' > build/_ts.v && cat build/_ts.v build/ws.v > build/ws_ts.v && mv build/ws_ts.v build/ws.v
	iverilog -o build/ws.vvp -s warp_scheduler_tb -g2012 $(WS_PARAMS) build/ws.v

test_warp_scheduler_lrr: WS_PARAMS=-Pwarp_scheduler_tb.POLICY=0
test_warp_scheduler_lrr: WS_TEST=lrr
test_warp_scheduler_lrr: compile_ws
	$(call cocotb,test.test_warp_scheduler_$(WS_TEST),build/ws.vvp)

test_warp_scheduler_gto: WS_PARAMS=-Pwarp_scheduler_tb.POLICY=1
test_warp_scheduler_gto: WS_TEST=gto
test_warp_scheduler_gto: compile_ws
	$(call cocotb,test.test_warp_scheduler_$(WS_TEST),build/ws.vvp)

test_warp_scheduler_gtlrr: WS_PARAMS=-Pwarp_scheduler_tb.POLICY=2
test_warp_scheduler_gtlrr: WS_TEST=gtlrr
test_warp_scheduler_gtlrr: compile_ws
	$(call cocotb,test.test_warp_scheduler_$(WS_TEST),build/ws.vvp)

test_%: compile
	$(call cocotb,test.test_$*,build/sim.vvp)

regression: compile
	@for t in matadd matmul vector_dot relu conv1d transpose divergent_relu tiled_matmul; do \
	  echo "=== $$t ==="; \
	  $(call cocotb,test.test_$$t,build/sim.vvp) || exit 1; \
	done

# A failing test, a module with no tests and a missing module must each fail the target.
harness_selftest: compile
	! ($(call cocotb,test.harness_fail,build/sim.vvp))
	! ($(call cocotb,test.harness_empty,build/sim.vvp))
	! ($(call cocotb,test.harness_missing,build/sim.vvp))
	@echo "harness_selftest: OK"

clean:
	rm -rf build test/logs/*.txt

bench_clean:
	rm -f bench/results/*.json
