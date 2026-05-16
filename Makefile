# tiny-gpu-extended Makefile
.PHONY: compile test_% clean bench bench_clean

SRC := $(wildcard src/*.sv)

# Pass parameters via DEFINES, e.g. `make test_matadd PARAMS="-PICACHE_LINES=16 -PCOALESCE=1"`.
PARAMS ?=

export PYGPI_PYTHON_BIN := $(shell which python3)
export LIBPYTHON_LOC    := $(shell cocotb-config --libpython)

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
	COCOTB_TEST_MODULES=test.test_warp_scheduler_$(WS_TEST) vvp -M $(shell cocotb-config --lib-dir) -m libcocotbvpi_icarus build/ws.vvp

test_warp_scheduler_gto: WS_PARAMS=-Pwarp_scheduler_tb.POLICY=1
test_warp_scheduler_gto: WS_TEST=gto
test_warp_scheduler_gto: compile_ws
	COCOTB_TEST_MODULES=test.test_warp_scheduler_$(WS_TEST) vvp -M $(shell cocotb-config --lib-dir) -m libcocotbvpi_icarus build/ws.vvp

test_warp_scheduler_gtlrr: WS_PARAMS=-Pwarp_scheduler_tb.POLICY=2
test_warp_scheduler_gtlrr: WS_TEST=gtlrr
test_warp_scheduler_gtlrr: compile_ws
	COCOTB_TEST_MODULES=test.test_warp_scheduler_$(WS_TEST) vvp -M $(shell cocotb-config --lib-dir) -m libcocotbvpi_icarus build/ws.vvp

test_%: compile
	COCOTB_TEST_MODULES=test.test_$* vvp -M $(shell cocotb-config --lib-dir) -m libcocotbvpi_icarus build/sim.vvp

regression: compile
	@for t in matadd matmul vector_dot relu conv1d transpose divergent_relu tiled_matmul; do \
	  test -f test/test_$$t.py || continue; \
	  echo "=== $$t ==="; \
	  COCOTB_TEST_MODULES=test.test_$$t vvp -M $$(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/sim.vvp || exit $$?; \
	done

clean:
	rm -rf build test/logs/*.txt

bench_clean:
	rm -f bench/results/*.json
