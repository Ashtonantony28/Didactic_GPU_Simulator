# tiny-gpu-extended Makefile
.PHONY: compile test_% clean bench bench_clean

SRC := $(wildcard src/*.sv)

# Pass parameters via DEFINES, e.g. `make test_matadd PARAMS="-PICACHE_LINES=16 -PCOALESCE=1"`.
PARAMS ?=

export PYGPI_PYTHON_BIN := $(shell which python3)
export LIBPYTHON_LOC    := $(shell cocotb-config --libpython)

build:
	mkdir -p build test/logs bench/results

compile: build
	sv2v -w build/gpu.v $(SRC)
	printf '`timescale 1ns/1ns\n' > build/_ts.v && cat build/_ts.v build/gpu.v > build/gpu_ts.v && mv build/gpu_ts.v build/gpu.v
	iverilog -o build/sim.vvp -s gpu -g2012 $(PARAMS) build/gpu.v

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
