# tiny-gpu-extended

Replication of [adam-maj/tiny-gpu](https://github.com/adam-maj/tiny-gpu)
extended with a bench harness, instruction cache, pipelined decode, memory
coalescing, shared memory with barriers, an L1-D with MeDiC memory-divergence
correction, a multi-warp core with a pluggable warp scheduler, a SIMT stack,
and int8 transformer kernels on a 32-bit build.

See [`docs/final-report.md`](docs/final-report.md) for results,
[`docs/architecture.md`](docs/architecture.md) for the design, and
[`ROADMAP.md`](ROADMAP.md) for per-phase status.

## Quick start

```bash
apt-get install -y iverilog
pip3 install cocotb==2.0.1 torch   # 2.1.x breaks the Makefile; torch + NANOGPT_DIR for test_all
# sv2v 0.0.13 binary into /usr/local/bin/

make test_matadd
make test_matmul
make regression                # exits non-zero on any failure
NANOGPT_DIR=... make test_all  # every test module at default params, xf pytest, export --check
make -j4 matrix                # kernel and xf tests under 13 configurations
python3 bench/run.py --phase 5 --label phase-5-coalesce \
    --params "ICACHE_LINES=16 COALESCE=1"
```
