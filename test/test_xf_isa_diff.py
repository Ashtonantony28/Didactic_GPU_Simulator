"""RTL vs xf.isasim on every 32-bit ISA fixture (sim_xf.vvp)."""
from .test_isa_diff import diff_test

test_xf_isa_diff = diff_test(32)
