"""test_l1d_indepth at the xf widths (WORD_BITS=32, 4096-word data memory)."""
from .test_l1d_indepth import (  # noqa: F401
    test_hot_address_hits, test_read_after_write_same_thread,
    test_write_read_by_later_block, test_l1d_disable_equivalence,
)
