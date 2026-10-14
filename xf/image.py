"""Data-memory image: named segments with a deterministic bump allocator.

JSON form: {word_bits, addr_bits, segments: {name: {base, shape, dtype, scale}}, words}.
Words are stored unsigned (two's complement mod 2**word_bits), row-major per segment.
"""
import json
import math

import numpy as np

DTYPES = {"i8": 8, "i32": 32}


class Image:
    def __init__(self, word_bits=32, addr_bits=12):
        self.word_bits, self.addr_bits = word_bits, addr_bits
        self.segments = {}
        self.words = []

    def alloc(self, name, shape, dtype, scale=None, data=None):
        if name in self.segments:
            raise ValueError(f"segment {name!r} already allocated")
        if DTYPES[dtype] > self.word_bits:
            raise ValueError(f"{dtype} does not fit {self.word_bits}-bit words")
        shape = [int(d) for d in np.atleast_1d(shape)]
        base, size = len(self.words), math.prod(shape)
        if base + size > 1 << self.addr_bits:
            raise MemoryError(f"{name}: {base + size} words exceed 2**{self.addr_bits}")
        self.segments[name] = {"base": base, "shape": shape, "dtype": dtype,
                               "scale": None if scale is None else float(scale)}
        self.words += [0] * size
        if data is not None:
            self.write(name, data)
        return base

    def write(self, name, data):
        seg = self.segments[name]
        a = np.asarray(data, dtype=np.int64)
        if list(a.shape) != seg["shape"]:
            raise ValueError(f"{name}: shape {list(a.shape)} != {seg['shape']}")
        bits = DTYPES[seg["dtype"]]
        if a.size and (a.min() < -(1 << (bits - 1)) or a.max() >= 1 << (bits - 1)):
            raise OverflowError(f"{name}: values outside {seg['dtype']}")
        mask = (1 << self.word_bits) - 1
        b = seg["base"]
        self.words[b:b + a.size] = [int(v) & mask for v in a.ravel()]

    def read(self, name, words=None):
        """Segment as a signed int64 array, from `words` (e.g. simulator memory) or the image."""
        seg = self.segments[name]
        words = self.words if words is None else words
        n = math.prod(seg["shape"])
        a = np.array(words[seg["base"]:seg["base"] + n], dtype=np.int64) & ((1 << self.word_bits) - 1)
        a = np.where(a >= 1 << (self.word_bits - 1), a - (1 << self.word_bits), a)
        return a.reshape(seg["shape"])

    def base(self, name):
        return self.segments[name]["base"]

    def to_dict(self):
        return {"word_bits": self.word_bits, "addr_bits": self.addr_bits,
                "segments": self.segments, "words": self.words}

    @classmethod
    def from_dict(cls, d):
        img = cls(d["word_bits"], d["addr_bits"])
        for name, seg in sorted(d["segments"].items(), key=lambda kv: kv[1]["base"]):
            if seg["base"] != len(img.words):
                raise ValueError(f"{name}: base {seg['base']} breaks the contiguous layout")
            img.alloc(name, seg["shape"], seg["dtype"], seg["scale"])
        if len(d["words"]) != len(img.words):
            raise ValueError(f"{len(d['words'])} words, segments cover {len(img.words)}")
        img.words = [int(w) for w in d["words"]]
        return img

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))

    def __eq__(self, other):
        return isinstance(other, Image) and self.to_dict() == other.to_dict()
