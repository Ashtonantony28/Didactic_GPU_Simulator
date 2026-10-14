"""Kernel test fixtures: test/fixtures/xf/<case>.json.

{case, image, inputs, golden, reference, tol}: `inputs` are written into
their image segments before launch, `golden` holds bit-exact int outputs and
`reference` float (PyTorch) outputs, both keyed by output segment name.
"""
import json
from pathlib import Path

import numpy as np

from .image import Image

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "xf"


def _lists(d):
    return {k: np.asarray(v).tolist() for k, v in d.items()}


def save(path, case, image, inputs, golden, reference, tol):
    d = {"case": case, "image": image.to_dict(), "inputs": _lists(inputs),
         "golden": _lists(golden), "reference": _lists(reference), "tol": float(tol)}
    Path(path).write_text(json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n")


def load(path):
    d = json.loads(Path(path).read_text())
    d["image"] = Image.from_dict(d["image"])
    for k in ("inputs", "golden"):
        d[k] = {n: np.asarray(v, dtype=np.int64) for n, v in d[k].items()}
    d["reference"] = {n: np.asarray(v, dtype=np.float64) for n, v in d["reference"].items()}
    return d
