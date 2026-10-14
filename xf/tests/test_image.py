import numpy as np
import pytest

from xf import fixture
from xf.image import Image


def _build(seed=0):
    rng = np.random.default_rng(seed)
    img = Image()
    img.alloc("x", [4, 8], "i8", scale=0.05, data=rng.integers(-128, 128, (4, 8)))
    img.alloc("w", [8, 16], "i8", scale=0.01, data=rng.integers(-128, 128, (8, 16)))
    img.alloc("M", [1], "i32", data=[(1 << 31) - 1])
    img.alloc("y", [4, 16], "i8")
    return img


def test_layout_is_sequential_and_deterministic():
    img = _build()
    assert [img.base(n) for n in ("x", "w", "M", "y")] == [0, 32, 160, 161]
    assert len(img.words) == 225
    assert _build().to_json() == img.to_json()


def test_json_round_trip():
    img = _build()
    back = Image.from_json(img.to_json())
    assert back == img and back.to_json() == img.to_json()
    for name in img.segments:
        assert np.array_equal(back.read(name), img.read(name))


@pytest.mark.parametrize("word_bits", [8, 32])
def test_signed_values_two_complement(word_bits):
    img = Image(word_bits=word_bits, addr_bits=8)
    img.alloc("v", [5], "i8", data=[-128, -1, 0, 1, 127])
    assert img.words == [w % (1 << word_bits) for w in (-128, -1, 0, 1, 127)]
    assert img.read("v").tolist() == [-128, -1, 0, 1, 127]


def test_i32_extremes():
    img = Image()
    img.alloc("v", [2], "i32", data=[-(1 << 31), (1 << 31) - 1])
    assert img.words == [1 << 31, (1 << 31) - 1]
    assert img.read("v").tolist() == [-(1 << 31), (1 << 31) - 1]


def test_read_from_simulator_words():
    img = _build()
    mem = list(img.words) + [0] * 10
    mem[img.base("y")] = 0xFFFFFF80
    assert img.read("y", mem)[0, 0] == -128


@pytest.mark.parametrize("data", [[128], [-129]])
def test_write_rejects_out_of_range(data):
    img = Image()
    img.alloc("v", [1], "i8")
    with pytest.raises(OverflowError):
        img.write("v", data)


def test_rejects_bad_alloc():
    img = Image(word_bits=8, addr_bits=4)
    with pytest.raises(ValueError):
        img.alloc("v", [1], "i32")
    img.alloc("a", [16], "i8")
    with pytest.raises(ValueError):
        img.alloc("a", [1], "i8")
    with pytest.raises(MemoryError):
        img.alloc("b", [1], "i8")
    with pytest.raises(ValueError):
        img.write("a", np.zeros(15))


def test_full_xf_memory_fits():
    img = Image()
    img.alloc("all", [4096], "i8")
    with pytest.raises(MemoryError):
        img.alloc("more", [1], "i8")


def test_from_dict_rejects_inconsistent():
    d = _build().to_dict()
    d["words"] = d["words"][:-1]
    with pytest.raises(ValueError):
        Image.from_dict(d)
    d = _build().to_dict()
    d["segments"]["y"]["base"] += 1
    with pytest.raises(ValueError):
        Image.from_dict(d)


def test_fixture_round_trip(tmp_path):
    img = _build()
    path = tmp_path / "case.json"
    golden = {"y": np.arange(64).reshape(4, 16) - 32}
    fixture.save(path, "case", img, {"x": img.read("x")}, golden, {"y": golden["y"] * 0.5}, 0.125)
    first = path.read_bytes()
    fixture.save(path, "case", img, {"x": img.read("x")}, golden, {"y": golden["y"] * 0.5}, 0.125)
    assert path.read_bytes() == first
    d = fixture.load(path)
    assert d["case"] == "case" and d["tol"] == 0.125 and d["image"] == img
    assert np.array_equal(d["inputs"]["x"], img.read("x"))
    assert np.array_equal(d["golden"]["y"], golden["y"])
    assert np.allclose(d["reference"]["y"], golden["y"] * 0.5)
