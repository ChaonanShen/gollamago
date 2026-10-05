import itertools
import os
import warnings

import pytest
import torch

import backends
import operators

pytestmark = pytest.mark.accelerator


@pytest.fixture
def maca_rope():
    if os.environ.get("RUN_ACCELERATOR_TESTS") != "1":
        pytest.skip("set RUN_ACCELERATOR_TESTS=1")
    if not getattr(torch.version, "maca", None):
        pytest.skip("requires MACA PyTorch")
    backends.configure_backend("maca_cpp", "cuda", "maca")
    assert operators.get_registered_operators("maca_cpp").get("rope") == "maca_cpp"
    from operators.maca_cpp import maca_kernels
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="(?s).*using torch", category=RuntimeWarning)
        yield maca_kernels.rope


SHAPES = [
    (b, s, h, 64)
    for h in (8, 32)
    for b, s in ((1, 1), (2, 3), (1, 31), (1, 32), (1, 33), (1, 63),
                 (1, 64), (1, 65), (5, 129), (2, 512), (1, 2048),
                 (1, 4096), (8, 1024))
] + [(3, 7, 5, 128), (1, 1, 24, 128), (2, 33, 8, 128),
     (1, 128, 32, 128), (2, 5, 3, 10), (2, 0, 3, 64)]


@pytest.mark.parametrize("shape", SHAPES)
def test_maca_rope_dispatch(maca_rope, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(shape[1] + 3, shape[-1] // 2, device="cuda")
    sin, cos = angles.sin().bfloat16(), angles.cos().bfloat16()
    saved = [t.clone() for t in (x, sin, cos)]
    expected = operators.dispatch("rope", x, sin, cos, backend="torch")
    actual = operators.dispatch("rope", x, sin, cos, backend="maca_cpp")
    torch.testing.assert_close(actual, expected, rtol=.04, atol=.04)
    for value, before in zip((x, sin, cos), saved):
        torch.testing.assert_close(value, before, rtol=0, atol=0)


@pytest.mark.parametrize("block_rows,threads", list(itertools.product(
    (1, 2, 4, 8, 16, 32, 64, 128), (64, 128, 256, 512))))
@pytest.mark.parametrize("head_dim", (64, 128))
def test_maca_rope_launch_configurations(maca_rope, block_rows, threads, head_dim):
    torch.manual_seed(1)
    x = torch.randn(3, 7, 5, head_dim, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(7, head_dim // 2, device="cuda")
    sin, cos = angles.sin().bfloat16(), angles.cos().bfloat16()
    expected = operators.dispatch("rope", x, sin, cos, backend="torch")
    actual = maca_rope(x, sin, cos, block_rows=block_rows, threads=threads)
    torch.testing.assert_close(actual, expected, rtol=.04, atol=.04)


@pytest.mark.parametrize("rotation", ("identity", "quarter-turn"))
def test_maca_rope_special_rotations(maca_rope, rotation):
    x = torch.randn(5, 65, 8, 64, device="cuda", dtype=torch.bfloat16)
    sin = torch.full((65, 32), float(rotation == "quarter-turn"),
                     device="cuda", dtype=x.dtype)
    cos = torch.full_like(sin, float(rotation == "identity"))
    expected = operators.dispatch("rope", x, sin, cos, backend="torch")
    torch.testing.assert_close(maca_rope(x, sin, cos), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", (
    {"block_rows": -1}, {"block_rows": 1025}, {"threads": 32},
    {"threads": 65}, {"threads": 1088},
))
def test_maca_rope_rejects_invalid_configuration(maca_rope, kwargs):
    x = torch.zeros(1, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
    table = torch.ones(1, 32, device="cuda", dtype=x.dtype)
    with pytest.raises(RuntimeError):
        maca_rope(x, table, table, **kwargs)


def test_maca_rope_current_stream(maca_rope):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(2, 33, 8, 64, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(33, 32, device="cuda")
        sin, cos = angles.sin().bfloat16(), angles.cos().bfloat16()
        actual = maca_rope(x, sin, cos)
        expected = operators.dispatch("rope", x, sin, cos, backend="torch")
        x.record_stream(stream)
        sin.record_stream(stream)
        cos.record_stream(stream)
        actual.record_stream(stream)
        expected.record_stream(stream)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=.04, atol=.04)


@pytest.mark.parametrize("name", ("rms_norm", "rope"))
def test_maca_launch_ignores_handled_prior_api_error(maca_rope, name):
    import ctypes
    from pathlib import Path
    from operators.maca_cpp import maca_kernels

    api = ctypes.CDLL(str(Path(os.environ.get("MACA_PATH", "/opt/maca")) / "lib" / "libmcruntime.so"))
    api.mcSetDevice.argtypes = [ctypes.c_int]
    api.mcSetDevice.restype = ctypes.c_int
    api.mcPeekAtLastError.restype = ctypes.c_int
    api.mcGetLastError.restype = ctypes.c_int
    if name == "rms_norm":
        x = torch.randn(2, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(2048, device="cuda", dtype=x.dtype)
        args = (x, w, 1e-5)
    else:
        x = torch.randn(2, 3, 8, 64, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(3, 32, device="cuda")
        args = (x, angles.sin().bfloat16(), angles.cos().bfloat16())
    function = getattr(maca_kernels, name)
    expected = operators.dispatch(name, *args, backend="torch")
    warmup = function(*args)
    torch.cuda.synchronize()
    del warmup
    try:
        # Invalid device selection changes no device and allocates no memory,
        # but leaves a handled error in MACA's thread-local runtime state.
        status = api.mcSetDevice(-1)
        assert status != 0 and api.mcPeekAtLastError() == status
        actual = function(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=.04, atol=.04)
        assert api.mcPeekAtLastError() == 0
    finally:
        api.mcGetLastError()
