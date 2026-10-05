import os
import warnings

import pytest
import torch

import backends
import operators


pytestmark = pytest.mark.accelerator

# Llama-3.2-1B only: hidden=2048, Q/K heads=32/8, head_dim=64.
# Include decode, prefill, and tails around RMSNorm rows=32 / RoPE B=4,S=64.
LLAMA1B_BATCH_SEQUENCE = [
    (1, 1), (2, 3), (2, 5), (4, 1), (5, 1), (8, 1),
    (1, 31), (1, 32), (1, 33), (1, 63), (1, 64), (1, 65),
    (4, 127), (4, 128), (5, 129), (2, 512),
    (1, 2048), (1, 4096), (8, 1024),
]


def require_accelerator_tests():
    if os.environ.get("RUN_ACCELERATOR_TESTS") != "1":
        pytest.skip("set RUN_ACCELERATOR_TESTS=1 to run accelerator tests")


@pytest.fixture(autouse=True)
def reject_torch_fallback():
    with warnings.catch_warnings():
        # Exercise dispatch, but fail if it tries to replace a backend with Torch.
        warnings.filterwarnings("error", message="(?s).*using torch", category=RuntimeWarning)
        yield


@pytest.fixture
def tilelang_device():
    require_accelerator_tests()
    config = backends.configure_backend("tilelang", "cuda", "auto")
    assert config.backend == "tilelang", "TileLang tests require an active TileLang backend"
    registered = operators.get_registered_operators("tilelang")
    assert registered.get("rms_norm") == "tilelang"
    assert registered.get("rope") == "tilelang"
    return config.device


@pytest.mark.parametrize(
    "shape",
    [(batch, sequence, 2048) for batch, sequence in LLAMA1B_BATCH_SEQUENCE],
    ids=[f"1b-b{batch}-s{sequence}" for batch, sequence in LLAMA1B_BATCH_SEQUENCE],
)
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        pytest.param(torch.float32, 1e-5, 1e-5, id="float32"),
        pytest.param(torch.float16, 4e-3, 7e-3, id="float16"),
        pytest.param(torch.bfloat16, 2e-2, 7e-2, id="bfloat16"),
    ],
)
def test_tilelang_rms_norm_on_detected_accelerator(tilelang_device, shape, dtype, rtol, atol):
    torch.manual_seed(0)
    input = torch.randn(shape, device=tilelang_device, dtype=dtype)
    weight = torch.randn(shape[-1], device=tilelang_device, dtype=dtype)
    input_before = input.clone()
    weight_before = weight.clone()

    expected = operators.dispatch("rms_norm", input, weight, 1e-5, backend="torch")
    actual = operators.dispatch("rms_norm", input, weight, 1e-5, backend="tilelang")

    assert actual.shape == input.shape
    assert actual.dtype == input.dtype
    assert actual.device == input.device
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(input, input_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (batch, sequence, heads, 64)
        for heads in (8, 32)
        for batch, sequence in LLAMA1B_BATCH_SEQUENCE
    ],
    ids=[
        f"1b-{'key' if heads == 8 else 'query'}-b{batch}-s{sequence}"
        for heads in (8, 32)
        for batch, sequence in LLAMA1B_BATCH_SEQUENCE
    ],
)
@pytest.mark.parametrize(
    "dtype,tolerance",
    [
        pytest.param(torch.float32, 1e-5, id="float32"),
        pytest.param(torch.float16, 4e-3, id="float16"),
        pytest.param(torch.bfloat16, 4e-2, id="bfloat16"),
    ],
)
def test_tilelang_rope_on_detected_accelerator(tilelang_device, shape, dtype, tolerance):
    torch.manual_seed(0)
    input = torch.randn(shape, device=tilelang_device, dtype=dtype)
    sequence, half = shape[1], shape[-1] // 2
    angles = torch.randn(sequence, half, device=tilelang_device)
    sin = angles.sin().to(dtype)
    cos = angles.cos().to(dtype)
    before = input.clone()
    sin_before, cos_before = sin.clone(), cos.clone()

    expected = operators.dispatch("rope", input, sin, cos, backend="torch")
    actual = operators.dispatch("rope", input, sin, cos, backend="tilelang")

    assert actual.shape == input.shape
    assert actual.dtype == input.dtype
    assert actual.device == input.device
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(input, before, rtol=0, atol=0)
    torch.testing.assert_close(sin, sin_before, rtol=0, atol=0)
    torch.testing.assert_close(cos, cos_before, rtol=0, atol=0)


@pytest.mark.parametrize("rotation", ["identity", "quarter-turn"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["float32", "float16", "bfloat16"])
def test_tilelang_rope_special_rotations(tilelang_device, rotation, dtype):
    torch.manual_seed(0)
    input = torch.randn(5, 65, 8, 64, device=tilelang_device, dtype=dtype)
    sin = torch.full((65, 32), float(rotation == "quarter-turn"), device=tilelang_device, dtype=dtype)
    cos = torch.full((65, 32), float(rotation == "identity"), device=tilelang_device, dtype=dtype)

    expected = operators.dispatch("rope", input, sin, cos, backend="torch")
    actual = operators.dispatch("rope", input, sin, cos, backend="tilelang")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_maca_cpp_rms_norm_on_mxmaca():
    require_accelerator_tests()
    if not getattr(torch.version, "maca", None):
        pytest.skip("requires a MACA-enabled PyTorch build")
    config = backends.configure_backend("maca_cpp", "cuda", "maca")
    assert config.backend == "maca_cpp"
    assert operators.get_registered_operators("maca_cpp").get("rms_norm") == "maca_cpp"
    input = torch.randn(2, 3, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)

    actual = operators.dispatch("rms_norm", input, weight, 1e-5, backend="maca_cpp")
    expected = operators.dispatch("rms_norm", input, weight, 1e-5, backend="torch")

    # FP32 square/normalization intermediates use the BF16 benchmark tolerance.
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=7e-2)


def test_ninetoothed_rms_norm_on_mxmaca():
    require_accelerator_tests()
    if not getattr(torch.version, "maca", None):
        pytest.skip("requires a MACA-enabled PyTorch build")
    config = backends.configure_backend("ninetoothed", "cuda", "maca")
    assert config.backend == "ninetoothed"
    assert operators.get_registered_operators("ninetoothed").get("rms_norm") == "ninetoothed"
    input = torch.randn(2, 3, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)

    actual = operators.dispatch("rms_norm", input, weight, 1e-5, backend="ninetoothed")
    expected = operators.dispatch("rms_norm", input, weight, 1e-5, backend="torch")

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.07)
