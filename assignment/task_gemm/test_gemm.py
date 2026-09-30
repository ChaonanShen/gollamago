import pytest
import torch

from solution import tl_gemm


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@pytest.mark.parametrize(
    "m,n,k",
    [
        (64, 64, 32),    # one tile
        (128, 64, 96),   # multiple output and reduction tiles
        (65, 129, 73),   # partial M, N, and K tiles
        (256, 128, 128), # rectangular matrix
    ],
)
def test_gemm(m, n, k):
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    out = tl_gemm(a, b, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    ref = torch.mm(a.float(), b.float())

    assert out.shape == (m, n)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
