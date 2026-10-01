import pytest
import torch

from ninetoothed_gemm import nt_gemm


@pytest.mark.parametrize(
    "m,n,k",
    [
        (512, 512, 512),
        (2048, 2048, 2048),
        (1, 1, 1),
        (3, 7, 5),
        (31, 65, 17),
        (64, 64, 64),
        (65, 63, 70),
        (129, 97, 131),
    ],
)
@pytest.mark.parametrize("transposed", [False, True])
def test_ninetoothed_gemm(m, n, k, transposed, monkeypatch):
    # Use the same float32 accumulation policy as the NineToothed kernel.
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False
    )
    torch.manual_seed(0)
    if transposed:
        lhs = torch.randn((k, m), device="cuda", dtype=torch.float16).T
        rhs = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    else:
        lhs = torch.randn((m, k), device="cuda", dtype=torch.float16)
        rhs = torch.randn((k, n), device="cuda", dtype=torch.float16)

    output = nt_gemm(lhs, rhs)
    reference = torch.mm(lhs, rhs)

    assert output.shape == (m, n)
    assert output.dtype == torch.float16
    assert output.device == lhs.device
    torch.testing.assert_close(output, reference, atol=0.025, rtol=0.025)
