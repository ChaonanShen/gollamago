import torch
import triton

from ninetoothed_gemm import nt_gemm


SIZES = [2**i for i in range(3, 13)]


if __name__ == "__main__":
    # Match the exercise's float32 accumulation requirement in the baseline.
    # cuBLAS may otherwise truncate partial sums to float16 for some shapes.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(0)
    print("M=N=K\tcorrect\tNineToothed(ms)\tPyTorch(ms)")

    for size in SIZES:
        lhs = torch.randn(
            (size, size),
            device="cuda",
            dtype=torch.float16,
        )
        rhs = torch.randn_like(lhs)

        output = nt_gemm(lhs, rhs)
        reference = torch.mm(lhs, rhs)
        assert torch.allclose(output, reference, atol=0.025, rtol=0.025)

        ninetoothed_ms = triton.testing.do_bench(lambda: nt_gemm(lhs, rhs))
        torch_ms = triton.testing.do_bench(lambda: torch.mm(lhs, rhs))
        print(f"{size}\tPASS\t{ninetoothed_ms:.4f}\t\t{torch_ms:.4f}")
