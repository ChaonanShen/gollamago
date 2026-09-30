import time

import torch

from solution import tl_gemm


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
CASES = [
    (64, 64, 32),
    (128, 64, 96),
    (65, 129, 73),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


def timed(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / repeat


if __name__ == "__main__":
    torch.manual_seed(0)
    print(
        f"{'M':>5} {'N':>5} {'K':>5} {'correct':>7} "
        f"{'TileLang(ms)':>14} {'PyTorch(ms)':>13} "
        f"{'max_abs_error':>15} {'compile(ms)':>12}"
    )
    for m, n, k in CASES:
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)

        compile_start = time.perf_counter()
        kernel = tl_gemm.compile(
            M=m, N=n, K=k, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
        compile_ms = (time.perf_counter() - compile_start) * 1e3

        out = kernel(a, b)
        ref = torch.mm(a.float(), b.float())
        assert out.shape == (m, n)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        max_error = (out - ref).abs().max().item()

        # Both paths return float32; the PyTorch baseline includes input conversion.
        tilelang_ms = timed(lambda: kernel(a, b))
        pytorch_ms = timed(lambda: torch.mm(a.float(), b.float()))
        print(
            f"{m:5d} {n:5d} {k:5d} {'PASS':>7} "
            f"{tilelang_ms:14.4f} {pytorch_ms:13.4f} "
            f"{max_error:15.3e} {compile_ms:12.1f}"
        )
