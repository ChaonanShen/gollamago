"""NineToothed GEMM exercise based on the official matrix-multiplication tutorial."""

import os

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor


AUTOTUNE = os.environ.get("NINETOOTHED_AUTOTUNE") == "1"
if AUTOTUNE:
    BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=32, upper_bound=128)
    BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=32, upper_bound=128)
    BLOCK_SIZE_K = ninetoothed.block_size(lower_bound=32, upper_bound=128)
else:
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64


def arrangement(lhs, rhs, output):
    """Map each output tile to an lhs tile row and an rhs tile column."""
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_arranged = lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    lhs_arranged = lhs_arranged.tile((1, -1))
    lhs_arranged = lhs_arranged.expand((-1, output_arranged.shape[1]))
    lhs_arranged.dtype = lhs_arranged.dtype.squeeze(0)

    rhs_arranged = rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    rhs_arranged = rhs_arranged.tile((-1, 1))
    rhs_arranged = rhs_arranged.expand((output_arranged.shape[0], -1))
    rhs_arranged.dtype = rhs_arranged.dtype.squeeze(1)

    return lhs_arranged, rhs_arranged, output_arranged


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])
    output = accumulator


_KERNEL = ninetoothed.make(
    arrangement,
    application,
    # Masked input loads use zero so a partial K tile contributes no padding.
    # NineToothed also masks stores outside the output matrix.
    (Tensor(2, other=0), Tensor(2, other=0), Tensor(2)),
)


def nt_gemm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]),
        device=lhs.device,
        dtype=torch.float16,
    )
    _KERNEL(lhs, rhs, output)
    return output
