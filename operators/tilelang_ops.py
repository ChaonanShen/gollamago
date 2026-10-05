"""TileLang RMSNorm and RoPE kernels."""

import functools
import warnings

import torch

import backends
from .registry import register_operator


_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _tilelang_dtype(dtype: torch.dtype) -> str:
    try:
        return _DTYPES[dtype]
    except KeyError as error:
        raise TypeError(f"TileLang examples do not support dtype {dtype}") from error


def _target() -> str:
    config = backends.get_active_backend()
    if config.backend != "tilelang" or config.target is None:
        raise RuntimeError("TileLang operator called without an active TileLang backend")
    return config.target

# Defaults measured on MetaX C500 with Llama-3.2-1B BF16 shapes.

@functools.lru_cache(maxsize=None)
def _compile_rms_norm(columns: int, eps: float, dtype: str, target: str):
    import tilelang
    import tilelang.language as T

    rows = T.dynamic("rows")
    block_rows = 1
    threads = 64  # One C500 warp reduces one complete row.

    @tilelang.jit(target=target, out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            input: T.Tensor((rows, columns), dtype), # (B*S, D)
            weight: T.Tensor((columns,), dtype),
            output: T.Tensor((rows, columns), dtype),
        ):
            with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as br:
                row = br * block_rows

                input_shared = T.alloc_shared((block_rows, columns), dtype)
                square_fragment = T.alloc_fragment((block_rows, columns), T.float32)
                square_sum = T.alloc_fragment((block_rows,), T.float32)
                mean = T.alloc_fragment((block_rows,), dtype)
                inverse_rms = T.alloc_fragment((block_rows,), dtype)
                weight_local = T.alloc_fragment((columns,), dtype)

                T.copy(weight, weight_local)

                T.clear(square_fragment)
                for i, j in T.Parallel(block_rows, columns):
                    T.copy(input[row+i, j], input_shared[i, j])
                for i, j in T.Parallel(block_rows, columns):
                    tmp = T.cast(input_shared[i, j], T.float32)
                    square_fragment[i, j] = tmp * tmp

                T.reduce_sum(square_fragment, square_sum, dim=1)
                for i in T.Parallel(block_rows):
                    mean[i] = square_sum[i] / columns
                    inverse_rms[i] = T.rsqrt(mean[i] + eps)

                for i, j in T.Parallel(block_rows, columns):
                    if row+i < rows:
                        output[row+i, j] = T.cast(input_shared[i, j], T.float32) * inverse_rms[i] * weight_local[j]

        return main

    return kernel()


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if input.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight size must match the final input dimension")
    if not input.is_contiguous() or not weight.is_contiguous():
        raise ValueError("TileLang rms_norm expects contiguous tensors")
    columns = input.shape[-1]
    kernel = _compile_rms_norm(columns, eps, _tilelang_dtype(input.dtype), _target())
    # input原本是[B, S, D]，传入kernel转为[B*S, D]
    return kernel(input.reshape(-1, columns), weight).reshape(input.shape)


register_operator("tilelang", "rms_norm", rms_norm)

@functools.lru_cache(maxsize=None)
def _compile_rope(
    heads: int, head_dim: int, dtype: str, target: str, block_rows: int | None = None,
):
    import tilelang
    import tilelang.language as T

    # A6000 sweep: 16 rows for D=64, 4 rows for D=128 across BF16/FP16/FP32.
    # Keep an explicit override for tuning; these defaults are not C500 measurements.
    if block_rows is None:
        block_rows = 4 if head_dim >= 128 else 16

    batch = T.dynamic("batch")
    sequence = T.dynamic("sequence")
    rows = batch * sequence * heads
    half = head_dim // 2

    @tilelang.jit(target=target, out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            input: T.Tensor((batch, sequence, heads, head_dim), dtype),
            sin_table: T.Tensor((sequence, half), dtype),
            cos_table: T.Tensor((sequence, half), dtype),
            output: T.Tensor((batch, sequence, heads, head_dim), dtype),
        ):
            # Buffer aliases flatten indexing without Python tensor views or copies.
            input_flat = T.reshape(input, (rows, head_dim))
            output_flat = T.reshape(output, (rows, head_dim))
            with T.Kernel(T.ceildiv(rows, block_rows), threads=128) as br:
                row_start = br * block_rows
                first_half = T.alloc_fragment((block_rows, half), T.float32)
                second_half = T.alloc_fragment((block_rows, half), T.float32)

                for i, d in T.Parallel(block_rows, half):
                    if row_start + i < rows:
                        first_half[i, d] = input_flat[row_start + i, d]
                        second_half[i, d] = input_flat[row_start + i, d + half]

                for i, d in T.Parallel(block_rows, half):
                    if row_start + i < rows:
                        # row = (batch_idx * sequence + seq_pos) * heads + head_idx
                        seq_pos = ((row_start + i) // heads) % sequence
                        sin = T.cast(sin_table[seq_pos, d], T.float32)
                        cos = T.cast(cos_table[seq_pos, d], T.float32)
                        first = first_half[i, d]
                        second = second_half[i, d]
                        first_half[i, d] = first * cos - second * sin
                        second_half[i, d] = first * sin + second * cos

                for i, d in T.Parallel(block_rows, half):
                    if row_start + i < rows:
                        output_flat[row_start + i, d] = first_half[i, d]
                        output_flat[row_start + i, d + half] = second_half[i, d]

        return main

    return kernel()

# rms_norm是attention/mlp之前做的，还没分heads
# rope是attention之中做的，已经分了heads

def rope(input: torch.Tensor, sin_table: torch.Tensor, cos_table: torch.Tensor) -> torch.Tensor:
    # The kernel aliases contiguous [B, S, H, D] as [B*S*H, D].
    _, sequence, H, D = input.shape
    if D <= 0 or D % 2 != 0:
        raise ValueError("ROPE dimension size should be positive and even")
    if sin_table.ndim != 2 or cos_table.ndim != 2:
        raise ValueError("RoPE tables must be two-dimensional")
    if D != sin_table.shape[1] * 2 or D != cos_table.shape[1] * 2:
        raise ValueError("ROPE dimension size mismatch")
    if sin_table.shape[0] < sequence or cos_table.shape[0] < sequence:
        raise ValueError("RoPE tables are too short for the input sequence")
    if not input.is_contiguous():
        raise ValueError("TileLang rope expects contiguous tensors")
    if input.numel() == 0:
        return torch.empty_like(input)
    if sin_table.shape[0] > sequence:
        sin_table = sin_table[:sequence]
    if cos_table.shape[0] > sequence:
        cos_table = cos_table[:sequence]
    kernel = _compile_rope(H, D, _tilelang_dtype(input.dtype), _target())
    return kernel(input, sin_table, cos_table)

register_operator("tilelang", "rope", rope)
