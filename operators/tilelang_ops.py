"""TileLang RMSNorm example."""

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

# 还要特别注意一点，在C500中一些参数可能不一样，要细调下

@functools.lru_cache(maxsize=None)
def _compile_rms_norm(columns: int, eps: float, dtype: str, target: str):
    import tilelang
    import tilelang.language as T

    rows = T.dynamic("rows")
    block_columns = 512

    @tilelang.jit(target=target, out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            input: T.Tensor((rows, columns), dtype), # (B*S, D)
            weight: T.Tensor((columns,), dtype),
            output: T.Tensor((rows, columns), dtype),
        ):
            # TODO: 一个block就处理一行？
            with T.Kernel(rows, threads=128) as row:
                input_shared = T.alloc_shared((block_columns,), dtype)
                square_fragment = T.alloc_fragment((block_columns,), T.float32)
                square_sum = T.alloc_fragment((1,), T.float32)
                mean = T.alloc_fragment((1,), dtype)
                inverse_rms = T.alloc_fragment((1,), dtype)

                T.clear(square_fragment)
                for chunk in range(T.ceildiv(columns, block_columns)):
                    T.copy(input[row, chunk * block_columns], input_shared)
                    for offset in T.Parallel(block_columns):
                        square_fragment[offset] += (
                            input_shared[offset] * input_shared[offset]
                        )

                T.reduce_sum(square_fragment, square_sum, dim=0)
                mean[0] = square_sum[0] / columns
                inverse_rms[0] = T.rsqrt(mean[0] + eps)

                for chunk in range(T.ceildiv(columns, block_columns)):
                    for offset in T.Parallel(block_columns):
                        index = chunk * block_columns + offset
                        if index < columns:
                            output[row, index] = T.cast(
                                input[row, index] * inverse_rms[0], dtype
                            ) * weight[index]

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
def _compile_rope(heads: int, head_dim: int, dtype: str, target: str):
    # llama3.2-1B的head_dim=64

    import tilelang 
    import tilelang.language as T 

    B = T.dynamic("B")
    S = T.dynamic("S")

    # BR = 4 # 一个block处理BR行 
    # BS = 4

    @tilelang.jit(target=target, out_idx=[-1])
    def kernel():
        @T.prim_func 
        def main(
            input: T.Tensor((B, S, heads, head_dim), dtype), # (B, S, H, D)
            sin_table: T.Tensor((S, head_dim//2), dtype),
            cos_table: T.Tensor((S, head_dim//2), dtype),
            output: T.Tensor((B, S, heads, head_dim), dtype)
        ):
            with T.Kernel(B, S, heads, threads=128) as (r, s, h):
                
                first_half = T.alloc_fragment((head_dim//2,), T.float32)
                second_half = T.alloc_fragment((head_dim//2,), T.float32)

                # 复制
                for d in T.Parallel(head_dim//2):
                    first_half[d] = input[r, s, h, d]
                    second_half[d] = input[r, s, h, d+head_dim//2]

                # 计算
                for d in T.Parallel(head_dim//2):
                    # 每一行D，乘以各自位置的sin/cos就行
                    sin = T.cast(sin_table[s, d], T.float32)
                    cos = T.cast(cos_table[s, d], T.float32)
                    first = first_half[d]
                    second = second_half[d]
                    first_half[d] = first * cos - second * sin
                    second_half[d] = first * sin + second * cos

                for d in T.Parallel(head_dim//2):
                    output[r, s, h, d] = first_half[d]
                    output[r, s, h, d+head_dim//2] = second_half[d]

        return main
    
    return kernel()

# rms_norm是attention/mlp之前做的，还没分heads
# rope是attention之中做的，已经分了heads

def rope(input: torch.Tensor, sin_table: torch.Tensor, cos_table: torch.Tensor) -> torch.Tensor:
    # input: [B, S, H, D], sin_table: [S, D//2]
    H, D = input.shape[-2], input.shape[-1]
    if D % 2 != 0:
        raise ValueError("ROPE dimension size should be even")
    if D != sin_table.shape[-1] * 2 or D != cos_table.shape[-1] * 2:
        raise ValueError("ROPE dimension size mismatch")
    if not input.is_contiguous():
        raise ValueError("TileLang rope expects contiguous tensors")
    kernel = _compile_rope(H, D, _tilelang_dtype(input.dtype), _target())
    return kernel(input, sin_table, cos_table)

register_operator("tilelang", "rope", rope)
