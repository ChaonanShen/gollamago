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
    block_rows = 4 # 一个block处理block_rows行，处理(block_rows, columns)小块

    @tilelang.jit(target=target, out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            input: T.Tensor((rows, columns), dtype), # (B*S, D)
            weight: T.Tensor((columns,), dtype),
            output: T.Tensor((rows, columns), dtype),
        ):
            with T.Kernel(T.ceildiv(rows, block_rows), threads=256) as br:
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
def _compile_rope(heads: int, head_dim: int, dtype: str, target: str):
    # llama3.2-1B的head_dim=64
    # 所以分块可以更多些，每行处理BR, BS

    import tilelang 
    import tilelang.language as T 

    B = T.dynamic("B")
    S = T.dynamic("S")

    BR = 1 # 一个block处理BR行 
    BS = 64

    @tilelang.jit(target=target, out_idx=[-1])
    def kernel():
        @T.prim_func 
        def main(
            input: T.Tensor((B, S, heads, head_dim), dtype), # (B, S, H, D)
            sin_table: T.Tensor((S, head_dim//2), dtype),
            cos_table: T.Tensor((S, head_dim//2), dtype),
            output: T.Tensor((B, S, heads, head_dim), dtype)
        ):
            with T.Kernel(T.ceildiv(B, BR), T.ceildiv(S, BS), heads, threads=128) as (br, bs, h):
                r = br * BR 
                s = bs * BS 
                
                first_half = T.alloc_fragment((BR, BS, head_dim//2,), T.float32)
                second_half = T.alloc_fragment((BR, BS, head_dim//2,), T.float32)

                # 复制
                for i, j, d in T.Parallel(BR, BS, head_dim//2):
                    first_half[i, j, d] = input[r+i, s+j, h, d]
                    second_half[i, j, d] = input[r+i, s+j, h, d+head_dim//2]

                # 计算
                for i, j, d in T.Parallel(BR, BS, head_dim//2):
                    # 每一行D，乘以各自位置的sin/cos就行
                    sin = T.cast(sin_table[s+j, d], T.float32)
                    cos = T.cast(cos_table[s+j, d], T.float32)
                    first = first_half[i, j, d]
                    second = second_half[i, j, d]
                    first_half[i, j, d] = first * cos - second * sin
                    second_half[i, j, d] = first * sin + second * cos

                for i, j, d in T.Parallel(BR, BS, head_dim//2):
                    output[r+i, s+j, h, d] = first_half[i, j, d]
                    output[r+i, s+j, h, d+head_dim//2] = second_half[i, j, d]

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
