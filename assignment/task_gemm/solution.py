import tilelang
import tilelang.language as T

@tilelang.jit
def tl_gemm(A, B, BLOCK_N: int, BLOCK_M: int, BLOCK_K: int, threads=256, num_stages=1):
    M, N, K = T.const("M, N, K")
    dtype = T.float16 
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), accum_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (bx, by):
        midx, nidx = by * BLOCK_M, bx * BLOCK_N 

        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        T.clear(C_local)

        for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
            kidx = k * BLOCK_K
            T.copy(A[midx, kidx], A_shared)
            T.copy(B[kidx, nidx], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[midx, nidx])

    return C


