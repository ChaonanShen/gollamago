import tilelang
import tilelang.language as T

@tilelang.jit
def tl_softmax(A, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")

    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    LOG2_E = 1.4426950408889634

    # Step 1: ceildiv 覆盖任意 N；Step 2: 分配 tile fragment。
    # Step 3: ceildiv 扫描任意 M；Step 4: 越界填充 -inf。
    # Step 5: reduce_max；Step 6: exp2 稳定指数；Step 7: reduce_sum。
    # Step 8: online 更新 running max/sum（或 lse）。
    # Step 9: 第二遍扫描并归一化；Step 10: 只写回有效行列。

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as bx:
        nidx = bx * BLOCK_N

        a_shared = T.alloc_shared((BLOCK_N, BLOCK_M), dtype)

        cur_sum = T.alloc_fragment((BLOCK_N,), dtype)
        denominator = T.alloc_fragment((BLOCK_N,), dtype)

        cur_max = T.alloc_fragment((BLOCK_N,), dtype)
        scale = T.alloc_fragment((BLOCK_N,), dtype)

        # 每一段的max记录下来，第二轮循环的时候进行scale
        block_max = T.alloc_fragment((BLOCK_N, T.ceildiv(M, BLOCK_M)), dtype)

        T.fill(block_max, -T.infinity(dtype))
        T.clear(denominator)

        # 不能先T.fill然后再T.copy，T.copy默认把越界位置填充为0
        T.annotate_safe_value({A: -T.infinity(dtype)})
        
        for m in T.Serial(T.ceildiv(M, BLOCK_M)):
            midx = m * BLOCK_M 

            
            T.copy(A[nidx, midx], a_shared)

            # 计算加入当前block之后的新max
            T.reduce_max(a_shared, cur_max, dim=1)

            # 把block_max[:,m-1]看作prev_max
            for i in T.Parallel(BLOCK_N):
                if m > 0:
                    cur_max[i] = T.max(cur_max[i], block_max[i, m-1])

            for i in T.Parallel(BLOCK_N):
                block_max[i, m] = cur_max[i]

            for i in T.Parallel(BLOCK_N):
                # 之前的sum要缩放 e^(m_old-m_new)
                if m > 0:
                    scale[i] = T.exp2((block_max[i, m-1] - cur_max[i]) * LOG2_E)
                else:
                    scale[i] = 1

            # block中的x => e^(x-m)=T.exp2((x-m(LOG2_E))
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                a_shared[i, j] = T.exp2((a_shared[i, j] - cur_max[i]) * LOG2_E)
                
            T.reduce_sum(a_shared, cur_sum, dim=1)

            for i in T.Parallel(BLOCK_N):
                denominator[i] = denominator[i] * scale[i] + cur_sum[i]

            T.copy(a_shared, B[nidx, midx])

        # 这个其实没必要Serial，但是用Parallel的话没法一下子装下整个B
        for m in T.Serial(T.ceildiv(M, BLOCK_M)):
            midx = m * BLOCK_M 

            T.copy(B[nidx, midx], a_shared)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                a_shared[i, j] = a_shared[i, j] * T.exp2((block_max[i, m] - cur_max[i]) * LOG2_E) / denominator[i]
            T.copy(a_shared, B[nidx, midx])
    return B


