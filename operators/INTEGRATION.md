# 算子接入指南

本仓库为 NineToothed、MXMACA 原生算子和 TileLang 算子预留了同一条接入路径。模型只调用
`operators.dispatch()`，后端由命令行参数选择；后端不可用、缺少某个算子或 kernel 运行出错时，
会提示警告并回退到 PyTorch。

## 接入点总览

```text
infer.py --backend <name> --target <target>
              |
       backends.configure_backend()
              |
       operators.dispatch("rms_norm", ...)
              |
       operators/registry.py
       (按 backend 懒加载模块并查找注册项)
          |
          +-- operators/ninetoothed_ops.py -> NineToothed kernel
          +-- operators/tilelang_ops.py    -> TileLang JIT
          +-- operators/maca_cpp/          -> setup.py + src/*.maca
                                             -> maca_kernels.rms_norm
```

当前示例算子签名是：

```python
rms_norm(input, weight, eps) -> output
```

框架还预置了 RoPE 接入点，固定签名为：

```python
rope(input, sin_table, cos_table) -> output
```

其中 `input` 为 `[batch, sequence, heads, head_dim]`，两个 table 为
`[sequence, head_dim // 2]`；输出 shape、dtype、device 与 `input` 相同。

新增算子时应先确定 Python 侧签名和张量契约（shape、dtype、device、contiguous 要求），
再让不同后端实现同一契约。

## 第一步：确定名称和调用位置

模型中的 `llama.RMSNorm.forward()` 通过 dispatch 调用算子：

```python
return operators.dispatch("rms_norm", input, self.weight, self.eps)
```

接入另一个算子时使用同样形式，例如 `operators.dispatch("my_op", input, weight, scale)`。
不要在模型里直接 import NineToothed、TileLang 或 MXMACA 扩展，否则 `--backend` 将无法
切换实现。

## 第二步：加入后端模块映射

在 `backends.py` 的 `BACKEND_NAMES` 增加后端名称（已有 `tilelang` 和 `maca_cpp` 时无需
重复添加），然后在 `operators/registry.py` 的 `_BACKEND_MODULES` 增加模块路径：

```python
_BACKEND_MODULES["my_maca"] = "operators.my_maca"
```

模块必须在导入时调用 `register_operator()`。注册表是显式的，重复注册会报错；模块或算子
不可用时框架会回退到 PyTorch，并打印警告。

## TileLang 接入

在 `operators/tilelang_ops.py` 编写 JIT kernel，并用 `backends.get_active_backend()` 读取
已解析的 target。编译函数建议用 `functools.lru_cache`，避免每次调用重新编译；wrapper
负责检查输入契约、整理 shape、调用 kernel 并恢复输出 shape。文件末尾注册：

```python
register_operator("tilelang", "my_op", my_op)
```

RoPE 不需要学员修改注册表或 `llama.py`：直接把预置的 `rope()` 槽位替换为 kernel
wrapper，并保持上述签名即可。在 kernel 尚未完成前，槽位会调用 PyTorch reference 并
打印一次警告，保证端到端命令可以运行；此时只能做正确性验证，不能做性能比较。

`configure_backend("tilelang", ..., target)` 会调用
`tilelang.utils.target.determine_target()`，因此 wrapper 不应自行猜测 `cuda`/`maca`。
在 MACA 上运行：

```shell
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend tilelang --target maca --device cuda
```

首次调用包含 JIT 编译，性能测试必须先预热。

## NineToothed 接入

在 `operators/ninetoothed_kernels/fused_rms_norm.py` 编写 kernel；
`operators/ninetoothed_ops.py` 中的 wrapper 负责检查输入契约、整理 shape、调用 kernel
并恢复输出 shape。文件末尾注册：

```python
register_operator("ninetoothed", "my_op", my_op)
```

RoPE 不需要学员修改注册表或 `llama.py`：直接把预置的 `rope()` 槽位替换为 kernel
wrapper，并保持上述签名即可。安装方法见
[NineToothed 文档](https://github.com/InfiniTensor/ninetoothed/blob/b77f930dc6c8b016e09adf33570d55a7bc8376c1/docs/source/installation.rst)，
更多算子示例见
[ninetoothed-examples](https://github.com/InfiniTensor/ninetoothed-examples/tree/e873474d4b4de8e4fa427bf245da4a02512a68b1/ops/ninetoothed/kernels)。

```shell
python -m pip install ninetoothed
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend ninetoothed --target maca --device cuda
```

首次调用包含编译开销，性能测试必须先预热。

## MXMACA 原生算子接入

推荐沿用 `operators/maca_cpp/` 的三层结构：

1. `src/my_op.maca`：实现 device kernel 和 launch 函数，使用 PyTorch 当前 stream。
2. `src/bindings.cpp`：检查 dtype、device、shape、contiguous，分配输出并暴露 Python
   函数；不支持的输入应明确报错，框架会在后端不可用时回退到 torch。
3. `setup.py`：用 `mxcc -x maca -offload-arch` 编译 `.maca`，再链接到
   `CUDAExtension`。可用 `MACA_PATH`、`MXCC`、`MACA_ARCH` 覆盖默认配置。
4. `__init__.py`：导入扩展并注册 Python 名称：

```python
from operators.registry import register_operator
from . import maca_kernels

register_operator("maca_cpp", "my_op", maca_kernels.my_op)
```

RoPE 已接入 BF16 原生实现，文件位于 `src/rope/rope.maca`，构建脚本会自动编译。
输入为 contiguous `[batch, sequence, heads, head_dim]`；两个 table 为 contiguous
`[table_sequence >= sequence, head_dim / 2]`，所有张量须在同一 MACA device 上。
kernel 展平前三维，一个 block 处理多行，每个线程旋转前后半维度的一对元素；
输出使用 FP32 中间计算后舍入到 BF16，因此与 eager Torch 的 BF16 中间舍入不保证逐位一致。

模型仍使用三个张量参数。C500 16 GiB 切片上联合扫描的默认配置为 D=64 时 8 行/256 线程，
D=128 时 4 行/256 线程；其他宽度沿用相近布局。独立实验时可显式覆盖：

```python
from operators.maca_cpp import maca_kernels
output = maca_kernels.rope(input, sin_table, cos_table, block_rows=4, threads=128)
```

`block_rows=0`、`threads=0` 表示自动选择；显式行数允许 1–1024，线程数允许
64–1024 中的 64 的倍数。D=64/128 使用专用 kernel，其他正偶数宽度使用通用路径。
C500 的配置扫描（结果保存在仓库外）：

```shell
python -m benchmarks.tune_maca_rope \
  --output /data/gollamago-data/rope-tuning/sweep.json
RUN_ACCELERATOR_TESTS=1 python -m pytest -q tests/test_maca_rope.py
```

扫描先检查每个候选的正确性，再用 CUDA Graph 测量 GPU 执行时间；
完整调用时间另行测量，包含输出分配和 Python/C++ 调用。最佳配置依输入大小与设备变化，
可用 `--block-rows` 和 `--threads` 调整候选列表。

构建和验证：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
python -c "from operators.maca_cpp import maca_kernels; print(maca_kernels.__file__)"
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend maca_cpp --target maca --device cuda
```

`maca_cpp` 强制要求 `--target maca`，MACA 版 PyTorch 仍通过 CUDA 兼容接口使用
`--device cuda`。编译生成的 `build/` 和 `.so` 是本机产物，不应提交。

## 常见问题

- 找不到算子：确认模块路径已加入 `_BACKEND_MODULES`，且模块末尾注册了完全相同的名称。
- 误跑 PyTorch：检查警告和输出 JSON 的 `backend`、`registered_operators` 字段；
  `torch_fallback` 不算原生实现。
- MACA 编译失败：检查 `MACA_PATH`、`MXCC`、`MACA_ARCH`，并确认 `torch.version.maca` 非空。
- TileLang 首次很慢：这是 JIT 编译开销；增加 warmup，不要把首次调用作为性能样本。

## 推荐的后续练习算子

1. **SwiGLU 融合 MLP**：签名可定为 `swiglu(gate, up) -> silu(gate) * up`，TileLang 中
   以向量化方式融合 SiLU、乘法和写回；MXMACA 中实现同一逐元素 kernel。这样能减少两次
   中间 tensor 读写，且容易与 PyTorch reference 做逐元素校验。
