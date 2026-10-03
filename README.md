
# 多范式算子开发实战营

模力方舟使用算力券租用沐曦C500的镜像选择：`TileLang / 0.1.9 / Python 3.12 / maca 3.3.0.4`

![镜像选择](image.png)

开始阶段先完成四个 Assignment，再进入下面的 Llama 算子练习项目。

| 任务 | 内容 | 目录 |
|---|---|---|
| 1 | 沐曦算力券、`mx-smi` | [`assignment/task1`](assignment/task1/README.md) |
| 2 | TileLang Add 与 NineToothed Vector Add：Tile、尾块和性能测试 | [`assignment/task2`](assignment/task2/README.md) |
| 3 | TileLang Softmax 与 NineToothed GEMM：归约、数值稳定性和矩阵乘 | [`assignment/task3`](assignment/task3/README.md) |
| 4 | AI Agent 辅助算子开发、验证与优化 | [`assignment/task4`](assignment/task4/README.md) |

```bash
cd assignment/task2
python -m pytest -q test_add.py test_ninetoothed_add.py
python benchmark_add.py
python benchmark_ninetoothed_add.py

cd ../task3
python -m pytest -q test_softmax.py test_ninetoothed_gemm.py
python benchmark_softmax.py
python benchmark_ninetoothed_gemm.py
```

完成四个 Assignment 后，继续 Llama 阶段：算子接入 → 正确性验证 → 性能优化 → 端到端评测。

## Llama 算子练习

这是一个小型 Llama 推理项目，用来练习 PyTorch、NineToothed、TileLang 和
MXMACA 算子。

## 快速开始

安装依赖：

```shell
python -m pip install -r requirements.txt
```

使用 NineToothed 时单独安装可选依赖：

```shell
python -m pip install ninetoothed
```

通过 ModelScope 下载模型：

```shell
python -m pip install modelscope
modelscope download --model LLM-Research/Llama-3.2-1B \
  --local_dir models/Llama-3.2-1B
```

## 选择后端

`--backend` 决定使用哪份算子代码：

| 后端 | 用途 |
|---|---|
| `torch` | 参考实现和 CPU 基线 |
| `tilelang` | TileLang kernel |
| `maca_cpp` | MXMACA `.maca` kernel |
| `ninetoothed` | NineToothed kernel |

`--target` 可选 `auto`、`cuda`、`maca`。MACA 版 PyTorch 仍使用 `--device cuda`。
如果后端、target、扩展或某个算子不可用，或者 kernel 运行出错，框架会打印警告并自动使用 PyTorch。
推理最终输出的 `registered_operators` 会列出当前后端已接入的算子；值为 `torch_fallback`
表示该算子暂时仍调用 PyTorch reference。

NineToothed 测试：

```shell
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend ninetoothed --target maca --device cuda
```

本仓库只保留 Llama 接入所需的薄包装和一个 RMSNorm kernel。安装和入门见
[NineToothed 文档](https://github.com/InfiniTensor/ninetoothed/blob/b77f930dc6c8b016e09adf33570d55a7bc8376c1/docs/source/installation.rst)、
[Add/Matmul 基础](https://github.com/InfiniTensor/ninetoothed/blob/b77f930dc6c8b016e09adf33570d55a7bc8376c1/docs/source/basics.rst)；
更多 RMSNorm、RoPE、SDPA、MM 和 SwiGLU 示例见
[ninetoothed-examples](https://github.com/InfiniTensor/ninetoothed-examples/tree/e873474d4b4de8e4fa427bf245da4a02512a68b1/ops/ninetoothed/kernels)。

TileLang测试：

先加载预装的 TileLang 开发环境。脚本会自动发现 `/app/tilelang-metax`：

```shell
source ./setup_env.sh
# 自定义源码位置：TILELANG_ROOT=/path/to/tilelang-metax source ./setup_env.sh
```

```shell
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend tilelang --target maca --device cuda
```

MXMACA测试：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
python infer.py --model models/Llama-3.2-1B --prompts "Hello" \
  --max-new-tokens 1 --backend maca_cpp --target maca --device cuda
```

## 学员要改什么

框架已经预置 `rms_norm` 和 `rope` 的调用、注册和测试入口。当前 TileLang
`rms_norm` 仅用于演示算子接入流程，并非性能最优实现；学员可以在此基础上继续优化
线程布局、访存和归约方式。

优化范围不限于现有示例或文档推荐的某一个算子。学员可以根据自己的能力和目标，自行
选择更多适合的模型算子进行实现和优化。RoPE 的接口是：

```python
rope(input, sin_table, cos_table) -> output
```

学员只需要实现对应的 NineToothed、TileLang 或 MXMACA kernel，还没有实现的槽位会暂时使用 PyTorch reference 并打印警告，所以示例可以直接
跑通；这种状态不能用于性能结论。完整说明见 [`operators/INTEGRATION.md`](operators/INTEGRATION.md)。

## 测试和性能

加载项目环境后，可以单独验证 TileLang RMSNorm 和 RoPE，不需要模型权重：

```shell
RUN_ACCELERATOR_TESTS=1 python -m pytest -q tests/test_backend_integration.py -k tilelang
```

这些 TileLang 测试通过 `operators.dispatch` 分别调用 TileLang 算子和 PyTorch 参考实现，
同时检查算子注册状态；回退到 PyTorch 的警告会让测试失败。RoPE 覆盖 Q/K 的 head 数、单 token 与多 batch/sequence、
FP32/FP16/BF16，以及恒等旋转和 90 度旋转。TileLang RMSNorm 同样覆盖单 token、多 batch/sequence
和三种精度，并检查输入及 weight 未被修改。MACA 和 NineToothed RMSNorm 测试也会检查注册并拒绝回退。
未设置 `RUN_ACCELERATOR_TESTS=1` 时会跳过加速器测试。

下面提供轻量性能对比，统一生成 16 个 token，使用 1 次 warmup、3 次测量，并保持模型、
prompt、seed、精度和设备完全一致。该配置用于快速反馈，结果波动较大，不作为正式性能结论。
下面命令假设在 MACA 机器上运行，模型目录是 `models/Llama-3.2-1B`。

先测 PyTorch 基线：

```shell
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 16 \
  --backend torch --target maca --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3 \
  --seed 0 \
  --output-json benchmarks/results/torch_maca.json
```

再测 TileLang：

```shell
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 16 \
  --backend tilelang --target maca --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3 \
  --seed 0 \
  --output-json benchmarks/results/tilelang_maca.json
```

NineToothed 使用同一组参数，运行以下命令并保存结果：

```shell
python -m pip install ninetoothed
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 16 \
  --backend ninetoothed --target maca --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3 \
  --seed 0 \
  --output-json benchmarks/results/ninetoothed_maca.json
```

MXMACA 原生算子需要先构建扩展：

```shell
python operators/maca_cpp/setup.py build_ext --inplace
python infer.py \
  --model models/Llama-3.2-1B \
  --prompts "Hello" \
  --max-new-tokens 16 \
  --backend maca_cpp --target maca --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3 \
  --seed 0 \
  --output-json benchmarks/results/maca_cpp_maca.json
```

比较 TileLang 和 PyTorch：

```shell
python benchmarks/compare_results.py \
  benchmarks/results/torch_maca.json \
  benchmarks/results/tilelang_maca.json \
  --output-json benchmarks/results/torch_vs_tilelang_maca.json
```

比较 NineToothed 和 PyTorch：

```shell
python benchmarks/compare_results.py \
  benchmarks/results/torch_maca.json \
  benchmarks/results/ninetoothed_maca.json \
  --output-json benchmarks/results/torch_vs_ninetoothed_maca.json
```




比较 MXMACA 原生实现和 PyTorch：

```shell
python benchmarks/compare_results.py \
  benchmarks/results/torch_maca.json \
  benchmarks/results/maca_cpp_maca.json \
  --output-json benchmarks/results/torch_vs_maca_cpp_maca.json
```

比较器会校验测试条件和生成的 token IDs，并输出吞吐 speedup、性能变化比例及实际替换
的算子。
 TileLang 调用包含 JIT 编译，不要直接
拿第一次运行的时间评价性能。如果 RoPE 还在使用临时 PyTorch reference，也不能把该
结果当作完整后端性能；应先实现对应 kernel。

## 目录

```text
.
├── assignment/                    # 四个入门实战
│   ├── README.md                  # Assignment 总览与通用要求
│   ├── task1/                     # 沐曦 GPU 与 MXMACA 环境
│   │   └── README.md
│   ├── task2/                     # TileLang Add 与 NineToothed Vector Add
│   │   ├── README.md
│   │   ├── solution.py            # TileLang kernel 作业入口
│   │   ├── ninetoothed_add.py     # NineToothed kernel 作业入口
│   │   ├── test_add.py            # TileLang 正确性测试
│   │   ├── test_ninetoothed_add.py # NineToothed 正确性测试
│   │   ├── benchmark_add.py       # TileLang 综合性能测试
│   │   └── benchmark_ninetoothed_add.py # NineToothed 性能测试
│   ├── task3/                     # TileLang Softmax 与 NineToothed GEMM
│   │   ├── README.md
│   │   ├── solution.py            # TileLang kernel 作业入口
│   │   ├── ninetoothed_gemm.py    # NineToothed kernel 作业入口
│   │   ├── test_softmax.py        # TileLang 正确性测试
│   │   ├── test_ninetoothed_gemm.py # NineToothed 正确性测试
│   │   ├── benchmark_softmax.py   # TileLang 综合性能测试
│   │   └── benchmark_ninetoothed_gemm.py # NineToothed 性能测试
│   └── task4/                     # AI Agent 辅助开发
│       ├── README.md
│       ├── prompts.md             # Prompt 记录模板
│       └── reflection.md          # 实践总结模板
│
├── infer.py                       # Llama 推理入口
├── llama.py                       # 模型与算子调用点
├── backends.py                    # 后端与 target 配置
├── setup_env.sh                   # TileLang/MXMACA 环境变量配置
├── requirements.txt               # Python 依赖
│
├── operators/                     # 算子实现
│   ├── registry.py                # 算子注册与分发
│   ├── torch_ops.py               # PyTorch 参考实现
│   ├── ninetoothed_ops.py         # NineToothed 薄包装
│   ├── ninetoothed_kernels/       # NineToothed 示例 kernel
│   ├── tilelang_ops.py            # TileLang 实现槽位
│   ├── maca_cpp/                  # MXMACA 原生扩展
│   └── INTEGRATION.md             # 算子接入教程
│
├── benchmarks/                    # 性能评测
│   ├── compare_results.py         # 结果比较工具
│   └── results/                   # 性能结果 JSON
│
└── tests/                         # 单元测试与后端集成测试
```
