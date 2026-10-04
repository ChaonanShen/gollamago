
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

MACA 镜像应保留预装的 MACA PyTorch 和 TileLang，先检查依赖是否已经齐全；
不要直接用 `requirements.txt` 中的 CUDA TileLang 版本覆盖镜像环境。
后端兼容 TileLang 0.1.14 和 MACA 镜像的 0.1.9 target helper 路径。

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

扩展正确性测试固定使用 1B 的 hidden=2048、Q/K heads=32/8、head_dim=64。
共 177 个 TileLang 用例（RMSNorm 57、RoPE 114、特殊旋转 6），覆盖 FP32/FP16/BF16、
19 组 batch/sequence，包括 31/32/33、63/64/65 等分块边界、batch=4/5、
sequence=2048/4096 和 batch=8、sequence=1024。还检查输入、weight 和 RoPE table 未被修改。

C500 16GB 上可按下面命令扩大独立算子性能测试范围；每个用例依次分配随机输入，无需模型权重。
日志保存在仓库外；数值校验或后端回退失败会以非零状态退出。

```shell
mkdir -p /data/gollamago-data/logs
(
  set -e
  for dtype in bfloat16 float16 float32; do
    batches="1 8 32 64 128"
    [ "$dtype" != float32 ] || batches="1 8 32 64"
    for batch in $batches; do
      python -m benchmarks.benchmark_operators --models llama3.2-1b \
        --backend tilelang --target maca --dtype "$dtype" --batch-size "$batch" \
        --seq-lens 128 512 2048 4906 \
        --warmup 20 --repeat 200
    done
  done
) > /data/gollamago-data/logs/llama1b-operators-expanded.log 2>&1
```

该命令共 168 组 RMSNorm / RoPE(Q) / RoPE(K) 对比。
BF16/FP16 的 batch=1/8/32/64/128，FP32 的 batch=1/8/32/64。
正确性和输入不变性检查采用分块比较，避免整份输出转换为 FP32 时产生大型临时张量；
全部元素仍需通过相同的容差检查，计时部分保持不变。


独立算子性能对比默认只测 Llama-3.2-1B；3B 和 8B 的代码保留，可通过 `--models` 显式选择（无需模型权重）：

| 性能用例 | RMSNorm hidden_size | RoPE Q heads | RoPE K heads | head_dim |
|---|---:|---:|---:|---:|
| [Llama-3.2-1B](https://modelscope.cn/models/LLM-Research/Llama-3.2-1B/resolve/master/config.json) | 2048 | 32 | 8 | 64 |
| [Llama-3.2-3B](https://modelscope.cn/models/LLM-Research/Llama-3.2-3B/resolve/master/config.json) | 3072 | 24 | 8 | 128 |
| [Llama-3.1-8B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B/resolve/master/config.json) | 4096 | 32 | 8 | 128 |

```shell
python -m benchmarks.benchmark_operators --backend tilelang --warmup 10 --repeat 100
# 专注 Llama-3.2-1B，或用 --models 选择多个型号：
python -m benchmarks.benchmark_operators --models llama3.2-1b --warmup 10 --repeat 100
# 仅测 RoPE，指定 batch、sequence 和精度：
python -m benchmarks.benchmark_operators --models llama3.2-1b --operator rope --batch-size 4 \
  --seq-lens 128 512 --dtype float16 --warmup 10 --repeat 100
```

性能测试的张量尺寸、RMSNorm eps 和默认 BF16 来自上述模型配置；输入和 weight 仍是随机数据，
不代表从模型抓取的真实 activation。RoPE table 复用 `llama.generate_sin_and_cos_tables`，
与本项目的模型调用一致；本项目当前未应用 config.json 中的 rope_scaling。
默认 batch=1、sequence=128/512/2048/4906，分别测 RMSNorm、RoPE(Q) 和 RoPE(K)，1B 共 12 组。
脚本先通过 dispatch 验证目标后端与 Torch 输出一致、输入未被修改，再分别预热和计时。
输出模型名、张量尺寸、平均调用延迟（微秒）、`torch_time / backend_time` 加速比和最大绝对误差。
JIT 编译和输入生成不计时；批量同步计时包含 dispatch、输出分配及 GPU 执行开销。
Torch baseline 是本仓库的 eager 参考实现；其多次 kernel 启动与中间张量开销也包含在计时内。
算子未注册或回退时直接失败。

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
