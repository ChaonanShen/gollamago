# GollamaGo 算子与端到端模型验证报告

## 实验范围与环境

本报告验证 TileLang 与 MACA C++ 两套 RMSNorm、RoPE 实现，均以仓库中的 eager PyTorch 版本为参考。截图脚本只测试 **batch_size=64、128，序列长度=512**，不包含其他尺寸或完整参数扫描。

| 项目 | 设置 |
|---|---|
| 实验平台 | MetaX C500，16GB 显存配额、25% Compute 切片 |
| Python 环境 | /data/gollamago-data/venv，使用预装 MACA PyTorch 与 TileLang |
| PyTorch | 2.8.0+metax3.5.3.9 |
| TileLang | 0.1.9+cuda.gitf1ca0fb9（MACA 开发版本） |
| target / device | maca / cuda（MACA 使用 CUDA 兼容接口） |
| 数据类型 | BF16 |
| 模型 | Llama-3.2-1B |
| batch_size / sequence | 64、128 / 512 |
| RMSNorm | input=[B,512,2048]，weight=[2048]，eps=1e-5 |
| RoPE(Q) | input=[B,512,32,64]，sin/cos=[512,32] |
| RoPE(K) | input=[B,512,8,64]，sin/cos=[512,32] |

两套后端必须实际注册并执行 RMSNorm 和 RoPE。脚本会把回退 Torch 的警告视为失败，避免将参考实现误记为优化后端的成绩。所有脚本结果保存为 JSON，日志、权重和缓存留在 Git 仓库之外。

本报告的算子数据来自 2026-10-05 的专用截图脚本实测；端到端数据来自此前相同模型、提示词、512 token 输入和 16 token 输出配置的实测。正式截图复测后，以新的 JSON 数据更新表格。

## 运行准备

在 C500 服务器执行：

```bash
cd /data/gollamago
source /data/gollamago-data/activate.sh
python operators/maca_cpp/setup.py build_ext --inplace
mx-smi
mkdir -p /data/gollamago-data/report-screenshots
set -o pipefail
```

确认没有其他 GPU 任务明显干扰本次测量。建议终端宽度至少 110 列，截图时保留脚本标题、后端、尺寸与结果表格。

## 1. 算子正确性验证

### 验证方法

对每个 batch 分别验证 RMSNorm、RoPE(Q)、RoPE(K)，每套后端共六个主要用例：

- 输出 shape、dtype、device 与输入一致。
- 与 PyTorch 参考实现逐元素比较，检查全部元素。
- 输入、RMSNorm weight、RoPE sin/cos table 未被修改。
- RoPE 另外检查恒等旋转和 90° 旋转，两种特殊旋转要求与参考逐元素完全一致。

BF16 的一般数值比较使用以下既定容差：

| 算子 | rtol | atol |
|---|---:|---:|
| RMSNorm | 0.02 | 0.07 |
| RoPE | 0.04 | 0.04 |

容差判断采用 `abs(actual-reference) <= atol + rtol*abs(reference)`。最大绝对误差是辅助指标，不能只拿它与 atol 单独比较。数值检查按块处理以限制临时显存，但覆盖全部元素。

### 截图命令

TileLang：

```bash
python -m benchmarks.report_correctness --backend tilelang \
  | tee /data/gollamago-data/report-screenshots/01-correctness-tilelang.log
```

MACA C++：

```bash
python -m benchmarks.report_correctness --backend maca_cpp \
  | tee /data/gollamago-data/report-screenshots/02-correctness-maca.log
```

### 实测结果

| batch | 算子 | TileLang | MACA C++ | 两后端最大绝对误差 |
|---:|---|---|---|---:|
| 64 | RMSNorm | PASS | PASS | 0.125 |
| 64 | RoPE(Q) | PASS | PASS | 0.03125 |
| 64 | RoPE(K) | PASS | PASS | 0.03125 |
| 128 | RMSNorm | PASS | PASS | 0.125 |
| 128 | RoPE(Q) | PASS | PASS | 0.03125 |
| 128 | RoPE(K) | PASS | PASS | 0.03125 |

所有 shape/dtype/device 检查、输入不变性检查通过；两后端 RoPE 的恒等与 90° 旋转检查均通过。一般随机数据是在上述 BF16 容差下通过，不声称与参考逐位一致。

截图位置：待补 `01-correctness-tilelang.png`、`02-correctness-maca.png`。

## 2. 算子性能验证

### 测量方法

每个用例先检查正确性与输入不变性，完成首次 JIT 编译后再计时。每轮预热 20 次、连续调用 100 次，共测量三轮，取三轮平均调用时间的中位数。每轮交换 Torch/后端的测试顺序。

计时包含 dispatch、输出分配与 GPU 执行，通过同步保证 GPU 工作完成；不包含输入生成、数值验证、首次编译。这里报告的是**完整调用时间**，不是 CUDA Graph 测得的纯 GPU kernel 时间。

```text
加速比 = Torch 平均调用耗时 / 优化后端平均调用耗时
```

大于 1 表示优化后端更快。

### 截图命令

TileLang：

```bash
python -m benchmarks.report_performance --backend tilelang \
  | tee /data/gollamago-data/report-screenshots/03-performance-tilelang.log
```

MACA C++：

```bash
python -m benchmarks.report_performance --backend maca_cpp \
  | tee /data/gollamago-data/report-screenshots/04-performance-maca.log
```

### TileLang 实测结果

| batch | 算子 | Torch（μs） | 后端（μs） | 相对 Torch 加速比 |
|---:|---|---:|---:|---:|
| 64 | RMSNorm | 679.76 | 190.21 | 3.57× |
| 64 | RoPE(Q) | 1040.21 | 230.85 | 4.51× |
| 64 | RoPE(K) | 297.52 | 220.77 | 1.35× |
| 128 | RMSNorm | 1311.91 | 370.40 | 3.54× |
| 128 | RoPE(Q) | 2024.47 | 441.99 | 4.58× |
| 128 | RoPE(K) | 544.78 | 215.43 | 2.53× |

### MACA C++ 实测结果

| batch | 算子 | Torch（μs） | 后端（μs） | 相对 Torch 加速比 |
|---:|---|---:|---:|---:|
| 64 | RMSNorm | 679.81 | 250.00 | 2.72× |
| 64 | RoPE(Q) | 1036.55 | 506.82 | 2.05× |
| 64 | RoPE(K) | 298.13 | 134.05 | 2.22× |
| 128 | RMSNorm | 1311.53 | 482.50 | 2.72× |
| 128 | RoPE(Q) | 2023.43 | 1003.07 | 2.02× |
| 128 | RoPE(K) | 545.50 | 258.25 | 2.11× |

TileLang 的 RMSNorm 与 RoPE(Q) 在本组大输入下取得更高加速比；RoPE(K) 的输入量较小，TileLang 的调用开销占比仍较明显，MACA C++ 在这两个 K 用例中更快。当前原生 RoPE 仍采用标量访存，而 TileLang 生成代码使用向量化访存，后续原生版本仍有优化空间。上述数字仅代表本平台与本组尺寸。

截图位置：待补 `03-performance-tilelang.png`、`04-performance-maca.png`。

## 3. 端到端模型正确性与加速比验证

### 测试条件

- 使用同一份 Llama-3.2-1B 权重与 BF16 精度，seed=0。
- 固定同一条恰好 512 token 的提示词，分别复制为 batch=64、128。
- 每个序列生成 16 个 token，采用仓库现有的 greedy generation。
- 每项预热一次，测量三次，报告平均完整生成耗时。
- 一次加载模型，依次切换 Torch 与所选后端；加载、输入准备和首次编译不计时。
- 对全部样本的全部生成 token ID 做精确比较，必须与同 batch 的 Torch 基线一致。
- 只替换已接入的 RMSNorm 和 RoPE，其余 GEMM、attention 等保持相同 Torch 路径。
- 当前仓库未实现 KV cache，每生成一个 token 都重算完整序列。本报告对应这一生成流程。

### 截图命令

可以一次测两套后端，避免重复加载模型和重复测 Torch 基线：

```bash
python -m benchmarks.report_e2e --backend all \
  | tee /data/gollamago-data/report-screenshots/05-e2e-all.log
```

该命令最终打印四行结果，同时包含 TileLang 与 MACA C++，可用同一张截图覆盖两套后端。

如需分别截图，可分别运行以下两条命令；每条都会重新测同 batch 的 Torch 基线：

```bash
python -m benchmarks.report_e2e --backend tilelang \
  | tee /data/gollamago-data/report-screenshots/05-e2e-tilelang.log

python -m benchmarks.report_e2e --backend maca_cpp \
  | tee /data/gollamago-data/report-screenshots/06-e2e-maca.log
```

默认模型路径为 `/data/gollamago-data/models/Llama-3.2-1B`，可通过 `--model` 指定其他位置。脚本只支持本报告的两个 batch 和固定输入长度；默认输出长度为 16，不需要额外传参。

若测试已完成，只需重新显示结果补截图，可执行：

```bash
python -m benchmarks.report_e2e \
  --show-json /data/gollamago-data/report-screenshots/e2e-all.json
```

此命令会明确标注“已保存结果显示”，不会重新执行推理。若此前分别运行后端，对应文件名为 `e2e-tilelang.json` 或 `e2e-maca_cpp.json`。

### 既有同配置实测结果

| batch | Torch（s） | TileLang（s） | TileLang 加速比 | MACA（s） | MACA 加速比 | 三者最大 PyTorch 显存峰值（GiB） |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 12.764 | 12.225 | 1.044× | 12.315 | 1.036× | 5.83 |
| 128 | 25.329 | 24.204 | 1.046× | 24.446 | 1.036× | 8.80 |

两种后端均成功运行 batch=64、128，全部生成 token 与 Torch 一致，没有后端回退。表中显存是 PyTorch 分配峰值，不包含运行时内部全部占用。

本组端到端加速约为 3.6%–4.6%。它小于单算子加速比，因为只优化了模型流程中的一部分算子，其他计算保持不变。复测后应以专用脚本的新 JSON 替换此表，而不是将单算子加速比直接当作模型整体加速比。

截图位置：待补 `05-e2e-all.png`，或分别补 `05-e2e-tilelang.png`、`06-e2e-maca.png`。

## 结果文件与复现说明

默认目录：`/data/gollamago-data/report-screenshots/`。JSON 保存平台版本、源文件 SHA-256、输入尺寸、原始测量值与检查结果：

| 内容 | JSON 文件 |
|---|---|
| TileLang 正确性 | correctness-tilelang.json |
| MACA 正确性 | correctness-maca_cpp.json |
| TileLang 性能 | performance-tilelang.json |
| MACA 性能 | performance-maca_cpp.json |
| 端到端两后端 | e2e-all.json |
| 端到端分别测量 | e2e-tilelang.json / e2e-maca_cpp.json |

若修改 `.maca`，先重新构建扩展，再开始截图。构建脚本已修正目标文件依赖，保证原生源文件变化触发重新链接。原生入口也已修正历史 runtime 错误状态误报，以保证测试不会误回退到 Torch。

本报告没有附带其他 batch、序列长度、dtype 或完整测试集的结果。所有截图应由实际运行输出补充。
