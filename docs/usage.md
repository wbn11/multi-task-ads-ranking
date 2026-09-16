# FIER 程序说明与运行命令

本页只覆盖 [README](../README.md) 中的主线：Ali-CCP 原始文件 → Parquet/Arrow 数据 →
FIER（代码名 `dcn_ple_esmm`）训练 → CTR/CVR/CTCVR 评估与校准。
以下命令在仓库根目录、Linux Bash 中运行；配置值以仓库当前 YAML 为准。
没有数据文件或 CUDA GPU 时，先完成环境与数据检查，不要直接启动全量训练。

## 1. 环境和原始文件

本项目的已验证服务器环境使用 Python 3.12、PyTorch 2.7.1+cu126 与 CUDA GPU。
`requirements-a6000.txt` 固定了相应 PyTorch wheel；其他驱动或 CUDA 环境应先核对兼容性。
虚拟环境必须在目标机器上新建，不能直接复制 Windows 的 `.venv`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-a6000.txt
python -c "import torch, pyarrow; print(torch.__version__, torch.cuda.is_available(), pyarrow.__version__)"
```

从[官方 Ali-CCP 页面](https://tianchi.aliyun.com/dataset/408)获取数据，将下列文件放在
`data/raw/`；也支持 [原始数据说明](../data/raw/README.md)列出的分目录解压布局。
原始 CSV、压缩包和处理后的数据均不提交 Git。

```text
data/raw/
├── sample_skeleton_train.csv
├── common_features_train.csv
├── sample_skeleton_test.csv
└── common_features_test.csv
```

## 2. 构造与验证数据

`scripts/preprocess.py` 读取四个原始文件及 `configs/data.yaml`，流式统计训练词表，
按 `common_feature_id` 关联曝光与公共特征，过滤 `click=0, conversion=1` 等异常标签，
写出 `data/processed/aliccp_full/` 下的 `vocab.json`、`manifest.json` 和三个 split 的
sample/common Parquet 分片。该步骤主要消耗 CPU、内存和磁盘，不需要 GPU。

```bash
python scripts/preprocess.py --config configs/data.yaml
```

`scripts/validate_dataset.py` 逐 split 检查标签、字段、计数与元数据；成功时标准输出的
JSON 含 `"valid": true`。三个 split 都通过后再开始训练。

```bash
python scripts/validate_dataset.py --config configs/data.yaml --split train
python scripts/validate_dataset.py --config configs/data.yaml --split validation
python scripts/validate_dataset.py --config configs/data.yaml --split test
```

正式数据划分由 `configs/data.yaml` 固定：官方 Train 用于训练；官方 Test 的前 400K 条
开发期已使用，正式评估排除它们；余下样本以固定哈希分入 Validation/Test。
更换训练 seed 不会重新划分 Validation/Test。

## 3. 检查真实 Batch 与模型前向

`scripts/check_dataloader.py` 从训练分片读取真实 Batch，检查 23 个字段的
`ids / values / offsets`、Embedding 输入及设备传输；标准输出报告启动耗时与预热后的
稳定吞吐。它只检查数据通路，不训练正式模型。

```bash
python scripts/check_dataloader.py \
  --config configs/data.yaml --split train \
  --batch-size 4096 --num-workers 8 --device cuda \
  --embedding-dim 16 --warmup-batches 10 --benchmark-batches 100
```

`scripts/overfit_dcn_ple_esmm.py` 在固定小批次上训练数百步，检查 loss 能下降、输出
与梯度有限、两项 Gate 和 CTCVR 乘积关系正确。输出 JSON 中的 `"valid": true` 是
正式全量训练前的最低正确性检查；它不代表泛化效果。

```bash
python scripts/overfit_dcn_ple_esmm.py \
  --config configs/dcn_ple_esmm.yaml --device cuda
```

## 4. 全样本训练 FIER

`scripts/train_dcn_ple_esmm.py` 读取处理后的 Train/Validation/Test、模型 YAML 和
训练参数。默认模型使用 16 维字段 Embedding、3 层 Cross Network、单层专家路由，
在完整曝光上优化 CTR 与 CTCVR。它按 Validation CTCVR AUC 选择 `best.pt`，
最后用该模型在 Test 上计算未经额外校准的指标。

```bash
OMP_NUM_THREADS=1 python scripts/train_dcn_ple_esmm.py \
  --config configs/dcn_ple_esmm.yaml \
  --device cuda --batch-size 8192 --num-workers 8
```

每次运行生成唯一的 `results/dcn_ple_esmm_aliccp_full_<timestamp>/`：

| 文件 | 作用 |
|---|---|
| `config.yaml` | 解析命令行覆盖后的完整配置快照 |
| `train.log` | 逐轮训练与 Validation 指标、早停记录 |
| `best.pt` | Validation CTCVR AUC 最优检查点 |
| `latest.pt` | 每轮保存的可恢复训练状态 |
| `metrics.json` | 最优轮次、Test 原始指标、耗时与模型参数 |

运行结束后，标准输出还有 `"valid": true` 和实际 `run_directory`；后续命令使用这个
准确路径，不根据日志时间猜目录名。`--batch-size` 与 `--num-workers` 可按本机吞吐和
内存调整；改变它们后应在 run 配置中保留记录。

### 多 seed 复验

`--seed` 改变训练随机性及训练负采样随机性，不改变已处理好的验证/测试划分。
为避免目录重名，每个 seed 指定独立 `--run-name`；可在不同机器上同时执行。
比较时保持模型、数据和其余训练参数一致。

```bash
OMP_NUM_THREADS=1 python scripts/train_dcn_ple_esmm.py \
  --config configs/dcn_ple_esmm.yaml --device cuda \
  --seed 2027 --dropout 0.2 --gate-dropout 0 \
  --batch-size 8192 --num-workers 8 \
  --run-name "full_seed_2027_$(date +%Y%m%d_%H%M%S)"
```

seed 2028 在另一台机器执行同一命令，只将 `--seed` 和 `--run-name` 中的 `2027`
改为 `2028`。不要把不同 seed 的 `best.pt` 当作同一个模型继续训练。

### 与主方案直接相关的对照

下面两个参数组均不修改 Validation/Test 分布；分别使用独立运行目录，并与相同
dropout/seed 的全样本运行比较：

```bash
# 对未点击曝光做 1:5 训练负采样
OMP_NUM_THREADS=1 python scripts/train_dcn_ple_esmm.py \
  --config configs/dcn_ple_esmm.yaml --device cuda \
  --seed 2026 --dropout 0.1 --negative-sampling-ratio 5 \
  --run-name "negative_1_5_$(date +%Y%m%d_%H%M%S)"

# 保留全部曝光，仅对点击空间辅助 CVR 损失做 1:20 采样
OMP_NUM_THREADS=1 python scripts/train_dcn_ple_esmm.py \
  --config configs/dcn_ple_esmm.yaml --device cuda \
  --seed 2026 --dropout 0.2 --no-negative-sampling \
  --auxiliary-cvr-weight 0.02 --auxiliary-cvr-negative-ratio 20 \
  --run-name "aux_cvr_1_20_w002_$(date +%Y%m%d_%H%M%S)"
```

正式大规模实验建议显式传入与主模型相同的 `--batch-size 8192 --num-workers 8`；
上例省略时使用 `configs/dcn_ple_esmm.yaml` 中的默认值。命令行仅覆盖本次运行，
不会改写基础 YAML。当前脚本全部可用参数见
`python scripts/train_dcn_ple_esmm.py --help`。

## 5. 评估与概率校准

训练脚本已写入原始 Test 指标。`scripts/evaluate_multitask.py` 读取某次运行的
`best.pt` 和配置，在 Validation 拟合 Platt/Isotonic 校准器，再在 Test 比较
AUC、LogLoss、Brier Score、ECE 与分箱曲线；训练集负采样的运行会先进行先验修正。
**Test 不参与校准器拟合**。

```bash
RUN_DIR="results/dcn_ple_esmm_aliccp_full_<实际运行后缀>"
OMP_NUM_THREADS=1 python scripts/evaluate_multitask.py \
  --run-directory "$RUN_DIR" --device cuda \
  --batch-size 8192 --num-workers 8 --calibration-bins 20
```

输出写入该运行目录：`evaluation_calibrated_multitask.json` 保存 Validation/Test 的
原始、先验修正和校准指标及分箱；`multitask_calibrators.json` 保存可复用的校准器参数。
CVR 仅在点击样本上计算，CTCVR 在所有曝光上计算；两个口径不可混用。

## 6. 汇总、恢复和检查

`scripts/summarize_results.py` 从本机 `results/*/metrics.json` 与可选的校准结果生成
可提交 Git 的紧凑表 `results/experiment_summary.csv`。它只汇总已存在的运行，
不会重新训练；来自其他机器的运行应先复制到本机 `results/` 再汇总。

```bash
python scripts/summarize_results.py
```

训练意外中断时，从同一运行目录的 `latest.pt` 继续；恢复时不要再传 `--run-name`。

```bash
OMP_NUM_THREADS=1 python scripts/train_dcn_ple_esmm.py \
  --config configs/dcn_ple_esmm.yaml --device cuda \
  --resume-from "results/dcn_ple_esmm_aliccp_full_<实际运行后缀>/latest.pt"
```

最后运行项目测试，核对改动没有破坏数据处理、模型前后向与指标计算：

```bash
python -m unittest discover -s tests -v
```

原始数据、Parquet 分片和 `best.pt` 不进入 Git；长期保存所需检查点请另行存储。
