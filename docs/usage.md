# FIER 程序说明与运行命令

本页只覆盖 [README](../README.md) 中的主线：Ali-CCP 原始文件 → Parquet/Arrow 数据 →
FIER（代码名 `dcn_ple_esmm`）训练 → CTR/CVR/CTCVR 评估与校准。
以下命令在仓库根目录、Linux Bash 中运行；配置值以仓库当前 YAML 为准。
没有数据文件或 CUDA GPU 时，先完成环境与数据检查，不要直接启动全量训练。

## 1. 环境和原始文件

本项目的已验证服务器环境使用 Python 3.12、PyTorch 2.7.1+cu126 与 CUDA GPU。
`requirements.txt` 包含相应 PyTorch wheel；目标机器的驱动需要与该 CUDA 构建兼容。
虚拟环境必须在目标机器上新建，不能直接复制 Windows 的 `.venv`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
python scripts/train_dcn_ple_esmm.py \
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
python scripts/train_dcn_ple_esmm.py \
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
python scripts/train_dcn_ple_esmm.py \
  --config configs/dcn_ple_esmm.yaml --device cuda \
  --seed 2026 --dropout 0.1 --negative-sampling-ratio 5 \
  --run-name "negative_1_5_$(date +%Y%m%d_%H%M%S)"

# 保留全部曝光，仅对点击空间辅助 CVR 损失做 1:20 采样
python scripts/train_dcn_ple_esmm.py \
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
python scripts/evaluate_multitask.py \
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
python scripts/train_dcn_ple_esmm.py \
  --config configs/dcn_ple_esmm.yaml --device cuda \
  --resume-from "results/dcn_ple_esmm_aliccp_full_<实际运行后缀>/latest.pt"
```

最后运行项目测试，核对改动没有破坏数据处理、模型前后向与指标计算：

```bash
python -m unittest discover -s tests -v
```

原始数据、Parquet 分片和 `best.pt` 不进入 Git；长期保存所需检查点请另行存储。

## 7. 可选：导出与本机 `/rank` 演示

这一步在离线训练与校准完成后进行，不是线上广告系统。导出程序读取同一次运行的
`config.yaml`、`best.pt`、`metrics.json`，以及该模型训练时的
`data/processed/aliccp_full/vocab.json`；缺少词表不能从权重恢复原始特征 ID。
它重建网络、严格加载权重，追踪 CTR/CVR/CTCVR 推理图，并用不同批大小、不同
历史长度复核 TorchScript 与原模型输出。导出目录包含 `ranker.ts`、`vocab.json`、
`manifest.json`，若存在验证集拟合的 Platt 参数，还包含 `platt.json`。不导出
优化器状态或训练集。只对自己信任的训练检查点运行导出程序。

```bash
RUN_DIR="results/dcn_ple_esmm_aliccp_full_dropout_02_20260915_154204"
python -m pip install "fastapi>=0.115,<1.0" "uvicorn>=0.30,<1.0" "pydantic>=2,<3"
python scripts/export_ranker.py \
  --run-directory "$RUN_DIR" \
  --processed-directory data/processed/aliccp_full \
  --output-directory "$RUN_DIR/serving" \
  --device cpu
python scripts/serve_ranker.py \
  --artifact-directory "$RUN_DIR/serving" \
  --device cuda --host 127.0.0.1 --port 8000 --torch-threads 1 \
  --dynamic-batching \
  --max-batch-requests 8 \
  --max-batch-candidates 256 \
  --max-batch-wait-ms 2
```

服务进程启动时只加载一次模型；另开一个同机终端调用与压测：

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/rank \
  -H 'Content-Type: application/json' \
  --data-binary @docs/examples/rank_request.json
python scripts/build_rank_payload.py \
  --processed-directory data/processed/aliccp_full \
  --max-candidates 8 --max-scanned 1000000 \
  --output "$RUN_DIR/serving/rank_request_from_test.json"
curl -sS -X POST http://127.0.0.1:8000/rank \
  -H 'Content-Type: application/json' \
  --data-binary @"$RUN_DIR/serving/rank_request_from_test.json"
python scripts/benchmark_service.py \
  --url http://127.0.0.1:8000/rank \
  --health-url http://127.0.0.1:8000/health \
  --payload-file "$RUN_DIR/serving/rank_request_from_test.json" \
  --batch-sizes 1 8 32 --concurrencies 1 8 \
  --requests 100 --warmup 10 \
  --output "$RUN_DIR/serving/benchmark_results.json"
```

`/rank` 将 `user_features`（公共用户字段）、`context_features`（字段 `301`）
与每个候选的 `features`（商品和组合字段）拼接。各字段接受原始
`feature_id`/数值 `value` 列表；依训练词表编码，未登录词和缺失字段遵循原训练
规则落到 `UNK=1`。组合字段 `508/509/702/853` 需要调用方按原数据口径提供，
服务不会臆造交叉特征。一个请求内最多 128 个候选，统一批量推理并按分数降序返回
`pctr`、点击后 `pcvr`、`pctcvr`、`score` 和 `rank`。默认使用**原始 CTCVR**
排序；将 `score_mode` 改为 `ctr` 可按 CTR 排序。若导出目录含 `platt.json`，
请求可设置 `probability_mode: "platt"` 获取分头校准并相乘后的概率；它可能改变
排序，不能把该分数等同于 README 中报告的未校准 Test AUC。

`build_rank_payload.py` 从已处理的官方 Test 中选择**同一用户、同一场景**的曝光，
将整数索引按训练词表反查为原始 token；它只读数据，不用标签训练或拟合校准器。
若扫描上限内不足 8 个候选，就输出找到的数量；压测脚本会循环这些真实候选
构造指定大小的批次。这比手写示例更适合做初步性能检查，但单个请求仍不能代表
全体用户和历史长度的分布。

压测输出的 QPS 为成功 HTTP 请求数除以计时总秒数，P95 为成功请求的客户端
端到端延迟（包含本机连接建立、特征编码、模型前向与响应传输；不含模型启动），
同时记录候选打分吞吐与失败数。手写示例只用于检验接口格式，示例 token 大多会
映射到 UNK；**正式报告性能前应检查请求是否具有代表性的候选数和历史长度**，
固定机器、设备、线程数和并发设置，并明确这是单机离线服务压测。保持服务仅监听
`127.0.0.1`；该演示没有鉴权，不应直接暴露公网。

初步单请求检查完成后，从多个 Test 用户生成请求数组。脚本不改变模型，
也不读取 Test 标签用于训练；它只为基线和动态合批生成同一份压测输入。

```bash
python scripts/build_rank_payload.py \
  --processed-directory data/processed/aliccp_full \
  --users 16 --max-candidates 8 --max-scanned 1000000 \
  --output "$RUN_DIR/serving/rank_requests_16_users.json"
```

服务默认启用动态微批处理。并发请求先进入有界队列，在最多
`max_batch_wait_ms` 的窗口内按 `max_batch_requests` 和
`max_batch_candidates` 合并；Ranker 对各请求独立校验和编码，将合法候选统一
`collate` 后只执行一次 TorchScript forward，再按候选数量切片、校准和排序。
某个请求的非法字段只返回给该请求，不会使同批其他请求失败。`/health` 返回累计的
批次数、平均每批请求/候选数、排队时间和执行时间；压测传入 `--health-url` 后，
每个 case 会把测量区间内的计数差写入 `service_batching`，用于确认是否真实发生合批。

动态批处理增加一个很短的等待窗口，低并发延迟可能略升；目标是在并发到达时减少
模型 forward 次数，提高 request QPS 或 candidate throughput。它不会修改模型参数、
特征编码、概率校准和排序规则。是否有效必须与关闭合批的同机基线对照，不能只报告
优化后的数字。

先在终端 A 启动关闭合批的基线服务：

```bash
python scripts/serve_ranker.py \
  --artifact-directory "$RUN_DIR/serving" \
  --device cuda --host 127.0.0.1 --port 8000 --torch-threads 1 \
  --no-dynamic-batching
```

在终端 B 用固定输入连续压测三轮：

```bash
python scripts/benchmark_service.py \
  --url http://127.0.0.1:8000/rank \
  --health-url http://127.0.0.1:8000/health \
  --payload-file "$RUN_DIR/serving/rank_requests_16_users.json" \
  --batch-sizes 1 8 32 --concurrencies 1 8 \
  --requests 500 --warmup 20 --repeats 3 \
  --output "$RUN_DIR/serving/benchmark_no_dynamic_batching.json"
```

在终端 A 按 `Ctrl+C` 停止基线服务，然后以相同模型启动动态批处理：

```bash
python scripts/serve_ranker.py \
  --artifact-directory "$RUN_DIR/serving" \
  --device cuda --host 127.0.0.1 --port 8000 --torch-threads 1 \
  --dynamic-batching \
  --max-batch-requests 8 \
  --max-batch-candidates 256 \
  --max-batch-wait-ms 2
```

在终端 B 使用完全相同的输入和并发参数：

```bash
python scripts/benchmark_service.py \
  --url http://127.0.0.1:8000/rank \
  --health-url http://127.0.0.1:8000/health \
  --payload-file "$RUN_DIR/serving/rank_requests_16_users.json" \
  --batch-sizes 1 8 32 --concurrencies 1 8 \
  --requests 500 --warmup 20 --repeats 3 \
  --output "$RUN_DIR/serving/benchmark_dynamic_batching.json"
```

压测程序会先按请求中的最大候选数估算特征 token 总量，使用与服务一致的默认上限
`--max-request-feature-tokens 16384`，固定保留对所有测试 case 都合法的模板集合；被过滤
模板的索引和估算 token 数会输出到终端并写入 `template_filter`。基线与动态合批必须
使用同一输入文件和相同过滤上限，不能只在发生 422 后跳过某一侧的失败请求。

为了把队列/预处理路径与真正的多请求合并分开，还应补充同路径无合并对照：

```bash
python scripts/serve_ranker.py \
  --artifact-directory "$RUN_DIR/serving" \
  --device cuda --host 127.0.0.1 --port 8000 --torch-threads 1 \
  --dynamic-batching \
  --max-batch-requests 1 \
  --max-batch-candidates 256 \
  --max-batch-wait-ms 0
```

使用相同压测命令，将输出另存为 `benchmark_batcher_no_merge.json`。完整实测中，原始
16 个 Test 用户模板有 9 个能在扩展至 32 候选后满足单请求 token 上限。并发 8 时：

| 候选数 | 无合并 QPS | 动态 QPS | 无合并 P95 | 动态 P95 | 动态平均请求/Batch |
|---:|---:|---:|---:|---:|---:|
| 1 | 61.73 | 130.63 | 183.86 ms | 113.26 ms | 4.03 |
| 8 | 30.83 | 74.76 | 325.49 ms | 164.02 ms | 4.00 |
| 32 | 15.80 | 33.98 | 602.92 ms | 306.23 ms | 3.97 |

压测的 `summary` 保存各轮 QPS 与 P95 的中位数及 P95 范围，`cases`
保留逐轮原始结果。`concurrency=1` 通常无法形成多请求批次，它用于量化
2ms 等待窗口带来的延迟；`concurrency=8` 才是主要观察点。若
`service_batching.batches` 约等于请求数，说明没有有效合批，应先检查客户端
是否真正并发，而不是直接调大等待时间。

选择的是扫描范围内先遇到的不同用户，每位用户仅保留同一场景的曝光；若某位用户
不足 8 条，压测时会循环其已找到的候选。此方法能检验多个真实用户的特征长度，
但不是对整个 Test 用户总体的随机抽样，不能写成生产环境的性能分布。
