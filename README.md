# Industrial Multi-Task Ads Ranking System

基于 Ali-CCP 的工业广告 CTR/CVR 多任务精排项目。项目覆盖原始曝光日志解析、
稀疏特征编码、CTR baseline、多任务学习、离线指标、负采样和概率校准。

当前已实现：

- LR、DeepFM、DCNv2 单任务 CTR 模型
- Shared Bottom、ESMM、MMoE、PLE 多任务模型
- AUC、GAUC、LogLoss、PR-AUC、Brier Score、ECE
- 均匀负样本降采样、先验修正、Platt Scaling、Isotonic Regression
- 流式 Ali-CCP 预处理、Parquet 分片和多 worker DataLoader

项目只保留一套正式数据管线，不包含旧 JSONL 或带版本号的数据实现。

## 项目目录

```text
configs/          数据与模型配置
scripts/          用户直接执行的程序入口
src/data/         数据解析、处理、Dataset和DataLoader
src/layers/       Embedding、FM、Cross Network、Expert/Gate
src/models/       CTR及多任务模型
src/losses/       多任务损失
src/metrics/      排序、分组和校准指标
src/calibration/  先验修正、Platt、Isotonic
src/trainer/      统一训练与评估流程
data/raw/         Ali-CCP原始文件
data/processed/   生成的模型数据
results/          实际实验配置、checkpoint、日志与指标
tests/            单元测试
.venv/            当前机器的虚拟环境，不提交、不跨系统复制
```

## 环境

Linux A6000/A100/RTX 4090 服务器：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-a6000.txt
```

检查环境：

```bash
python -c "import torch, pyarrow; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(pyarrow.__version__)"
```

## 原始数据

`data/raw/`需要包含：

```text
sample_skeleton_train.csv
common_features_train.csv
sample_skeleton_test.csv
common_features_test.csv
```

`sample_skeleton`每行是一条广告曝光，包含sample ID、click、conversion、
`common_feature_id`和商品/组合/场景特征。`common_features`通过
`common_feature_id`提供用户及历史行为特征。field 101的原始匿名用户ID会单独
保留给GAUC分组，但模型仍使用其训练集词表编码。

原始文件没有显式时间戳。正式数据划分使用官方train作为训练集；官方test前
400K条已在早期开发实验中使用，因此从最终数据排除。剩余官方test通过稳定哈希
划分为10% Validation和90% Test，避免不同运行产生不同划分。

## 数据处理流程

```text
sample_skeleton_train.csv
  -> 第1遍流式扫描
  -> 丢弃 click=0, conversion=1 的异常漏斗样本
  -> 统计标签、sample侧token频次、common_id曝光复用次数

common_features_train.csv
  -> 扫描训练曝光引用的common记录
  -> common token频次乘以对应曝光复用次数
  -> SQLite保存 common_id -> 原文件字节偏移

训练频次
  -> 每个field建立独立词表
  -> PAD=0、UNK=1、频次低于min_frequency映射到UNK
  -> vocab.json（只使用训练集拟合）

再次流式读取官方train
  -> 按SQLite偏移读取当前shard需要的common记录
  -> 编码sample/common特征
  -> 每250K曝光写一个sample Parquet和一个去重common Parquet

官方test的未使用部分
  -> blake2b(sample_id, seed)稳定划分Validation/Test
  -> 复用训练词表，未见token映射到UNK
  -> 写Parquet分片

Parquet shard
  -> AdsDataset为每个DataLoader worker分配不同shard
  -> shard内通过局部common_index完成Join
  -> Arrow批量take sample/common行
  -> ListArray直接转换为ids/values/offsets Tensor
  -> 模型训练batch
```

输出目录：

```text
data/processed/aliccp_full/
├── vocab.json
├── manifest.json
├── cache/
│   ├── train_common_offsets.sqlite3
│   └── test_common_offsets.sqlite3
├── train/
│   ├── metadata.json
│   ├── samples/part-*.parquet
│   └── common/part-*.parquet
├── validation/
│   ├── metadata.json
│   ├── samples/part-*.parquet
│   └── common/part-*.parquet
└── test/
    ├── metadata.json
    ├── samples/part-*.parquet
    └── common/part-*.parquet
```

sample Parquet保存曝光标签、商品/组合/场景特征和局部`common_index`；配套
common Parquet只保存该shard使用的去重用户特征。长用户历史不会在每条曝光中
重复展开。SQLite只用于预处理定位原始common行，训练阶段不查询SQLite。

## 构造与验证数据

全量处理一次完成Train、Validation和Test：

```bash
python scripts/preprocess.py --config configs/data.yaml
```

预处理只使用CPU、内存和磁盘，不占GPU。处理完成后检查每个split：

```bash
python scripts/validate_dataset.py --config configs/data.yaml --split train
python scripts/validate_dataset.py --config configs/data.yaml --split validation
python scripts/validate_dataset.py --config configs/data.yaml --split test
```

检查真实DataLoader batch、CUDA传输和23个字段的EmbeddingBag输入：

```bash
python scripts/check_dataloader.py \
  --config configs/data.yaml \
  --split train \
  --batch-size 4096 \
  --num-workers 8 \
  --device cuda \
  --warmup-batches 10 \
  --benchmark-batches 100
```

报告分别给出首批启动耗时和预热后的稳定`batches/s`、`samples/s`，用于选择
`batch_size`与`num_workers`，避免把Parquet打开时间误认为持续训练吞吐。
正式配置默认启用`columnar_batching: true`：它保留既有分片、Shuffle、负采样和
跨分片组Batch语义，但跳过逐样本Python字典和`to_pylist()`特征展开。报告中的
`batching_mode`应为`arrow_columnar`。如需做等价或性能回退对比，可增加：

```bash
python scripts/check_dataloader.py \
  --config configs/data.yaml \
  --split train \
  --batch-size 4096 \
  --num-workers 8 \
  --device cuda \
  --no-columnar-batching
```

模型收到的batch结构：

```text
features[field].ids       int64   [N_tokens]
features[field].values    float32 [N_tokens]
features[field].offsets   int64   [B+1]
click                     float32 [B]
conversion                float32 [B]
ctcvr                     float32 [B]
user_group_id              int64   [B]（稳定哈希分组，只用于GAUC）
```

## 训练流程

每个模型正式训练前先运行固定小批次过拟合，验证shape、输出范围、loss、梯度和
NaN。示例：

```bash
python scripts/overfit_lr.py --config configs/lr.yaml --device cuda
python scripts/overfit_deepfm.py --config configs/deepfm.yaml --device cuda
python scripts/overfit_dcnv2.py --config configs/dcnv2.yaml --device cuda
python scripts/overfit_shared_bottom.py --config configs/shared_bottom.yaml --device cuda
python scripts/overfit_esmm.py --config configs/esmm.yaml --device cuda
python scripts/overfit_mmoe.py --config configs/mmoe.yaml --device cuda
python scripts/overfit_ple.py --config configs/ple.yaml --device cuda
```

正式训练：

```bash
python scripts/train_lr.py --config configs/lr.yaml --device cuda
python scripts/train_deepfm.py --config configs/deepfm.yaml --device cuda
python scripts/train_dcnv2.py --config configs/dcnv2.yaml --device cuda
python scripts/train_shared_bottom.py --config configs/shared_bottom.yaml --device cuda
python scripts/train_esmm.py --config configs/esmm.yaml --device cuda
python scripts/train_mmoe.py --config configs/mmoe.yaml --device cuda
python scripts/train_ple.py --config configs/ple.yaml --device cuda
```

训练入口支持AMP、early stopping、gradient clipping、checkpoint、训练日志和配置
快照。`--batch-size`与`--num-workers`可以根据服务器资源覆盖配置。每个完整epoch
保存可恢复的`latest.pt`，指标最优模型保存为`best.pt`。中断后从原实验目录继续：

```bash
python scripts/train_lr.py \
  --config configs/lr.yaml \
  --device cuda \
  --resume-from results/lr_aliccp_full_<run>/latest.pt
```

多任务模型统一使用验证集CTCVR AUC选择`best.pt`，同时完整报告CTR、点击空间CVR
与曝光空间CTCVR指标。

## 模型与目标

单任务CTR组用于比较特征交互能力：

- LR：线性基线，验证数据和指标链路。
- DeepFM：一阶项、FM二阶交互和DNN高阶交互。
- DCNv2：Matrix Cross Network显式交叉与DNN并行。

多任务组用于比较CTR/CVR任务关系：

- Shared Bottom：共享底层，多任务基线。
- ESMM：在完整曝光空间优化CTR与CTCVR，缓解CVR选择偏差。
- MMoE：任务Gate对共享专家进行不同加权。
- PLE：共享专家与任务专属专家进一步解耦信息流。

ESMM主要目标：

```text
pCTCVR = pCTR * pCVR
Loss = BCE(click, pCTR) + BCE(click * conversion, pCTCVR)
```

## 负采样与校准

训练集负采样保留全部点击正例，并在每个epoch对未点击曝光重新进行均匀
Bernoulli采样。Validation和Test始终保持原始分布。负采样配置产生的
`negative_keep_probability`用于先验修正：

```text
p_true = alpha * p_sampled / (1 - p_sampled + alpha * p_sampled)
```

其中`alpha`是负例保留概率。先验修正恢复采样改变的类别先验；Platt和Isotonic
进一步学习模型校准误差。校准使用Validation拟合，只在Test评估，比较LogLoss、
Brier Score和ECE。AUC主要评估排序，校准概率用于`pCTR * bid`或
`pCTR * pCVR * value`。

## 实验结果

以下均为实际全量实验结果。训练集包含42,299,905条有效曝光；Validation包含
4,263,506条曝光；Test包含38,353,110条曝光。训练集中有1,644,256条点击和
8,802条转化，说明CVR/CTCVR远比CTR稀疏。早期100K和1M结果不用于最终结论。

### 单任务CTR对比

| Model | Test AUC | Test GAUC | Test LogLoss |
|---|---:|---:|---:|
| LR | 0.569636 | 0.581511 | 0.189346 |
| DeepFM | 0.584801 | 0.588494 | 0.183517 |
| DCNv2 | **0.627173** | **0.590622** | **0.161830** |

DCNv2显著优于LR和DeepFM，表明这批匿名稀疏字段的显式高阶交叉具有价值。

### 多任务模型对比

| Model | CTR AUC | CVR AUC | CTCVR AUC | CTCVR GAUC | CTCVR LogLoss |
|---|---:|---:|---:|---:|---:|
| Shared Bottom | 0.620490 | 0.670444 | 0.656806 | 0.605391 | 0.002058 |
| ESMM | 0.622897 | 0.669692 | 0.653902 | 0.615818 | **0.001994** |
| MMoE | 0.622986 | 0.672177 | 0.661048 | 0.611964 | 0.002070 |
| PLE | 0.618929 | 0.680956 | 0.663300 | 0.625665 | 0.002137 |
| Target-aware DCN-PLE-ESMM | **0.627093** | 0.680538 | 0.670384 | **0.627006** | 0.002047 |
| **DCN-PLE-ESMM** | 0.625479 | **0.691803** | **0.672664** | 0.623419 | 0.002014 |

最终主模型选择DCN-PLE-ESMM：16维field embedding先经过3层Matrix Cross Network
学习显式特征交叉，再送入包含2个共享专家、1个CTR专属专家和1个CVR专属专家的
PLE，最后使用ESMM的CTR+CTCVR全曝光空间目标训练。它取得最高CVR AUC与全局
CTCVR AUC。单任务DCNv2的CTR AUC略高，Target-aware版本的CTCVR GAUC略高，
因此这里不宣称所有指标全面领先。

### 概率校准

最终模型在Validation上拟合校准器，在Test上评估：

| Task | Raw LogLoss | Platt LogLoss | Raw ECE | Platt ECE |
|---|---:|---:|---:|---:|
| CTR | 0.161615 | **0.160325** | 0.007770 | **0.000920** |
| CVR | 0.033390 | **0.032677** | 0.001977 | **0.000571** |
| CTCVR | 0.002014 | **0.001971** | 0.000114 | **0.000023** |

Platt Scaling基本保持排序能力，同时明显改善概率误差。Isotonic同样改善校准，但在
极稀疏转化任务中产生更多相同分数并轻微降低AUC，因此最终推荐Platt。

### 负采样与模型消融

- 对未点击曝光做1:10均匀降采样后，CTR AUC为0.625726，基本保持；但CTCVR AUC
  降至0.652472。原始平均pCTR从真实0.038828偏到0.107481，说明负采样会改变
  类别先验。先验修正和Platt能恢复概率尺度，但不能恢复已经损失的排序能力。
- 为ESMM增加点击空间CVR 1:20辅助损失后，CTCVR AUC降至0.662290，平均pCVR
  达到0.043994，而真实CVR只有0.005479；CVR Gate同时塌缩到单个共享专家。
  该实验说明强辅助监督会破坏PLE的信息解耦，因此不进入最终方案。
- Gate Dropout没有带来稳定的全局提升：MMoE的CTCVR AUC从0.661048降至
  0.659244；PLE的AUC近似持平但GAUC下降。
- Target-aware历史注意力提高CTR AUC和CTCVR GAUC，但全局CTCVR AUC低于
  主模型且训练更慢，因此作为消融保留。

这些结果共同说明最终方案不是按模块数量选择，而是以完整曝光空间CTCVR排序为
主指标，通过对照实验保留有效模块、排除无收益或造成概率偏移的模块。

### 实验产物

每次实际运行保存：

```text
results/<experiment>_<timestamp>/
├── config.yaml
├── best.pt
├── metrics.json
└── train.log
```

仓库保留所有全量实验的配置、指标和日志。约1GB的模型checkpoint不提交Git；本地
只需保留最终DCN-PLE-ESMM的`best.pt`，其他模型均可通过对应配置重新训练。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖解析、词表、Parquet分片、Join、DataLoader、模型forward/backward、
多任务损失、GAUC、负采样、先验修正和概率校准。
