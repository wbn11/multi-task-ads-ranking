# 工业级多目标广告精排系统——Codex 项目总提示词

## 项目名称

**Industrial Multi-Task Ads Ranking System**
**工业级多目标广告 CTR/CVR 精排系统**

---

## 1. 项目背景与目标

这是一个面向秋招搜广推/广告算法岗位的个人项目。

项目目标不是简单复现某一个 CTR 模型，而是尽可能模拟工业广告推荐系统中的离线建模流程，完成：

**原始曝光日志 → 数据处理 → 稀疏特征建模 → CTR 预估 → CVR 预估 → 多任务学习 → 离线评估 → 负采样 → 概率校准 → 消融实验 → 推理排序**

完整链路。

项目主要用于展示以下能力：

1. 广告推荐系统数据处理能力
2. CTR / CVR / CTCVR 建模能力
3. Sparse Feature / Dense Feature 特征工程
4. DeepFM、DCNv2 等 CTR 模型实现能力
5. ESMM、MMoE、PLE 等多任务学习模型能力
6. DataLoader 与大规模稀疏数据训练工程能力
7. AUC、GAUC、LogLoss、PR-AUC 等广告指标理解
8. Negative Sampling 对训练与概率分布的影响
9. CTR Probability Calibration
10. 模型消融、模型对比和实验分析能力
11. PyTorch 工程化开发能力

本项目不追求搭建真正的线上广告服务集群，而是实现一个具有较完整工业思路的 **Offline Ads Ranking Pipeline**。

已有另一个项目覆盖 LLM、Semantic ID、SFT、GRPO 和生成式推荐，因此本项目应尽量聚焦：

**传统工业搜广推基本功 + 多任务广告建模**

不要为了使用大模型而强行加入 LLM 模块。

---

## 2. 硬件环境

主要训练设备：

- NVIDIA RTX A6000
- 48GB GPU Memory

预计服务器配置：

- Linux
- Python 3.10+
- CUDA
- PyTorch
- CPU RAM 视服务器情况调整

代码需要支持：

- GPU
- CUDA
- Mixed Precision
- Large Batch Training

尽量避免将全部数据一次性加载到内存。

对于大数据，应优先考虑：

- chunk processing
- parquet
- mmap
- iterable dataset
- efficient DataLoader

---

## 3. 数据集

主要使用：

**Ali-CCP（Alibaba Click and Conversion Prediction Dataset）**

项目任务包含：

```text
Impression
   ↓
Click
   ↓
Conversion
```

其中：

### CTR

```text
P(click = 1 | impression)
```

### CVR

```text
P(conversion = 1 | click = 1)
```

### CTCVR

```text
P(click = 1, conversion = 1 | impression)
```

满足：

```text
pCTCVR = pCTR × pCVR
```

需要特别关注广告 CVR 建模中的：

- Sample Selection Bias
- Data Sparsity

---

## 4. 项目整体技术路线

整体 Pipeline：

```text
Ali-CCP Raw Data
        │
        ▼
Data Parsing / Join
        │
        ▼
Feature Processing
        │
        ├── Sparse Features
        │
        ├── Dense Features
        │
        └── Labels
        │
        ▼
Feature Dictionary / Vocabulary
        │
        ▼
PyTorch Dataset / DataLoader
        │
        ▼
Embedding Layer
        │
        ├────────────────────────────┐
        │                            │
        ▼                            ▼
    DeepFM                        DCNv2
        │                            │
        └────────────┬───────────────┘
                     ▼
                CTR Baseline
                     │
                     ▼
              Multi-Task Learning
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
        ESMM        MMoE         PLE
          │          │           │
          └──────────┴───────────┘
                     ▼
          CTR / CVR / CTCVR
                     │
                     ▼
           Offline Evaluation
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
       AUC          GAUC        LogLoss
                                   │
                     PR-AUC / Calibration
                                   │
                                   ▼
                         Negative Sampling
                                   │
                                   ▼
                             Ablation Study
```

---

## 5. 项目目录结构

建议严格按照如下工程结构开发：

```text
multi-task-ads-ranking/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/
│   ├── data.yaml
│   ├── deepfm.yaml
│   ├── dcnv2.yaml
│   ├── esmm.yaml
│   ├── mmoe.yaml
│   └── ple.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── scripts/
│   ├── download_data.sh
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiments.sh
│
├── src/
│   │
│   ├── data/
│   │   ├── parser.py
│   │   ├── preprocess.py
│   │   ├── feature_encoder.py
│   │   ├── dataset.py
│   │   ├── sampler.py
│   │   └── dataloader.py
│   │
│   ├── layers/
│   │   ├── embedding.py
│   │   ├── mlp.py
│   │   ├── fm.py
│   │   ├── cross_network.py
│   │   └── expert.py
│   │
│   ├── models/
│   │   ├── lr.py
│   │   ├── deepfm.py
│   │   ├── dcnv2.py
│   │   ├── shared_bottom.py
│   │   ├── esmm.py
│   │   ├── mmoe.py
│   │   └── ple.py
│   │
│   ├── losses/
│   │   └── multitask_loss.py
│   │
│   ├── metrics/
│   │   ├── auc.py
│   │   ├── gauc.py
│   │   ├── ranking.py
│   │   └── calibration.py
│   │
│   ├── calibration/
│   │   ├── platt.py
│   │   └── isotonic.py
│   │
│   ├── trainer/
│   │   ├── trainer.py
│   │   ├── multitask_trainer.py
│   │   └── callbacks.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── seed.py
│       ├── config.py
│       └── checkpoint.py
│
├── experiments/
│   ├── baseline/
│   ├── multitask/
│   ├── sampling/
│   ├── calibration/
│   └── ablation/
│
├── notebooks/
│   ├── data_analysis.ipynb
│   └── result_analysis.ipynb
│
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_metrics.py
│
└── results/
    ├── logs/
    ├── checkpoints/
    ├── figures/
    └── metrics/
```

不要一开始把所有文件全部实现。

根据开发阶段逐步创建。

---

## 6. 第一阶段：数据 Pipeline

这是项目最重要的工程模块之一。

不要直接调用别人已经处理好的 Tensor Dataset。

尽量完成 Ali-CCP 原始数据到模型输入的处理过程。

### 6.1 数据解析

实现：

```text
Raw Ali-CCP
    ↓
sample_skeleton
+
common_features
    ↓
Join
    ↓
Training Samples
```

需要解析：

- user features
- ad/item features
- context features
- click label
- conversion label

输出统一格式。

例如：

```python
{
    "sparse_features": ...,
    "dense_features": ...,
    "click": 0,
    "conversion": 0
}
```

同时需要检查标签逻辑，例如统计并处理：

```text
click = 0 && conversion = 1
```

这类不符合正常点击后转化链路的异常样本。

---

## 7. Feature Engineering

广告数据以 categorical feature 为主。

实现统一 Feature Schema。

例如：

```python
SparseFeature(
    name="field_101",
    vocab_size=100000,
    embedding_dim=16
)
```

```python
DenseFeature(
    name="xxx",
    dimension=1
)
```

Ali-CCP 中大量字段是匿名化 field，因此除非官方资料明确给出业务语义，不要自行将 field 强行解释为年龄、性别、价格等真实业务字段。

所有 categorical feature：

```text
raw value
   ↓
frequency filter
   ↓
vocabulary
   ↓
feature id
   ↓
embedding
```

低频特征需要支持：

```text
UNK
```

或者：

```text
Hash Bucket
```

配置项例如：

```yaml
min_frequency: 5
embedding_dim: 16
hash_bucket: false
```

---

## 8. Dataset / DataLoader

实现：

```python
AdsDataset
```

返回：

```python
{
    "features": ...,
    "click": ...,
    "conversion": ...
}
```

需要兼容：

- batch training
- shuffle
- num_workers
- pin_memory
- GPU transfer

后期如果数据过大，再增加：

- IterableDataset
- Parquet Dataset
- Chunk Loading

不要一开始过度工程化。

---

## 9. 数据划分

严格避免随机泄漏。

优先按照数据原始时间顺序：

```text
Train
Validation
Test
```

例如：

```text
80%
10%
10%
```

如果 Ali-CCP 数据结构允许，应优先采用其官方 train/test 时间划分。

---

## 10. 第一组 Baseline

第一阶段只做简单模型验证 Pipeline 正确。

顺序：

```text
LR
↓
DeepFM
↓
DCNv2
```

---

## 11. Logistic Regression Baseline

用于验证：

- 数据正确
- label 正确
- metric 正确
- pipeline 正确

结构：

```text
Sparse Feature
     ↓
Embedding / Linear Weight
     ↓
Linear Sum
     ↓
Sigmoid
     ↓
CTR
```

损失：

```text
Binary Cross Entropy
```

---

## 12. DeepFM

手动实现 DeepFM 核心模块，不直接调用 DeepCTR-Torch 模型。

结构：

```text
                  Input
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   First Order              Embedding
        │                       │
        │               ┌───────┴───────┐
        │               ▼               ▼
        │              FM              DNN
        │               │               │
        └───────────────┼───────────────┘
                        ▼
                      Logit
                        ↓
                     Sigmoid
```

实现：

```python
class FM(nn.Module)
```

二阶交互不要使用显式 O(n²) 枚举。

使用公式：

```text
0.5 * [
    (sum(v_i))²
    -
    sum(v_i²)
]
```

实现：

```python
class DeepFM(nn.Module)
```

---

## 13. DCNv2

实现 Cross Network。

输入：

```text
x0
```

每层：

```text
x_{l+1}
=
x0 * f(x_l)
+
x_l
```

优先实现 DCNv2 中 Matrix Cross Network。

最终：

```text
Embedding
    │
    ├──── Cross Network ────┐
    │                       │
    └──── Deep Network ─────┤
                            ↓
                         Concatenate
                            ↓
                           CTR
```

目的：

研究显式 feature crossing 对 CTR 预测的作用。

---

## 14. CTR Baseline 实验

至少比较：

| Model | AUC | LogLoss | Params | Train Time |
|---|---:|---:|---:|---:|
| LR | | | | |
| DeepFM | | | | |
| DCNv2 | | | | |

不要提前伪造任何结果。

所有实验结果必须实际运行后记录。

---

## 15. 多任务学习

完成单目标 CTR 后进入项目核心部分：

```text
CTR + CVR
```

依次实现：

```text
Shared Bottom
ESMM
MMoE
PLE
```

---

## 16. Shared Bottom

共享底层：

```text
Input
  ↓
Shared Embedding
  ↓
Shared MLP
  │
 ┌┴───────────┐
 ↓            ↓
CTR Tower    CVR Tower
 ↓            ↓
CTR          CVR
```

主要作为多任务 baseline。

---

## 17. ESMM

这是项目重点模型之一。

需要正确理解：

CVR 真实含义：

```text
P(conversion | click)
```

但是传统 CVR 只使用点击样本训练会导致：

```text
Sample Selection Bias
```

同时 conversion 数据非常稀疏。

ESMM 在整个曝光空间中训练：

```text
CTR Task
+
CTCVR Task
```

计算：

```text
pCTR = CTR Tower(x)

pCVR = CVR Tower(x)

pCTCVR = pCTR * pCVR
```

Loss：

```text
Loss =
BCE(click, pCTR)
+
BCE(click * conversion, pCTCVR)
```

注意：

不要直接使用：

```text
BCE(conversion, pCVR)
```

作为标准 ESMM 主要训练目标。

实现：

```python
class ESMM(nn.Module)
```

forward 返回：

```python
{
    "ctr": p_ctr,
    "cvr": p_cvr,
    "ctcvr": p_ctcvr
}
```

---

## 18. MMoE

实现 Multi-gate Mixture-of-Experts。

输入：

```text
Shared Input
```

多个 Expert：

```text
Expert 1
Expert 2
...
Expert N
```

CTR Gate：

```text
Softmax Gate
      ↓
Weighted Experts
```

CVR Gate：

```text
Softmax Gate
      ↓
Weighted Experts
```

然后：

```text
CTR Tower
CVR Tower
```

实现：

```python
class Expert(nn.Module)
class Gate(nn.Module)
class MMoE(nn.Module)
```

要求支持配置：

```yaml
num_experts: 4
expert_hidden_dims:
  - 256
  - 128

tower_hidden_dims:
  - 64
```

---

## 19. PLE

如果项目时间允许，再实现 PLE。

结构：

```text
Shared Experts

CTR Specific Experts

CVR Specific Experts
```

通过不同 Gate 控制任务信息流。

至少实现：

```text
1 Layer PLE
```

如果开发复杂度过高，不需要立即做多层 CGC。

正确性优先于复杂度。

---

## 20. 多任务实验

最终形成：

| Model | CTR AUC | CVR AUC | CTCVR AUC | LogLoss |
|---|---:|---:|---:|---:|
| Shared Bottom | | | | |
| ESMM | | | | |
| MMoE | | | | |
| PLE | | | | |

需要分析：

- Shared Bottom 是否存在 Task Conflict
- MMoE 是否缓解 Task Conflict
- PLE 是否进一步解耦任务
- ESMM 是否改善 CVR 数据稀疏与 Selection Bias

---

## 21. Evaluation Metrics

必须实现：

- AUC
- LogLoss
- PR-AUC

推荐进一步实现：

- GAUC

GAUC 按用户计算：

```text
GAUC =
Σ weight(user) × AUC(user)
/
Σ weight(user)
```

权重可以使用：

```text
user impression count
```

如果一个用户的数据全部为正或者全部为负，则无法计算用户 AUC，需要跳过。

---

## 22. Ranking Metrics

为了模拟广告排序过程，可以增加：

- Precision@K
- Recall@K
- Lift@K

例如：

```text
按照 pCTR 对曝光广告排序
```

观察：

```text
Top 1%
Top 5%
Top 10%
```

正样本浓度。

---

## 23. Negative Sampling

真实广告场景负样本远多于正样本。

实现可配置 Negative Sampling：

```yaml
negative_sampling:
  enabled: true
  ratio: 10
```

实验：

```text
All Samples

Positive : Negative
1 : 5

1 : 10

1 : 20
```

比较：

| Sampling | AUC | LogLoss | Training Time |
|---|---:|---:|---:|
| Full | | | |
| 1:5 | | | |
| 1:10 | | | |
| 1:20 | | | |

重点分析：

- 训练速度
- Ranking Ability
- Probability Bias

---

## 24. Probability Calibration

这是项目非常重要的工业化模块。

CTR 排序中：

```text
AUC
```

主要衡量：

```text
Ranking Ability
```

但广告系统经常需要使用：

```text
pCTR
```

直接参与业务价值计算，例如：

```text
score = pCTR × bid
```

或者：

```text
score = pCTR × pCVR × value
```

因此模型预测概率应该尽量接近真实概率。

实现：

- Platt Scaling
- Isotonic Regression

输入：

```text
raw predicted CTR
```

输出：

```text
calibrated CTR
```

评估：

- LogLoss
- Brier Score
- ECE

并绘制：

```text
Calibration Curve
```

横轴：

```text
Predicted CTR
```

纵轴：

```text
Actual CTR
```

---

## 25. Training Framework

实现统一 Trainer。

支持：

- train
- validation
- test
- checkpoint
- early stopping
- gradient clipping
- mixed precision

核心配置：

```yaml
training:
  epochs: 10
  batch_size: 4096
  learning_rate: 0.001
  weight_decay: 0.00001
  num_workers: 8
  amp: true
  early_stopping_patience: 2
```

batch size 根据 A6000 显存实际情况动态调整。

不要假设 4096 一定最佳。

---

## 26. Optimizer

默认：

- Adam
- 或 AdamW

后续可以尝试：

- Adagrad

因为广告推荐中的大规模 sparse embedding historically 常使用 Adagrad 类优化器。

但项目初期不需要同时测试很多优化器。

---

## 27. Mixed Precision

A6000 支持 AMP。

训练时使用：

```python
torch.autocast
```

以及相应 scaler。

需要确保：

- Loss
- Metric
- Output

数值稳定。

---

## 28. Reproducibility

提供：

```python
set_seed()
```

固定：

- Python
- NumPy
- PyTorch
- CUDA

随机种子。

README 中说明：

```text
实验结果在随机种子固定情况下可复现。
```

---

## 29. Logging

训练过程必须记录：

- epoch
- train loss
- validation loss
- CTR AUC
- CVR AUC
- CTCVR AUC
- learning rate
- training time

推荐使用：

```text
Python logging
+
TensorBoard
```

第一版不要强制依赖 W&B。

---

## 30. Experiment Management

每次实验保存：

- config
- checkpoint
- metrics.json
- training.log

目录：

```text
results/
└── deepfm_exp001/
    ├── config.yaml
    ├── best.pt
    ├── metrics.json
    └── train.log
```

确保以后可以追踪：

```text
哪个结果对应哪个配置。
```

---

## 31. 消融实验

项目后期至少完成以下消融：

### Experiment 1

```text
LR
vs
DeepFM
vs
DCNv2
```

研究：

```text
feature interaction
```

### Experiment 2

```text
Shared Bottom
vs
MMoE
vs
PLE
```

研究：

```text
multi-task architecture
```

### Experiment 3

```text
Traditional CVR
vs
ESMM
```

研究：

```text
sample selection bias
+
data sparsity
```

### Experiment 4

```text
Full Negative Samples
vs
Negative Sampling
```

研究：

```text
efficiency
+
prediction bias
```

### Experiment 5

```text
Before Calibration
vs
After Calibration
```

研究：

```text
ranking quality
vs
probability accuracy
```

---

## 32. 最终推荐实验矩阵

控制项目规模，不无限增加模型。

核心模型：

- LR
- DeepFM
- DCNv2
- Shared Bottom
- ESMM
- MMoE
- PLE

其中重点：

- DeepFM
- ESMM
- MMoE

必须完整。

PLE 可以作为进一步增强。

---

## 33. 实验推进顺序

不要一次实现全部模型。

按照：

```text
Stage 1
Data Pipeline
↓
LR
↓
Metric
```

确认数据正确。

然后：

```text
Stage 2
DeepFM
↓
DCNv2
```

然后：

```text
Stage 3
Shared Bottom
↓
ESMM
```

然后：

```text
Stage 4
MMoE
↓
PLE
```

然后：

```text
Stage 5
Negative Sampling
```

然后：

```text
Stage 6
Calibration
```

最后：

```text
Stage 7
Ablation
+
README
+
Result Visualization
```

---

## 34. 数据规模推进

开发阶段不要一开始使用全量数据。

按照：

```text
Debug / Small Dataset
~ 100K samples
```

用于：

- 检查数据解析与 Join
- 检查代码、shape 和 loss
- 验证模型能够正常学习

然后：

```text
Medium Dataset
~ 1M samples
```

用于：

```text
模型比较
```

最后根据：

- RAM
- SSD
- 训练时间

决定是否扩大数据规模。

注意 Ali-CCP conversion label 极其稀疏，因此 100K 小数据主要用于调试，不能用来得出严肃的 CVR/ESMM 实验结论。

---

## 35. Debug 要求

实现任何模型后首先完成 overfit small batch test。

例如：

```text
取 512 个样本
```

让模型不断训练这一批数据。

如果模型连这 512 个样本都无法明显降低 loss：

```text
优先认为代码存在问题。
```

不要直接进行大规模训练。

---

## 36. 单元测试

至少给以下模块写简单测试：

- FeatureEncoder
- Dataset
- FM
- DeepFM
- ESMM
- MMoE
- GAUC

重点测试：

- shape
- NaN
- forward
- backward
- loss

---

## 37. 编码规范

整个项目使用：

- Python
- PyTorch

不要同时混用：

- TensorFlow
- JAX

核心模型尽量自己实现。

允许使用：

- numpy
- pandas
- polars
- scikit-learn
- PyYAML
- tqdm
- matplotlib

对于大规模数据，优先考虑：

- Polars
- PyArrow
- Parquet

---

## 38. 模型 API 统一

所有模型尽量统一：

```python
output = model(batch)
```

单任务模型：

```python
{
    "ctr": ...
}
```

多任务模型：

```python
{
    "ctr": ...,
    "cvr": ...,
    "ctcvr": ...
}
```

避免 Trainer 为每个模型写完全不同的代码。

---

## 39. Config Driven

尽量不要把超参数硬编码。

例如：

```yaml
model:
  name: esmm

embedding:
  dim: 16

network:
  hidden_dims:
    - 512
    - 256
    - 128

training:
  batch_size: 4096
  learning_rate: 0.001
  epochs: 10
```

---

## 40. README 最终目标

README 不能只写：

```text
如何运行代码。
```

最终应该包含：

1. Project Introduction
2. Advertising Ranking Background
3. Dataset
4. System Architecture
5. Data Pipeline
6. Models
7. Multi-task Learning
8. Experiments
9. Results
10. Ablation
11. Calibration
12. How to Run
13. Project Structure

需要至少有：

- 整体架构图
- 模型结构图
- 实验表格
- Calibration Curve

---

## 41. 最终项目亮点

### 亮点 1

完整构建：

```text
Ali-CCP
→ 数据解析
→ Feature Engineering
→ Dataset
→ Training
→ Evaluation
```

### 亮点 2

实现：

```text
DeepFM
DCNv2
```

用于广告 CTR 建模和特征交叉。

### 亮点 3

实现：

```text
ESMM
MMoE
PLE
```

完成：

```text
CTR
CVR
CTCVR
```

多任务学习。

### 亮点 4

针对 CVR：

```text
Sample Selection Bias
Data Sparsity
```

进行分析。

### 亮点 5

加入：

```text
Negative Sampling
+
Calibration
```

模拟真实广告建模中的工程问题。

---

## 42. 面试目标

项目代码和实验必须能够支撑回答：

- CTR 和 CVR 有什么区别？
- 为什么 CVR 比 CTR 更难训练？
- 什么是 Sample Selection Bias？
- ESMM 为什么在整个曝光空间训练？
- pCTCVR 为什么等于 pCTR × pCVR？
- DeepFM 为什么可以学习二阶特征交叉？
- DCNv2 和 DeepFM 的区别是什么？
- Shared Bottom 有什么问题？
- MMoE 如何缓解多任务冲突？
- MMoE 的 Gate 是怎么工作的？
- PLE 相比 MMoE 做了什么？
- AUC 为什么适合 CTR？
- GAUC 为什么更符合推荐场景？
- 为什么 Accuracy 不适合 CTR？
- 负采样会不会影响 AUC？
- 为什么负采样以后 CTR 概率会偏？
- 什么是 Calibration？
- 为什么广告系统需要校准后的 pCTR？
- pCTR 在广告排序中如何参与 eCPM？

---

## 43. 禁止事项

开发过程中不要：

1. 一开始就生成几十个无用文件。
2. 为追求“高级”强行加入 LLM。
3. 同时实现十几个 CTR 模型。
4. 直接复制 DeepCTR-Torch 整个实现。
5. 伪造实验结果。
6. 在没有验证数据 Pipeline 的情况下直接训练复杂模型。
7. 使用随机数字填充 README 实验结果。
8. 为了追求工程复杂度而过早加入 Spark、Kafka、Redis。
9. 把离线项目包装成真实线上生产系统。
10. 在不理解 Ali-CCP 字段含义的情况下猜测 Feature Schema。

---

## 44. Codex 工作方式要求

你是这个项目的协作开发助手。

在整个开发过程中，请遵守：

### 第一

在修改代码前先检查当前仓库已有实现。

不要重复创建已有功能。

### 第二

每次只完成当前阶段需要的模块。

不要一次生成整个项目。

### 第三

修改已有代码时：

```text
优先复用
最小修改
保持接口一致
```

### 第四

每次实现新模块后告诉我：

1. 修改了哪些文件
2. 每个文件的作用
3. 核心实现逻辑
4. 如何运行
5. 预期输出
6. 下一步推荐做什么

### 第五

如果遇到错误：

不要只是绕开错误。

需要：

```text
定位原因
解释原因
修改代码
验证修改
```

### 第六

模型实现优先：

```text
正确
↓
可运行
↓
可复现
↓
性能
↓
工程优化
```

不要为了优化训练速度牺牲代码正确性。

### 第七

任何模型正式大规模训练前都要先：

- 检查 tensor shape
- 检查 label
- 检查 output range
- 检查 NaN
- 检查 loss
- 完成 small batch overfit

### 第八

不要自行虚构实验数据。

只有实际训练以后才能填写：

- AUC
- GAUC
- LogLoss
- 训练时间
- 模型提升比例

---

## 45. 当前开发总目标

最终形成一个能够放在秋招简历和 GitHub 上的完整项目：

**基于多任务学习的广告点击与转化预估系统**

GitHub 仓库建议名称：

```text
multi-task-ads-ranking
```

技术关键词：

```text
PyTorch
Ali-CCP
CTR Prediction
CVR Prediction
DeepFM
DCNv2
ESMM
MMoE
PLE
Multi-Task Learning
Sparse Features
Negative Sampling
Probability Calibration
AUC
GAUC
LogLoss
Advertising Ranking
```

项目重点不是追求 SOTA，而是：

**完整性 + 工业逻辑 + 模型理解 + 工程实现 + 实验分析。**

在后续所有开发任务中，请始终以这个项目目标和架构作为上下文，不要随意偏离技术路线。

---

## 46. 第一个开发任务

现在不要直接实现 DeepFM、ESMM 或 MMoE。

首先：

1. 检查当前仓库目录。
2. 如果仓库为空，则创建最小必要目录。
3. 研究 Ali-CCP 数据文件结构。
4. 设计数据处理 Pipeline。
5. 明确：
   - 原始文件分别是什么
   - 如何关联
   - click label 如何获得
   - conversion label 如何获得
   - 哪些字段属于 sparse feature
   - 是否存在 dense feature
6. 对匿名 field 保持匿名，不要自行猜测真实业务语义。
7. 检查 `click = 0 && conversion = 1` 等异常标签组合。
8. 给出数据处理模块设计方案。
9. 创建最小可运行的数据解析代码。
10. 直接抽取约 10 万条样本验证 Pipeline，不再单独生成 1 万条调试数据。
11. 输出：
    - 样本数量
    - CTR
    - CVR
    - CTCVR
    - 异常标签数量
    - Feature 数量
    - 每类 Feature 的 cardinality
12. 在确认数据正确以前，不进入模型开发阶段。

---

## 47. 后期扩展要求

完成核心离线建模流程后，必须继续遵守 [Online Serving + A/B Experiment Simulation 开发要求](ONLINE_SERVING_AB_SIMULATION.md)。该扩展只能在离线评估、负采样和概率校准稳定后开始，不得提前创建 Serving 或 Experiment 模块。

---

## 48. 后续模块讲解与实现要求

后续实现任何数据、模型、训练、评估、校准、Serving 或实验模块时，必须先向用户详细介绍设计，再编写程序。讲解至少包含：

1. 为什么当前阶段需要这个模块，以及它解决的具体问题。
2. 模块在广告 CTR/CVR 精排完整链路中的作用。
3. 核心原理、关键公式、必要假设和容易出错的地方。
4. 程序输入：文件、配置、字段、Tensor shape、数据类型和前置依赖。
5. 程序输出：文件、对象、字段、Tensor shape、指标及其后续消费者。
6. 完整工作流程，尤其是数据从原始输入到最终输出的逐步变化。
7. 当前程序与已有模块、后续模块之间的调用关系和数据依赖关系。
8. 为什么选择当前实现方案，以及暂不采用的替代方案和取舍。

编写完成后还要说明：

1. 实际修改了哪些文件。
2. 每个文件和关键类/函数的职责。
3. 如何运行和验证。
4. 预期输出是什么。
5. 实际验证结果及其是否符合预期。
6. 当前限制、未完成事项和下一步建议。

对于复杂数据流程，优先使用清晰的文本流程图展示模块关系，例如：

```text
Raw File
   ↓ parser.py
Parsed Record
   ↓ preprocess.py
Encoded Sample
   ↓ dataset.py / dataloader.py
Training Batch
   ↓ model.py
Prediction
   ↓ metrics.py
Evaluation Result
```

不能只给出代码或笼统描述。必须让用户能够理解每个程序为什么存在、接收什么、产出什么，以及它如何接入整个工程。
