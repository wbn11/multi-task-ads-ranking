# FIER：广告点击与转化联合预估的离线精排实践

广告精排需要判断一次曝光能否带来点击，以及点击后是否产生转化。本项目基于
[Ali-CCP](https://tianchi.aliyun.com/dataset/408) 原始曝光数据，构建从流式预处理、
稀疏特征编码到 CTR/CVR/CTCVR 训练、评估与概率校准的完整离线流程。
核心方案 FIER 将 DCNv2 显式特征交叉、PLE 任务专家与 ESMM 全曝光训练目标结合，
并用同条件对照、负采样和多随机种子实验检验它的收益与局限。

## 1. 问题定义

广告行为通常遵循“曝光 → 点击 → 转化”。对于当前曝光特征 $x$：

$$
p_{CTR}=P(click=1\mid x),\quad
p_{CVR}=P(conversion=1\mid click=1,x),\quad
p_{CTCVR}=P(click=1,conversion=1\mid x)=p_{CTR}p_{CVR}.
$$

CTR 和 CTCVR 面向全部曝光评估；点击后 CVR 只在已点击曝光上评估。
转化正例远少于点击正例。如果只在点击样本上训练 CVR，模型面对的是经过点击筛选的
特征分布，与对全部曝光做排序的场景不同，容易出现样本选择偏差。FIER 使用 CTR 与
CTCVR 的全曝光监督缓解这一问题，但不保证消除所有估计偏差。

## 2. 数据与处理链路

```text
sample_skeleton（曝光、标签、sample 侧特征）
             + common_features（用户及历史特征）
             → 按 common_feature_id 关联
             → 流式扫描、清理不符合点击→转化漏斗的标签
             → 仅用训练集建立逐字段词表（PAD=0、UNK=1）
             → sample/common 配对 Parquet 分片
             → Arrow 列式 Batch，在分片内按 common_index 关联
             → ids / values / offsets → GPU 模型
```

配对分片对公共特征去重，不在每条曝光中重复展开长用户历史。模型接收 23 个匿名稀疏字段；
不根据字段编号猜测真实业务语义。预处理及 DataLoader 入口见[运行文档](docs/usage.md)。

| 划分 | 有效曝光数 | 来源 |
|---|---:|---|
| Train | 42,299,905 | 官方 Train |
| Validation | 4,263,506 | 官方 Test 剩余样本的哈希划分结果 |
| Test | 38,353,110 | 官方 Test 剩余样本的哈希划分结果 |

具体做法是：先从**官方 Test** 排除开发期实验用过的前 400K 条，再以固定种子对
剩余样本的 `sample_id` 做稳定哈希，约 10% 分入 Validation，其余分入本项目的 Test。
原始文件没有显式时间戳，因而这里**不是时间切分**。
词表只拟合 Train，验证与测试中的未见 token 映射为 UNK。

## 3. FIER 模型

```text
23 个字段 → 16 维 Embedding / 字段加权均值 → 拼接输入 x₀
                                            ↓
                                3 层 DCNv2 Matrix Cross
                                            ↓
                                        LayerNorm
                                            ↓
                          单层 PLE 式专家与任务 Gate
                    ┌───────────────────────┴───────────────────────┐
             共享专家 ×2 + CTR 专家 ×1                    共享专家 ×2 + CVR 专家 ×1
                    ↓                                               ↓
              CTR Tower → pCTR                                 CVR Tower → pCVR
                    └───────────────────────┬───────────────────────┘
                                        pCTCVR = pCTR × pCVR
```

两侧使用的是**同一组**共享专家，并非各自复制两份。下图将 2 个共享专家、
1 个 CTR 专家和 1 个 CVR 专家分别画出，展示它们如何进入两个 Gate：

![FIER 模型结构图：DCNv2 交叉网络、四个独立专家、双任务 Gate 与 CTCVR 乘积](docs/figures/fier_architecture.svg)

DCNv2 Cross 学习显式特征交互；两个任务的 Gate 分别融合共享专家与本任务专家。
这里实现的是**单层** PLE 式路由，而非原论文的完整多层结构。两个输出满足
`pCTCVR = pCTR × pCVR`，默认在全部曝光上优化：

```text
Loss = BCE(click, pCTR) + BCE(click × conversion, pCTCVR)
```

网络前向可使用 AMP，损失计算保持 FP32。DCNv2、PLE 与 ESMM 均为已有方法；
本项目的重点是将它们接入同一数据和评估流程，并验证组合效果。

## 4. 实验协议

训练使用固定数据划分、batch size 8192、最多 10 个 epoch 和 early stopping patience 3。
每次运行仅按 **Validation CTCVR AUC** 选择 `best.pt`；Test 不参与选轮次或校准器拟合。
CVR 指标只在点击样本上计算，CTR/CTCVR 指标在全部曝光上计算；GAUC 按符合计算条件的
用户曝光数加权。以下同条件组件对照统一使用 seed 2026、`dropout=0.1`、
`gate_dropout=0` 和完整曝光训练；另行比较 dropout 时保持其他设置不变。

结果来自实际全量实验。[实验汇总 CSV](results/experiment_summary.csv) 记录运行名、
配置、耗时和测试指标；除专门注明的先验修正与校准结果外，下文 Test 指标均使用
模型的未校准输出。历史探索已多次查看 Test，因此这些对照是回顾性分析，
不能视为完全未触碰测试集的盲测结论。

## 5. 核心结果

### 同条件组件对照（Test；dropout=0.1，seed=2026）

| 方案 | 训练目标 | CTR AUC | CVR AUC | CTCVR AUC | CTCVR GAUC | CTCVR LogLoss |
|---|---|---:|---:|---:|---:|---:|
| DCNv2 + PLE | CTR + 点击空间 CVR | 0.619646 | 0.676244 | 0.662116 | 0.625034 | 0.002168 |
| PLE + ESMM | CTR + 全曝光 CTCVR | 0.624086 | 0.688508 | 0.667359 | **0.629610** | **0.002012** |
| **FIER（DCNv2 + PLE + ESMM）** | CTR + 全曝光 CTCVR | **0.625479** | **0.691803** | **0.672664** | 0.623419 | 0.002014 |

DCNv2 + PLE 与 FIER 具有相同骨干，主要对照点击空间 CVR 损失与全曝光 CTCVR 损失；
PLE + ESMM 与 FIER 主要对照是否加入 Cross Network，但参数量也随之变化，
不能把差异全归因于“交叉”本身。此次 FIER 的全局 CTCVR AUC 最高；PLE + ESMM 的
CTCVR GAUC 和 LogLoss 略好，没有一个方案在所有指标上占优。

### Dropout 与随机种子

FIER 分别以 0.1 和 0.2 跑了相同的三个 seed。下表 Test 栏为 CTCVR AUC，
“±”后是三个 seed 的**样本标准差**；Validation 栏为各 seed 最优轮次 AUC 的均值。

| Dropout | Validation AUC 均值 | Test：2026 | Test：2027 | Test：2028 | Test 均值 ± 标准差 |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.655295 | 0.672664 | 0.668533 | 0.655896 | 0.665698 ± 0.008736 |
| 0.2 | 0.656173 | 0.677874 | 0.669411 | 0.660807 | 0.669364 ± 0.008533 |

目前默认保留 `dropout=0.2`，但 Validation 均值仅高 0.000878，Test 均值差
0.003666 也小于组内波动，**不能据此声称 0.2 稳定优于 0.1**。下文概率校准使用
0.2、seed 2026 的运行 `dcn_ple_esmm_aliccp_full_dropout_02_20260915_154204`。

## 6. 关键取舍

### 未点击曝光负采样

以下比较固定 `dropout=0.1`、seed 2026；Validation/Test 始终保持原分布。
只对训练集的未点击曝光做均匀负采样，所有点击曝光保留。

| 训练曝光 | 训练流程耗时 | Test CTCVR AUC | 原始 CTCVR LogLoss | 先验修正后 CTCVR LogLoss |
|---|---:|---:|---:|---:|
| 全样本 | 2,880 s | **0.672664** | **0.002014** | — |
| 点击 : 未点击 = 1 : 5 | 636 s | 0.658952 | 0.003423 | 0.002115 |
| 点击 : 未点击 = 1 : 10 | 932 s | 0.652472 | 0.002346 | 0.002044 |
| 点击 : 未点击 = 1 : 20 | 1,855 s | 0.670302 | 0.002071 | 0.002078 |

1:5 大幅缩短耗时，但 AUC 下降；1:20 更接近全样本表现，时间收益也更小。
均匀负采样改变了类别先验，先验修正通常可调整概率尺度，却**不保证**每次都改善
LogLoss（如 1:20），更不能恢复采样训练已经失去的排序信息。耗时还受机器和 I/O 影响，
不能全部归因于采样算法。默认保留全样本训练。

### 辅助 CVR、Gate Dropout 与历史注意力

下表均固定 `dropout=0.2`、seed 2026、完整曝光训练，每行只改动所列模块：

| 设置 | 点击空间 CVR AUC | CTCVR AUC | CTCVR GAUC | CTCVR LogLoss |
|---|---:|---:|---:|---:|
| FIER 默认 | 0.692857 | **0.677874** | 0.622393 | 0.002049 |
| + 辅助 CVR 损失（权重 0.02，负采样 1:20） | **0.693999** | 0.676627 | 0.623652 | 0.002352 |
| + Gate Dropout 0.1 | 0.676000 | 0.665352 | **0.628898** | **0.002035** |
| + Target-aware 历史注意力 | 0.679687 | 0.668247 | 0.619018 | 0.002080 |

辅助 CVR 略提高点击空间 CVR AUC，却没有提高主指标且使 CTCVR LogLoss 变差；
Gate Dropout 提高 GAUC，但降低全局 CTCVR AUC；历史注意力也未改善主指标。
因此默认配置不加入这三项。它们目前只有单 seed 的同条件对照，不能据此判断稳定收益。

### 概率校准

对 CTR/CVR 分别在 Validation 拟合 Platt Scaling 与 Isotonic Regression，
再相乘得到校准后的 CTCVR；Test **只用于评估**。下表来自上述 FIER 0.2、seed 2026
运行的 `evaluation_calibrated_multitask.json`，ECE 使用 20 个等频分箱：

| CTCVR 概率 | Val LogLoss | Val ECE | Test AUC | Test LogLoss | Test ECE |
|---|---:|---:|---:|---:|---:|
| 未校准 | 0.002070 | 0.00012058 | **0.677874** | 0.002049 | 0.00011723 |
| Platt | 0.001978 | **0.00003690** | 0.675274 | **0.001970** | **0.00001990** |
| Isotonic | **0.001975** | 0.00003722 | 0.672833 | 0.001973 | 0.00002562 |

以 **Validation ECE** 为当前报告的选择口径，概率输出暂选 Platt：它在该口径
略优于 Isotonic，Test LogLoss/ECE 也更低；若以 Validation LogLoss 为标准，
则应选 Isotonic。两种方法差距很小，且当前 Validation 同时承担校准器拟合与方法比较，
选择结果不是独立验证的结论；更严格的实验应另留校准方法选择集。
分头校准再相乘不保证保持 CTCVR 排序，两种方法的 Test AUC 均低于未校准预测。
因此排序评估保留原始分数；如需概率输出，本报告暂推荐 Platt。评估程序仍保存
两种校准结果，并未自动部署其中一种。下图只展示原始概率与 Platt 的 Test 分箱曲线；
校准器没有使用 Test 标签拟合。

![Test CTCVR 原始概率与 Platt 校准曲线；20 个等频分箱](docs/figures/ctcvr_calibration.svg)

## 7. 局限

- 本项目只有离线曝光日志；没有真实 bid/value、线上流量或真实 A/B 实验。
- 哈希划分保证运行间一致，不能替代时间切分；正式评估已排除开发期接触过的样本。
- 项目探索已多次查看 Test；主线组件与采样实验多为单次运行，不能宣称统计显著或严格盲测。
- 当前校准器在 Validation 上拟合并以同一划分比较方法，尚缺独立的校准方法选择集。
- 模型使用单层 PLE 式专家路由；方法模块源自已有研究，当前实验不证明组合在其他数据集上普适。

## 8. Quick Start

准备 Python 3.12、与驱动兼容的 PyTorch/CUDA 环境以及官方 Ali-CCP 原始文件。
从数据处理到 FIER 训练、评估和断点续训的输入、命令及输出，见
**[完整程序说明与运行命令](docs/usage.md)**。

## 9. 方法来源

- [DCN V2：显式特征交叉](https://arxiv.org/abs/2008.13535)
- [PLE：共享与任务专家](https://doi.org/10.1145/3383313.3412236)
- [ESMM：全曝光空间点击与转化建模](https://arxiv.org/abs/1804.07931)

以上论文提供方法基础；本仓库的指标仅对应本仓库记录的 Ali-CCP 离线实验。
