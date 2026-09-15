# Online Serving + A/B Experiment Simulation 开发要求

在完成核心离线建模流程后，继续为项目增加在线推理服务、A/B 分桶模拟和服务性能评估。

本项目没有真实生产流量，因此禁止将模拟实验描述为真实线上 A/B Test。项目中统一使用：

- A/B Experiment Simulation
- Offline Experiment
- Traffic Bucketing Simulation
- Offline Replay Experiment
- Offline Model Comparison

## 1. Online Serving

完成离线训练后，将最佳模型部署为一个可调用的广告排序服务：

```text
Offline Trained Model
        ↓
    Model Export
        ↓
TorchScript / ONNX
        ↓
   FastAPI Service
        ↓
 Feature Processing
        ↓
 Batch Inference
        ↓
 Ranking Score
        ↓
   Ranking Result
```

优先使用 PyTorch、FastAPI 和 Uvicorn。模型导出顺序为 PyTorch checkpoint → TorchScript；只有实现稳定后再考虑 ONNX，不要为了使用 ONNX 增加不必要复杂度。

## 2. Ranking API

实现最小可运行的广告排序接口：

```text
POST /rank
```

输入包含 user features、candidate ads 和 context features。概念示例：

```json
{
  "user_id": 10001,
  "candidates": [
    {"ad_id": 101, "features": {}},
    {"ad_id": 102, "features": {}}
  ]
}
```

输出示例：

```json
{
  "request_id": "...",
  "model_version": "...",
  "results": [
    {
      "ad_id": 102,
      "pctr": 0.082,
      "pcvr": 0.014,
      "score": 0.082,
      "rank": 1
    }
  ]
}
```

如果使用 ESMM、MMoE 或 PLE 等多任务模型，应支持输出 pCTR、pCVR 和 pCTCVR。

## 3. Ranking Score

项目至少支持：

```text
score = pCTR
```

进一步通过配置支持：

```text
score = pCTR × pCVR
score = pCTR × bid
score = pCTR × pCVR × value
```

如果 Ali-CCP 本身没有真实 bid/value 数据，只能使用明确标记为 simulated 的字段演示排序逻辑，禁止将其描述为真实数据。

## 4. A/B Experiment Simulation

实现基于 User ID 的稳定哈希分桶：

```text
bucket = stable_hash(user_id) % 100
```

同一个用户在不同请求和不同进程中必须稳定进入相同实验组。不能直接使用受 Python 随机化影响的内置 `hash()`；应使用 MD5、SHA256 或 xxhash 等稳定算法。

示例分组：

```text
0–49   → Control Group
50–99  → Treatment Group
```

Control 可以部署 DeepFM/ESMM，Treatment 可以部署 MMoE/PLE，但必须使用实际训练所得模型，不能虚构模型效果。

## 5. Experiment Config

A/B 实验必须配置驱动，并方便调整 90/10、80/20、50/50 等流量比例：

```yaml
experiment:
  name: ranking_model_v1
  enabled: true
  control:
    model: esmm
    traffic: 50
  treatment:
    model: mmoe
    traffic: 50
```

## 6. Exposure Logging

实验日志至少记录：

- timestamp
- request_id
- experiment_id
- experiment_group
- user_id
- model_version
- ad_id
- prediction_score
- rank

离线数据回放可以进一步记录 click 和 conversion，形成：

```text
request
→ bucket
→ model
→ ranking
→ exposure log
→ click/conversion label
→ experiment analysis
```

## 7. Offline A/B Replay

由于没有真实线上用户流量，使用 Ali-CCP Test Set 进行 Offline Replay Experiment。将测试样本分别通过 Control Model 和 Treatment Model，对比：

- CTR AUC
- CTR GAUC
- CVR AUC
- CTCVR AUC
- LogLoss
- PR-AUC
- Lift@K
- Calibration

这一本质上是 Offline Model Comparison，不是严格意义上的 Online A/B Test。README 和代码注释必须保持这一表述准确。

## 8. Statistical Analysis

可以通过 Bootstrap Confidence Interval 分析 AUC、GAUC 和 LogLoss 差异。建议输出：

```text
Treatment - Control
mean difference
95% confidence interval
```

如果实现显著性检验，必须明确统计方法、实验单位和假设，不能仅凭指标点估计上涨就声称“显著提升”。

## 9. Load Testing

在线推理服务完成后，至少统计：

- QPS
- P50 Latency
- P95 Latency
- P99 Latency

建议测试：

```text
Concurrency = 1 / 8 / 16 / 32
Batch size = 1 / 8 / 32 / 128
```

观察吞吐量、延迟和 GPU 利用率。

## 10. Benchmark Script

增加独立性能测试脚本：

```text
scripts/benchmark_service.py
```

如果引入 Locust 会明显增加复杂度，优先自行实现简单并发 benchmark。实际生成：

- `benchmark_results.json`
- Latency vs Concurrency 图表
- QPS vs Concurrency 图表

不得伪造 benchmark 数据。

## 11. 推荐新增目录

以下目录只能在项目进入相应阶段后逐步创建：

```text
src/
├── serving/
│   ├── app.py
│   ├── predictor.py
│   ├── feature_service.py
│   ├── ranking.py
│   └── schemas.py
└── experiment/
    ├── bucketing.py
    ├── experiment.py
    └── logger.py

scripts/
├── export_model.py
├── serve.py
├── benchmark_service.py
└── analyze_experiment.py
```

## 12. 开发顺序

```text
Stage 1  Data Pipeline
Stage 2  CTR Baseline
Stage 3  ESMM
Stage 4  MMoE / PLE
Stage 5  Negative Sampling
Stage 6  Calibration
Stage 7  Model Export
Stage 8  FastAPI Ranking Service
Stage 9  A/B Bucketing Simulation
Stage 10 Load Testing
Stage 11 Experiment Analysis
```

不要提前开发 Serving。必须等 Data Pipeline、DeepFM、ESMM、MMoE 和 Offline Evaluation 稳定以后再实现。

## 13. 项目最终完整链路

```text
Ali-CCP
   ↓
Raw Data Parsing
   ↓
Feature Engineering
   ↓
Sparse Feature Encoding
   ↓
Dataset / DataLoader
   ↓
DeepFM / DCNv2
   ↓
ESMM / MMoE / PLE
   ↓
CTR / CVR / CTCVR
   ↓
Negative Sampling
   ↓
Probability Calibration
   ↓
Offline Evaluation
   ↓
Model Export
   ↓
FastAPI Ranking Service
   ↓
User Hash Bucketing
   ↓
A/B Experiment Simulation
   ↓
Exposure Logging
   ↓
Offline Replay Analysis
   ↓
QPS / P50 / P95 / P99 Benchmark
```

## 14. README 表述要求

README 中禁止写：

```text
完成真实线上 A/B Test
```

应该写：

```text
Implemented a user-level traffic bucketing and A/B experiment simulation framework for comparing ranking model versions.
```

或者：

```text
实现基于 UserID 稳定哈希分桶的 A/B 实验模拟框架，并结合离线测试集完成不同排序模型版本的实验对比。
```

在线服务可以准确描述为：

```text
部署 FastAPI 广告排序推理服务，并通过 QPS、P50/P95/P99 延迟评估模型在线推理性能。
```

## 15. 最终目标

增加这一模块是为了覆盖：

```text
Offline Training
+ Model Evaluation
+ Model Export
+ Online Inference
+ Traffic Bucketing
+ Experiment Analysis
+ Serving Benchmark
```

目标是形成较完整的广告排序离线训练与在线推理工程 Pipeline，而不是将个人离线项目包装成真实生产系统。任何没有真实生产流量支撑的结果，都不能声称为真实线上业务结果。
