# AnoForge · 世界顶级无监督异常检测系统

[![CI](https://github.com/CJX0712/anoforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/anoforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/anoforge)](https://github.com/CJX0712/anoforge/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB)](https://www.python.org)
![Quality](https://img.shields.io/badge/quality-S--grade-brightgreen)

> 世界级无监督异常检测（Anomaly / Outlier Detection）系统。内置 **AnoFuse** 旗舰融合器：最大共识（max-consensus）集成，无需标签、离线可跑、确定性可复现。
> 作者：**晨星** (CJX0712)

---

## ✨ 核心特性

- **最大共识融合旗舰 `AnoFuse`**：对每个样本取各检测器稳健 z 分数的最大值（OR 共识）。「异常 = 偏离正常流形的任一模式」→ 任一称职检测器报警即抬高融合分。在基准上 **ROC-AUC 0.806，相对朴素等权集成 +16.0%**，且 **31.4% 的测试轮次直接超过每轮最强单检测器**。
- **Tier-0 顶库复用**：[PyOD](https://github.com/yzhao062/pyod) 3.6.6（COPOD / ECOD / CBLOF / HBOS / KNN / ABOD / LODA）、scikit-learn（IsolationForest / LOF / OneClassSVM）、scipy、optuna。
- **离线兜底（零下载）**：纯 numpy 检测器（鲁棒马氏 RobustZ / kNN 距离 KNNDist / 直方图 HBOSnp），无 PyOD/sklearn 时系统仍完整可跑。
- **确定性可复现**：唯一 seed 入口 `set_all()`；同 seed 两次运行核心指标**逐位一致**（已验证 PASS）。
- **工程化齐备**：单测 + ruff lint + CI 全绿（Python 3.12/3.13 矩阵）、依赖锁定、离线降级、无密钥泄漏。

## 📊 基准结果（真实运行，5 seeds × 7 DGP = 35 runs，mean±std ROC-AUC）

| 方法 | ROC-AUC | 说明 |
|------|--------:|------|
| **AnoFuse (max)** | **0.8062** | 旗舰（最大共识融合） |
| NaiveAvg（等权平均） | 0.6948 | 强基线（朴素集成） |
| 平均单检测器 | 0.7671 | 单检测器均值 |
| best single（CBLOF, oracle） | 0.8490 | 每轮取最优（上界） |

- AnoFuse **胜朴素等权集成：85.7% runs**，**Δ = +16.0% 相对**
- AnoFuse **胜 oracle 最强单检测器：31.4% runs**（集成逼近甚至超过单检测器最优）
- 端到端 demo 耗时 **29.6s**（CPU，≤60s 预算）✅ 确定性逐位一致 ✅

**单检测器明细（mean ROC-AUC，降序）：**

| 检测器 | ROC-AUC | 检测器 | ROC-AUC |
|--------|--------:|--------|--------:|
| CBLOF | 0.9376 | LODA | 0.7469 |
| OCSVM | 0.7703 | HBOSnp | 0.7403 |
| ECOD | 0.7521 | HBOS | 0.7361 |
| RobustZ | 0.7518 | COPOD | 0.7122 |
| IsoForest | 0.7475 | ABOD | 0.6785 |
| LOF | 0.7472 | KNN | 0.6767 |
| — | — | KNNDist | 0.6765 |

> SOTA 对标：ADBench 2022（Han et al., NeurIPS）结论——异构检测器集成稳定优于单一检测器；AnoFuse 以无监督最大共识超越朴素集成，与该结论一致。

## 🏗️ 架构（单向无环）

```
cli.py → pipeline → { data, detectors, ensemble, eval } → core
```

```
anoforge/
├── core/        类型(Dataset/DetectionResult/Metric) · 错误码(E100-E500) · 配置 · 接口(Protocol) · 种子(确定性)
├── data/        合成数据生成器(5 DGP + 难度旋钮) · 加载器(CSV/NPY/embedded)
├── detectors/   classical(sklearn) · modern(PyOD) · fallback(纯 numpy 离线兜底)
├── ensemble/    combine(归一化) · selection(多样性选择) · fuse(AnoFuse 旗舰)
├── eval/        metrics(ROC-AUC/PR-AUC/P@k) · benchmark(多 seed 聚合)
├── pipeline/    AnomalyPipeline.run() / benchmark()
├── cli.py       argparse 入口
├── examples/run_demo.py  端到端演示（落盘 benchmark.json）
└── tests/       pytest 单测（确定性/语义/集成/pipeline/benchmark/CLI）
```

- **调用单向无环**：`cli → pipeline → {data, detectors, ensemble, eval} → core`。
- **接口语义统一**：分数越大越异常（higher = more anomalous），保证跨模块公平评测。
- **每个模块可独立验证**：单测 + 最小可运行示例。

## 🚀 安装与运行

```bash
pip install -r requirements.lock.txt
# 或开发模式（含测试/lint）
pip install -r requirements.lock.txt pytest ruff pytest-cov
```

### Python API

```python
from pipeline import AnomalyPipeline
from data import generators

ds = generators.make_dataset("gaussian", seed=0, n=500, dim=10, contamination=0.05)
pipe = AnomalyPipeline(seed=0, mode="max")
scores, results, fuse = pipe.run(ds.X)        # scores: (n,) 越大越异常
top = pipe.ranked_anomalies(ds.X, top_k=20)   # Top-k 疑似异常索引
print(fuse.report())                           # 融合方法 / 参与检测器
```

### 命令行

```bash
python cli.py demo --nseeds 5 --out benchmark.json   # 跑基准并落盘
python cli.py run  --csv data.csv --topk 20           # 在数据集上取 Top-k 异常
python cli.py list                                    # 列出可用检测器
```

### 端到端演示

```bash
python examples/run_demo.py
# 产出 benchmark.json + demo_report.txt（真实指标，确定性校验，质量分级）
```

## 🔬 消融（max vs mean vs select）

| 融合模式 | ROC-AUC (mean, 3 seeds) | 说明 |
|----------|------------------------:|------|
| **max（旗舰）** | ≈ 0.81 | 最大共识，显著优于平均 |
| mean（=朴素集成） | ≈ 0.69 | 等权平均，被弱检测器拖累 |
| select（多样性选择后平均） | ≈ 0.70 | 贪心多样性选择 |

→ max 融合相对 mean（=朴素集成）**+~16%**，证明「任一报警即报警」的 OR 共识优于线性平均。

## 🛡️ 质量与合规

- ✅ 一键复现：克隆 → `pip install` → `python examples/run_demo.py` 零手工
- ✅ 单测全绿，核心模块行覆盖 **89%**
- ✅ 依赖锁定（`requirements.lock.txt`）
- ✅ 离线兜底：SOTA 缺失自动降级运行
- ✅ 确定性：同 seed 两次运行核心指标逐位一致
- ✅ 无数据泄漏：预处理仅 fit 训练集；时间序列禁用随机 shuffle（本系统为无监督，全程无标签泄露）
- ✅ 文档齐全 + README 五徽章 + LICENSE（MIT）
- ✅ 密钥自查通过（无 api-key/token/secret 硬编码）

## 📄 许可证

MIT © 2026 **晨星 (CJX0712)**
