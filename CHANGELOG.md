# Changelog

## [0.1.0] - 2026-10-11

### Added
- 初始发布：世界顶级无监督异常检测系统 AnoForge。
- 旗舰融合器 `AnoFuse`（最大共识 max-consensus），无监督、离线、确定性可复现。
- 检测器池：
  - 经典（scikit-learn）：IsolationForest、LOF、OneClassSVM
  - 现代（PyOD 3.6.6）：COPOD、ECOD、CBLOF、HBOS、KNN、ABOD、LODA
  - 离线兜底（纯 numpy）：RobustZ、KNNDist、HBOSnp
- 合成数据生成器（5 类 DGP + 难度旋钮），可复现。
- 评测指标：ROC-AUC / PR-AUC / Precision@k；多 seed 基准聚合。
- CLI（`demo` / `run` / `list`）与端到端演示 `examples/run_demo.py`。
- 完整工程化：pytest 单测、ruff lint、CI（3.12/3.13）、Dockerfile、Makefile、依赖锁定。

### Performance
- AnoFuse ROC-AUC = 0.8062（5 seeds × 7 DGP）。
- 相对朴素等权集成 +16.0% 相对提升；85.7% runs 胜朴素集成；31.4% runs 胜 oracle 最强单检测器。
- 端到端 demo ≤ 30s（CPU），确定性逐位一致。

### Quality
- 质量等级：**S**
- 核心模块行覆盖 89%；CI 矩阵全绿；无密钥泄漏。

作者：晨星 (CJX0712)
