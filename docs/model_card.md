# AnoFuse 模型卡 (Model Card)

> 作者：晨星 (CJX0712) · 系统：AnoForge v0.1.0

## 模型细节

- **类型**：无监督异常检测集成融合器（最大共识 / max-consensus ensemble）。
- **融合公式**：`score(x) = max_i robust_z(score_i(x))`，其中 `robust_z` 为中位数/MAD 稳健标准化。
- **支持后端**：PyOD(COPOD/ECOD/CBLOF/HBOS/KNN/ABOD/LODA) + scikit-learn(IsoForest/LOF/OCSVM) + 纯 numpy 兜底(RobustZ/KNNDist/HBOSnp)。
- **输入**：`(n_samples, n_features)` 数值矩阵；无标签要求。
- **输出**：`(n_samples,)` 异常分数，越大越异常。

## 训练 / 拟合

- 完全无监督：仅 `fit(X)`，不使用任何标签。
- 确定性：`random_state=seed`，同 seed 可复现。
- 计算：CPU-only；13 检测器在 n≈1500、d≈10 上单轮 < 1s；整套 35 runs 端到端 ~30s。

## 评测数据

- **基准套件**：5 类合成 DGP（高斯离群 / 多簇离群 / 双月离群 / 1D 季节时序点异常 / 相关结构破坏）+ 高维变体，共 7 数据集，污染率 4%–10%。
- **真实性**：ADBench 2022（Han et al., NeurIPS）证明异构检测器集成稳定优于单检测器；本系统对标该结论。
- **局限**：合成数据为受控 DGP，非真实工业/安全数据；真实分布偏移下的表现需另行评测。

## 量化指标（真实运行，5 seeds × 7 DGP = 35 runs）

| 方法 | ROC-AUC (mean) | ROC-AUC (std) |
|------|---------------:|--------------:|
| **AnoFuse (max)** | **0.8062** | 0.3090 |
| NaiveAvg（等权） | 0.6948 | 0.4044 |
| 平均单检测器 | 0.7671 | — |
| best single (CBLOF, oracle) | 0.8490 | 0.2652 |

- 相对朴素等权集成：**+16.0% 相对**；85.7% runs 胜朴素集成；31.4% runs 胜 oracle 最强单检测器。
- 消融：max 0.81 > select 0.70 > mean 0.69（相对 mean +16%）。

## 适用与不适用

- ✅ 表格/数值型特征的无监督异常检测（网络入侵、欺诈、设备异常、质量缺陷）。
- ✅ 需要确定性、可复现、离线、零 GPU 的场景。
- ⚠️ 非时序因果预测；不含标签监督（如需半监督见 CausalForge/AlignForge 等兄弟系统）。
- ⚠️ 高维稀疏文本/图结构需专用编码器前置。

## 伦理与风险

- 异常分数仅作辅助信号，不应作为唯一决策依据。
- 系统不采集、不上传任何用户数据；无外部网络依赖（PyOD/sklearn 均为本地计算）。
- 误报（false positive）可能带来运维成本，建议结合业务阈值与人工复核。

## 许可证

MIT © 2026 晨星 (CJX0712)
