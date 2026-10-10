# AnoForge 架构设计

> 作者：晨星 (CJX0712)

## 1. 设计原则

1. **无监督**：推断与融合阶段绝不触碰标签；标签仅用于离线评测（benchmark）。
2. **确定性**：所有随机性由 `core.seed.set_all(seed)` 一次设齐（numpy / random），保证复现。
3. **单向无环**：`cli → pipeline → {data, detectors, ensemble, eval} → core`，无反向依赖。
4. **接口语义统一**：分数越大越异常（higher = more anomalous），跨模块公平评测。
5. **离线优先**：无 PyOD / HF 权重亦可运行（纯 numpy 兜底检测器）。
6. **不伪造数字**：基准所有指标来自真实运行输出，禁止手填。

## 2. 分层职责

| 层 | 职责 | 关键契约 |
|----|------|---------|
| `core` | 类型 / 错误码 / 配置 / 接口 / 种子 | `Dataset`, `DetectionResult`, `Metric`, `AnomalyDetector`(Protocol), `set_all` |
| `data` | 合成数据 + 加载 | `make_suite`, `make_dataset`, `load_csv/npy/embedded` |
| `detectors` | 检测器实现 + 池构建 | `BaseDetector`, `REGISTRY`, `build_pool`, `list_available` |
| `ensemble` | 归一化 / 选择 / 融合 | `robust_z`, `to_ranks`, `spearman_diversity`, `AnoFuse` |
| `eval` | 指标 / 基准 | `roc_auc`, `pr_auc`, `precision_at_k`, `benchmark` |
| `pipeline` | 编排 | `AnomalyPipeline.run / benchmark` |
| `cli` | 入口 | `demo` / `run` / `list` |

## 3. AnoFuse 融合算法

输入：各检测器输出 `scores_i`（越大越异常），形状 `(n,)`。

```
1. 稳健 z 归一化：z_i = robust_z(scores_i)
   z_i = (s_i - median(s_i)) / (1.4826 * MAD(s_i))   # 保留「越大越异常」符号
2. 融合（默认 max 模式）：
   final = max_i z_i
   语义：异常 = 偏离正常流形的任一模式 → 任一检测器报警即抬高融合分（OR 共识）
3. 选择代替（select 模式，供消融）：
   贪心边际多样性选 k 个后等权平均（spearman 分歧度 (1-spearman)/2）
```

**为何 max 优于 mean**：mean 会被弱/错误检测器线性稀释，且对「自信但错误」的离群检测器无免疫；max 取各点的最强异常证据，等价于「任一称职检测器认定异常即判异常」，与异常检测的本质定义一致。基准实测 max 相对 mean（=朴素集成）+16.0%。

## 4. 确定性保证

- `set_all(seed)` 在 pipeline 入口与各 benchmark 轮次首行调用。
- 检测器（IsolationForest / CBLOF / KNN / LODA 等）均传入 `random_state=seed`；纯 numpy 兜底检测器本身确定性。
- 数据生成器使用 `np.random.default_rng((seed + hash(tag)) & 0xFFFFFFFF)`，与全局 numpy 状态解耦，避免跨模块随机流纠缠。
- 验证：同 seed 两次运行 `benchmark.json` 核心指标逐位一致（除 `elapsed_sec`）。

## 5. 故障降级（无泄漏、不崩）

- 单检测器在某 DGP 上拟合失败（如 CBLOF 无法聚类）→ `benchmark` 逐数据集 `try/except` 跳过该检测器并计入 `skipped`，不污染融合、不伪造、不中断整体。
- PyOD / sklearn 缺失 → `available` 属性返回 False，池自动缩减；纯 numpy 兜底检测器始终可用。

## 6. 评测严谨性

- ≥3 seeds（默认 5）报 mean±std。
- 强基线 = 朴素等权集成（NaiveAvg）；oracle 最强单检测器（best_single）仅作上界对照，不计入基线。
- 消融：AnoFuse(max) vs AnoFuse(mean) vs AnoFuse(select) 三方对照。
- 失败案例：取 AnoFuse 明显落后 oracle 最强单检测器的轮次（gap>0.05），附原因归因。
- 性能预算：demo 端到端 ≤ 60s（CPU）；实测 ~30s。

## 7. 可扩展点

- 新增检测器：继承 `BaseDetector`，实现 `fit/score`，`@_register` 即自动入池。
- 新增 DGP：在 `data/generators.py` 的 `DATASET_REGISTRY` 注册，纳入 `make_suite`。
- 新增融合模式：在 `AnoFuse.__init__` 的 `method` 分支扩展。
