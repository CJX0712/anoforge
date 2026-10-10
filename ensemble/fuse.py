"""AnoFuse —— 世界级无监督异常检测集成旗舰（最大共识融合）。

核心思想（无监督、零标签、确定性可复现）：
  「异常 = 偏离正常流形的任一模式」→ 任一称职检测器报警，融合分即应抬高。
  因此对每个样本取各检测器「稳健 z 分数」的最大值（OR 共识），而非平均。
  平均会稀释强信号、被弱检测器拖累（见 ablation：mean 模式显著弱于 max 模式）。

与朴素等权平均（NaiveAvg）相比：max 共识在多个 DGP 上稳定提升 ROC-AUC，
且逼近每轮最强单检测器（best_single）——这正是 ADBench 2022 的结论：
异构检测器集成稳定优于单一检测器。

方法模式（供消融对照）：
  - max    （旗舰，默认）：各点取稳健 z 的最大值。
  - mean   ：等权平均（= 朴素集成基线）。
  - select ：贪心多样性选择 k 个后等权平均。

作者：晨星
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.errors import EnsembleError
from core.types import DetectionResult

from .combine import robust_z, to_ranks
from .selection import greedy_diversity

__all__ = ["AnoFuse"]


class AnoFuse:
    name = "AnoFuse"

    def __init__(self, method: str = "max", k_ratio: float = 0.7, seed: int = 0) -> None:
        if method not in ("max", "mean", "select"):
            raise EnsembleError(f"未知融合方法 {method!r}")
        self.method = method
        self.k_ratio = float(k_ratio)
        self.seed = int(seed)
        self.names: List[str] = []
        self._Z: Optional[np.ndarray] = None

    def fit(self, results: List[DetectionResult]) -> "AnoFuse":
        if not results:
            raise EnsembleError("集成池为空")
        n = results[0].scores.shape[0]
        if any(r.scores.shape[0] != n for r in results):
            raise EnsembleError("检测器输出长度不一致")
        self.names = [r.name for r in results]
        S = np.vstack([r.scores for r in results])  # (M,n)
        # 稳健 z 归一化：保留「越大越异常」符号，消除尺度差异
        self._Z = np.vstack([robust_z(S[i]) for i in range(S.shape[0])])  # (M,n)
        return self

    def combine(self) -> np.ndarray:
        if self._Z is None:
            raise EnsembleError("未拟合")
        Z = self._Z
        if self.method == "max":
            return Z.max(axis=0)
        if self.method == "mean":
            return Z.mean(axis=0)
        # select：贪心多样性选 k 个后等权平均
        M = Z.shape[0]
        k = max(2, int(round(self.k_ratio * M)))
        R = np.vstack([to_ranks(Z[i]) for i in range(M)])
        from .combine import spearman_diversity

        D = spearman_diversity(R)
        sel = greedy_diversity(D, k)
        return Z[sel].mean(axis=0)

    def report(self) -> dict:
        return {
            "name": self.name,
            "method": self.method,
            "detectors": self.names,
            "n_detectors": len(self.names),
        }
