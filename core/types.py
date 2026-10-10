"""核心数据类型（dataclass）。

约定：异常分数语义全局统一为「分数越大越异常」(higher = more anomalous)，
保证跨模块公平评测与融合。

作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class Dataset:
    """一个异常检测数据集。

    X: (n_samples, n_features)
    y: (n_samples,) 二值标签，1 = 异常 (anomaly / outlier)，0 = 正常 (inlier)。
    contamination: 异常占比（真实或可假定）。
    name: 数据集标识（用于报告）。
    """

    X: np.ndarray
    y: np.ndarray
    name: str = "unnamed"
    contamination: float = 0.05

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.int64)
        if self.X.ndim != 2:
            raise ValueError(f"Dataset {self.name!r}: X 必须为 2D，得到 {self.X.shape}")
        if self.y.ndim != 1 or self.y.shape[0] != self.X.shape[0]:
            raise ValueError(f"Dataset {self.name!r}: y 必须与 X 同样本数")
        if self.contamination <= 0 or self.contamination >= 1:
            # 从标签推断
            self.contamination = float(np.mean(self.y == 1))


@dataclass
class DetectionResult:
    """单个检测器在数据集上的输出。"""

    name: str
    scores: np.ndarray  # (n_samples,) 分数越大越异常
    fit_time: float = 0.0
    predict_time: float = 0.0

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64).ravel()
        # 数值保护：所有分数有限，避免 inf/nan 污染融合
        if not np.all(np.isfinite(self.scores)):
            med = np.nanmedian(self.scores)
            self.scores = np.where(np.isfinite(self.scores), self.scores, med)


@dataclass
class Metric:
    """评测指标。统一语义：越大越好（ROC-AUC / PR-AUC 越接近 1 越好）。"""

    roc_auc: float
    pr_auc: float  # Average Precision
    precision_at_k: float
    k: int


@dataclass
class BenchmarkRow:
    dataset: str
    n_samples: int
    n_features: int
    contamination: float
    metrics: Dict[str, Metric] = field(default_factory=dict)  # name -> Metric
