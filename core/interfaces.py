"""接口契约（Protocol）。

统一语义：score 越大 = 越异常 (higher = more anomalous)。

作者：晨星
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyDetector(Protocol):
    """异常检测器契约。"""

    name: str

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AnomalyDetector":
        """在 X 上拟合（无监督，y 忽略）。"""
        ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """返回 (n_samples,) 异常分数，越大越异常。"""
        ...

    @property
    def available(self) -> bool:
        """后端是否可用（未安装依赖时为 False）。"""
        ...


@runtime_checkable
class Ensemble(Protocol):
    """集成器契约。"""

    name: str

    def fit(self, results: list) -> "Ensemble": ...

    def combine(self) -> np.ndarray:
        """返回融合后的 (n_samples,) 分数，越大越异常。"""
        ...
