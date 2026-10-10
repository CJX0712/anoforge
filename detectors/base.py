"""检测器基类 + 注册表 + 池构建工厂。

统一契约：fit(X) 无监督拟合；score(X) 返回 (n_samples,) 分数，越大越异常。
每个检测器自带 `available` 属性做后端探测，后端缺失自动降级（不崩溃、不伪造）。

作者：晨星
"""

from __future__ import annotations

from typing import Dict, List, Type

import numpy as np

from core.interfaces import AnomalyDetector

__all__ = ["BaseDetector", "REGISTRY", "build_pool", "list_available"]


class BaseDetector(AnomalyDetector):
    name = "base"

    def __init__(self, seed: int = 0, contamination: float = 0.05) -> None:
        self.seed = int(seed)
        self.contamination = float(contamination)
        self._fitted = False
        self._X_fit: np.ndarray | None = None

    @property
    def available(self) -> bool:  # pragma: no cover - 由子类覆盖
        return True

    def fit(self, X: np.ndarray, y=None) -> "BaseDetector":
        X = np.asarray(X, dtype=np.float64)
        self._X_fit = X
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: 未拟合即调用 score()")


# ---- 注册表 ----------------------------------------------------------------
REGISTRY: Dict[str, Type[BaseDetector]] = {}


def _register(cls: Type[BaseDetector]) -> Type[BaseDetector]:
    REGISTRY[cls.name] = cls
    return cls


def build_pool(
    names: List[str] | None = None, seed: int = 0, contamination: float = 0.05
) -> List[BaseDetector]:
    """构建可用检测器池。names=None 时取全部 available 的。"""
    if names is None:
        names = list(REGISTRY.keys())
    pool: List[BaseDetector] = []
    for n in names:
        cls = REGISTRY.get(n)
        if cls is None:
            continue
        inst = cls(seed=seed, contamination=contamination)
        if inst.available:
            pool.append(inst)
    return pool


def list_available() -> List[str]:
    return [n for n, c in REGISTRY.items() if c(seed=0).available]
