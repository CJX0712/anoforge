"""现代检测器（PyOD 后端，Tier-0 顶级开源）。

ADBench 2022 头部方法：COPOD / ECOD 在无监督设定下常优于深度方法；
CBLOF / HBOS / KNN / ABOD / LODA 提供异构视角，增强集成多样性。
PyOD 的 decision_scores_ 已满足「分数越大越异常」语义，直接复用。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector, _register

# PyOD 类名映射（仅在实际可用时导入）
_PYOD_MAP = {
    "COPOD": "pyod.models.copod.COPOD",
    "ECOD": "pyod.models.ecod.ECOD",
    "CBLOF": "pyod.models.cblof.CBLOF",
    "HBOS": "pyod.models.hbos.HBOS",
    "KNN": "pyod.models.knn.KNN",
    "ABOD": "pyod.models.abod.ABOD",
    "LODA": "pyod.models.loda.LODA",
}


def _import(name: str):
    import importlib

    mod, cls = _PYOD_MAP[name].rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def _make(cls_name: str, seed: int):
    Cls = _import(cls_name)
    try:
        return Cls(random_state=seed)
    except TypeError:
        return Cls()


class _PyODWrapper(BaseDetector):
    pyod_name = "COPOD"

    @property
    def available(self) -> bool:
        try:
            _import(self.pyod_name)
            return True
        except Exception:  # noqa: BLE001
            return False

    def fit(self, X, y=None):
        super().fit(X, y)
        self.clf_ = _make(self.pyod_name, self.seed)
        self.clf_.fit(self._X_fit)
        return self

    def score(self, X):
        self._check_fitted()
        return np.asarray(
            self.clf_.decision_function(np.asarray(X, dtype=np.float64)), dtype=np.float64
        )


@_register
class COPODDet(_PyODWrapper):
    name = "COPOD"
    pyod_name = "COPOD"


@_register
class ECODDet(_PyODWrapper):
    name = "ECOD"
    pyod_name = "ECOD"


@_register
class CBLOFDet(_PyODWrapper):
    name = "CBLOF"
    pyod_name = "CBLOF"


@_register
class HBOSDet(_PyODWrapper):
    name = "HBOS"
    pyod_name = "HBOS"


@_register
class KNNDet(_PyODWrapper):
    name = "KNN"
    pyod_name = "KNN"


@_register
class ABODDet(_PyODWrapper):
    name = "ABOD"
    pyod_name = "ABOD"


@_register
class LODADet(_PyODWrapper):
    name = "LODA"
    pyod_name = "LODA"
