"""经典检测器（scikit-learn 后端，Tier-0）。

统一把 sklearn 的「分数越大越正常」翻转为「分数越大越异常」。
IsolationForest / LOF / OneClassSVM 均确定性可复现（random_state=seed）。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector, _register


@_register
class IsoForestDet(BaseDetector):
    name = "IsoForest"

    @property
    def available(self) -> bool:
        try:
            import sklearn  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def fit(self, X, y=None):
        from sklearn.ensemble import IsolationForest

        super().fit(X, y)
        self.clf_ = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=1,
        )
        self.clf_.fit(self._X_fit)
        return self

    def score(self, X):
        self._check_fitted()
        return -self.clf_.score_samples(np.asarray(X, dtype=np.float64))


@_register
class LOFDet(BaseDetector):
    name = "LOF"

    @property
    def available(self) -> bool:
        try:
            import sklearn  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def fit(self, X, y=None):
        from sklearn.neighbors import LocalOutlierFactor

        super().fit(X, y)
        self.clf_ = LocalOutlierFactor(
            n_neighbors=20, contamination=self.contamination, novelty=True
        )
        self.clf_.fit(self._X_fit)
        return self

    def score(self, X):
        self._check_fitted()
        return -self.clf_.score_samples(np.asarray(X, dtype=np.float64))


@_register
class OCSVMDet(BaseDetector):
    name = "OCSVM"

    @property
    def available(self) -> bool:
        try:
            import sklearn  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def fit(self, X, y=None):
        from sklearn.svm import OneClassSVM

        super().fit(X, y)
        self.clf_ = OneClassSVM(nu=max(1e-3, min(0.5, self.contamination)), gamma="scale")
        self.clf_.fit(self._X_fit)
        return self

    def score(self, X):
        self._check_fitted()
        return -self.clf_.score_samples(np.asarray(X, dtype=np.float64))
