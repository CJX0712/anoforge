"""离线兜底检测器（纯 numpy，零第三方依赖，Tier-1）。

即便 PyOD / sklearn 均不可用，系统仍可运行。同时作为池中的多样性成员。
全部确定性，无任何随机状态。语义：分数越大越异常。

1. RobustZ  —— 每维稳健 z 分数（中位数/MAD）的 L2 范数（参数型 / 全局视角）
2. KNNDist  —— 到 k 近邻的平均距离（密度 / 局部视角）
3. HBOSnp   —— 直方图异常分数（频率 / 全局分箱视角）

作者：晨星
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector, _register


def _robust_scale(X: np.ndarray):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < 1e-12, np.std(X, axis=0) + 1e-12, scale)
    return (X - med) / scale


@_register
class RobustZDet(BaseDetector):
    name = "RobustZ"

    def fit(self, X, y=None):
        super().fit(X, y)
        self._rz = _robust_scale(self._X_fit)
        return self

    def score(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        rz = _robust_scale(X)
        # 稳健马氏式：缩放后 L2 范数
        return np.sqrt(np.sum(rz**2, axis=1))


@_register
class KNNDistDet(BaseDetector):
    name = "KNNDist"
    k = 10

    def fit(self, X, y=None):
        super().fit(X, y)
        return self

    def score(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        from scipy.spatial.distance import cdist

        d = cdist(X, self._X_fit, metric="euclidean")
        # 排除自身（对角线置 inf），取 k 个最小距离均值
        np.fill_diagonal(d, np.inf)
        k = min(self.k, d.shape[1] - 1)
        part = np.partition(d, k, axis=1)[:, :k]
        return np.mean(part, axis=1)


@_register
class HBOSnpDet(BaseDetector):
    name = "HBOSnp"
    n_bins = 12

    def fit(self, X, y=None):
        super().fit(X, y)
        Xf = self._X_fit
        self._edges = []
        self._dens = []
        n = Xf.shape[0]
        for j in range(Xf.shape[1]):
            col = Xf[:, j]
            edges = np.histogram_bin_edges(col, bins=self.n_bins)
            counts, _ = np.histogram(col, bins=edges)
            dens = counts / n
            self._edges.append(edges)
            self._dens.append(dens)
        return self

    def score(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        total = np.zeros(X.shape[0])
        eps = 1e-6
        for j in range(X.shape[1]):
            idx = np.clip(np.digitize(X[:, j], self._edges[j][1:-1]), 0, len(self._dens[j]) - 1)
            dens = self._dens[j][idx]
            total += -np.log(dens + eps)
        return total
