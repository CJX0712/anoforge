"""合成异常数据生成器（DGP 多样化 + 难度旋钮 + 固定 seed 可复现）。

所有生成器返回 Dataset，y: 1=异常 / 0=正常。异常分数语义：越大越异常（仅评测用标签）。

难度旋钮 `separation` 控制异常与正常流形的间距：太近→全崩无区分度，太远→全满分；
默认取甜点区，使基线明显低于天花板，保证 benchmark 有区分度。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from core.types import Dataset

__all__ = ["make_dataset", "make_suite", "DATASET_REGISTRY"]


def _rng(seed: int, tag: str) -> np.random.Generator:
    h = abs(hash(tag)) % (2**31)
    return np.random.default_rng((seed + h) & 0xFFFFFFFF)


def gaussian_outliers(
    seed: int = 0,
    n: int = 1000,
    dim: int = 10,
    contamination: float = 0.05,
    separation: float = 4.0,
    tag: str = "gaussian",
) -> Dataset:
    """多数类为标准正态，异常为远离原点的高斯球（Mahalanobis 型）。"""
    rng = _rng(seed, tag)
    n_anom = max(1, int(n * contamination))
    n_in = n - n_anom
    X_in = rng.standard_normal((n_in, dim))
    # 异常：在随机方向、固定半径处的高斯云
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction) + 1e-9
    X_anom = direction * separation + rng.standard_normal((n_anom, dim)) * 0.3
    X = np.vstack([X_in, X_anom])
    y = np.array([0] * n_in + [1] * n_anom)
    return Dataset(X=X, y=y, name=f"{tag}_d{dim}_s{separation:.1f}", contamination=contamination)


def clustered_outliers(
    seed: int = 0,
    n: int = 1200,
    dim: int = 5,
    contamination: float = 0.06,
    separation: float = 6.0,
    tag: str = "clustered",
) -> Dataset:
    """多个高斯簇作为正常流形，均匀背景点作为异常。"""
    rng = _rng(seed, tag)
    n_anom = max(1, int(n * contamination))
    n_in = n - n_anom
    n_clusters = 4
    centers = rng.standard_normal((n_clusters, dim)) * 3.0
    sizes = rng.multinomial(n_in, [1 / n_clusters] * n_clusters)
    parts = []
    for c, sz in enumerate(sizes):
        if sz == 0:
            continue
        parts.append(centers[c] + rng.standard_normal((sz, dim)) * 0.6)
    X_in = np.vstack(parts)
    # 异常：超立方体外的均匀点
    X_anom = rng.uniform(-separation, separation, (n_anom, dim))
    # 去掉落在簇附近的异常（保持纯净）
    keep = []
    for row in X_anom:
        if np.min(np.linalg.norm(centers - row, axis=1)) > 2.0:
            keep.append(row)
    if len(keep) < n_anom:
        extra = rng.uniform(-separation, separation, (n_anom - len(keep), dim))
        keep = list(keep) + list(extra)
    X_anom = np.array(keep[:n_anom])
    X = np.vstack([X_in, X_anom])
    y = np.array([0] * X_in.shape[0] + [1] * X_anom.shape[0])
    return Dataset(X=X, y=y, name=f"{tag}_d{dim}_s{separation:.1f}", contamination=contamination)


def moons_outliers(
    seed: int = 0,
    n: int = 900,
    contamination: float = 0.05,
    separation: float = 2.5,
    tag: str = "moons",
) -> Dataset:
    """两弯月（正常流形）+ 散落离群点（异常）。2D 可视化友好。"""
    from sklearn.datasets import make_moons

    rng = _rng(seed, tag)
    n_anom = max(1, int(n * contamination))
    n_in = n - n_anom
    X_in, _ = make_moons(n_samples=n_in, noise=0.1, random_state=int(seed % (2**31)))
    X_anom = rng.uniform(-separation, separation, (n_anom, 2))
    keep = [r for r in X_anom if np.min(np.linalg.norm(X_in - r, axis=1)) > 0.8]
    if len(keep) < n_anom:
        keep += list(rng.uniform(-separation, separation, (n_anom - len(keep), 2)))
    X_anom = np.array(keep[:n_anom])
    X = np.vstack([X_in, X_anom])
    y = np.array([0] * n_in + [1] * n_anom)
    return Dataset(X=X, y=y, name=f"{tag}_s{separation:.1f}", contamination=contamination)


def ts_point_anomalies(
    seed: int = 0,
    n: int = 1500,
    contamination: float = 0.04,
    separation: float = 6.0,
    tag: str = "ts",
) -> Dataset:
    """1D 季节性时间序列 + 点异常（幅值突变）与上下文异常（相位偏移）。"""
    rng = _rng(seed, tag)
    t = np.arange(n)
    trend = 0.002 * t
    season = np.sin(2 * np.pi * t / 50.0)
    noise = rng.standard_normal(n) * 0.2
    x = trend + season + noise
    y = np.zeros(n, dtype=int)
    n_anom = max(1, int(n * contamination))
    idx = rng.choice(n, size=n_anom, replace=False)
    for i in idx:
        # 50% 点异常（幅值突变），50% 上下文异常（局部反向）
        if rng.random() < 0.5:
            x[i] += separation * (1 if rng.random() < 0.5 else -1)
        else:
            x[i] = -x[i] + 2 * trend[i]
        y[i] = 1
    return Dataset(
        X=x.reshape(-1, 1), y=y, name=f"{tag}_s{separation:.1f}", contamination=contamination
    )


def correlated_outliers(
    seed: int = 0,
    n: int = 1000,
    dim: int = 8,
    contamination: float = 0.05,
    separation: float = 3.0,
    tag: str = "corr",
) -> Dataset:
    """相关特征（正常服从低秩相关高斯），异常为相关结构破坏的点。"""
    rng = _rng(seed, tag)
    n_anom = max(1, int(n * contamination))
    n_in = n - n_anom
    # 低秩协方差
    A = rng.standard_normal((dim, 3))
    cov = A @ A.T + np.eye(dim) * 0.1
    L = np.linalg.cholesky(cov)
    X_in = rng.standard_normal((n_in, dim)) @ L.T
    # 异常：相关结构破坏（对角缩放 + 偏置）
    X_anom = rng.standard_normal((n_anom, dim)) * separation
    X = np.vstack([X_in, X_anom])
    y = np.array([0] * n_in + [1] * n_anom)
    return Dataset(X=X, y=y, name=f"{tag}_d{dim}_s{separation:.1f}", contamination=contamination)


DATASET_REGISTRY = {
    "gaussian": gaussian_outliers,
    "clustered": clustered_outliers,
    "moons": moons_outliers,
    "ts": ts_point_anomalies,
    "corr": correlated_outliers,
}


def make_dataset(name: str, seed: int = 0, **kwargs) -> Dataset:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"未知数据集 {name!r}，可选: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](seed=seed, tag=name, **kwargs)


def make_suite(seed: int = 0) -> list[Dataset]:
    """多 DGP 基准套件，覆盖不同结构与难度（甜点区）。"""
    return [
        gaussian_outliers(seed=seed, dim=10, contamination=0.05, separation=4.0, tag="gaussian"),
        gaussian_outliers(seed=seed, dim=20, contamination=0.10, separation=3.5, tag="gaussian_hd"),
        clustered_outliers(seed=seed, dim=5, contamination=0.06, separation=6.0, tag="clustered"),
        moons_outliers(seed=seed, contamination=0.05, separation=2.5, tag="moons"),
        ts_point_anomalies(seed=seed, contamination=0.04, separation=6.0, tag="ts"),
        correlated_outliers(seed=seed, dim=8, contamination=0.05, separation=3.0, tag="corr"),
        correlated_outliers(seed=seed, dim=12, contamination=0.08, separation=2.5, tag="corr_hd"),
    ]
