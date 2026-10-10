"""分数归一化工具：把不同检测器异构分数对齐到可比空间。

语义统一为「分数越大越异常」。

作者：晨星
"""

from __future__ import annotations

import numpy as np

__all__ = ["robust_z", "to_ranks", "minmax", "spearman_diversity"]


def robust_z(scores: np.ndarray) -> np.ndarray:
    """稳健 z 分数：中位数为 0，MAD 缩放，避免极端值主导。"""
    s = np.asarray(scores, dtype=np.float64).ravel()
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = np.std(s) + 1e-12
    return (s - med) / (scale + 1e-12)


def to_ranks(scores: np.ndarray) -> np.ndarray:
    """平均秩（ties 取均值）。异常分数越高 → 秩越大。"""
    s = np.asarray(scores, dtype=np.float64).ravel()
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based 平均秩
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def minmax(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel()
    lo, hi = np.min(s), np.max(s)
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def spearman_diversity(rank_mat: np.ndarray) -> np.ndarray:
    """输入 (M, n) 秩矩阵，返回 (M, M) 两两分歧度 = (1 - spearman) / 2。

    spearman 为对秩向量的 pearson 相关，范围 [-1, 1]：
      - 完全一致(同序) → 分歧 0
      - 完全反序(逆转) → 分歧 1（最大分歧）
      - 不相关       → 分歧 0.5
    """
    # 居中秩
    R = rank_mat - rank_mat.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(R**2, axis=1))
    norm = np.where(norm < 1e-12, 1e-12, norm)
    Rn = R / norm[:, None]
    corr = Rn @ Rn.T  # (M, M)
    corr = np.clip(corr, -1.0, 1.0)
    return (1.0 - corr) / 2.0
