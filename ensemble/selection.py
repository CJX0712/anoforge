"""多样性选择：无监督贪心边际多样性（MMR 风格）。

用于 AnoFuse 的 select 模式与消融实验。不依赖标签。

作者：晨星
"""

from __future__ import annotations

import numpy as np

__all__ = ["greedy_diversity", "diversity_weights"]


def greedy_diversity(D: np.ndarray, k: int) -> list[int]:
    """给定 (M,M) 分歧度矩阵 D（越大越不同），贪心选 k 个互异度最大的检测器。

    首轮选「与全体平均分歧」最大者（信息性代理），其后每轮选与已选集合总分歧最大者。
    """
    M = D.shape[0]
    k = min(k, M)
    if k <= 0:
        return []
    # 首轮：平均分歧最大
    first = int(np.argmax(D.mean(axis=1)))
    selected = [first]
    while len(selected) < k:
        best, best_val = -1, -1.0
        for c in range(M):
            if c in selected:
                continue
            val = float(np.sum(D[c, selected]))
            if val > best_val:
                best_val, best = val, c
        if best < 0:
            break
        selected.append(best)
    return selected


def diversity_weights(z: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """无监督权重：信息性(分数离散度) × 多样性(平均分歧)。

    z: (M,n) robust-z；ranks: (M,n) 秩。返回 (M,) 归一化权重。
    """
    from .combine import spearman_diversity

    M = z.shape[0]
    if M == 1:
        return np.array([1.0])
    conf = z.std(axis=1)  # 分数离散度代理信息性
    conf = np.where(conf < 1e-12, 1e-12, conf)
    D = spearman_diversity(ranks)
    div = D.mean(axis=1)  # 与全体平均分歧
    div = np.where(div < 0, 0.0, div)
    if div.sum() < 1e-12:
        div = np.ones(M)
    w = conf * div
    s = w.sum()
    if s < 1e-12:
        w = np.ones(M)
        s = M
    return w / s
