"""评测指标。语义统一：越大越好（ROC-AUC / PR-AUC 越接近 1 越好）。

优先用 scikit-learn（精确），缺失时回退纯 numpy 实现，保证离线可复现。

作者：晨星
"""

from __future__ import annotations

import numpy as np

__all__ = ["roc_auc", "pr_auc", "precision_at_k", "all_metrics"]


def _ranks(scores: np.ndarray) -> np.ndarray:
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
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if np.all(y == y[0]):
        return float("nan")
    ranks = _ranks(scores)
    pos = y == 1
    P = int(pos.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    r_pos = ranks[pos].sum()
    return float((r_pos - P * (P + 1) / 2.0) / (P * N))


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, scores))
    except Exception:  # noqa: BLE001
        order = np.argsort(-scores, kind="mergesort")
        ysorted = y[order]
        P = int((y == 1).sum())
        if P == 0:
            return float("nan")
        tp = np.cumsum(ysorted)
        precision = tp / (np.arange(len(y)) + 1)
        # PR-AUC via trapezoid on recall grid
        recall = tp / P
        ap = 0.0
        prev_r = 0.0
        prev_p = 1.0
        for r, p in zip(recall, precision):
            ap += (r - prev_r) * (p + prev_p) / 2.0
            prev_r, prev_p = r, p
        return float(ap)


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int | None = None) -> float:
    y = np.asarray(y, dtype=int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    P = int((y == 1).sum())
    if k is None:
        k = max(1, P)
    k = min(k, len(y))
    order = np.argsort(-scores, kind="mergesort")[:k]
    return float(y[order].sum() / k)


def all_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    P = int((y == 1).sum())
    return {
        "roc_auc": roc_auc(y, scores),
        "pr_auc": pr_auc(y, scores),
        "precision_at_k": precision_at_k(y, scores, P),
        "k": P,
    }
