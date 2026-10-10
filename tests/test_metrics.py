import numpy as np

from eval.metrics import pr_auc, precision_at_k, roc_auc


def test_roc_auc_perfect():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # 越大越异常
    assert abs(roc_auc(y, s) - 1.0) < 1e-9


def test_roc_auc_reversed():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])  # 反转 → 0
    assert roc_auc(y, s) < 0.1


def test_roc_auc_random_about_half():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    s = rng.random(100)
    a = roc_auc(y, s)
    assert 0.35 < a < 0.65


def test_pr_auc_range():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert 0.9 < pr_auc(y, s) <= 1.0


def test_precision_at_k():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # 索引 3-7 为异常（共 5 个）
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6])  # 越高越异常
    # top5 = 索引 {4,5,6,7,3}，全部为异常 → precision = 5/5
    p = precision_at_k(y, s)
    assert abs(p - 1.0) < 1e-9
