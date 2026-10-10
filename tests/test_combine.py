import numpy as np

from ensemble.combine import minmax, robust_z, spearman_diversity, to_ranks


def test_robust_z_zero_mean():
    z = robust_z(np.array([1.0, 2, 3, 4, 5]))
    assert abs(z.mean()) < 1e-9


def test_to_ranks_average_ties():
    r = to_ranks(np.array([1.0, 2, 2, 3.0]))
    assert r[0] == 1.0
    assert r[1] == r[2] == 2.5
    assert r[3] == 4.0


def test_minmax():
    m = minmax(np.array([0.0, 5.0, 10.0]))
    assert m[0] == 0.0 and m[2] == 1.0


def test_spearman_diversity_identical_zero():
    R = np.array([[1.0, 2, 3], [1.0, 2, 3]])
    D = spearman_diversity(R)
    assert abs(D[0, 1]) < 1e-9  # 完全相同 → 分歧 0


def test_spearman_diversity_reversed_one():
    R = np.array([[1.0, 2, 3], [3.0, 2, 1]])
    D = spearman_diversity(R)
    assert abs(D[0, 1] - 1.0) < 1e-9  # 完全反序 → 分歧 1
