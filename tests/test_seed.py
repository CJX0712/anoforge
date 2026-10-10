import numpy as np

from core.seed import get_seed, set_all


def test_set_all_deterministic():
    set_all(7)
    a = np.random.rand(5)
    set_all(7)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
    assert get_seed() == 7


def test_set_all_changes_seed():
    set_all(1)
    x = np.random.rand(5)
    set_all(2)
    y = np.random.rand(5)
    # 极小概率完全相同
    assert not np.array_equal(x, y)
