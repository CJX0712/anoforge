import numpy as np

from core.types import DetectionResult
from ensemble import AnoFuse
from ensemble.combine import robust_z, to_ranks


def _results(n=50, m=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(m):
        s = rng.standard_normal(n) + i
        out.append(DetectionResult(f"det{i}", s))
    return out


def test_fuse_shape():
    res = _results()
    af = AnoFuse(method="max", seed=0).fit(res)
    final = af.combine()
    assert final.shape[0] == 50


def test_fuse_deterministic():
    res = _results(seed=3)
    a = AnoFuse(method="max", seed=3).fit(res).combine()
    b = AnoFuse(method="max", seed=3).fit(res).combine()
    assert np.array_equal(a, b)


def test_max_beats_mean_on_separable():
    """当某检测器清晰分离时，max 共识应强于 mean。"""
    n = 200
    rng = np.random.default_rng(0)
    good = np.concatenate([rng.standard_normal(180), rng.standard_normal(20) + 8.0])  # 明显双模
    bad = rng.standard_normal(n)
    res = [DetectionResult("good", good), DetectionResult("bad", bad)]
    mx = AnoFuse(method="max", seed=0).fit(res).combine()
    mean = AnoFuse(method="mean", seed=0).fit(res).combine()
    # max 在分离处给更高峰值
    assert mx.max() > mean.max()


def test_combine_helpers_finite():
    s = np.array([1.0, 2.0, 3.0, 100.0])
    z = robust_z(s)
    r = to_ranks(s)
    assert np.all(np.isfinite(z))
    assert np.array_equal(r, np.array([1.0, 2.0, 3.0, 4.0]))
