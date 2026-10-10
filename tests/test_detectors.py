import numpy as np

from core.seed import set_all
from core.types import Dataset
from detectors import build_pool


def _obvious_outlier_dataset():
    rng = np.random.default_rng(0)
    X_in = rng.standard_normal((200, 5))
    outlier = np.array([[100.0, -100.0, 100.0, -100.0, 100.0]])
    X = np.vstack([X_in, outlier])
    y = np.array([0] * 200 + [1])
    return Dataset(X=X, y=y, name="obvious", contamination=1 / 201)


def test_pool_nonempty():
    pool = build_pool(None, seed=0)
    assert len(pool) >= 5


def test_all_available_detectors_run():
    pool = build_pool(None, seed=0)
    d = _obvious_outlier_dataset()
    for det in pool:
        det.fit(d.X)
        s = det.score(d.X)
        assert s.shape[0] == d.X.shape[0]
        assert np.all(np.isfinite(s))


def test_fuse_flags_obvious_outlier():
    """语义校验（系统级）：明显离群点应被 AnoFuse 排到最高异常分。"""
    from core.types import DetectionResult
    from ensemble import AnoFuse

    set_all(0)
    d = _obvious_outlier_dataset()
    pool = build_pool(None, seed=0, contamination=d.contamination)
    results = []
    for det in pool:
        try:
            det.fit(d.X)
            results.append(DetectionResult(det.name, det.score(d.X)))
        except Exception:
            continue
    af = AnoFuse(method="max", seed=0).fit(results)
    s = af.combine()
    order = np.argsort(-s)
    assert 200 in order[:3]


def test_offline_fallback_detectors():
    """纯 numpy 兜底检测器在无 PyOD/sklearn 之外也可独立运行。"""
    from detectors.fallback import HBOSnpDet, KNNDistDet, RobustZDet

    set_all(0)
    d = _obvious_outlier_dataset()
    for cls in (RobustZDet, KNNDistDet, HBOSnpDet):
        det = cls(seed=0, contamination=d.contamination)
        det.fit(d.X)
        s = det.score(d.X)
        assert np.all(np.isfinite(s))
        assert int(np.argmax(s)) == 200
