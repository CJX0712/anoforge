import numpy as np

from core.seed import set_all
from core.types import DetectionResult
from data import generators
from detectors import build_pool
from ensemble import AnoFuse
from ensemble.combine import robust_z, to_ranks
from ensemble.combine import spearman_diversity
from ensemble.selection import diversity_weights, greedy_diversity


def test_greedy_diversity_picks_orthogonal():
    R = np.array(
        [
            [1.0, 2, 3, 4, 5],
            [1.0, 2, 3, 4, 5],  # 与第一行相同
            [5.0, 4, 3, 2, 1],  # 反序（最大分歧）
        ]
    )
    D = spearman_diversity(R)
    sel = greedy_diversity(D, 2)
    assert len(sel) == 2
    assert 2 in sel  # 反序检测器必被选中
    assert set(sel).issubset({0, 1, 2})


def test_diversity_weights_downweights_flat():
    Z = np.array([[1.0, 2, 3], [3, 2, 1], [2, 2, 2]])  # 第三行零标准差
    R = np.array([[1.0, 2, 3], [3, 2, 1], [1.0, 3, 2]])
    w = diversity_weights(Z, R)
    assert np.isclose(w.sum(), 1.0)
    assert w[2] < w[0]  # 零方差检测器被降权


def test_anofuse_select_runs():
    set_all(0)
    d = generators.make_dataset("gaussian", seed=0, n=200, dim=5, contamination=0.05)
    results = []
    for det in build_pool(None, seed=0, contamination=0.05):
        try:
            det.fit(d.X)
            results.append(DetectionResult(det.name, det.score(d.X)))
        except Exception:
            continue
    af = AnoFuse(method="select", seed=0).fit(results)
    final = af.combine()
    assert final.shape[0] == d.X.shape[0]
    assert af.report()["method"] == "select"


def test_combine_helpers():
    s = np.array([1.0, 2, 3, 4, 5])
    z = robust_z(s)
    assert abs(z.mean()) < 1e-9
    r = to_ranks(np.array([1.0, 2, 2, 3.0]))
    assert r[1] == r[2] == 2.5  # 并列取平均秩
