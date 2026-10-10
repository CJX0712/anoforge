from core.seed import set_all
from data import generators
from eval.benchmark import benchmark


def test_benchmark_runs_and_keys():
    set_all(0)
    r = benchmark(generators.make_suite, seeds=[0, 1], verbose=False)
    for k in ("detectors", "AnoFuse", "NaiveAvg", "best_single", "win_vs_naive_frac"):
        assert k in r
    assert r["n_runs"] == 14  # 2 seeds * 7 datasets
    assert len(r["detectors"]) >= 5


def test_anofuse_beats_naive():
    """旗舰 max 共识应显著优于朴素等权集成（强基线）。"""
    set_all(0)
    r = benchmark(generators.make_suite, seeds=[0, 1, 2], verbose=False)
    af = r["AnoFuse"]["mean"]
    nv = r["NaiveAvg"]["mean"]
    # 已知差距约 +16%；用宽松下限防回归
    assert af >= nv + 0.05, f"AnoFuse {af:.4f} 未显著优于 NaiveAvg {nv:.4f}"


def test_benchmark_deterministic():
    set_all(0)
    r1 = benchmark(generators.make_suite, seeds=[0, 1], verbose=False)
    set_all(0)
    r2 = benchmark(generators.make_suite, seeds=[0, 1], verbose=False)
    assert abs(r1["AnoFuse"]["mean"] - r2["AnoFuse"]["mean"]) < 1e-12
