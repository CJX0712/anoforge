from core.seed import set_all
from data import generators
from eval.metrics import all_metrics
from pipeline import AnomalyPipeline, quick_eval


def test_pipeline_run_shape():
    set_all(0)
    d = generators.make_dataset("gaussian", seed=0, n=300, dim=6, contamination=0.05)
    pipe = AnomalyPipeline(seed=0)
    scores, results, fuse = pipe.run(d.X)
    assert scores.shape[0] == d.X.shape[0]
    assert len(results) >= 5
    assert pipe.fuse_report is not None


def test_pipeline_high_separation_auc():
    """清晰分离数据上 AnoFuse 应达高 AUC（语义级保证）。"""
    set_all(0)
    d = generators.make_dataset("gaussian", seed=0, n=400, dim=5, contamination=0.1, separation=6.0)
    pipe = AnomalyPipeline(seed=0)
    scores, _, _ = pipe.run(d.X)
    m = all_metrics(d.y, scores)
    assert m["roc_auc"] > 0.9


def test_quick_eval_returns_metrics():
    set_all(1)
    d = generators.make_dataset("moons", seed=1)
    out = quick_eval(d.X, d.y, seed=1)
    assert "metrics" in out
    assert "fuse_report" in out
    assert out["metrics"]["roc_auc"] > 0.5
