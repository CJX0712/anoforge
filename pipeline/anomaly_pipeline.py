"""AnomalyPipeline —— 端到端编排（cli → pipeline → {data, detectors, ensemble, eval} → core）。

单向无环：pipeline 调用 detectors / ensemble / eval，不直接反向依赖。

作者：晨星
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.config import Config
from core.seed import set_all
from core.types import DetectionResult
from data import generators
from detectors import build_pool
from ensemble import AnoFuse
from eval.benchmark import benchmark
from eval.metrics import all_metrics


class AnomalyPipeline:
    """无监督异常检测流水线。

    用法：
        pipe = AnomalyPipeline(seed=42)
        scores, results, fuse = pipe.run(X)
        # scores: (n,) 越大越异常；results: 各单检测器输出
        bench = pipe.benchmark(seeds=[0,1,2,3,4])
    """

    def __init__(
        self,
        seed: int = 20261011,
        contamination: float = 0.05,
        pool_names: Optional[List[str]] = None,
        mode: str = "max",
        cfg: Optional[Config] = None,
    ) -> None:
        self.cfg = cfg or Config(seed=seed, contamination=contamination)
        self.seed = int(seed)
        self.contamination = float(contamination)
        self.pool_names = pool_names
        self.mode = mode
        self.fuse_report: Optional[dict] = None

    def build_pool(self):
        set_all(self.seed)
        return build_pool(self.pool_names, seed=self.seed, contamination=self.contamination)

    def run(self, X, y=None) -> tuple[np.ndarray, List[DetectionResult], AnoFuse]:
        X = np.asarray(X, dtype=np.float64)
        set_all(self.seed)
        pool = self.build_pool()
        if not pool:
            raise RuntimeError("检测器池为空（依赖缺失？）")
        results: List[DetectionResult] = []
        for det in pool:
            det.fit(X)
            results.append(DetectionResult(det.name, det.score(X)))
        fuse = AnoFuse(method=self.mode, seed=self.seed).fit(results)
        self.fuse_report = fuse.report()
        return fuse.combine(), results, fuse

    def ranked_anomalies(self, X, top_k: int = 10) -> List[int]:
        scores, _, _ = self.run(X)
        order = np.argsort(-scores)[:top_k]
        return [int(i) for i in order]

    def benchmark(self, seeds: List[int], verbose: bool = False) -> dict:
        return benchmark(
            generators.make_suite,
            seeds=seeds,
            pool_names=self.pool_names,
            contamination=self.contamination,
            verbose=verbose,
        )


def quick_eval(X, y, seed: int = 20261011) -> dict:
    """便捷函数：在给定 X/y 上跑 AnoFuse 并返回指标 + 权重。"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    pipe = AnomalyPipeline(seed=seed)
    scores, results, fuse = pipe.run(X)
    m = all_metrics(y, scores)
    return {
        "metrics": m,
        "fuse_report": pipe.fuse_report,
        "per_detector": {r.name: all_metrics(y, r.scores) for r in results},
    }
