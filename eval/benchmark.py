"""多 seed 基准评测 + 集成对比。

对比对象（均为真实运行输出，禁止手填）：
  - 各单检测器（池成员）
  - NaiveAvg：等权 robust-z 平均（组件基线，用于证成旗舰权重设计）
  - AnoFuse：多样性加权融合（旗舰）
  - best_single：每轮 ROC-AUC 最高的单检测器（强基线）

统计：≥3 seeds 报 mean±std；记录 AnoFuse 胜 best_single / 胜 NaiveAvg 的比例。

作者：晨星
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.seed import set_all
from core.types import Dataset, DetectionResult
from detectors import build_pool
from ensemble import AnoFuse, robust_z

from .metrics import all_metrics

__all__ = ["benchmark"]


def benchmark(
    datasets_factory: Callable[[int], List[Dataset]],
    seeds: List[int],
    pool_names: Optional[List[str]] = None,
    contamination: float = 0.05,
    n_jobs_dummy: int = 1,
    verbose: bool = False,
) -> dict:
    """对多 DGP 套件、多 seed 跑基准，返回聚合 dict（可直接落盘 JSON）。"""
    seeds = list(seeds)
    # 每个检测器名的逐次 roc 序列
    det_roc: Dict[str, List[float]] = {}
    anofuse_roc: List[float] = []
    naive_roc: List[float] = []
    best_single_roc: List[float] = []
    win_vs_best = 0
    win_vs_naive = 0
    n_runs = 0
    skipped: Dict[str, int] = {}
    dataset_names: List[str] = []
    per_run: List[dict] = []

    for seed in seeds:
        set_all(seed)
        dsets = datasets_factory(seed)
        pool = build_pool(pool_names, seed=seed, contamination=contamination)
        if verbose:
            print(f"  seed={seed}: {len(pool)} 检测器, {len(dsets)} 数据集")
        for d in dsets:
            if d.name not in dataset_names:
                dataset_names.append(d.name)
            results: List[DetectionResult] = []
            roc_by_name: Dict[str, float] = {}
            for det in pool:
                try:
                    t0 = time.perf_counter()
                    det.fit(d.X)
                    s = det.score(d.X)
                    dt = time.perf_counter() - t0
                    results.append(DetectionResult(det.name, s, fit_time=dt))
                    m = all_metrics(d.y, s)
                    roc_by_name[det.name] = m["roc_auc"]
                    det_roc.setdefault(det.name, []).append(m["roc_auc"])
                except Exception:  # noqa: BLE001
                    # 单检测器在个别 DGP 上失败（如 CBLOF 无法聚类）
                    # 诚实跳过该检测器在此数据集上的运行，不伪造、不崩整体
                    skipped[det.name] = skipped.get(det.name, 0) + 1
                    continue
            if not results:
                # 该数据集全部检测器失败，跳过
                continue

            # NaiveAvg：等权 robust-z 平均
            Z = np.vstack([robust_z(r.scores) for r in results])
            naive_scores = Z.mean(axis=0)
            naive_m = all_metrics(d.y, naive_scores)
            naive_roc.append(naive_m["roc_auc"])

            # AnoFuse：最大共识融合（旗舰）
            fuse = AnoFuse(method="max", seed=seed).fit(results)
            af_scores = fuse.combine()
            af_m = all_metrics(d.y, af_scores)
            anofuse_roc.append(af_m["roc_auc"])

            # 最强单检测器
            best_name = max(roc_by_name, key=lambda k: roc_by_name[k])
            bs = roc_by_name[best_name]
            best_single_roc.append(bs)

            n_runs += 1
            if af_m["roc_auc"] >= bs:
                win_vs_best += 1
            if af_m["roc_auc"] >= naive_m["roc_auc"]:
                win_vs_naive += 1
            per_run.append(
                {
                    "dataset": d.name,
                    "seed": seed,
                    "roc": roc_by_name,
                    "AnoFuse": af_m["roc_auc"],
                    "NaiveAvg": naive_m["roc_auc"],
                    "best_single": bs,
                    "best_name": best_name,
                }
            )

    def agg(seq: List[float]) -> dict:
        a = np.array([x for x in seq if np.isfinite(x)], dtype=np.float64)
        if a.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(a.mean()), "std": float(a.std())}

    det_summary = {name: agg(v) for name, v in det_roc.items()}
    # 最强单检测器名（按 mean 排序）
    best_name = max(det_summary, key=lambda k: det_summary[k]["mean"])
    return {
        "seeds": seeds,
        "n_datasets": len(set(dataset_names)),
        "datasets": sorted(set(dataset_names)),
        "n_runs": n_runs,
        "detectors": det_summary,
        "AnoFuse": agg(anofuse_roc),
        "NaiveAvg": agg(naive_roc),
        "best_single": {**agg(best_single_roc), "name": best_name},
        "win_vs_best_frac": win_vs_best / max(1, n_runs),
        "win_vs_naive_frac": win_vs_naive / max(1, n_runs),
        "skipped": skipped,
        "per_run": per_run,
    }
