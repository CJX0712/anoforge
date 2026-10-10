"""端到端演示：跑基准 → 落盘 benchmark.json → 确定性校验 → 消融对照 → 失败案例 → 性能预算与质量分级。

使用方法：
  python examples/run_demo.py

产出：
  benchmark.json         —— 真实运行指标（禁止手填）
  demo_report.txt        —— 人读摘要

作者：晨星
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.seed import set_all  # noqa: E402
from pipeline import AnomalyPipeline  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
ABLATION_SEEDS = [0, 1, 2]


def determinism_check(pipe: AnomalyPipeline) -> bool:
    """同 seed 两次跑基准，核心指标须逐位一致（elapsed 除外）。"""
    set_all(0)
    r1 = pipe.benchmark(SEEDS[:3], verbose=False)
    set_all(0)
    r2 = pipe.benchmark(SEEDS[:3], verbose=False)
    for k in ["AnoFuse", "NaiveAvg", "best_single"]:
        if abs(r1[k]["mean"] - r2[k]["mean"]) > 1e-12:
            return False
    for name in r1["detectors"]:
        if abs(r1["detectors"][name]["mean"] - r2["detectors"][name]["mean"]) > 1e-12:
            return False
    return True


def failure_cases(result: dict) -> list:
    """取 AnoFuse 明显落后最强单检测器的典型轮次（含原因）。"""
    cases = []
    for row in result["per_run"]:
        gap = row["best_single"] - row["AnoFuse"]
        if gap > 0.05:
            cases.append(
                {
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "best_name": row["best_name"],
                    "best_roc": round(row["best_single"], 4),
                    "anofuse_roc": round(row["AnoFuse"], 4),
                    "gap": round(gap, 4),
                    "reason": f"{row['best_name']} 该 DGP 上单点远超集成（oracle 上界，非常规基线）",
                }
            )
    cases.sort(key=lambda c: -c["gap"])
    return cases[:5]


def grade(af, naive, mean_single, det_ok, elapsed) -> str:
    t1 = af >= naive * 1.01
    t2 = af >= mean_single
    if t1 and det_ok and elapsed <= 60:
        return "S" if (af >= naive * 1.05) else "A"
    if t2 and det_ok:
        return "B"
    return "C"


def main() -> int:
    # ---- 主基准（旗舰 max）----
    pipe = AnomalyPipeline(seed=0, contamination=0.05, mode="max")
    t0 = time.perf_counter()
    result = pipe.benchmark(SEEDS, verbose=True)
    elapsed = time.perf_counter() - t0

    with open(os.path.join(ROOT, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- 消融（max/mean/select）----
    abl = {}
    for mode in ["max", "mean", "select"]:
        # 直接构造对应 method 的集成分（benchmark 内 AnoFuse 固定 max，这里单独算）
        from core.seed import set_all
        from core.types import DetectionResult
        from detectors import build_pool
        from ensemble import AnoFuse
        from eval.metrics import all_metrics

        rocs = []
        for seed in ABLATION_SEEDS:
            set_all(seed)
            from data import generators

            for d in generators.make_suite(seed):
                results = []
                for det in build_pool(None, seed=seed, contamination=0.05):
                    try:
                        det.fit(d.X)
                        results.append(DetectionResult(det.name, det.score(d.X)))
                    except Exception:
                        continue
                if not results:
                    continue
                af = AnoFuse(method=mode, seed=seed).fit(results)
                rocs.append(all_metrics(d.y, af.combine())["roc_auc"])
        abl[mode] = float(sum(rocs) / len(rocs))

    det_ok = determinism_check(pipe)
    cases = failure_cases(result)

    # 关键汇总
    af = result["AnoFuse"]["mean"]
    naive = result["NaiveAvg"]["mean"]
    bs = result["best_single"]["mean"]
    mean_single = sum(v["mean"] for v in result["detectors"].values()) / len(result["detectors"])
    rel_vs_naive = (af - naive) / max(1e-9, naive) * 100
    rel_vs_single = (af - mean_single) / max(1e-9, mean_single) * 100
    g = grade(af, naive, mean_single, det_ok, elapsed)

    lines = []
    lines.append("=" * 76)
    lines.append(" AnoForge 端到端演示报告")
    lines.append("=" * 76)
    lines.append(
        f" seeds={result['seeds']}  datasets={result['n_datasets']}  runs={result['n_runs']}"
    )
    lines.append(
        f" 端到端耗时(CPU): {elapsed:.2f}s  (预算 ≤60s)  | 确定性: {'PASS' if det_ok else 'FAIL'}"
    )
    lines.append(
        f" 检测器数: {len(result['detectors'])}  | 跳过(单检测器个别 DGP 失败): {result.get('skipped', {})}"
    )
    lines.append("-" * 76)
    lines.append(f" {'方法':<14}{'ROC-AUC mean':>16}{'std':>10}")
    for name, v in sorted(result["detectors"].items(), key=lambda kv: -kv[1]["mean"]):
        lines.append(f" {name:<14}{v['mean']:>16.4f}{v['std']:>10.4f}")
    lines.append("-" * 76)
    lines.append(f" {'NaiveAvg(等权)':<14}{naive:>16.4f}{result['NaiveAvg']['std']:>10.4f}")
    lines.append(f" {'平均单检测器':<14}{mean_single:>16.4f}{' ':>10}")
    lines.append(
        f" {'best_single(oracle)':<14}{bs:>16.4f}{result['best_single']['std']:>10.4f} ({result['best_single']['name']})"
    )
    lines.append(f" {'AnoFuse(max)':<14}{af:>16.4f}{result['AnoFuse']['std']:>10.4f}")
    lines.append("-" * 76)
    lines.append(
        " 消融（ROC-AUC mean, 3 seeds）: "
        + " | ".join(f"{m}={abl[m]:.4f}" for m in ["max", "mean", "select"])
    )
    lines.append(
        f"   → max 融合相对 mean(=朴素): {(abl['max'] - abl['mean']) / abl['mean'] * 100:+.1f}%"
    )
    lines.append("-" * 76)
    lines.append(f" AnoFuse 胜朴素等权集成: {result['win_vs_naive_frac'] * 100:.1f}% runs")
    lines.append(f" AnoFuse 胜 oracle 最强单检测器: {result['win_vs_best_frac'] * 100:.1f}% runs")
    lines.append("-" * 76)
    lines.append(" 门槛判定:")
    lines.append(
        f"   T1 强基线(NaiveAvg)+1%: {'PASS' if af >= naive * 1.01 else 'FAIL'}  (Δ={rel_vs_naive:+.2f}%相对)"
    )
    lines.append(
        f"   T2 胜平均单检测器:      {'PASS' if af >= mean_single else 'FAIL'}  (Δ={rel_vs_single:+.2f}%相对)"
    )
    lines.append(
        f"   性能预算 ≤60s:         {'PASS' if elapsed <= 60 else 'FAIL'}  ({elapsed:.2f}s)"
    )
    lines.append(f"   确定性逐位一致:        {'PASS' if det_ok else 'FAIL'}")
    lines.append(f"   ⇒ 质量等级: {g}")
    if cases:
        lines.append("-" * 76)
        lines.append(" 失败/落后案例（典型，被 oracle 最强单检测器领先）:")
        for c in cases:
            lines.append(
                f"   {c['dataset']} s{c['seed']}: AnoFuse={c['anofuse_roc']} vs {c['best_name']}={c['best_roc']} (gap {c['gap']})"
            )
            lines.append(f"     原因: {c['reason']}")
    lines.append("=" * 76)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(ROOT, "demo_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(
        f"\n[ok] benchmark.json / demo_report.txt 已生成; 等级={g}; 确定性={'PASS' if det_ok else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
