"""AnoForge 命令行入口。

样例：
  python cli.py demo --seeds 0 1 2 3 4 --out benchmark.json
  python cli.py run  --csv data.csv --topk 20
  python cli.py list

作者：晨星
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 允许以脚本方式直接运行（python cli.py），同时兼容包内导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config  # noqa: E402
from data import generators, loaders  # noqa: E402
from detectors import list_available  # noqa: E402
from pipeline import AnomalyPipeline  # noqa: E402


def cmd_demo(args) -> int:
    cfg = Config(seed=args.seed0, contamination=args.contamination)
    pipe = AnomalyPipeline(
        seed=cfg.seed, contamination=cfg.contamination, pool_names=args.pool, mode=args.mode
    )
    seeds = list(range(args.seed0, args.seed0 + args.nseeds))
    result = pipe.benchmark(seeds, verbose=True)
    out = args.out or "benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _print_summary(result)
    print(f"\n[ok] 已落盘 {out}")
    return 0


def cmd_run(args) -> int:
    cfg = Config(seed=args.seed, contamination=args.contamination)
    if args.csv:
        ds = loaders.load_csv(args.csv, contamination=cfg.contamination)
    else:
        ds = generators.make_suite(seed=cfg.seed)[0]
    pipe = AnomalyPipeline(
        seed=cfg.seed, contamination=cfg.contamination, pool_names=args.pool, mode=args.mode
    )
    scores, results, fuse = pipe.run(ds.X)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[: args.topk]
    print(
        f"数据集 {ds.name}: {ds.X.shape[0]} 样本 / {ds.X.shape[1]} 维 / 污染率 {ds.contamination:.2%}"
    )
    print(f"AnoFuse 模式: {args.mode}  参与检测器: {[r.name for r in results]}")
    print(f"Top-{args.topk} 疑似异常索引: {order}")
    m = __import__("eval.metrics", fromlist=["all_metrics"]).all_metrics(ds.y, scores)
    print(f"若以标签评测 ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
    return 0


def cmd_list(_args) -> int:
    print("可用检测器：")
    for n in list_available():
        print(f"  - {n}")
    return 0


def _print_summary(result: dict) -> None:
    print("\n" + "=" * 72)
    print(" AnoForge 基准摘要")
    print("=" * 72)
    print(f" seeds={result['seeds']}  datasets={result['n_datasets']}  runs={result['n_runs']}")
    print(f" {'方法':<12}{'ROC-AUC mean':>16}{'std':>10}")
    print("-" * 72)
    for name, v in sorted(result["detectors"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f" {name:<12}{v['mean']:>16.4f}{v['std']:>10.4f}")
    print("-" * 72)
    print(f" {'NaiveAvg':<12}{result['NaiveAvg']['mean']:>16.4f}{result['NaiveAvg']['std']:>10.4f}")
    print(
        f" {'best_single':<12}{result['best_single']['mean']:>16.4f}{result['best_single']['std']:>10.4f} ({result['best_single']['name']})"
    )
    print(f" {'AnoFuse':<12}{result['AnoFuse']['mean']:>16.4f}{result['AnoFuse']['std']:>10.4f}")
    print("-" * 72)
    print(f" AnoFuse 胜单检测器最强: {result['win_vs_best_frac'] * 100:.1f}% runs")
    print(f" AnoFuse 胜朴素等权集成: {result['win_vs_naive_frac'] * 100:.1f}% runs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anoforge", description="AnoForge 无监督异常检测系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="跑基准并落盘 benchmark.json")
    d.add_argument("--seed0", type=int, default=0)
    d.add_argument("--nseeds", type=int, default=5)
    d.add_argument("--contamination", type=float, default=0.05)
    d.add_argument("--pool", nargs="*", default=None)
    d.add_argument("--mode", choices=["max", "mean", "select"], default="max")
    d.add_argument("--out", default="benchmark.json")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="在数据集上跑 AnoFuse 并给 Top-k 异常")
    r.add_argument("--csv", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--contamination", type=float, default=0.05)
    r.add_argument("--pool", nargs="*", default=None)
    r.add_argument("--mode", choices=["max", "mean", "select"], default="max")
    r.add_argument("--topk", type=int, default=20)
    r.set_defaults(func=cmd_run)

    lp = sub.add_parser("list", help="列出可用检测器")
    lp.set_defaults(func=cmd_list)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
