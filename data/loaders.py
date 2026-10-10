"""数据加载器：支持内置合成套件、本地 CSV/NPY、以及嵌入的小型真实数据集。

离线优先：不依赖任何外部下载。CSV 约定首列可选 label（由 `label_col` 指定），
其余为特征。

作者：晨星
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np

from core.errors import DataError
from core.types import Dataset

__all__ = ["load_csv", "load_npy", "load_embedded", "EMBEDDED_DIR"]


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


EMBEDDED_DIR = os.path.join(_here(), "embedded")


def load_csv(path: str, label_col: Optional[int] = None, contamination: float = 0.05) -> Dataset:
    """加载 CSV。label_col=None 时假设最后一列是标签（1=异常）。"""
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.reader(f):
            if not r or r[0].strip().startswith("#"):
                continue
            rows.append([float(v) for v in r])
    arr = np.array(rows, dtype=np.float64)
    if arr.size == 0:
        raise DataError(f"{path}: 空数据集")
    if label_col is None:
        X = arr[:, :-1]
        y = arr[:, -1].astype(int)
    else:
        cols = [c for c in range(arr.shape[1]) if c != label_col]
        X = arr[:, cols]
        y = arr[:, label_col].astype(int)
    name = os.path.splitext(os.path.basename(path))[0]
    return Dataset(X=X, y=y, name=name, contamination=float(np.mean(y == 1)) or contamination)


def load_npy(path: str, label_path: Optional[str] = None, contamination: float = 0.05) -> Dataset:
    data = np.load(path)
    if label_path is not None:
        y = np.load(label_path).astype(int)
    else:
        y = data[:, -1].astype(int)
        data = data[:, :-1]
    name = os.path.splitext(os.path.basename(path))[0]
    return Dataset(X=data, y=y, name=name, contamination=float(np.mean(y == 1)) or contamination)


def load_embedded(name: str, contamination: float = 0.05) -> Dataset:
    """加载仓库内 embedded/ 下的数据（如有）。"""
    if not os.path.isdir(EMBEDDED_DIR):
        raise DataError("embedded/ 不存在")
    for ext in (".csv", ".npy"):
        p = os.path.join(EMBEDDED_DIR, name + ext)
        if os.path.exists(p):
            if ext == ".csv":
                return load_csv(p, contamination=contamination)
            return load_npy(p, contamination=contamination)
    raise DataError(f"embedded 数据 {name!r} 不存在")
