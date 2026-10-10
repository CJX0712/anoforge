"""数据层：合成生成器 + 加载器。

作者：晨星
"""

from . import generators, loaders
from .generators import DATASET_REGISTRY, make_dataset, make_suite
from .loaders import EMBEDDED_DIR, load_csv, load_embedded, load_npy

__all__ = [
    "generators",
    "loaders",
    "DATASET_REGISTRY",
    "make_dataset",
    "make_suite",
    "load_csv",
    "load_npy",
    "load_embedded",
    "EMBEDDED_DIR",
]
