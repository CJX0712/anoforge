"""检测器包：导入即注册到 REGISTRY。

作者：晨星
"""

from . import base, classical, fallback, modern
from .base import REGISTRY, BaseDetector, build_pool, list_available

__all__ = [
    "base",
    "classical",
    "modern",
    "fallback",
    "BaseDetector",
    "REGISTRY",
    "build_pool",
    "list_available",
]
