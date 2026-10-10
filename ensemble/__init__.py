"""集成包。

作者：晨星
"""

from . import combine, fuse, selection
from .combine import minmax, robust_z, spearman_diversity, to_ranks
from .fuse import AnoFuse
from .selection import diversity_weights, greedy_diversity

__all__ = [
    "combine",
    "fuse",
    "selection",
    "AnoFuse",
    "robust_z",
    "to_ranks",
    "minmax",
    "spearman_diversity",
    "diversity_weights",
    "greedy_diversity",
]
