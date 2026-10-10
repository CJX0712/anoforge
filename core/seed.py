"""全局确定性种子。

所有随机性（numpy / random / 部分库级 random_state）由唯一入口 `set_all` 一次设齐，
保证 demo 两次运行核心指标逐位一致（elapsed_sec 除外）。

作者：晨星
"""

from __future__ import annotations

import os
import random
from typing import Optional

__all__ = ["set_all", "get_seed", "_SEED"]

_SEED: int = 20261011  # 默认确定性种子


def set_all(seed: Optional[int] = None) -> int:
    """设全局确定性种子，返回实际生效种子。"""
    global _SEED
    if seed is None:
        seed = _SEED
    # 允许环境变量覆盖（仅当未显式传入）
    env_seed = os.environ.get("ANOFORGE_SEED")
    if seed is None and env_seed is not None:
        seed = int(env_seed)
    _SEED = int(seed)
    try:
        import numpy as np

        np.random.seed(_SEED)
    except Exception:  # noqa: BLE001
        pass
    random.seed(_SEED)
    try:
        import torch  # noqa: F401

        torch.manual_seed(_SEED)
    except Exception:  # noqa: BLE001
        pass
    return _SEED


def get_seed() -> int:
    return _SEED
