"""配置：ENV_ANOFORGE_* 覆盖 + 校验。

作者：晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from .errors import ConfigError


@dataclass
class Config:
    """全局配置。

    可被环境变量 ANOFORGE_* 覆盖（命令行 / 测试时用）。
    """

    seed: int = 20261011
    contamination: float = 0.05
    # 集成池中选用的检测器；空 = 全部可用
    pool: List[str] = field(default_factory=list)
    # 融合时保留的多样性加权窗口
    diversity_weight: float = 1.0
    # 确定性校验开关
    strict_determinism: bool = True
    # 性能预算
    max_fit_sec: float = 30.0

    @staticmethod
    def from_env() -> "Config":
        cfg = Config()
        _int = ("seed",)
        _float = ("contamination", "diversity_weight", "max_fit_sec")
        for key in _int:
            env = os.environ.get(f"ANOFORGE_{key.upper()}")
            if env is not None:
                try:
                    setattr(cfg, key, int(env))
                except ValueError as e:
                    raise ConfigError(f"ANOFORGE_{key.upper()}={env!r} 非整数") from e
        for key in _float:
            env = os.environ.get(f"ANOFORGE_{key.upper()}")
            if env is not None:
                try:
                    setattr(cfg, key, float(env))
                except ValueError as e:
                    raise ConfigError(f"ANOFORGE_{key.upper()}={env!r} 非浮点") from e
        if cfg.contamination <= 0 or cfg.contamination >= 1:
            raise ConfigError(f"contamination 必须在 (0,1)，得到 {cfg.contamination}")
        if cfg.diversity_weight < 0:
            raise ConfigError(f"diversity_weight 必须 >=0，得到 {cfg.diversity_weight}")
        return cfg

    def validate(self) -> None:
        if not (0 < self.contamination < 1):
            raise ConfigError(f"contamination 必须在 (0,1)，得到 {self.contamination}")


def load_config() -> Config:
    try:
        return Config.from_env()
    except ConfigError:
        return Config()
