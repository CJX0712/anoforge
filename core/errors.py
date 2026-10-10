"""错误码与异常层级（E100~E500）。

作者：晨星
"""

from __future__ import annotations


class AnoForgeError(Exception):
    """基类。"""

    code = "E000"
    message = "AnoForge 内部错误"


class ConfigError(AnoForgeError):
    code = "E100"
    message = "配置错误"


class DataError(AnoForgeError):
    code = "E200"
    message = "数据错误（形状/标签/污染率非法）"


class DetectorError(AnoForgeError):
    code = "E300"
    message = "检测器错误（未安装后端 / 拟合失败）"


class EnsembleError(AnoForgeError):
    code = "E400"
    message = "集成错误（池为空 / 融合失败）"


class EvalError(AnoForgeError):
    code = "E500"
    message = "评测错误（标签缺失 / 指标计算失败）"


def fmt(err: Exception) -> str:
    if isinstance(err, AnoForgeError):
        return f"[{err.code}] {err.message}: {err}"
    return f"[E000] {type(err).__name__}: {err}"
