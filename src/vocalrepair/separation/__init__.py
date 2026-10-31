"""分离层：人声 / 伴奏拆分。

当前实现的三种引擎（质量与依赖递增）：

| 引擎 | 依赖 | 质量 | 说明 |
|---|---|---|---|
| ``hpss`` | 无 | 差 | 谐波/打击中值滤波，**没有人声语义** |
| ``center`` | 无 | 中 | 中置声道提取，利用"人声居中"的空间线索，秒级 |
| ``demucs`` | torch + demucs | 好 | htdemucs 神经网络，CPU 上较慢 |

三者都**不区分主唱与和声**——4-stem 模型只输出 vocals/no_vocals。
"""

from __future__ import annotations
