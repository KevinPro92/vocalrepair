"""修复层：削波修复、音准修正、WORLD 重合成、音色与动态。

模块分工
--------
- ``declip``        : 削波平台检测 + 三次样条插值（短段）/ 保守衰减（长段）
- ``pitch_correct`` : 音符级音准修正（死区 16 cents、残差 8 cents、保颤音）
- ``world_vocoder`` : WORLD 分解与重合成，保共振峰的变调
- ``postproc``      : 包络 EQ、齿音控制、压缩、软限幅
- ``pipeline``      : 把上述步骤串成一条可复现的管线，并产出前后指标
"""

from __future__ import annotations

from .declip import ClippingReport, detect_clipping, repair_clipping
from .pitch_correct import CorrectionPlan, plan_correction
from .postproc import band_gain_db, compress, enhance_envelope, sibilance_mask, soft_limit
from .world_vocoder import (
    WorldFeatures,
    resynth_transpose,
    shift_formants,
    world_analyze,
    world_synthesize,
)

__all__ = [
    "ClippingReport",
    "CorrectionPlan",
    "WorldFeatures",
    "band_gain_db",
    "compress",
    "detect_clipping",
    "enhance_envelope",
    "plan_correction",
    "repair_clipping",
    "resynth_transpose",
    "sibilance_mask",
    "shift_formants",
    "soft_limit",
    "world_analyze",
    "world_synthesize",
]
