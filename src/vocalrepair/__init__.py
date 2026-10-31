"""vocalrepair —— 开源的人声频谱分析与修复工具。

分层结构
--------
- ``dsp``       : 纯数字信号处理基础（音高刻度、基频检测、后续的 PSOLA/相位声码器）
- ``analysis``  : 频谱分析、颤音、嗓音质量、共振峰、伪影检测
- ``separation``: 人声/伴奏分离（DSP 基线 + 可选神经网络）
- ``repair``    : 音准、破音、音色修复
- ``synth``     : 带真值的合成测试信号（用于客观验证）

约定
----
- 音高一律以 A4 = 440 Hz 为基准（见 :mod:`vocalrepair.dsp.pitch_scale`）
- 所有误差用音分（cents）表达，因为人耳对音高是近似对数感知
- 核心功能不依赖 torch；深度学习分离属于可选 extras
"""

from __future__ import annotations

__version__ = "0.5.0"

from .dsp.f0 import F0Result, yin_f0
from .dsp.pitch_scale import (
    A4_HZ,
    cents_between,
    cents_deviation,
    hz_to_cents,
    hz_to_midi,
    hz_to_name,
    midi_to_hz,
    midi_to_name,
    note_name_to_midi,
    snap_to_scale,
    snap_to_semitone,
)
from .synth import SynthSignal, make_note_sequence, synth_vowel

__all__ = [
    "__version__",
    "A4_HZ",
    "F0Result",
    "SynthSignal",
    "cents_between",
    "cents_deviation",
    "hz_to_cents",
    "hz_to_midi",
    "hz_to_name",
    "make_note_sequence",
    "midi_to_hz",
    "midi_to_name",
    "note_name_to_midi",
    "snap_to_scale",
    "snap_to_semitone",
    "synth_vowel",
    "yin_f0",
]
