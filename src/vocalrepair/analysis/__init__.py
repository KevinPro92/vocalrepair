"""分析层：把频谱与基频轨迹变成可读、可核验的统计量。

模块分工
--------
- ``spectral``      : 谱峰/谱谷、谱平坦度/对比度/倾斜、长时平均谱 LTAS
- ``vibrato``       : 颤音速率/深度/范围/规律性
- ``voice_quality`` : HNR、CPP、jitter、shimmer、H1-H2
- ``notes``         : 音符事件表（逐音统计）

所有指标只描述"测量到了什么"，不给出总分或评分——艺术质量无法用单一
数字量化，这一条是项目约定。
"""

from __future__ import annotations

from .notes import NoteEvent, segment_notes
from .spectral import frame_spectral_stats, ltas, peak_valley_summary
from .vibrato import VibratoResult, VibratoSegment, analyze_vibrato
from .voice_quality import (
    cpp_db,
    h1_h2_db,
    hnr_db,
    jitter_percent,
    shimmer_percent,
    voice_quality_summary,
)

__all__ = [
    "NoteEvent",
    "VibratoResult",
    "VibratoSegment",
    "analyze_vibrato",
    "cpp_db",
    "frame_spectral_stats",
    "h1_h2_db",
    "hnr_db",
    "jitter_percent",
    "ltas",
    "peak_valley_summary",
    "segment_notes",
    "shimmer_percent",
    "voice_quality_summary",
]
