"""调性估计：色度向量与 Krumhansl-Kessler 音调轮廓的相关系数。

方法
----
把整段音频的 chroma 按时间平均，得到 12 维"音级能量分布"；再与 24 个候选
（12 个主音 × 大/小调）的 Krumhansl-Kessler 经验轮廓做皮尔逊相关，取最大者。

诚实标注
--------
轮廓数值取自 Krumhansl & Kessler (1982) 的经典实验，是目前多数开源实现的
通用取值；若要写进论文，必须按 BibTeX 单一真源规则核对原文数值后再引用。
本函数只做"报告用"的估计——修音是否使用调内吸附由用户显式指定，不依赖它。
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_key", "KRUMHANSL_MAJOR", "KRUMHANSL_MINOR"]

#: Krumhansl-Kessler 大调轮廓（按 C D E F G A B 的音级顺序 12 维）
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
#: Krumhansl-Kessler 小调轮廓
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_EPS = 1e-12


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    den = float(np.sqrt(np.sum(a0**2) * np.sum(b0**2)))
    return 0.0 if den <= _EPS else float(np.sum(a0 * b0) / den)


def estimate_key(chroma: np.ndarray, min_confidence: float = 0.45) -> dict:
    """由 chroma（12 × 帧数）估计调性。

    返回 {"key": "A minor", "tonic_pc": 9, "mode": "minor",
          "confidence": 0.78, "is_reliable": true}
    """
    c = np.asarray(chroma, dtype=float)
    if c.ndim != 2 or c.shape[0] != 12:
        raise ValueError("chroma 形状必须为 (12, 帧数)")
    profile = c.mean(axis=1)

    best = {"score": -2.0, "tonic_pc": 0, "mode": "major"}
    for tonic in range(12):
        rotated = np.roll(profile, -tonic)
        for mode, ref in (("major", KRUMHANSL_MAJOR), ("minor", KRUMHANSL_MINOR)):
            s = _corr(rotated, ref)
            if s > best["score"]:
                best = {"score": s, "tonic_pc": tonic, "mode": mode}

    return {
        "key": f"{NOTE_NAMES[best['tonic_pc']]} {best['mode']}",
        "tonic_pc": int(best["tonic_pc"]),
        "mode": best["mode"],
        "confidence": round(float(best["score"]), 3),
        "is_reliable": bool(best["score"] >= min_confidence),
        "note": "报告用估计，不参与修音决策；调内吸附需显式指定主音",
    }
