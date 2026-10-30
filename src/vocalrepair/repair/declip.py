"""削波（破音）检测与修复。

物理边界（必须先说清）
----------------------
削波 = 波形超过满量程被硬性截断。被截掉的那部分**信息真的丢了**，任何修复
都是"根据上下文估计"，不是"还原"。所以：

- 短平台（<= 10 ms）：用三次样条从两侧未削波样本插值，能显著改善听感
- 长平台（> 10 ms）：只做保守处理（衰减到安全电平）并**标记为严重削波**，
  不假装恢复。报告里会如实记录段数与总时长

为什么不用"整体限幅"糊弄过去：那是把问题藏起来，波形上仍然有平台，
频谱上仍然有大量奇次谐波（3f, 5f, 7f...），听起来依然"脏"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["ClippingReport", "detect_clipping", "repair_clipping"]

_EPS = 1e-12


@dataclass
class ClippingReport:
    """削波统计，直接写进 report.json 的 clipping 字段。"""

    before_samples: int = 0
    after_samples: int = 0
    n_segments: int = 0
    severe_segments: int = 0
    severe_duration_s: float = 0.0
    ceiling: float = 0.98

    def to_dict(self) -> dict:
        return {
            "before_samples": int(self.before_samples),
            "after_samples": int(self.after_samples),
            "n_segments": int(self.n_segments),
            "severe_segments": int(self.severe_segments),
            "severe_duration_s": round(float(self.severe_duration_s), 4),
            "ceiling": float(self.ceiling),
            "reduction_pct": (
                round(100.0 * (1.0 - self.after_samples / self.before_samples), 2)
                if self.before_samples
                else 0.0
            ),
        }


def detect_clipping(x: np.ndarray, ceiling: float = 0.98, min_run: int = 3) -> list[tuple[int, int]]:
    """返回削波平台列表 [(start, end), ...]（左闭右开）。

    ceiling = 0.98 而不是 0.999：考虑到重采样/编码后的峰值会被轻微改变，
    真正的数字削波在 ±0.98 以上通常已经形成平台。
    """
    sig = np.abs(np.asarray(x, dtype=float))
    flag = sig >= ceiling
    if flag.size < min_run or not flag.any():
        return []

    edges = np.diff(flag.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if flag[0]:
        starts = np.concatenate(([0], starts))
    if flag[-1]:
        ends = np.concatenate((ends, [flag.size]))

    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_run]


def repair_clipping(
    x: np.ndarray,
    sr: int,
    ceiling: float = 0.98,
    max_segment_ms: float = 10.0,
    context: int = 6,
    safe_level: float = 0.95,
) -> tuple[np.ndarray, ClippingReport]:
    """修复削波，返回 (修复后波形, 统计报告)。

    context : 插值时每侧使用的未削波样本数（6 点足够约束三次样条）
    """
    y = np.asarray(x, dtype=float).copy()
    segs = detect_clipping(y, ceiling=ceiling)
    rep = ClippingReport(before_samples=int(np.count_nonzero(np.abs(y) >= ceiling)), ceiling=ceiling)
    rep.n_segments = len(segs)
    if not segs:
        rep.after_samples = rep.before_samples
        return y, rep

    max_len = int(max_segment_ms * 1e-3 * sr)
    n = y.size

    for s, e in segs:
        sign = float(np.sign(np.mean(y[s:e]))) or 1.0
        length = e - s

        if length <= max_len and s >= context and e + context <= n:
            # 短平台：用两侧未削波样本做三次样条插值
            left = np.arange(s - context, s)
            right = np.arange(e, min(e + context, n))
            xs = np.concatenate([left, right]).astype(float)
            ys = np.concatenate([y[left], y[right]])
            try:
                spline = CubicSpline(xs, ys, bc_type="natural")
                y[s:e] = spline(np.arange(s, e, dtype=float))
            except Exception:
                y[s:e] = sign * safe_level
        else:
            # 长平台：只做保守衰减并如实标记
            rep.severe_segments += 1
            rep.severe_duration_s += length / float(sr)
            y[s:e] = sign * safe_level * 0.85

        # 插值可能越界，做一次软约束
        np.clip(y[s:e], -safe_level, safe_level, out=y[s:e])

    rep.after_samples = int(np.count_nonzero(np.abs(y) >= ceiling))
    return y, rep
