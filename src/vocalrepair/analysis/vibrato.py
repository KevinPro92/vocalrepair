"""颤音（vibrato）分析：速率、深度、范围、规律性。

为什么颤音要单独测量而不是只说"保留"
--------------------------------------
自然歌唱颤音是中心音高上的准周期调制：速率 4-7 Hz（美声可到 6.5 Hz）、
深度 20-80 cents。它是"人声是否自然"的核心线索之一，也是教学里最常被
训练的要素。所以除了在修音时保护它，还要能报出：

- rate / Hz        调制频率（对 f0 的 cents 轨迹做 FFT 取峰）
- depth / cents    调制幅度（正弦假设下 depth = RMS * sqrt(2)）
- extent / cents   峰峰值范围（P95 - P5，对非正弦调制更稳健）
- regularity       调制能量的集中度，0-1。越接近 1 越"规整"，偏低说明
                   气息不稳或存在不规则抖动

判定阈值：rate 落在 3-9 Hz、depth >= 10 cents、regularity >= 0.30 才判为颤音；
否则报告为"音高漂移 / 不稳"而不是颤音。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["VibratoSegment", "VibratoResult", "analyze_vibrato"]

_EPS = 1e-12


@dataclass
class VibratoSegment:
    """一段连续浊音上的颤音测量结果。"""

    start_s: float
    end_s: float
    center_hz: float
    rate_hz: float
    depth_cents: float
    extent_cents: float
    regularity: float
    is_vibrato: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "center_hz": round(self.center_hz, 2),
            "rate_hz": round(self.rate_hz, 3),
            "depth_cents": round(self.depth_cents, 2),
            "extent_cents": round(self.extent_cents, 2),
            "regularity": round(self.regularity, 3),
            "is_vibrato": bool(self.is_vibrato),
        }


@dataclass
class VibratoResult:
    segments: list[VibratoSegment] = field(default_factory=list)

    @property
    def vibrato_segments(self) -> list[VibratoSegment]:
        return [s for s in self.segments if s.is_vibrato]

    def summary(self) -> dict:
        vb = self.vibrato_segments
        if not vb:
            return {
                "n_segments": len(self.segments),
                "n_vibrato": 0,
                "median_rate_hz": None,
                "median_depth_cents": None,
                "median_regularity": None,
                "note": "未检测到符合定义的颤音（3-9 Hz 且深度 >= 10 cents）",
            }
        return {
            "n_segments": len(self.segments),
            "n_vibrato": len(vb),
            "median_rate_hz": round(float(np.median([s.rate_hz for s in vb])), 3),
            "median_depth_cents": round(float(np.median([s.depth_cents for s in vb])), 2),
            "median_regularity": round(float(np.median([s.regularity for s in vb])), 3),
        }


def _segments_of(voiced: np.ndarray) -> list[tuple[int, int]]:
    """把布尔序列切成连续 True，每段必须连续。"""
    out: list[tuple[int, int]] = []
    n = voiced.size
    i = 0
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def analyze_vibrato(
    f0_hz: np.ndarray,
    times: np.ndarray,
    min_duration: float = 0.40,
    rate_range: tuple[float, float] = (3.0, 9.0),
    min_depth_cents: float = 10.0,
    min_regularity: float = 0.30,
) -> VibratoResult:
    """在 f0 轨迹上逐段分析颤音。

    min_duration 设为 0.40 s 是频率分辨率的硬约束：段长 T 上 FFT 的频率
    分辨率为 1/T，要分辨 3 Hz 与 9 Hz 之间的峰至少需要 0.4 s（分辨率 2.5 Hz）。
    """
    f0 = np.asarray(f0_hz, dtype=float)
    t = np.asarray(times, dtype=float)
    if f0.size < 8 or f0.size != t.size:
        return VibratoResult()

    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.005
    if dt <= 0:
        return VibratoResult()
    fs_v = 1.0 / dt

    cents = np.where(np.isfinite(f0) & (f0 > 0), 1200.0 * np.log2(np.maximum(f0, _EPS) / 440.0), np.nan)
    voiced = np.isfinite(cents)

    result = VibratoResult()
    for i0, i1 in _segments_of(voiced):
        dur = (i1 - i0) * dt
        if dur < min_duration:
            continue
        seg = cents[i0:i1]
        center = float(np.median(seg))

        # 去趋势：减去线性回归，避免"整句上滑"被误当成慢颤音
        x = np.arange(seg.size, dtype=float)
        slope, intercept = np.polyfit(x, seg, 1)
        mod = seg - (slope * x + intercept)
        mod = mod - mod.mean()

        # 调制谱：加 Hann 窗减少泄漏
        win = np.hanning(mod.size)
        spec = np.abs(np.fft.rfft(mod * win))
        f_axis = np.fft.rfftfreq(mod.size, d=dt)
        band = (f_axis >= rate_range[0]) & (f_axis <= rate_range[1])
        if not band.any() or spec.max() <= _EPS:
            result.segments.append(
                VibratoSegment(t[i0], t[i1 - 1], float(np.median(f0[i0:i1])), 0.0, 0.0, 0.0, 0.0, False)
            )
            continue

        band_idx = np.flatnonzero(band)
        k = band_idx[int(np.argmax(spec[band]))]
        # 抛物线插值细化频率
        rate = float(f_axis[k])
        if 0 < k < spec.size - 1:
            y0, y1, y2 = np.log(max(spec[k - 1], _EPS)), np.log(max(spec[k], _EPS)), np.log(max(spec[k + 1], _EPS))
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-12:
                rate += 0.5 * (y0 - y2) / denom * (f_axis[1] - f_axis[0])

        depth = float(np.sqrt(2.0) * np.std(mod))
        extent = float(np.percentile(mod, 95) - np.percentile(mod, 5))
        total = float(np.sum(spec**2)) + _EPS
        regularity = float(np.sum(spec[band] ** 2) / total)

        is_vib = (
            rate_range[0] <= rate <= rate_range[1]
            and depth >= min_depth_cents
            and regularity >= min_regularity
        )
        result.segments.append(
            VibratoSegment(
                start_s=float(t[i0]),
                end_s=float(t[i1 - 1]),
                center_hz=float(np.median(f0[i0:i1])),
                rate_hz=rate,
                depth_cents=depth,
                extent_cents=extent,
                regularity=regularity,
                is_vibrato=bool(is_vib),
            )
        )
    return result
