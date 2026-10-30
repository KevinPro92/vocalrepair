"""八度错误（octave error）检测与可选修正。

为什么必须处理
--------------
YIN / pYIN / CREPE 在弱能量、气声、低音男声上会把基频判成 2f 或 f/2。
后果不是"差一点"，而是整句被修高或修低一个八度——这是修音里最严重的
错误类型，比跑调严重得多。

判据
----
真实演唱的八度跳进是**突变且有意**的，而检测错误表现为：
"某一段的 f0 恰好在局部中值的 2 倍（或 0.5 倍）附近，且持续若干帧"。

因此用 ±(1/12 半音*7) 容差内的 2 倍/0.5 倍关系 + 最短持续时间作为判据。
默认只**报告**不修正（可解释优先），修正需显式开启。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["OctaveSuspect", "find_octave_suspects", "fix_octave_errors"]

_EPS = 1e-12


@dataclass
class OctaveSuspect:
    start_s: float
    end_s: float
    factor: float        # 2.0 表示疑似高一个八度，0.5 表示低一个八度
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "factor": self.factor,
            "duration_s": round(self.duration_s, 3),
            "hint": "疑似高一个八度" if self.factor > 1 else "疑似低一个八度",
        }


def _local_reference(cents: np.ndarray, window: int) -> np.ndarray:
    """局部中值作为参考轨迹（cents 域，NaN 用 nanmedian）。"""
    k = window + 1 if window % 2 == 0 else window
    k = max(3, k)
    half = k // 2
    padded = np.pad(cents, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


def find_octave_suspects(
    f0_hz: np.ndarray,
    times: np.ndarray,
    window: int = 11,
    tol_cents: float = 60.0,
    min_duration: float = 0.06,
) -> list[OctaveSuspect]:
    """找出"相对局部中值恰差一个八度"的连续段。

    tol_cents = 60 意味着允许 ±60 cents 的偏差（真实八度跳进不完全精确，
    且颤音本身有 ±40 cents 的摆动）。
    """
    f0 = np.asarray(f0_hz, dtype=float)
    t = np.asarray(times, dtype=float)
    if f0.size != t.size or f0.size < 5:
        return []
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01

    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(np.maximum(f0, _EPS) / 440.0)
    cents = np.where(np.isfinite(f0) & (f0 > 0), cents, np.nan)
    ref = _local_reference(cents, window)

    diff = cents - ref
    is_up = np.isfinite(diff) & (np.abs(diff - 1200.0) <= tol_cents)
    is_down = np.isfinite(diff) & (np.abs(diff + 1200.0) <= tol_cents)

    out: list[OctaveSuspect] = []
    for flag, factor in ((is_up, 2.0), (is_down, 0.5)):
        i = 0
        while i < flag.size:
            if not flag[i]:
                i += 1
                continue
            j = i
            while j < flag.size and flag[j]:
                j += 1
            if (j - i) * dt >= min_duration:
                out.append(OctaveSuspect(float(t[i]), float(t[j - 1]), factor, float((j - i) * dt)))
            i = j
    out.sort(key=lambda s: s.start_s)
    return out


def fix_octave_errors(
    f0_hz: np.ndarray,
    times: np.ndarray,
    suspects: list[OctaveSuspect] | None = None,
    **kwargs,
) -> np.ndarray:
    """把疑似八度段的 f0 除以对应倍数，返回修正后的副本。

    只在明确开启时才调用；真实演唱的八度跳进也会被这段代码"修掉"，
    所以它是可选步骤而不是默认行为。
    """
    f0 = np.array(f0_hz, dtype=float, copy=True)
    t = np.asarray(times, dtype=float)
    if suspects is None:
        suspects = find_octave_suspects(f0, t, **kwargs)
    for s in suspects:
        sel = (t >= s.start_s) & (t <= s.end_s)
        f0[sel] = f0[sel] / s.factor
    return f0
