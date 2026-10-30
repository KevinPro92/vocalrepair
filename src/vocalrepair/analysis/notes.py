"""音符事件表：把一条 f0 轨迹切成"音符"，逐音给出可读统计。

这是"分析每个音的音高"的正确实现形式。一条连续的 f0 曲线对人不友好；
真正有用的是：

| 起止时间 | 音名 | 实测 Hz | 偏差 cents | 时长 | 颤音 rate/depth | 稳定度 |

稳定度用"去掉颤音调制后残余的 cents 标准差"衡量：它把"音准漂移"与
"有意颤音"分开——这正是修音时决定要不要动这一帧的依据。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..dsp.pitch_scale import hz_to_midi, hz_to_name, midi_to_name, cents_between

__all__ = ["NoteEvent", "segment_notes"]

_EPS = 1e-12


@dataclass
class NoteEvent:
    """一个音符事件。"""

    start_s: float
    end_s: float
    midi: int
    name: str
    mean_hz: float
    cents_dev: float          # 相对最近半音的平均偏差（正 = 偏高）
    stability_cents: float    # 去趋势后残余标准差，越小越稳
    vibrato_rate_hz: float | None
    vibrato_depth_cents: float | None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "midi": int(self.midi),
            "name": self.name,
            "mean_hz": round(self.mean_hz, 2),
            "cents_dev": round(self.cents_dev, 1),
            "stability_cents": round(self.stability_cents, 2),
            "vibrato_rate_hz": None if self.vibrato_rate_hz is None else round(self.vibrato_rate_hz, 2),
            "vibrato_depth_cents": None if self.vibrato_depth_cents is None else round(self.vibrato_depth_cents, 1),
        }


def _split_runs(midi_smooth: np.ndarray, valid: np.ndarray, jump_semitones: float) -> list[tuple[int, int]]:
    """按"平滑音高发生大于阈值的跳变"切段。"""
    runs: list[tuple[int, int]] = []
    n = midi_smooth.size
    start = None
    for i in range(n):
        if not valid[i]:
            if start is not None:
                runs.append((start, i))
                start = None
            continue
        if start is None:
            start = i
            continue
        if abs(midi_smooth[i] - midi_smooth[i - 1]) > jump_semitones:
            runs.append((start, i))
            start = i
    if start is not None:
        runs.append((start, n))
    return runs


def _median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    """保边中值滤波（NaN 用 nanmedian）。

    整窗全是 NaN（静音段）时 nanmedian 会抛 RuntimeWarning，这里静默处理：
    全 NaN 输入返回 NaN 是正确语义，不是异常。
    """
    if k < 3:
        return x
    k = k + 1 if k % 2 == 0 else k
    half = k // 2
    padded = np.pad(x, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


def segment_notes(
    f0_hz: np.ndarray,
    times: np.ndarray,
    min_duration: float = 0.08,
    jump_semitones: float = 0.55,
    smooth_frames: int = 9,
) -> list[NoteEvent]:
    """把 f0 轨迹切成音符事件。

    参数
    ----
    min_duration : 短于此时长的段丢弃（避免把毛刺当成音符）
    jump_semitones : 相邻帧平滑音高跳变超过该值即切段。取 0.55 是折中：
        自然颤音（±40 cents = ±0.4 半音）不会触发，而换音（≥1 半音）一定触发
    smooth_frames : 切段前的中值平滑窗（帧）。约一个颤音周期，防止颤音峰值
        被误判成"换音"
    """
    f0 = np.asarray(f0_hz, dtype=float)
    t = np.asarray(times, dtype=float)
    if f0.size != t.size or f0.size < 4:
        return []

    valid = np.isfinite(f0) & (f0 > 0)
    midi = np.where(valid, hz_to_midi(np.maximum(f0, _EPS)), np.nan)
    midi_smooth = _median_smooth(midi, smooth_frames)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01

    events: list[NoteEvent] = []
    for i0, i1 in _split_runs(midi_smooth, valid, jump_semitones):
        if (i1 - i0) * dt < min_duration:
            continue
        seg_f0 = f0[i0:i1]
        seg_f0 = seg_f0[np.isfinite(seg_f0)]
        if seg_f0.size < 3:
            continue

        mean_hz = float(np.median(seg_f0))
        midi_cont = float(hz_to_midi(mean_hz))
        midi_int = int(round(midi_cont))
        cents_dev = float((midi_cont - midi_int) * 100.0)

        # 稳定度：先去掉线性漂移与颤音（中值），再看残余标准差
        c = 1200.0 * np.log2(seg_f0 / mean_hz)
        resid = c - _median_smooth(c, 7)
        stability = float(np.std(resid)) if resid.size > 3 else 0.0

        # 颤音参数（段内粗估，精细分析见 analysis.vibrato）
        rate = depth = None
        if seg_f0.size >= 20:
            x = np.arange(seg_f0.size, dtype=float)
            mod = c - np.polyval(np.polyfit(x, c, 1), x)
            spec = np.abs(np.fft.rfft(mod * np.hanning(mod.size)))
            fax = np.fft.rfftfreq(mod.size, d=dt)
            band = (fax >= 3.0) & (fax <= 9.0)
            if band.any() and spec.max() > _EPS:
                rate = float(fax[band][int(np.argmax(spec[band]))])
                depth = float(np.sqrt(2.0) * np.std(mod))

        events.append(
            NoteEvent(
                start_s=float(t[i0]),
                end_s=float(t[i1 - 1] + dt),
                midi=midi_int,
                name=midi_to_name(midi_int),
                mean_hz=mean_hz,
                cents_dev=cents_dev,
                stability_cents=stability,
                vibrato_rate_hz=rate,
                vibrato_depth_cents=depth,
            )
        )
    return events
