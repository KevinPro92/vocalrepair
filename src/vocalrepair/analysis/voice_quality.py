"""嗓音质量客观指标：HNR、CPP、jitter、shimmer、H1-H2。

这几个量是嗓音医学与声乐教学的标准描述符（Praat Voice Report 同类），
用途是"客观地说明问题"，而不是打分：

- **HNR**（谐波噪声比，dB）：周期成分与噪声成分的能量比。越低越沙哑/漏气
- **CPP**（倒谱峰突出度，dB）：倒谱在基音周期处的峰高出回归线多少。
  它是信噪比与周期性的联合度量，被认为是稳健的嗓音质量单一指标
- **jitter**（%，周期扰动）：相邻周期长度的相对变化，反映声带振动稳定性
- **shimmer**（%，幅度扰动）：相邻周期幅度的相对变化
- **H1-H2**（dB）：第一与第二谐波的能量差。差值越大越偏"气声/虚声"

诚实标注（重要）
----------------
本项目从**帧级** f0 轨迹估计 jitter / shimmer，而 Praat / pyreaper 用
**逐周期标记**。帧级结果的绝对值不能与 Praat 数值直接比较，只能用于
同一音频内部的前后对比（例如"修音前后 jitter 是否变差"）。这是能力边界。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hnr_db",
    "cpp_db",
    "jitter_percent",
    "shimmer_percent",
    "h1_h2_db",
    "voice_quality_summary",
]

_EPS = 1e-12


def _frames(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (x.size - frame_len) // hop)
    if n < 1:
        return np.zeros((1, frame_len))
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def hnr_db(
    x: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    frame_len: int = 1764,
    hop: int = 441,
) -> np.ndarray:
    """逐帧 HNR（自相关法），返回 dB 数组；非浊音帧为 NaN。

    原理：归一化自相关 r(tau) 在基音周期处的高度反映"周期性成分占比"。
    若信号 = 周期成分 + 白噪声，则 r_max = P_s/(P_s + P_n)，于是
        HNR = 10*log10(r_max / (1 - r_max))
    """
    sig = np.asarray(x, dtype=float).ravel()
    fr = _frames(sig, frame_len, hop)
    n = min(fr.shape[0], f0_hz.size)
    out = np.full(n, np.nan)

    for i in range(n):
        f0 = float(f0_hz[i])
        if not np.isfinite(f0) or f0 <= 0:
            continue
        seg = fr[i] - fr[i].mean()
        energy = float(np.dot(seg, seg))
        if energy <= _EPS:
            continue
        period = sr / f0
        lo = max(2, int(np.floor(0.9 * period)))
        hi = min(seg.size - 2, int(np.ceil(1.1 * period)))
        if hi <= lo:
            continue
        tau = np.arange(lo, hi + 1)
        r = np.array([np.dot(seg[:-t], seg[t:]) for t in tau]) / energy
        r_max = float(np.clip(np.max(r), 0.0, 0.999999))
        out[i] = 10.0 * np.log10(max(r_max, _EPS) / max(1.0 - r_max, _EPS))
    return out


def cpp_db(
    x: np.ndarray,
    sr: int,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    frame_len: int = 1764,
    hop: int = 441,
    voiced: np.ndarray | None = None,
) -> np.ndarray:
    """逐帧 CPP（倒谱峰突出度，dB），返回 dB 数组。

    步骤：加 Hann 窗 → 取 ln|X| → 实倒谱 c(q) → 在 q 对应 [f0_min, f0_max]
    的区间找峰 → 减去该处的线性回归基线。因为输入是对数幅度谱，倒谱单位
    是 neper，乘 20/ln10 换算成 dB。
    """
    sig = np.asarray(x, dtype=float).ravel()
    fr = _frames(sig, frame_len, hop)
    n = fr.shape[0] if voiced is None else min(fr.shape[0], len(voiced))
    win = np.hanning(frame_len)
    out = np.full(n, np.nan)

    q_min = int(np.floor(sr / f0_max))
    q_max = int(np.ceil(sr / f0_min))
    q_max = min(q_max, frame_len // 2 - 2)

    for i in range(n):
        if voiced is not None and not bool(voiced[i]):
            continue
        seg = (fr[i] - fr[i].mean()) * win
        if float(np.dot(seg, seg)) <= _EPS:
            continue
        log_mag = np.log(np.maximum(np.abs(np.fft.rfft(seg, n=frame_len)), _EPS))
        ceps = np.fft.irfft(log_mag, n=frame_len)[: frame_len // 2]
        ceps_db = 20.0 / np.log(10.0) * ceps  # ln -> dB

        seg_q = ceps_db[q_min : q_max + 1]
        if seg_q.size < 3:
            continue
        k = int(np.argmax(seg_q))
        q = np.arange(q_min, q_max + 1, dtype=float)

        # 线性回归基线（Praat 的做法：对整段倒谱做一次线性拟合）
        idx = np.arange(frame_len // 2, dtype=float)
        coeff = np.polyfit(idx, ceps_db, 1)
        baseline = np.polyval(coeff, q[k])
        out[i] = float(ceps_db[q_min + k] - baseline)
    return out


def jitter_percent(f0_hz: np.ndarray) -> float:
    """帧级 jitter（%）：相邻浊音帧周期间隔的相对变化均值。

    只在相邻帧音高变化小于 1 个半音时计入，避免换音/滑音的大跳变被误算。
    """
    f = np.asarray(f0_hz, dtype=float)
    valid = np.isfinite(f) & (f > 0)
    if valid.sum() < 3:
        return float("nan")
    fv = f[valid]
    T = 1.0 / fv
    ratio = np.abs(np.diff(fv)) / fv[:-1]
    keep = ratio < 0.06
    if keep.sum() < 2:
        return float("nan")
    dT = np.abs(np.diff(T))[keep]
    return float(np.mean(dT / np.mean(T)) * 100.0)


def shimmer_percent(rms_db: np.ndarray) -> float:
    """帧级 shimmer（%）：相邻帧 RMS 的相对变化均值。"""
    v = np.asarray(rms_db, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    lin = 10.0 ** (v / 20.0)
    d = np.abs(np.diff(lin))
    return float(np.mean(d / np.maximum(lin[1:], _EPS)) * 100.0)


def h1_h2_db(mag: np.ndarray, freqs: np.ndarray, f0_hz: np.ndarray) -> np.ndarray:
    """逐帧 H1-H2（dB）：在 f0 与 2*f0 附近各取 ±3% 带宽的峰值之差。"""
    m = np.asarray(mag, dtype=float)
    n = min(m.shape[1], f0_hz.size)
    out = np.full(n, np.nan)
    for i in range(n):
        f0 = float(f0_hz[i])
        if not np.isfinite(f0) or f0 <= 0:
            continue

        def peak_near(target: float) -> float:
            sel = (freqs >= target * 0.97) & (freqs <= target * 1.03)
            return float(np.max(m[sel, i])) if sel.any() else 0.0

        a1 = peak_near(f0)
        a2 = peak_near(2.0 * f0)
        if a1 > _EPS and a2 > _EPS:
            out[i] = 20.0 * np.log10(a1 / a2)
    return out


def voice_quality_summary(
    x: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    mag: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
) -> dict:
    """汇总嗓音质量指标（全部取浊音帧中位数）。"""
    f0 = np.asarray(f0_hz, dtype=float)
    voiced = np.isfinite(f0) & (f0 > 0)
    if voiced.sum() < 5:
        return {"status": "no_voiced_frames"}

    hnr = hnr_db(x, sr, f0_hz)
    cpp = cpp_db(x, sr, voiced=voiced)
    out: dict = {
        "hnr_db": _nanmedian(hnr),
        "cpp_db": _nanmedian(cpp),
        "jitter_percent_framewise": _round_or_none(jitter_percent(f0_hz)),
        "note": "jitter/shimmer 为帧级近似，不可与 Praat 周期级数值直接比较",
    }
    if mag is not None and freqs is not None:
        out["h1_h2_db"] = _nanmedian(h1_h2_db(mag, freqs, f0_hz))
    return out


def _nanmedian(a: np.ndarray) -> float | None:
    v = np.asarray(a, dtype=float)
    v = v[np.isfinite(v)]
    return None if v.size == 0 else round(float(np.median(v)), 2)


def _round_or_none(v: float) -> float | None:
    return None if not np.isfinite(v) else round(float(v), 3)

