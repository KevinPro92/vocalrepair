"""共振峰追踪：LPC 极点法（Praat 的标准做法）。

为什么不用"谱上找峰"
--------------------
谱上的峰既有共振峰、也有谐波。当基频变化时（例如变调后），谐波位置移动，
"带内最大值"会被邻近谐波抢走，导致同样的物理声道测出不同的"共振峰"。
本项目的实测就踩过这个坑：+3 半音时带内峰值法报出 +36.7% 的假偏移，
而 +5 半音只有 -10%，明显不自洽。

LPC 的做法是把每帧建模成全极点滤波器
    H(z) = 1 / (1 + a1 z^-1 + ... + ap z^-p)
其极点（求根）对应声道的共振模式，与激励（谐波）无关，因此不受 f0 影响。
极点 z = r*exp(j*theta) 给出
    频率 F = theta/(2*pi) * sr
    带宽 B = -sr/pi * ln(r)
带宽过小（< 50 Hz）或过大的极点通常是伪峰，需要过滤。
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["lpc_coefficients", "formants_from_lpc", "track_formants"]

_EPS = 1e-12


def lpc_coefficients(frame: np.ndarray, order: int) -> np.ndarray:
    """自相关法 + Levinson-Durbin 递推求 LPC 系数 a（a[0] = 1）。

    Levinson-Durbin 递推：
        k_i = -(r[i] + sum_{j=1}^{i-1} a_j * r[i-j]) / E_{i-1}
        a_j^(i) = a_j^(i-1) + k_i * a_{i-j}^(i-1),  a_i^(i) = k_i
        E_i = E_{i-1} * (1 - k_i^2)
    """
    x = np.asarray(frame, dtype=float).ravel()
    x = x - x.mean()
    n = x.size
    if n <= order + 1:
        raise ValueError("帧长必须大于 LPC 阶数")

    r = np.correlate(x, x, mode="full")[n - 1 : n + order]
    r[0] = r[0] * (1.0 + 1e-9) + _EPS  # 加一点白噪声底，避免奇异

    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    for i in range(1, order + 1):
        acc = r[i] + float(np.dot(a[1:i], r[i - 1 : 0 : -1])) if i > 1 else r[i]
        k = -acc / err
        if not np.isfinite(k):
            break
        a_prev = a.copy()
        a[1:i] = a_prev[1:i] + k * a_prev[i - 1 : 0 : -1]
        a[i] = k
        err *= 1.0 - k * k
        if err <= _EPS:
            break
    return a


def formants_from_lpc(
    a: np.ndarray,
    sr: int,
    n_formants: int = 3,
    min_freq: float = 90.0,
    max_freq: float = 5500.0,
    min_bandwidth: float = 20.0,
    max_bandwidth: float = 900.0,
) -> np.ndarray:
    """由 LPC 系数求共振峰频率，返回长度 n_formants 的数组（缺失处为 NaN）。

    带宽阈值的取法（实测依据）：F3 这类高频共振峰在包络上可以很窄
    （实测 /a/ 的 F3 极点 |z| = 0.9967 → 带宽仅 46 Hz），若把下限设成 50 Hz
    会把它整条丢掉；而 > 900 Hz 的宽带极点通常是两个共振峰融合后的伪峰
    （实测 2026 Hz 处出现 985 Hz 带宽的极点）。
    """
    coeffs = np.asarray(a, dtype=float)
    if coeffs.size < 3 or abs(coeffs[0]) < _EPS:
        return np.full(n_formants, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        roots = np.roots(coeffs)

    # 只保留单位圆内、正频率的复根
    pos = roots[np.imag(roots) > 0]
    if pos.size == 0:
        return np.full(n_formants, np.nan)
    freqs = np.angle(pos) * sr / (2.0 * np.pi)
    bws = -sr / np.pi * np.log(np.maximum(np.abs(pos), _EPS))

    keep = (freqs >= min_freq) & (freqs <= max_freq) & (bws >= min_bandwidth) & (bws <= max_bandwidth)
    freqs, bws = freqs[keep], bws[keep]
    if freqs.size == 0:
        return np.full(n_formants, np.nan)

    order = np.argsort(freqs)
    freqs = freqs[order][:n_formants]
    out = np.full(n_formants, np.nan)
    out[: freqs.size] = freqs
    return out


def track_formants(
    x: np.ndarray,
    sr: int,
    voiced: np.ndarray | None = None,
    frame_len: int = 1102,
    hop: int = 441,
    order: int = 18,
    preemphasis: float = 0.97,
    n_formants: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """逐帧共振峰轨迹，返回 (freqs[n_frames, n_formants], times[n_frames])。

    默认 order = 18（大约 sr/1000 + 2 的常用经验值，44.1 kHz 下能覆盖
    约 4-5 个共振峰）；frame_len 25 ms 是"能分辨 F1/F2 与相邻谐波"与
    "帧内声道近似不变"之间的折中。
    """
    sig = np.asarray(x, dtype=float).ravel()
    sig = np.append(sig[0], sig[1:] - preemphasis * sig[:-1]) if sig.size > 1 else sig
    n_frames = 1 + max(0, (sig.size - frame_len) // hop)
    out = np.full((max(1, n_frames), n_formants), np.nan)
    times = np.zeros(max(1, n_frames))
    win = np.hanning(frame_len)

    for i in range(n_frames):
        s = i * hop
        times[i] = (s + frame_len / 2.0) / float(sr)
        if voiced is not None and i < len(voiced) and not bool(voiced[i]):
            continue
        seg = sig[s : s + frame_len] * win
        if float(np.dot(seg, seg)) <= _EPS:
            continue
        try:
            a = lpc_coefficients(seg, order)
            out[i] = formants_from_lpc(a, sr, n_formants=n_formants)
        except ValueError:
            continue
    return out, times
