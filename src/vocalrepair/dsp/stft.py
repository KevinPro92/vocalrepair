"""频谱基础层：STFT / 逆变换 / 谱峰与谱谷检测 / 对数频率谱图。

物理约定
--------
- 时间分辨率与频率分辨率的乘积有下界（Gabor 极限）：df = sr/N, dt = N/sr。
  44.1 kHz、N = 2048 时 df ≈ 21.5 Hz、dt ≈ 46 ms。这决定了"能看清多细"，
  也再次说明为什么音高不能从 bin 序号读（见 dsp.f0 的说明）。
- 谱峰检测用三点抛物线插值：峰值真实位置落在两 bin 之间时，用对数幅度
  的抛物线顶点偏移量修正，精度从 1 bin 提升到约 0.1 bin。
- 对数频率谱图（log-frequency spectrogram）是对 STFT 幅度谱在频率轴上做
  对数重采样得到的**近似** CQT：视觉上每八度等宽、可直接叠加音高网格，
  但它不是真正的常数 Q 变换（不做变窗长滤波），因此不用于定量测量，
  只用于可视化与 chroma 统计。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

__all__ = [
    "stft_mag_phase",
    "istft",
    "spectral_peaks",
    "spectral_valleys",
    "log_frequency_spectrogram",
    "chroma_from_spectrum",
]

_EPS = 1e-12


def stft_mag_phase(
    x: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """短时傅里叶变换，返回 (幅度谱, 相位谱, 频率轴, 时间轴)。

    幅度谱形状 (频率 bin 数, 帧数)，n_fft 为偶数时频率 bin 数 = n_fft//2 + 1。
    """
    sig = np.asarray(x, dtype=float).ravel()
    if hop_length is None:
        hop_length = n_fft // 4
    win = get_window(window, n_fft, fftbins=True)

    n_frames = 1 + max(0, (sig.size - n_fft) // hop_length)
    if n_frames < 1:
        sig = np.pad(sig, (0, n_fft - sig.size))
        n_frames = 1
    # 尾部补零，保证最后一帧完整
    pad = (n_frames - 1) * hop_length + n_fft - sig.size
    if pad > 0:
        sig = np.pad(sig, (0, pad))

    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = sig[idx] * win[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1).T

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = (np.arange(n_frames) * hop_length + n_fft / 2.0) / float(sr)
    return np.abs(spec), np.angle(spec), freqs, times


def istft(
    mag: np.ndarray,
    phase: np.ndarray,
    n_fft: int,
    hop_length: int,
    length: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """重叠相加逆变换（加权 OLA，窗平方归一化，保证幅度不失真）。"""
    spec = mag * np.exp(1j * phase)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1)
    win = get_window(window, n_fft, fftbins=True)

    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * hop_length + n_fft
    out = np.zeros(out_len, dtype=float)
    norm = np.zeros(out_len, dtype=float)
    for i in range(n_frames):
        s = i * hop_length
        out[s : s + n_fft] += frames[i] * win
        norm[s : s + n_fft] += win * win
    out = np.divide(out, np.maximum(norm, 1e-8))
    if length is not None:
        out = out[:length] if out.size >= length else np.pad(out, (0, length - out.size))
    return out


def spectral_peaks(
    mag_frame: np.ndarray,
    freqs: np.ndarray,
    n_peaks: int = 8,
    fmin: float = 50.0,
    fmax: float = 12000.0,
    min_db_below_max: float = 40.0,
) -> list[tuple[float, float]]:
    """在单帧幅度谱上检测谱峰，返回 [(频率/Hz, 幅度), ...]，按幅度降序。

    抛物线插值：设峰在 bin k，取对数幅度 y(k-1), y(k), y(k+1)，顶点偏移
        delta = 0.5*(y(k-1) - y(k+1)) / (y(k-1) - 2*y(k) + y(k+1))
    则真实频率 = freqs[k] + delta * (freqs[1] - freqs[0])。
    """
    m = np.asarray(mag_frame, dtype=float)
    band = (freqs >= fmin) & (freqs <= fmax) & (freqs < 0.5 * 0 + np.inf)
    if not band.any():
        return []
    lo = int(np.argmax(band))
    hi = int(len(freqs) - 1 - np.argmax(band[::-1]))
    if hi - lo < 3:
        return []

    seg = m[lo : hi + 1]
    ref = float(np.max(seg)) if seg.size else 0.0
    if ref <= _EPS:
        return []
    is_peak = np.zeros(seg.size, dtype=bool)
    is_peak[1:-1] = (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
    if not is_peak.any():
        return []

    df = float(freqs[1] - freqs[0])
    order = np.argsort(seg[is_peak])[::-1]
    peak_positions = np.flatnonzero(is_peak)
    out: list[tuple[float, float]] = []
    for p in peak_positions[order][:n_peaks]:
        amp = float(seg[p])
        if 20.0 * np.log10(amp / ref) < -min_db_below_max:
            continue
        # 用对数幅度做抛物线插值
        y0, y1, y2 = (
            np.log(max(seg[p - 1], _EPS)),
            np.log(max(seg[p], _EPS)),
            np.log(max(seg[p + 1], _EPS)),
        )
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        out.append((float(freqs[lo + p] + delta * df), amp))
    return out


def spectral_valleys(
    mag_frame: np.ndarray,
    freqs: np.ndarray,
    n_valleys: int = 6,
    fmin: float = 50.0,
    fmax: float = 12000.0,
) -> list[tuple[float, float]]:
    """谱谷检测：峰之间的局部极小，代表"频谱空洞"（浑浊或抵消的位置）。"""
    m = np.asarray(mag_frame, dtype=float)
    band = (freqs >= fmin) & (freqs <= fmax)
    lo = int(np.argmax(band))
    hi = int(len(freqs) - 1 - np.argmax(band[::-1]))
    if hi - lo < 3:
        return []
    seg = m[lo : hi + 1]
    is_valley = np.zeros(seg.size, dtype=bool)
    is_valley[1:-1] = (seg[1:-1] < seg[:-2]) & (seg[1:-1] <= seg[2:])
    if not is_valley.any():
        return []
    positions = np.flatnonzero(is_valley)
    out = [(float(freqs[lo + p]), float(seg[p])) for p in positions]
    out.sort(key=lambda t: t[1])
    return out[:n_valleys]


def log_frequency_spectrogram(
    mag: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 55.0,
    fmax: float = 8000.0,
    bins_per_octave: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """对 STFT 幅度谱做对数频率重采样（近似 CQT），返回 (幅度, 频率轴)。

    为什么需要它：线性频率轴上，一个八度在低频只占几百 Hz、在高频占几千 Hz，
    叠加音高网格时会挤成一团。对数轴上每八度等宽，才能直接画出十二平均律
    的横线（A3/A4/A5 等距）。

    实现：在 dB 域对相邻两 bin 做线性插值。这是**近似** CQT（不做变窗长
    滤波），因此只用于可视化与 chroma 统计，不用于定量频率测量。
    """
    db = 20.0 * np.log10(np.maximum(mag, _EPS))
    n_octaves = float(np.log2(fmax / fmin))
    n_bins = max(12, int(round(n_octaves * bins_per_octave)))
    log_freqs = fmin * np.power(2.0, np.arange(n_bins) / float(bins_per_octave))

    idx = np.searchsorted(freqs, log_freqs)
    idx = np.clip(idx, 1, len(freqs) - 1)
    f_lo = freqs[idx - 1]
    f_hi = freqs[idx]
    w = np.divide(
        log_freqs - f_lo,
        np.maximum(f_hi - f_lo, _EPS),
        out=np.zeros(n_bins),
        where=(f_hi - f_lo) > 0,
    )
    out = (1.0 - w)[:, None] * db[idx - 1, :] + w[:, None] * db[idx, :]
    return out, log_freqs


def chroma_from_spectrum(
    mag: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 55.0,
    fmax: float = 5000.0,
) -> np.ndarray:
    """12 维色度（chroma）向量，形状 (12, 帧数)。

    把每个频率 bin 的能量按"最接近的十二平均律音级"累加（A4 = 440 Hz 为基准）。
    色度对八度不敏感，因此适合估计调性、画音准热力图。
    """
    m = np.asarray(mag, dtype=float)
    mask = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if not mask.any():
        return np.zeros((12, m.shape[1]))
    f = freqs[mask]
    # 连续音高号相对 A4 的半音距离，四舍五入到最近半音，再取模 12
    pc = np.mod(np.round(12.0 * np.log2(f / 440.0)).astype(int), 12)
    sub = m[mask, :]
    chroma = np.zeros((12, m.shape[1]), dtype=float)
    for k in range(12):
        sel = pc == k
        if sel.any():
            chroma[k, :] = sub[sel, :].sum(axis=0)
    total = chroma.sum(axis=0, keepdims=True)
    return np.divide(chroma, np.maximum(total, _EPS))

