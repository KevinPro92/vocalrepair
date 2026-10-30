"""频谱统计：谱峰/谱谷、谱平坦度、谱对比度、谱倾斜、长时平均谱（LTAS）。

这些量都是"把一帧频谱压缩成几个数"，用于诊断而非修复：

- 谱质心 (centroid)：能量的"重心频率"，粗略对应明亮度
- 谱平坦度 (flatness)：几何均值 / 算术均值。接近 1 = 噪声样（气声、齿音），
  接近 0 = 纯谐波样（浊音）。这是区分"气声"与"实声"最直接的指标
- 谱滚降 (rolloff)：85% 能量所在频率以下，粗略对应"高频含量"
- 谱倾斜 (slope)：对数幅度对对数频率的回归斜率，单位 dB/oct。人声越"亮"
  斜率越平（接近 -6 dB/oct），越"闷"越陡（-12 dB/oct 以下）
- 谱对比度 (contrast)：分频带取"峰带均值 − 谷带均值"，衡量谐波结构的清晰度
"""

from __future__ import annotations

import numpy as np

__all__ = ["frame_spectral_stats", "ltas", "peak_valley_summary"]

_EPS = 1e-12


def frame_spectral_stats(
    mag: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 50.0,
    fmax: float = 12000.0,
) -> dict[str, np.ndarray]:
    """逐帧计算频谱标量特征，返回字典（每个值长度为帧数）。"""
    m = np.asarray(mag, dtype=float)
    band = (freqs >= fmin) & (freqs <= fmax)
    sub = m[band, :]
    f = freqs[band]
    if sub.size == 0:
        n = m.shape[1]
        return {k: np.zeros(n) for k in ("centroid", "flatness", "rolloff", "slope", "contrast")}

    energy = np.maximum(sub**2, _EPS)
    total = energy.sum(axis=0) + _EPS

    centroid = (f[:, None] * energy).sum(axis=0) / total

    # 平坦度：对数域的均值指数化 = 几何均值
    log_mean = np.exp(np.mean(np.log(np.maximum(sub, _EPS)), axis=0))
    arith_mean = np.mean(sub, axis=0) + _EPS
    flatness = log_mean / arith_mean

    # 85% 滚降
    cum = np.cumsum(energy, axis=0)
    thr = 0.85 * cum[-1, :]
    rolloff = f[np.argmax(cum >= thr[None, :], axis=0)]

    # 谱倾斜：log10(f) 对 20log10(mag) 的最小二乘斜率，换算成 dB/oct
    lf = np.log10(np.maximum(f, _EPS))
    lf0 = lf - lf.mean()
    db = 20.0 * np.log10(np.maximum(sub, _EPS))
    db0 = db - db.mean(axis=0, keepdims=True)
    num = (lf0[:, None] * db0).sum(axis=0)
    den = float((lf0**2).sum()) + _EPS
    slope_db_per_decade = num / den
    slope_db_per_octave = slope_db_per_decade * np.log10(2.0)

    # 谱对比度：按八度分带，峰带（前 20%）与谷带（后 20%）均值之差
    contrast = _band_contrast(sub, f)

    return {
        "centroid": centroid,
        "flatness": flatness,
        "rolloff": rolloff,
        "slope": slope_db_per_octave,
        "contrast": contrast,
    }


def _band_contrast(sub: np.ndarray, f: np.ndarray, n_bands: int = 6, f0: float = 100.0) -> np.ndarray:
    """按倍频程分带的谱对比度（Jiang 等人的标准定义，取峰/谷 20% 分位）。"""
    n_frames = sub.shape[1]
    out = np.zeros(n_frames)
    for b in range(n_bands):
        lo, hi = f0 * 2.0**b, f0 * 2.0 ** (b + 1)
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            continue
        band_db = 20.0 * np.log10(np.maximum(sub[sel, :], _EPS))
        q_lo = np.percentile(band_db, 20, axis=0)
        q_hi = np.percentile(band_db, 80, axis=0)
        out += q_hi - q_lo
    return out / max(1, n_bands)


def ltas(mag: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """长时平均谱（Long-Term Average Spectrum），返回 (dB, 频率轴)。

    LTAS 是"整首歌的平均音色指纹"：参考对齐的 conservative 模式就是用
    LTAS 相减求 EQ 增益曲线（见 repair.timbre）。
    """
    m = np.asarray(mag, dtype=float)
    avg = np.mean(m**2, axis=1) if m.ndim == 2 else m**2
    db = 10.0 * np.log10(np.maximum(avg, _EPS))
    return db, np.asarray(freqs, dtype=float)


def peak_valley_summary(
    mag: np.ndarray,
    freqs: np.ndarray,
    clip_db: float = -45.0,
    fmin: float = 50.0,
    fmax: float = 12000.0,
) -> dict[str, float | int]:
    """整段音频的谱峰/谱谷统计摘要（用于报告的 "spectrum" 字段）。

    做法：把每帧 dB 谱按 -45 dB 相对峰值截断后累加，得到"平均形状"，再数
    局部极大/极小的个数与峰谷比。峰谷比越大，说明谐波结构越清晰。
    """
    m = np.asarray(mag, dtype=float)
    band = (freqs >= fmin) & (freqs <= fmax)
    sub = m[band, :]
    if sub.size == 0:
        return {"n_peaks": 0, "n_valleys": 0, "peak_to_valley_db": 0.0, "mean_flatness": 0.0}

    db = 20.0 * np.log10(np.maximum(sub, _EPS))
    db = db - db.max(axis=0, keepdims=True)
    shape = np.mean(np.maximum(db, clip_db), axis=1)

    is_peak = np.zeros(shape.size, dtype=bool)
    is_peak[1:-1] = (shape[1:-1] > shape[:-2]) & (shape[1:-1] >= shape[2:])
    is_valley = np.zeros(shape.size, dtype=bool)
    is_valley[1:-1] = (shape[1:-1] < shape[:-2]) & (shape[1:-1] <= shape[2:])

    peak_vals = shape[is_peak]
    valley_vals = shape[is_valley]
    ratio = float(np.mean(peak_vals) - np.mean(valley_vals)) if peak_vals.size and valley_vals.size else 0.0

    stats = frame_spectral_stats(m, freqs, fmin, fmax)
    return {
        "n_peaks": int(is_peak.sum()),
        "n_valleys": int(is_valley.sum()),
        "peak_to_valley_db": round(ratio, 2),
        "mean_flatness": round(float(np.mean(stats["flatness"])), 4),
        "mean_slope_db_per_octave": round(float(np.mean(stats["slope"])), 2),
        "mean_centroid_hz": round(float(np.mean(stats["centroid"])), 1),
    }
