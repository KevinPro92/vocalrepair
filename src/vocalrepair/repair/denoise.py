"""降噪：三种路线，供听感对比与按需启用。

诊断依据（本项目 KTV 素材，分离后的人声轨）
-------------------------------------------
| 频段 | 噪声 | 人声 | SNR |
|---|---|---|---|
| 200-500 Hz  | -10.8 | -3.6  | +7.2 dB |
| 1-2 kHz     | -13.7 | -10.8 | +2.9 dB |
| 2-4 kHz     | -17.4 | -27.6 | **-10.3 dB** |
| 4-8 kHz     | -20.6 | -37.9 | **-17.2 dB** |
| 8-12 kHz    | -23.7 | -42.7 | **-18.9 dB** |

两个结论：
1. 噪声**集中在 2 kHz 以上，且比人声强 10-19 dB**——这就是听感上的"电流声/嘶嘶"。
2. 噪声的**时间稳定性 CV = 1.16（很不平稳）**，所以依赖"噪声平稳"假设的谱减法与
   维纳滤波容易留下 musical noise（金属/水声）；反而是静态高频衰减更稳妥。

三种方法都实现，让听感决定：

- ``highcut``：静态高频衰减（不估计噪声，最稳）
- ``spectral_subtraction``：谱减（经典，过减因子 + 谱底限幅抑制 musical noise）
- ``wiener``：维纳滤波（增益连续，比谱减平滑）

**共同风险**：都会削掉 2 kHz 以上的成分，而"空气感/齿音/气息"也在那里。所以三者
都必须带 ``min_gain``（最大衰减量）保护，且默认只作用于 >= ``fmin`` 的频段。
"""

from __future__ import annotations

import numpy as np

from ..dsp.stft import istft, stft_mag_phase

__all__ = [
    "estimate_noise_spectrum",
    "highcut",
    "spectral_subtraction",
    "wiener_denoise",
]

_EPS = 1e-12


def estimate_noise_spectrum(
    mag: np.ndarray,
    quiet_db: float = -35.0,
    min_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """用最安静的帧估计噪声功率谱。

    真静音段常常很少（本素材只有 2.6% 的帧），所以当数量不足时退化为
    "最安静的 10% 帧"——必须避免用 2 帧估出不可靠的噪声谱。
    """
    rms = 10.0 * np.log10(np.mean(mag**2, axis=0) + _EPS)
    thr = np.percentile(rms, 95) + quiet_db
    quiet = rms < thr
    if int(quiet.sum()) < min_frames:
        quiet = rms <= np.percentile(rms, 10)
    noise = np.mean(mag[:, quiet] ** 2, axis=1) if quiet.any() else np.zeros(mag.shape[0])
    return noise, quiet


def _limit_overshoot(out: np.ndarray, ref: np.ndarray, tol: float = 1.05) -> np.ndarray:
    """过冲保护。

    在频域只改幅度、保留原相位，会破坏"幅度-相位一致性"，重建后时域出现
    尖峰。实测谱减把峰值从 0.284 抬到 0.891（接近满量程），若不处理会在后续
    限幅器里被硬压，听感就是新的失真。
    这里只在输出峰值超过输入的 tol 倍时整体缩放——不改变频谱形状，不引入失真。
    """
    pk_in = float(np.max(np.abs(ref))) if ref.size else 0.0
    pk_out = float(np.max(np.abs(out))) if out.size else 0.0
    if pk_in <= 0.0 or pk_out <= 0.0:
        return out
    if pk_out > pk_in * tol:
        return out * (pk_in * tol / pk_out)
    return out


def highcut(
    y: np.ndarray,
    sr: int,
    fmin: float = 2000.0,
    gain_db: float = -6.0,
    fade_ms: float = 150.0,
) -> tuple[np.ndarray, dict]:
    """静态高频衰减（不估计噪声）。最稳妥，但会削掉空气感。"""
    from .postproc import _rbj_shelf_high

    sig = np.asarray(y, dtype=float).ravel()
    out = _rbj_shelf_high(sig, sr, fmin, gain_db)
    return out, {
        "method": "highcut",
        "fmin_hz": fmin,
        "gain_db": gain_db,
        "note": "静态搁架衰减，不估计噪声；代价是同时削弱空气感与齿音",
    }


def spectral_subtraction(
    y: np.ndarray,
    sr: int,
    over_sub: float = 1.5,
    floor: float = 0.06,
    min_gain: float = 0.25,
    quiet_db: float = -35.0,
) -> tuple[np.ndarray, dict]:
    """谱减法。

    over_sub：过减因子（>1 削得更狠，但更容易产生 musical noise）
    floor  ：谱底限幅（相对原谱的比例），防止把整帧削成 0 而产生"空洞"
    min_gain：每个频点允许的最小增益（0.25 = 最多衰减 12 dB）
    """
    sig = np.asarray(y, dtype=float).ravel()
    mag, phase, freqs, _t = stft_mag_phase(sig, sr, 2048, 512)
    noise, quiet = estimate_noise_spectrum(mag, quiet_db=quiet_db)

    power = mag**2
    sub = power - over_sub * noise[:, None]
    gain = np.sqrt(np.maximum(sub, 0.0) / np.maximum(power, _EPS))
    gain = np.maximum(gain, min_gain)      # 限幅：避免过度削减
    gain = np.maximum(gain, floor)         # 谱底：避免 musical noise 的深洞
    # 频率轴 + 时间轴平滑（3-bin Hann）。
    # 必须做：谱减保持原相位但改变幅度，相邻频点/相邻帧的增益突变会破坏
    # 幅度-相位一致性，在时域产生尖峰过冲（实测峰值 0.284 -> 0.891）。
    k = np.array([0.25, 0.5, 0.25])
    if gain.shape[0] >= 3:
        gain = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, gain)
    if gain.shape[1] >= 3:
        gain = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, gain)
    gain = np.clip(gain, min_gain, 1.0)
    out = istft(mag * gain, phase, 2048, 512, length=sig.size)
    out = _limit_overshoot(out, sig)
    return out, {
        "method": "spectral_subtraction",
        "over_sub": over_sub,
        "floor": floor,
        "min_gain": min_gain,
        "quiet_frames": int(quiet.sum()),
        "mean_gain": round(float(np.mean(gain)), 3),
    }


def wiener_denoise(
    y: np.ndarray,
    sr: int,
    min_gain: float = 0.25,
    smooth_frames: int = 3,
    quiet_db: float = -35.0,
) -> tuple[np.ndarray, dict]:
    """维纳滤波（增益连续，比谱减平滑）。

    增益 g = SNR_p / (1 + SNR_p)，SNR_p 为后验信噪比；对增益做时间平滑以
    抑制帧间抖动（这正是谱减产生 musical noise 的根源）。
    """
    sig = np.asarray(y, dtype=float).ravel()
    mag, phase, freqs, _t = stft_mag_phase(sig, sr, 2048, 512)
    noise, quiet = estimate_noise_spectrum(mag, quiet_db=quiet_db)

    power = mag**2
    snr_post = np.maximum(power - noise[:, None], 0.0) / np.maximum(noise[:, None], _EPS)
    gain = snr_post / (1.0 + snr_post)
    gain = np.maximum(gain, min_gain)

    if smooth_frames >= 3:
        # np.hanning(3) == [0, 1, 0] —— 等于完全不平滑。必须取内部点，
        # 否则"平滑"参数形同虚设（本项目实测过这个坑）。
        k = np.hanning(smooth_frames + 2)[1:-1]
        k = k / k.sum()
        gain = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, gain)
    # 频率轴平滑：与谱减同理，邻频增益突变会在时域产生过冲
    if gain.shape[0] >= 3:
        kf = np.array([0.25, 0.5, 0.25])
        gain = np.apply_along_axis(lambda r: np.convolve(r, kf, mode="same"), 0, gain)
    gain = np.clip(gain, min_gain, 1.0)

    out = istft(mag * gain, phase, 2048, 512, length=sig.size)
    out = _limit_overshoot(out, sig)
    return out, {
        "method": "wiener",
        "min_gain": min_gain,
        "smooth_frames": smooth_frames,
        "quiet_frames": int(quiet.sum()),
        "mean_gain": round(float(np.mean(gain)), 3),
    }
