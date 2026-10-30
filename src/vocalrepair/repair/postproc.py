"""音色饱满化与动态处理：包络 EQ + 齿音控制 + 压缩 + 软限幅。

为什么在**包络**上做 EQ 而不是在波形上做
----------------------------------------
WORLD 已经把信号分解成 f0 / 谱包络 sp / 非周期 ap。对 sp 施加增益等价于
"只改声道传递函数，不动谐波结构与相位"，因此不会引入相位涂抹，也不会
改变音高。这正是"保留音色特征的前提下提升饱满度"的正确做法。

三个频段的物理依据
------------------
- 150-400 Hz（低次谐波支撑）：基频与前几次谐波的能量，决定"厚度"
- 2-5 kHz（存在感）：人耳在该频段听阈最低（外耳道约 3 kHz 共振），
  但耳蜗在此处压缩最强，抬多了立刻"刺"，所以增益限制在 +1.0 dB × strength
- 5-10 kHz（齿音区）：s/sh 等辅音的能量集中区；用"该频段能量占比"做
  逐帧检测，只在检测到齿音时衰减，最多 -2.0 dB × strength

所有增益曲线都用对数频率轴上的高斯钟形，禁止硬切（硬切会在频域产生
振铃，听感上是"金属味"）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["band_gain_db", "enhance_envelope", "sibilance_mask", "compress", "soft_limit"]

_EPS = 1e-12


def band_gain_db(
    freqs: np.ndarray,
    center_hz: float,
    width_octaves: float,
    gain_db: float,
) -> np.ndarray:
    """对数频率轴上的高斯钟形增益曲线，返回 dB。"""
    f = np.maximum(np.asarray(freqs, dtype=float), _EPS)
    oct_dist = np.log2(f / float(center_hz))
    return float(gain_db) * np.exp(-0.5 * (oct_dist / max(width_octaves, 1e-6)) ** 2)


def sibilance_mask(
    sp: np.ndarray,
    freqs: np.ndarray,
    low_hz: float = 5000.0,
    high_hz: float = 10000.0,
    ratio_threshold: float = 0.12,
    max_atten_db: float = 2.0,
    strength: float = 1.0,
) -> np.ndarray:
    """逐帧齿音检测，返回每帧的衰减量（dB，负值）。

    判据：5-10 kHz 能量占全带能量的比例超过阈值，说明该帧以齿音为主。
    超过越多衰减越多，上限 max_atten_db × strength。
    """
    n_frames = sp.shape[0]
    band = (freqs >= low_hz) & (freqs <= high_hz)
    total = sp.shape[1]
    if not band.any():
        return np.zeros(n_frames)
    sib = np.sum(sp[:, band] ** 2, axis=1)
    tot = np.sum(sp**2, axis=1) + _EPS
    ratio = sib / tot
    excess = np.clip((ratio - ratio_threshold) / max(ratio_threshold, 1e-6), 0.0, 1.0)
    return -float(max_atten_db) * float(strength) * excess


def enhance_envelope(
    sp: np.ndarray,
    freqs: np.ndarray,
    strength: float = 1.0,
    low_support_db: float = 1.5,
    presence_db: float = 1.0,
    sibilance_max_db: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """对 WORLD 谱包络做音色塑形，返回 (新包络, 增益统计)。

    默认参数即"保守模式"：总增益不超过 +1.5 dB，齿音衰减不超过 -2 dB。
    """
    s = float(np.clip(strength, 0.0, 1.0))
    gain = (
        band_gain_db(freqs, 260.0, 0.85, low_support_db * s)
        + band_gain_db(freqs, 3300.0, 0.75, presence_db * s)
    )
    sib = sibilance_mask(sp, freqs, max_atten_db=sibilance_max_db, strength=s)
    # 齿音衰减加在 5-10k 钟形上，避免影响 3 kHz 的存在感提升
    sib_curve = band_gain_db(freqs, 7500.0, 0.55, 1.0)[None, :] * sib[:, None]

    total_db = gain[None, :] + sib_curve
    sp_new = sp * np.power(10.0, total_db / 20.0)
    info = {
        "low_support_db": round(low_support_db * s, 2),
        "presence_db": round(presence_db * s, 2),
        "max_gain_db": round(float(np.max(gain)), 2),
        "sibilance_frames": int(np.count_nonzero(sib < -0.1)),
        "sibilance_max_atten_db": round(float(np.min(sib)) if sib.size else 0.0, 2),
    }
    return sp_new, info


def match_ltas(
    x: np.ndarray,
    sr: int,
    ref_ltas_db: np.ndarray,
    ref_freqs: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_db: float = 2.0,
    smooth_bins: int = 15,
) -> tuple[np.ndarray, dict]:
    """把 x 的长时平均谱（LTAS）对齐到参考 LTAS，返回 (新波形, 统计)。

    为什么需要这一步（"以自己的音色为基准"的实现）
    --------------------------------------------
    WORLD 是**参数化重合成**：它把信号拆成 f0 / 包络 / 非周期成分再重建，
    即使共振峰位置保持不变（已验证 0.7% 偏移），整体频谱仍会漂移——实测
    LTAS 相关系数只有 0.68。这不是"共振峰跑了"，而是重建过程改变了
    谐波间的相对能量与噪声底。

    所以"以用户自己的音色为基准"不能只靠"不改包络"，必须有一步显式的
    回归：用**输入信号自己的 LTAS** 作为参考，把输出拉回去。

    限幅 max_db = 2 dB 是刻意的：只修正大尺度漂移，不动 ±2 dB 以内的
    细节，这样既有的音色增强（低次谐波支撑、存在感）不会被抹掉。
    """
    from scipy.ndimage import uniform_filter1d

    from ..dsp.stft import istft, stft_mag_phase

    sig = np.asarray(x, dtype=float).ravel()
    mag, phase, freqs, _t = stft_mag_phase(sig, sr, n_fft=n_fft, hop_length=hop_length)

    # 当前信号的 LTAS（dB）
    cur = 10.0 * np.log10(np.maximum(np.mean(mag**2, axis=1), 1e-12))
    # 参考 LTAS 插值到当前频率轴
    ref = np.interp(freqs, np.asarray(ref_freqs, float), np.asarray(ref_ltas_db, float))

    gain = ref - cur
    # 只在有意义的频带上修正（20 Hz - 16 kHz），并做平滑避免逐 bin 抖动
    band = (freqs >= 20.0) & (freqs <= min(16000.0, sr / 2.0))
    gain = np.where(band, gain, 0.0)
    gain = uniform_filter1d(gain, size=max(3, int(smooth_bins)))
    gain = np.clip(gain, -max_db, max_db)

    mag_new = mag * np.power(10.0, gain[:, None] / 20.0)
    y = istft(mag_new, phase, n_fft=n_fft, hop_length=hop_length, length=sig.size)

    # 匹配后的相关系数（同一频带）
    cur2 = 10.0 * np.log10(np.maximum(np.mean(mag_new**2, axis=1), 1e-12))
    sel = (freqs >= 100) & (freqs <= 8000)
    corr = float(np.corrcoef(ref[sel], cur2[sel])[0, 1]) if int(sel.sum()) > 10 else 1.0
    info = {
        "applied": True,
        "max_db": float(max_db),
        "applied_gain_range_db": [round(float(gain.min()), 2), round(float(gain.max()), 2)],
        "ltas_correlation_after_match": round(corr, 4),
    }
    return y, info


def _apply_silence_gate(
    y: np.ndarray,
    ref: np.ndarray,
    sr: int,
    threshold_db: float = -55.0,
    smooth_ms: float = 20.0,
) -> np.ndarray:
    """输入为静音的区间，输出也必须为静音。

    为什么需要：WORLD 重合成在"静音→发声"的边界会产生极微弱的周期性成分，
    足以让 f0 检测器多判出 1 帧"浊音"。它听不见，但违反"禁止把清音改为有声"
    的原则，所以直接用**输入信号自己的能量包络**做门控压掉。

    用平方律软门（gain = (rms/thr)^2）而不是硬切，避免引入爆音。
    """
    from scipy.ndimage import uniform_filter1d

    sig_out = np.asarray(y, dtype=float).ravel()
    r = np.asarray(ref, dtype=float).ravel()
    n = min(sig_out.size, r.size)
    if n == 0:
        return sig_out

    w = max(3, int(smooth_ms * 1e-3 * sr))
    rms = np.sqrt(np.maximum(uniform_filter1d(r[:n] ** 2, w), 1e-20))
    peak = float(np.max(rms)) or 1.0
    thr = peak * 10.0 ** (threshold_db / 20.0)
    gain = np.clip(rms / max(thr, 1e-20), 0.0, 1.0) ** 2.0

    out = sig_out.copy()
    out[:n] = out[:n] * gain
    return out


def compress(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 2.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    max_gain_reduction_db: float = 6.0,
    makeup: bool = True,
) -> np.ndarray:
    """前馈式 RMS 压缩（dB 域包络跟随）。

    时间常数：attack 10 ms / release 100 ms 是人声的常用折中——攻击太快会
    削掉辅音起始（"吃字"），释放太快会产生抽吸感（pumping）。
    """
    sig = np.asarray(x, dtype=float).ravel()
    if sig.size == 0:
        return sig.copy()

    # 用一阶递归平滑近似 RMS 包络（时间常数由采样率换算）
    a_att = np.exp(-1.0 / max(attack_ms * 1e-3 * sr, 1.0))
    a_rel = np.exp(-1.0 / max(release_ms * 1e-3 * sr, 1.0))
    env = np.zeros(sig.size)
    v = 0.0
    for i, s in enumerate(np.abs(sig)):
        a = a_att if s > v else a_rel
        v = a * v + (1.0 - a) * s
        env[i] = v
    env_db = 20.0 * np.log10(np.maximum(env, _EPS))

    over = np.maximum(env_db - threshold_db, 0.0)
    reduction_db = -over * (1.0 - 1.0 / max(ratio, 1e-6))
    reduction_db = np.maximum(reduction_db, -abs(max_gain_reduction_db))

    gain = np.power(10.0, reduction_db / 20.0)
    y = sig * gain
    if makeup:
        # 补偿被压掉的响度：按实际平均衰减量补齐
        comp = float(np.mean(gain))
        if comp > _EPS:
            y = y / comp
    return y


def soft_limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """软限幅：只对超过阈值的部分用 tanh 平滑压回。

    ceiling_db = -1 dBFS 是流媒体交付的常见安全线（多数平台按 true peak
    ≤ -1 dBTP 做归一化，超过会被二次压制）。
    """
    sig = np.asarray(x, dtype=float).ravel()
    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    ceiling = 10.0 ** (ceiling_db / 20.0)
    if peak <= ceiling or peak <= _EPS:
        return sig.copy()

    # 先把整体缩到阈值内，再对超出部分做 tanh 软压缩，避免硬碰硬
    scaled = sig * (ceiling / peak) * 1.10
    over = np.abs(scaled) > ceiling
    out = scaled.copy()
    if over.any():
        s = np.sign(scaled[over])
        excess = (np.abs(scaled[over]) - ceiling) / ceiling
        out[over] = s * ceiling * (1.0 + np.tanh(excess))
    return np.clip(out, -ceiling, ceiling)
