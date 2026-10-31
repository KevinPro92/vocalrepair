"""瞬态检测：找出爆破辅音、音符起音、换声点等"非稳态"位置。

为什么需要它
------------
用户指定的三处残留问题：3-4 s 的 E4、10 s 的 #G4-#F4-E4-D4 下行音阶、
16 s 的 #F4 真声转 B4 假声——听起来"像爆破音"。

这三处的共同点是**都不是稳态**：
- 爆破辅音（p/t/k）本身是宽带瞬态，没有周期结构
- 快速下行音阶每个周期的内容都不同（滑音）
- 真假声转换时音高与共振峰同时突变

PSOLA 假设"相邻周期波形相似"，在这三处必然拼接错乱。降噪只处理平稳噪声，
对它们无效，所以必须在**修音阶段就把它们排除掉**。

判据
----
**半波整流谱通量**（spectral flux）：相邻帧幅度谱的正向增量之和。
稳态浊音的谱形状帧间几乎不变（通量低）；爆破辅音、起音、换声点的谱形状
突变（通量高）。这是 onset detection 的标准做法，对"突变"最敏感。
"""

from __future__ import annotations

import numpy as np

from ..dsp.stft import stft_mag_phase

__all__ = ["detect_transients"]

_EPS = 1e-12


def detect_transients(
    x: np.ndarray,
    sr: int,
    times: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    k_sigma: float = 1.6,
    rel_threshold: float = 1.3,
    min_gap_ms: float = 30.0,
) -> np.ndarray:
    """返回与 ``times`` 等长的布尔数组：True 表示该帧处于瞬态。

    参数
    ----
    k_sigma : 峰值判定用 ``median + k_sigma * MAD``（MAD 抗离群，比 std 稳）
    rel_threshold : 通量还需超过全局通量中位数的多少倍，避免把安静段的小起伏
        当成瞬态
    min_gap_ms : 两个瞬态之间的最小间隔，小于此值视为同一次瞬态（防止抖动）
    """
    sig = np.asarray(x, dtype=float).ravel()
    mag, _ph, _freqs, tf = stft_mag_phase(sig, sr, n_fft, hop_length)

    if mag.shape[1] < 3:
        return np.zeros(np.asarray(times).size, dtype=bool)

    # 半波整流谱通量：只统计"变大"的部分，对起音敏感、对衰减不敏感
    flux = np.sum(np.maximum(0.0, np.diff(mag, axis=1)), axis=0)
    flux = np.concatenate(([0.0], flux))

    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med))) + _EPS
    thr = max(med + k_sigma * 1.4826 * mad, rel_threshold * med)
    strong = flux > thr

    # 映射到 f0 帧网格
    out = np.interp(np.asarray(times), tf, strong.astype(float), left=0.0, right=0.0) > 0.5

    # 每次瞬态前后各扩 2 帧（约 20 ms）：瞬态的边沿同样不适合拼接
    if out.any():
        g = out.copy()
        for shift in (-2, -1, 1, 2):
            shifted = np.roll(out, shift)
            if shift > 0:
                shifted[:shift] = False
            else:
                shifted[shift:] = False
            g |= shifted
        out = g

    # 合并过近的瞬态（< min_gap_ms 视为同一事件）
    dt = float(np.median(np.diff(times))) if np.asarray(times).size > 1 else 0.01
    min_gap_frames = max(1, int(round(min_gap_ms * 1e-3 / max(dt, 1e-9))))
    idx = np.flatnonzero(out)
    if idx.size > 1:
        keep = [int(idx[0])]
        for i in idx[1:]:
            if i - keep[-1] >= min_gap_frames:
                keep.append(int(i))
            else:
                out[keep[-1] : i] = True
        out[:] = out  # keep 已并入 out，无需额外处理
    return out
