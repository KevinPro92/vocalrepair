"""基频（f0）估计：YIN 算法（de Cheveigné & Kawahara, 2002）的实现。

为什么不直接用"STFT 找谱峰"
---------------------------
STFT 的频率分辨率 df = sr/N（44.1 kHz / 2048 ≈ 21.5 Hz）。在 200 Hz 处
21.5 Hz 相当于 186 cents，完全无法用于音准判断。YIN 在时域直接测周期
f0 = sr/tau，tau 经抛物线插值后精度约 0.01 个采样点，在 200 Hz 处折合
0.15 cents —— 差三个数量级。

算法四步
--------
1. 差分函数      d(tau) = sum_j (x[j] - x[j+tau])^2
2. 累积均值归一化 c(tau) = d(tau)*tau / sum_{k=1..tau} d(k)   <- 抗倍频错误的关键
3. 绝对阈值      取第一个低于 threshold 的局部极小
4. 抛物线插值    在 tau* 邻域三点拟合，得到亚采样精度的周期

第 1 步等价于 d(tau) = e0 + e(tau) - 2*r(tau)，其中 r(tau) 用 FFT 互相关
计算，避免逐点暴力搜索。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["F0Result", "yin_f0", "difference_function"]


@dataclass
class F0Result:
    """帧级基频分析结果，各字段等长；未检出（清音/静音）处 f0_hz 为 NaN。"""

    f0_hz: np.ndarray          # 基频 / Hz
    times: np.ndarray          # 每帧中心时刻 / s
    aperiodicity: np.ndarray   # CMND 最小值，越小越"浊音"（0 = 完全周期）
    voiced: np.ndarray         # 清/浊音判决（bool）
    frame_length: int          # 分析窗长 / 采样点
    hop_length: int            # 帧移 / 采样点
    sr: int                    # 采样率 / Hz

    @property
    def dt(self) -> float:
        """帧移对应的时间 / s。"""
        return self.hop_length / self.sr

    def cents(self, ref_hz: float | None = None) -> np.ndarray:
        """相对 A4（或指定参考）的音分轨迹，用于画图与统计。"""
        from .pitch_scale import A4_HZ, hz_to_cents

        ref = A4_HZ if ref_hz is None else ref_hz
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(hz_to_cents(self.f0_hz, ref), dtype=float)


def difference_function(segment: np.ndarray, window: int, tau_max: int) -> np.ndarray:
    """计算差分函数 d(tau)，tau = 0..tau_max。

    参数
    ----
    segment : 一维片段（已去均值），长度 >= window + tau_max + 1
    window  : 参考段长度 W
    tau_max : 最大滞后（对应最小基频）

    返回长度 tau_max+1 的非负数组。
    """
    a = segment[:window]
    b = segment[: window + tau_max + 1]

    # 互相关 r(tau) = sum_{j=0}^{W-1} a[j] * b[tau+j]，用 FFT 计算
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    spec = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    r = np.fft.irfft(spec, n)[: tau_max + 1]

    # 能量项：e0 为常数，e(tau) 用前缀和 O(1) 取得
    e0 = float(np.dot(a, a))
    cs = np.concatenate(([0.0], np.cumsum(b * b)))
    e_tau = cs[window : window + tau_max + 1] - cs[: tau_max + 1]

    d = e0 + e_tau - 2.0 * r
    np.maximum(d, 0.0, out=d)  # 数值误差可能给出极小负值
    return d


def _cmnd(d: np.ndarray) -> np.ndarray:
    """累积均值归一化差分（YIN 第 2 步），c(0) 定义为 1。"""
    c = np.ones_like(d)
    if d.size <= 1:
        return c
    cum = np.cumsum(d[1:])
    k = np.arange(1, d.size, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        c[1:] = np.where(cum > 0, d[1:] * k / cum, 1.0)
    return c


def _pick_tau(c: np.ndarray, tau_min: int, tau_max: int, threshold: float) -> tuple[int, float]:
    """YIN 第 3、4 步：绝对阈值选峰 + 抛物线插值，返回 (tau, CMND 最小值)。"""
    hi = min(tau_max, c.size - 2)
    if hi <= tau_min:
        tau_min, hi = 1, max(2, c.size - 2)
    idx = np.arange(tau_min, hi + 1)
    is_local_min = (c[idx] < c[idx - 1]) & (c[idx] <= c[idx + 1])

    if not is_local_min.any():
        tau = int(idx[np.argmin(c[idx])])
    else:
        candidates = idx[is_local_min]
        below = candidates[c[candidates] < threshold]
        tau = int(below[0]) if below.size else int(candidates[np.argmin(c[candidates])])

    # 抛物线插值：三点拟合抛物线顶点，得到亚采样周期的偏移量
    tau_hat = float(tau)
    if 0 < tau < c.size - 1:
        y0, y1, y2 = float(c[tau - 1]), float(c[tau]), float(c[tau + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            tau_hat = tau + float(np.clip(delta, -0.5, 0.5))
    return tau, tau_hat


def yin_f0(
    x: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 1100.0,
    frame_length: int | None = None,
    hop_length: int | None = None,
    threshold: float = 0.15,
    relative_energy_db: float = -50.0,
) -> F0Result:
    """用 YIN 估计整段信号的 f0 轨迹。

    搜索范围为什么是 65-1100 Hz（C2 - C6 附近）
    -------------------------------------------
    - 下限 65 Hz（C2）：常规人声最低约 E2 = 82 Hz，极端男低音到 C2 = 65 Hz。
      **不要取 C1 = 32.7 Hz**：那一带没有有效人声，只有低频隆隆声与空调噪声，
      放开下限只会让倍周期错误变多。
    - 上限 1100 Hz（略高于 C6 = 1046.5 Hz）：覆盖女高音高音区。哨音（C7 以上）
      不在修音的目标范围内，故意不覆盖。

    **这个范围限制能解决什么、不能解决什么（必须说清）**
    - 能：排除"跑到人声范围之外"的倍周期/半周期错误（例如把 900 Hz 判成 1800 Hz）。
    - 不能：排除**范围内的八度错误**。实测真实案例是 220 Hz 被判成 110 Hz——
      两个候选都在 65-1100 Hz 之内，砍范围毫无作用；真正起作用的是
      ``dsp.f0_track`` 里的"周期惩罚 + 多候选 Viterbi"。

    参数
    ----
    x : 单声道 float 波形，范围约 [-1, 1]
    frame_length : 分析窗长 / 采样点。默认 40 ms：要包住 6-8 个周期，
        同时必须 >= 2 倍最大周期（1100 Hz 时约 40 采样，65 Hz 时约 679 采样）
    hop_length : 帧移 / 采样点。默认 10 ms：要能跟住 4-7 Hz 的颤音
    threshold : YIN 绝对阈值，越小越严格（常用 0.1-0.2）
    relative_energy_db : 相对最响帧的能量门限，低于此值判为静音

    返回
    ----
    F0Result
    """
    sig = np.asarray(x, dtype=float).ravel()
    if sig.size == 0:
        raise ValueError("输入信号为空")

    tau_min = max(1, int(np.floor(sr / float(fmax))))
    tau_max = max(tau_min + 2, int(np.ceil(sr / float(fmin))))

    if frame_length is None:
        frame_length = int(round(0.040 * sr))
    frame_length = max(int(frame_length), 2 * tau_max + 2)
    if hop_length is None:
        hop_length = int(round(0.010 * sr))
    hop_length = max(1, int(hop_length))

    span = frame_length + tau_max + 1
    if sig.size < span:
        sig = np.pad(sig, (0, span - sig.size))

    n_frames = 1 + (sig.size - span) // hop_length
    starts = np.arange(n_frames) * hop_length

    f0 = np.full(n_frames, np.nan, dtype=float)
    aper = np.ones(n_frames, dtype=float)
    rms = np.zeros(n_frames, dtype=float)

    for i, start in enumerate(starts):
        seg = sig[start : start + span]
        seg = seg - seg.mean()  # 去直流，避免偏置污染相关峰
        rms[i] = float(np.sqrt(np.mean(seg[:frame_length] ** 2)))

        d = difference_function(seg, frame_length, tau_max)
        if not np.any(d > 0):
            continue
        c = _cmnd(d)
        tau, tau_hat = _pick_tau(c, tau_min, tau_max, threshold)
        aper[i] = float(c[tau])
        if tau_hat > 0 and aper[i] < 2.0 * threshold:
            f0[i] = sr / tau_hat

    # 能量门限：相对最响帧低于 relative_energy_db 的帧一律判为非浊音
    peak = float(rms.max()) if rms.size else 0.0
    if peak > 0:
        with np.errstate(divide="ignore"):
            rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12) / peak)
    else:
        rms_db = np.full_like(rms, -np.inf)
    voiced = np.isfinite(f0) & (rms_db > relative_energy_db)
    f0 = np.where(voiced, f0, np.nan)

    times = (starts + frame_length / 2.0) / float(sr)
    return F0Result(
        f0_hz=f0,
        times=times,
        aperiodicity=aper,
        voiced=voiced,
        frame_length=frame_length,
        hop_length=hop_length,
        sr=int(sr),
    )
