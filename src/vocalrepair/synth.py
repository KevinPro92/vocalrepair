"""合成测试信号：给算法提供"精确已知真值"的输入。

为什么必须先做这个
------------------
真实录音的真实基频未知，用它验证 f0 检测或修音精度属于循环论证。
本模块用源-滤波器模型（S(f) = E(f) * H(f)）合成"元音化"的歌声片段：

- E(f)：声门脉冲串，谐波幅度近似 1/k（幅度谱约 -6 dB/oct 的倾斜）
- H(f)：声道共振峰包络，用若干高斯峰叠加
- 可注入：颤音、滑音、跑调、气声噪声、削波失真

于是每一时刻的真实 f0 完全已知，可用于断言
"检测误差 < 5 cents"、"修音后偏差中位数 < 5 cents"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 元音共振峰 (F1, F2, F3) 频率 / Hz 与带宽 / Hz（典型成人声道的量级）
VOWEL_FORMANTS: dict[str, tuple[tuple[float, float], ...]] = {
    "a": ((730.0, 90.0), (1090.0, 110.0), (2440.0, 160.0)),
    "e": ((530.0, 80.0), (1840.0, 100.0), (2480.0, 150.0)),
    "i": ((270.0, 60.0), (2290.0, 100.0), (3010.0, 150.0)),
    "o": ((570.0, 80.0), (840.0, 100.0), (2410.0, 160.0)),
    "u": ((300.0, 60.0), (870.0, 90.0), (2240.0, 150.0)),
}


@dataclass
class SynthSignal:
    """合成结果。f0_true / f0_center 均为逐采样真值。"""

    x: np.ndarray             # 波形，峰值归一化到 peak 参数
    sr: int                   # 采样率 / Hz
    f0_true: np.ndarray       # 逐采样瞬时 f0 / Hz（含颤音）
    f0_center: np.ndarray     # 逐采样中心 f0 / Hz（不含颤音）
    times: np.ndarray         # 逐采样时刻 / s

    @property
    def duration(self) -> float:
        return self.x.size / float(self.sr)


def formant_envelope(freqs: np.ndarray, formants) -> np.ndarray:
    """由共振峰参数生成幅度包络（高斯峰叠加）。

    freqs 与 formants 单位均为 Hz。共振峰是声道截面积函数的频域表象，
    决定"哪个元音"，与基频无关——这也是修音必须保留共振峰的原因。
    """
    f = np.asarray(freqs, dtype=float)
    env = np.zeros_like(f)
    for center, bandwidth in formants:
        env += np.exp(-0.5 * ((f - center) / max(bandwidth, 1e-6)) ** 2)
    return env


def synth_vowel(
    f0_per_sample: np.ndarray,
    sr: int,
    vowel: str = "a",
    n_harmonics: int = 40,
    noise_snr_db: float = 40.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """按逐采样基频合成一个元音，返回波形（未归一化）。

    做法：相位 phi(t) = 2*pi*cumsum(f0)/sr 逐采样累积，第 k 次谐波取
    sin(k*phi)，幅度 = (1/k) * 包络(k*f0)。逐采样更新 f0 意味着谐波
    频率随基频一起平滑移动，这正是"自然的声音"在短时谱上的样子。
    """
    f0 = np.asarray(f0_per_sample, dtype=float)
    n = f0.size
    formants = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS["a"])
    rng = np.random.default_rng(0) if rng is None else rng

    phase = 2.0 * np.pi * np.cumsum(f0) / float(sr)
    out = np.zeros(n, dtype=float)
    for k in range(1, n_harmonics + 1):
        inst = k * f0
        amp = formant_envelope(inst, formants) / float(k)
        # 高于 Nyquist 的谐波直接丢弃，避免混叠
        amp = np.where(inst < 0.5 * sr, amp, 0.0)
        if not np.any(amp):
            continue
        out += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

    if noise_snr_db is not None and np.isfinite(noise_snr_db):
        sig_rms = float(np.sqrt(np.mean(out**2))) or 1.0
        noise = rng.normal(scale=sig_rms * 10 ** (-noise_snr_db / 20.0), size=n)
        out = out + noise

    return out


def make_note_sequence(
    midi_notes,
    sr: int = 44100,
    note_dur: float = 0.60,
    gap: float = 0.08,
    vibrato_rate: float = 5.5,
    vibrato_depth_cents: float = 35.0,
    detune_cents=None,
    vowel: str = "a",
    noise_snr_db: float = 40.0,
    seed: int = 0,
) -> SynthSignal:
    """合成一段"音符阶梯"，逐采样真值已知。

    参数
    ----
    midi_notes : MIDI 音高号序列，如 [57, 59, 60, 62, 60]
    note_dur, gap : 每个音的时长与音间静音 / s
    vibrato_rate, vibrato_depth_cents : 颤音速率 / Hz 与深度 / cents。
        自然歌唱约 4-7 Hz、20-80 cents
    detune_cents : 每个音的音准偏差 / cents。正 = 偏高。用来制造"跑调"，
        使修音算法有明确的修正目标
    vowel : 元音类型（决定共振峰），见 VOWEL_FORMANTS
    noise_snr_db : 气声噪声的信噪比 / dB
    """
    from .dsp.pitch_scale import midi_to_hz

    sr = int(sr)
    notes = np.asarray(midi_notes, dtype=float).ravel()
    if notes.size == 0:
        raise ValueError("midi_notes 不能为空")
    if detune_cents is None:
        detune = np.zeros(notes.size, dtype=float)
    else:
        detune = np.asarray(detune_cents, dtype=float).ravel()
        if detune.size == 1:
            detune = np.repeat(detune, notes.size)
        if detune.size != notes.size:
            raise ValueError("detune_cents 长度必须为 1 或与音符数相同")

    note_len = max(1, int(round(note_dur * sr)))
    gap_len = max(0, int(round(gap * sr)))
    step = note_len + gap_len
    n = notes.size * step + gap_len

    f0_true = np.zeros(n, dtype=float)
    f0_center = np.zeros(n, dtype=float)
    env = np.zeros(n, dtype=float)
    ramp = min(int(0.05 * sr), note_len // 2)

    for i, note in enumerate(notes):
        start = i * step
        center = float(midi_to_hz(note + detune[i] / 100.0))
        t = np.arange(note_len, dtype=float) / sr
        # 颤音：以中心音高乘以 2^(cents/1200) 的准周期调制
        vib = np.power(
            2.0,
            (vibrato_depth_cents * np.sin(2.0 * np.pi * vibrato_rate * t + 0.7 * i)) / 1200.0,
        )
        f0_true[start : start + note_len] = center * vib
        f0_center[start : start + note_len] = center

        # 余弦升降包络，避免起止处的阶跃（阶跃会产生宽带爆音）
        e = np.ones(note_len, dtype=float)
        if ramp > 1:
            w = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / (ramp - 1))
            e[:ramp] = w
            e[-ramp:] = w[::-1]
        env[start : start + note_len] = e

    # 间隙处 f0 用最近有效值填充，保证相位累积连续（否则 cumsum 会跳变）
    idx = np.arange(n)
    valid = f0_center > 0
    if valid.any():
        f0_center = np.interp(idx, idx[valid], f0_center[valid])
    f0_true = np.where(f0_true > 0, f0_true, f0_center)

    x = synth_vowel(
        f0_true,
        sr,
        vowel=vowel,
        noise_snr_db=noise_snr_db,
        rng=np.random.default_rng(seed),
    )
    x = x * env
    peak = float(np.max(np.abs(x))) or 1.0
    x = 0.9 * x / peak

    return SynthSignal(
        x=x,
        sr=sr,
        f0_true=f0_true,
        f0_center=f0_center,
        times=idx / float(sr),
    )


def apply_clipping(x: np.ndarray, drive_db: float = 12.0, ceiling: float = 0.25) -> np.ndarray:
    """制造削波（"破音"）：先增益过头，再硬限幅。

    物理上，削波 = 波形顶部被截断，谐波结构被注入高阶谐波（宽带失真）。
    这是"修复破音"要处理的对象：被截断的顶部信息只能估计、不能无损恢复。
    """
    sig = np.asarray(x, dtype=float)
    gain = 10.0 ** (drive_db / 20.0)
    driven = sig * gain
    clipped = np.clip(driven, -ceiling, ceiling)
    # 限幅后再归一化回原峰值量级，便于 A/B 对比（只改波形形状，不改响度）
    peak_out = float(np.max(np.abs(clipped))) or 1.0
    peak_in = float(np.max(np.abs(sig))) or 1.0
    return clipped / peak_out * peak_in


def clipped_sample_mask(x: np.ndarray, ceiling: float = 0.999) -> np.ndarray:
    """检测削波采样：连续 >= 3 个样本贴到满量程。

    必须按"连续平台"整体标记，而不是只标记中间样本：削波平台上的每一个
    样本都是被截断的，全部需要参与重建。只取三连点会漏掉平台首尾各一个
    样本（早期版本的缺陷）。
    """
    sig = np.abs(np.asarray(x, dtype=float)) >= ceiling
    mask = np.zeros(sig.shape, dtype=bool)
    if sig.size < 3:
        return mask

    edges = np.diff(sig.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if sig[0]:
        starts = np.concatenate(([0], starts))
    if sig[-1]:
        ends = np.concatenate((ends, [sig.size]))

    for s, e in zip(starts, ends):
        if e - s >= 3:
            mask[s:e] = True
    return mask


def synth_vowel_allpole(
    f0_per_sample: np.ndarray,
    sr: int,
    vowel: str = "a",
    formants=None,
    bandwidths=None,
    noise_snr_db: float = 45.0,
    seed: int = 0,
) -> np.ndarray:
    """用"全极点声道模型"合成元音——更接近真实发声，用于验证共振峰保持。

    为什么需要它
    ------------
    ``synth_vowel`` 的做法是"给每个谐波按包络赋幅度"，得到的是一个**静态谱
    形状**，并不是真正的声道滤波结果：它没有极点，也没有共振峰的群延迟特性。
    用这种信号做 LPC 分析会得到错误的共振峰（实测 /a/ 的 F1=730、F2=1090
    被融合成一个 913 Hz 的极点，因为 220 Hz 的谐波间距太粗，两点之间分辨不出
    两个共振峰）。

    真实发声可以用经典的源-滤波器模型描述：
        声门脉冲串 → 声道（全极点滤波器）→ 唇辐射（一阶差分）
    而这个模型恰好就是 LPC 的生成模型，所以 LPC 能准确恢复共振峰，
    因此它适合作为"变调是否保住共振峰"的验证信号。

    参数
    ----
    formants, bandwidths : 共振峰中心 / Hz 与带宽 / Hz；默认取 VOWEL_FORMANTS
    """
    from scipy.signal import lfilter

    f0 = np.asarray(f0_per_sample, dtype=float).ravel()
    n = f0.size
    if formants is None or bandwidths is None:
        preset = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS["a"])
        formants = [f for f, _ in preset]
        bandwidths = [b for _, b in preset]

    # 声道：每个共振峰对应一对共轭极点，串成级联全极点滤波器
    a = np.array([1.0])
    for F, BW in zip(formants, bandwidths):
        r = float(np.exp(-np.pi * BW / float(sr)))
        theta = 2.0 * np.pi * float(F) / float(sr)
        a = np.convolve(a, [1.0, -2.0 * r * np.cos(theta), r * r])

    # 声门激励：F0 相位每跨过一个整数就发一个脉冲（脉冲串）
    phase = np.cumsum(f0) / float(sr)
    pulse_idx = np.flatnonzero(np.diff(np.floor(phase), prepend=np.floor(phase[0])) > 0)
    exc = np.zeros(n, dtype=float)
    exc[pulse_idx[pulse_idx < n]] = 1.0

    y = lfilter([1.0], a, exc)
    y = np.diff(y, prepend=0.0)  # 唇辐射：一阶差分，带来约 +6 dB/oct 的高频提升

    if noise_snr_db is not None and np.isfinite(noise_snr_db):
        rng = np.random.default_rng(seed)
        rms = float(np.sqrt(np.mean(y**2))) or 1.0
        y = y + rng.normal(scale=rms * 10 ** (-noise_snr_db / 20.0), size=n)

    peak = float(np.max(np.abs(y))) or 1.0
    return 0.9 * y / peak
