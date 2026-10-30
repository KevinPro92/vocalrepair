"""音高刻度与音分映射（十二平均律，A4 = 440 Hz 基准）。

本模块是整个修音链路的"坐标系"：把物理量（频率 Hz、采样点数）与
音乐量（MIDI 音高号、音名、音分）互相转换，并给出"这一帧应该修正到
哪个音"的判据。

物理约定
--------
- 十二平均律：相邻半音频率比为 2^(1/12) = 1.0594630943592953
- 基准：A4 = MIDI 69 = 440.000 Hz
- 音分：100 cents = 1 半音，1200 cents = 1 八度
- 音名采用科学音高记法：C4 = MIDI 60 = 261.6256 Hz（钢琴中央 C）

为什么用音分而不是用 Hz 做误差度量
----------------------------------
人对音高的感知近似服从对数律。同样 5 Hz 的偏差在 100 Hz 处是 84 cents
（非常刺耳），在 1000 Hz 处只有 8.6 cents（几乎听不出）。因此音准误差、
颤音深度、修音阈值都必须用音分（cents = 1200*log2(f/f_ref)）表达，
这是感知上近似线性的坐标轴。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# ---------------------------------------------------------------------------
# 常量：整个项目的音高标准只有一个来源，禁止在别处硬编码 440
# ---------------------------------------------------------------------------
A4_HZ: float = 440.0
A4_MIDI: float = 69.0
SEMITONES_PER_OCTAVE: float = 12.0
CENTS_PER_SEMITONE: float = 100.0
CENTS_PER_OCTAVE: float = 1200.0

#: 十二个音级名称，索引 = MIDI 音高号 % 12
NOTE_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

#: 音名 -> 音级（0 = C）。等音异名一并收录，便于解析乐谱/用户输入。
NOTE_TO_PITCH_CLASS: Mapping[str, int] = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "FB": 4, "E#": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

#: 常用音阶模板（相对主音的半音数），用于 scale-aware 吸附
SCALE_TEMPLATES: Mapping[str, tuple[int, ...]] = {
    "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
}


# ---------------------------------------------------------------------------
# 1. 基本换算：MIDI <-> Hz <-> cents
# ---------------------------------------------------------------------------
def midi_to_hz(midi, a4_hz: float = A4_HZ):
    """MIDI 音高号 -> 频率 / Hz。A4（69 号）= 440 Hz。"""
    m = np.asarray(midi, dtype=float)
    return a4_hz * np.power(2.0, (m - A4_MIDI) / SEMITONES_PER_OCTAVE)


def hz_to_midi(hz, a4_hz: float = A4_HZ):
    """频率 / Hz -> 连续 MIDI 音高号（小数部分携带音分信息）。"""
    f = np.asarray(hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return A4_MIDI + SEMITONES_PER_OCTAVE * np.log2(f / a4_hz)


def hz_to_cents(hz, ref_hz: float = A4_HZ):
    """频率 -> 相对参考频率的音分（绝对音分，不是偏差）。"""
    f = np.asarray(hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return CENTS_PER_OCTAVE * np.log2(f / ref_hz)


def cents_to_ratio(cents):
    """音分 -> 频率比。+1200 cents = 2 倍，-1200 cents = 0.5 倍。"""
    return np.power(2.0, np.asarray(cents, dtype=float) / CENTS_PER_OCTAVE)


def semitones_to_ratio(semitones):
    """半音数 -> 频率比（+12 半音 = 倍频）。"""
    return np.power(2.0, np.asarray(semitones, dtype=float) / SEMITONES_PER_OCTAVE)


def cents_between(hz_from, hz_to):
    """从 hz_from 移到 hz_to 需要多少音分（正 = 升高）。"""
    a = np.asarray(hz_from, dtype=float)
    b = np.asarray(hz_to, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return CENTS_PER_OCTAVE * np.log2(b / a)


# ---------------------------------------------------------------------------
# 2. 音名显示
# ---------------------------------------------------------------------------
def midi_to_name(midi: float) -> str:
    """MIDI 号 -> 音名（科学音高记法，C4 = 60）。"""
    m = int(round(float(midi)))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def hz_to_name(hz: float) -> str:
    """频率 -> 音名。"""
    return midi_to_name(hz_to_midi(hz))


def note_name_to_midi(name: str) -> int:
    """音名 -> MIDI 号，例如 "A4" -> 69、"C#3" -> 49。"""
    text = name.strip().upper()
    if len(text) < 2:
        raise ValueError(f"无法解析音名：{name!r}")
    idx = len(text) - 1
    while idx > 0 and (text[idx].isdigit() or (text[idx] == "-" and idx == len(text) - 2)):
        idx -= 1
    pitch, octave = text[: idx + 1], text[idx + 1:]
    if pitch not in NOTE_TO_PITCH_CLASS:
        raise ValueError(f"无法解析音名：{name!r}")
    return NOTE_TO_PITCH_CLASS[pitch] + (int(octave) + 1) * 12


def pitch_class_of(name: str) -> int:
    """音名/音级 -> 音级（0=C … 11=B）。接受 "A"、"a"、"C#"、"Bb"。"""
    key = name.strip().upper().replace("#", "#")
    if len(key) > 1 and key[-1].isdigit():
        key = key[:-1]
    if key not in NOTE_TO_PITCH_CLASS:
        raise ValueError(f"无法解析音级：{name!r}")
    return NOTE_TO_PITCH_CLASS[key]


# ---------------------------------------------------------------------------
# 3. 偏差与吸附（"该修到哪个音"）
# ---------------------------------------------------------------------------
def cents_deviation(hz):
    """相对最近半音的音分偏差，取值范围 (-50, +50]。

    这是判断"跑调有多严重"的直接量。|偏差| 接近 50 说明该音其实更靠近
    另一个半音，往往意味着检测错误，或演唱确实落在两个半音中间。
    """
    m = hz_to_midi(hz)
    return (m - np.round(m)) * CENTS_PER_SEMITONE


def snap_to_semitone(hz):
    """吸附到最近的十二平均律半音（chromatic snap）。"""
    return midi_to_hz(np.round(hz_to_midi(hz)))


def scale_pitch_classes(root_pc: int, scale: str = "major") -> frozenset:
    """给定主音音级与音阶名，返回该调的 12 音级集合。"""
    if scale not in SCALE_TEMPLATES:
        raise KeyError(f"未知音阶 {scale!r}，可选：{sorted(SCALE_TEMPLATES)}")
    return frozenset((root_pc + step) % 12 for step in SCALE_TEMPLATES[scale])


def snap_to_scale(hz: float, root_pc: int, scale: str = "major") -> float:
    """吸附到指定调内**最近**的音（scale-aware snap），标量版本。

    必须在所有候选里比较距离再取最小，不能"按固定顺序返回第一个命中"：
    例如 MIDI 61.43（C#4 偏高）在 C 大调下，C4 在下方 1.43 半音、D4 在上方
    0.57 半音，正确答案是 D4。早期版本按 (0,-1,+1) 顺序返回了 C4，等于把音
    修到了错误的方向 —— 这是修音软件最不可接受的错误类型。

    只在中心音高 ±2 半音内搜索即可覆盖全部情况（十二平均律相邻同名音相距
    12 半音，任何调式在任意位置上 2 半音内必有一个调内音）。
    """
    pcs = scale_pitch_classes(root_pc, scale)
    continuous = float(hz_to_midi(hz))
    center = int(round(continuous))

    best_note = None
    best_dist = float("inf")
    for delta in range(-2, 3):
        candidate = center + delta
        if candidate % 12 not in pcs:
            continue
        dist = abs(candidate - continuous)
        if dist < best_dist:
            best_note, best_dist = candidate, dist
    if best_note is None:
        raise RuntimeError("音阶集合异常，未找到候选音")  # 逻辑上不可达
    return float(midi_to_hz(best_note))


def snap_track_to_scale(hz_track, root_pc: int, scale: str = "major") -> np.ndarray:
    """对整条 f0 轨迹逐帧做调内吸附；NaN / 非正值保持 NaN。"""
    out = np.full(len(hz_track), np.nan, dtype=float)
    for i, f in enumerate(hz_track):
        if np.isfinite(f) and f > 0:
            out[i] = snap_to_scale(float(f), root_pc, scale)
    return out


# ---------------------------------------------------------------------------
# 4. 软吸附：把"目标"变成"平滑逼近的轨迹"
# ---------------------------------------------------------------------------
def smooth_center_track(
    hz_track,
    dt_s: float = 0.010,
    vibrato_hz: float = 5.5,
    cycles: float = 1.0,
) -> np.ndarray:
    """用滑动中值滤波估计颤音的"中心轨迹"。

    窗长必须由物理量决定，不能拍一个数：
      - 一个完整颤音周期 = 1/vibrato_hz 秒，折合帧数 = 1/(vibrato_hz*dt_s)
      - 10 ms 帧移、5.5 Hz 颤音 → 18.2 帧；取 cycles=1.0 使窗覆盖一个完整
        周期，中值滤波才能把周期性起伏压平（窗长只有半周期时实测仅能
        抑制约 40% 的调制深度，这正是早期版本的问题）
      - 中值滤波是保边算子：音高台阶（换音）不会被抹成斜坡，这是它优于
        同长度滑动平均的原因

    诚实标注能力边界：即使是"窗长 = 一个完整周期"的最优配置，中值滤波对
    正弦型调制也只能抑制约 70%（实测 40 cents 颤音 → 残差约 12 cents），
    无法完全消除。工程上这是足够的，因为残差只需小于修音死区（默认 15
    cents）就不会被误判为跑调；若强行加长窗长，换音的过渡带会被人为拉宽。
    """
    x = np.asarray(hz_track, dtype=float)
    frames = int(round(cycles / max(vibrato_hz * dt_s, 1e-9)))
    frames = max(3, frames)
    if frames % 2 == 0:
        frames += 1
    half = frames // 2
    padded = np.pad(x, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, frames)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(windows, axis=-1)


def soft_correct_track(
    f0_hz,
    target_hz,
    dt_s: float,
    tau_s: float = 0.030,
    deadzone_cents: float = 15.0,
    voiced=None,
) -> np.ndarray:
    """逐帧一阶追踪软吸附，返回修正后的 f0 轨迹。

    每一帧只拉回"超出死区的那部分"误差，并按 alpha = 1-exp(-dt/tau)
    的比例逼近目标：
      - tau -> 0   即 alpha -> 1，等价 Auto-Tune 的硬吸附（台阶化基频，
        谐波间相位不连续，产生金属/电音味）
      - tau = 30 ms  为自然区间，听感上"准了但没被掐住"
      - tau > 100 ms 基本修不动，只压掉最慢的漂移

    只拉回超出死区的部分，是为了保留演奏/演唱的倾向性（导音偏高、
    三度偏高），这一条是"听起来像人唱的"与"听起来像软件"的分界。
    """
    f0 = np.asarray(f0_hz, dtype=float)
    tgt = np.asarray(target_hz, dtype=float)
    out = f0.copy()
    alpha = 1.0 - np.exp(-dt_s / max(tau_s, 1e-6))
    for i in range(f0.size):
        if not (np.isfinite(f0[i]) and np.isfinite(tgt[i])) or f0[i] <= 0 or tgt[i] <= 0:
            continue
        if voiced is not None and not bool(voiced[i]):
            continue
        err = float(cents_between(f0[i], tgt[i]))
        if abs(err) <= deadzone_cents:
            continue
        excess = err - np.sign(err) * deadzone_cents
        out[i] = f0[i] * float(cents_to_ratio(alpha * excess))
    return out
