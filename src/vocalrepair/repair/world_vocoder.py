"""WORLD 声码器：保共振峰的变调与重合成。

物理原理（这是"升 key 不尖锐、降 key 不水牛"的全部依据）
-------------------------------------------------------
人声按源-滤波器模型分解：S(f) = E(f) * H(f)
  - E(f)：声带振动产生的谐波梳，峰位在 f0, 2f0, 3f0 ...（随基频移动）
  - H(f)：声道传递函数，峰位就是共振峰 F1/F2/F3（只由声道形状决定）

WORLD 的 cheaptrick 给出的是**与 f0 无关的平滑谱包络 sp**（即 H 的估计）。
变调时只改 f0 数组，sp 与 ap 原封不动 → synthesize 重新生成的是新的谐波梳，
而共振峰位置不变。这就是"保共振峰"的实现方式。

对照：相位声码器是在频率轴上**整体缩放**频谱，E 和 H 一起被乘了同一个系数，
共振峰跟着跑，于是变花栗鼠或水牛音。必须额外做"包络除法再乘回"才能修正。

一处实测修正（重要）
--------------------
不要用 pyworld.harvest 求 f0：本机实测 harvest 对 220 Hz 纯音报 214.17 Hz
（偏 −47 cents），因为 harvest 的代价函数偏向真人声而非合成音。正确做法是
用本项目自己的 YIN 结果（实测误差中位数 2.77 cents）插值到 WORLD 的 5 ms 帧
网格上，再交给 cheaptrick / d4c 分析包络。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WorldFeatures", "world_analyze", "world_synthesize", "resynth_transpose"]

_EPS = 1e-12


@dataclass
class WorldFeatures:
    """WORLD 分析结果（全部为 WORLD 帧网格上的数组）。"""

    f0: np.ndarray        # 帧级基频 / Hz，0 表示非周期
    sp: np.ndarray        # 谱包络 (帧数, 频率 bin 数)，线性幅度
    ap: np.ndarray        # 非周期成分比例 (帧数, 频率 bin 数)，0-1
    t: np.ndarray         # 帧中心时刻 / s
    sr: int
    frame_period: float   # 帧移 / ms

    @property
    def n_bins(self) -> int:
        return int(self.sp.shape[1])

    @property
    def freqs(self) -> np.ndarray:
        """sp 的频率轴（0 .. sr/2 等间隔）。"""
        return np.linspace(0.0, self.sr / 2.0, self.n_bins)


def _f0_on_world_grid(
    x: np.ndarray,
    sr: int,
    n_frames: int,
    frame_period: float,
    f0_hz: np.ndarray | None,
    times: np.ndarray | None,
) -> np.ndarray:
    """把外部 f0（如 YIN 结果）插值到 WORLD 帧网格；缺失时回退 harvest。"""
    t_w = np.arange(n_frames) * frame_period / 1000.0
    if f0_hz is None or times is None:
        import pyworld

        f0, _ = pyworld.harvest(
            np.asarray(x, dtype=np.float64), int(sr), f0_floor=60.0, f0_ceil=1100.0,
            frame_period=float(frame_period),
        )
        f0 = np.asarray(f0, dtype=np.float64)
        if f0.size >= n_frames:
            return f0[:n_frames]
        return np.pad(f0, (0, n_frames - f0.size), mode="edge")

    f = np.asarray(f0_hz, dtype=float)
    t = np.asarray(times, dtype=float)
    valid = np.isfinite(f) & (f > 0)
    if valid.sum() < 2:
        return np.zeros(n_frames, dtype=np.float64)

    # 只在有效帧上插值，无效区域填 0（WORLD 用 0 表示非周期）
    out = np.interp(t_w, t[valid], f[valid])
    # 用有效区间之外的部分置零，避免把首尾静音误当成有声
    outside = (t_w < t[valid].min()) | (t_w > t[valid].max() + 1e-9)
    out[outside] = 0.0
    return out.astype(np.float64)


def world_analyze(
    x: np.ndarray,
    sr: int,
    f0_hz: np.ndarray | None = None,
    times: np.ndarray | None = None,
    frame_period: float = 5.0,
) -> WorldFeatures:
    """分解为 f0 / 谱包络 sp / 非周期 ap。

    f0_hz 与 times 为帧级 f0（本项目 YIN 的结果）；不提供则回退 harvest。
    """
    import pyworld

    sig = np.asarray(x, dtype=np.float64).ravel()
    n_frames = max(1, int(np.ceil(sig.size / (sr * frame_period / 1000.0))))
    f0 = _f0_on_world_grid(sig, sr, n_frames, frame_period, f0_hz, times)

    # cheaptrick / d4c 要求时间轴连续且帧数一致
    t = np.arange(f0.size, dtype=np.float64) * (frame_period / 1000.0)
    sp = pyworld.cheaptrick(sig, f0, t, int(sr))
    ap = pyworld.d4c(sig, f0, t, int(sr))
    return WorldFeatures(
        f0=f0,
        sp=np.asarray(sp, dtype=np.float64),
        ap=np.asarray(ap, dtype=np.float64),
        t=t,
        sr=int(sr),
        frame_period=float(frame_period),
    )


def warp_ap(ap: np.ndarray, f0_old: np.ndarray, f0_new: np.ndarray) -> np.ndarray:
    """按 f0 比率在频率轴上重采样非周期成分 ap。

    为什么必须做（这是"糊掉 + 吱呀 + 水流声"的直接原因）
    --------------------------------------------------
    ``ap[k]`` 表示"第 k 个频点有多少比例是非周期成分"。它的结构是**梳状**的：
    谐波位置 ap 低（周期性最强），谐波之间 ap 高（气声/噪声）。f0 改变后谐波梳
    会移动；若 ap 不同步移动，就会出现：
      - 新谐波落在原来"气声多"的格子上 → 谐波被当成噪声 → **人声糊掉**
      - 原谐波位置（ap 低）现在落在谐波之间 → 噪声被当成谐波 → **吱呀/水流声**
    实测素材 73-77% 的频点 ap > 0.5（手机录音的气声与底噪），所以这个失配
    非常明显。

    做法与保共振峰的 sp 不同：sp 是**包络**（与 f0 无关，绝对不能动），
    ap 是**谐波结构的一部分**（必须跟着 f0 走）。这一区分是 WORLD 变调的要点。
    """
    n_frames, n_bins = ap.shape
    out = np.asarray(ap, dtype=np.float64).copy()
    src = np.arange(n_bins, dtype=float)
    fo = np.asarray(f0_old, dtype=float)
    fn = np.asarray(f0_new, dtype=float)

    for i in range(min(n_frames, fo.size, fn.size)):
        if fo[i] <= 0 or fn[i] <= 0:
            continue
        ratio = fn[i] / fo[i]
        if abs(ratio - 1.0) < 1e-4:
            continue
        dst = src / ratio
        out[i] = np.interp(dst, src, ap[i], left=ap[i, 0], right=ap[i, -1])
    return out


def world_synthesize(
    feat: WorldFeatures,
    f0_new: np.ndarray | None = None,
    sp_new: np.ndarray | None = None,
    ap_new: np.ndarray | None = None,
) -> np.ndarray:
    """用 WORLD 重新合成波形。

    ap_new 通常必须与 f0_new 同步变化（见 ``warp_ap``）：只改 f0 而不改 ap
    会产生梳状失配，听感上是"糊掉 + 吱呀声"。
    """
    import pyworld

    f0 = feat.f0 if f0_new is None else np.asarray(f0_new, dtype=np.float64)
    sp = feat.sp if sp_new is None else np.asarray(sp_new, dtype=np.float64)
    ap = feat.ap if ap_new is None else np.asarray(ap_new, dtype=np.float64)
    if f0.shape[0] != sp.shape[0]:
        raise ValueError(f"f0 帧数 {f0.shape[0]} 与 sp 帧数 {sp.shape[0]} 不一致")
    y = pyworld.synthesize(f0, sp, ap, feat.sr, feat.frame_period)
    return np.asarray(y, dtype=np.float64)


def shift_formants(sp: np.ndarray, ratio: float) -> np.ndarray:
    """对谱包络做频率轴缩放：sp_new(f) = sp(f / ratio)。

    ratio > 1 使共振峰上移（声道变短，声音更"亮/年轻"）；
    ratio < 1 使共振峰下移（声道变长，更"厚/暗"）。

    注意这与"变调"是两件独立的事：变调改谐波梳，这里改的是包络本身。
    音域拓宽功能（超出自然音域时轻微上移共振峰模拟"唱上去"的声道变化）
    就是调用本函数，且必须限幅（本项目上限 8%，见 repair.pitch_correct）。
    """
    if abs(ratio - 1.0) < 1e-6:
        return sp
    n_bins = sp.shape[1]
    src = np.arange(n_bins, dtype=float)
    dst = src / float(ratio)
    out = np.empty_like(sp)
    for i in range(sp.shape[0]):
        row = sp[i]
        out[i] = np.interp(dst, src, row, left=row[0], right=row[-1])
    return out


def resynth_transpose(
    x: np.ndarray,
    sr: int,
    semitones: float,
    f0_hz: np.ndarray | None = None,
    times: np.ndarray | None = None,
    frame_period: float = 5.0,
    formant_shift: float | None = None,
    max_semitones: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """变调重合成（保共振峰），返回 (波形, 信息字典)。

    参数
    ----
    semitones : 半音数，正升负降。**超过 max_semitones 会被截断并记录**——
        超过 ±5 半音时，即使共振峰保持，音色也会明显改变（声道长度是固定的
        物理量），这是物理限制而不是实现缺陷
    formant_shift : 额外的共振峰缩放比例（None = 1.0，完全保持）
    """
    sig = np.asarray(x, dtype=np.float64).ravel()
    requested = float(semitones)
    applied = float(np.clip(requested, -max_semitones, max_semitones))
    clipped = abs(requested - applied) > 1e-9

    feat = world_analyze(sig, sr, f0_hz=f0_hz, times=times, frame_period=frame_period)
    f0_new = feat.f0 * (2.0 ** (applied / 12.0))
    # 只在原本就浊音的帧上保持浊音，绝不把清音帧变成有声
    f0_new = np.where(feat.f0 > 0, f0_new, 0.0)

    ratio = 1.0 if formant_shift is None else float(formant_shift)
    sp_new = shift_formants(feat.sp, ratio) if abs(ratio - 1.0) > 1e-9 else feat.sp

    # ap 也要随 f0 缩放（与 pipeline 同一理由：否则梳状结构失配 → 糊掉/吱呀）
    ap_new = warp_ap(feat.ap, feat.f0, f0_new)
    y = world_synthesize(feat, f0_new=f0_new, sp_new=sp_new, ap_new=ap_new)
    if y.size > sig.size:
        y = y[: sig.size]
    elif y.size < sig.size:
        y = np.pad(y, (0, sig.size - y.size))

    info = {
        "requested_semitones": round(requested, 3),
        "applied_semitones": round(applied, 3),
        "clipped_to_limit": bool(clipped),
        "max_semitones_allowed": float(max_semitones),
        "formant_shift_ratio": round(ratio, 4),
        "resonance_preserved": bool(abs(ratio - 1.0) < 1e-6),
        "voiced_frames": int(np.count_nonzero(feat.f0 > 0)),
        "frame_period_ms": float(frame_period),
        "length_in": int(sig.size),
        "length_out": int(y.size),
    }
    return y, info

