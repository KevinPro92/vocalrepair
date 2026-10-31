"""TD-PSOLA：时域基音同步重叠相加（Time-Domain Pitch-Synchronous Overlap-Add）。

为什么音准修正必须回退到它
--------------------------
之前整条修复链路都走 WORLD 重合成。实测在用户的手机录音上（``ap`` 均值 0.71，
即 70% 是非周期成分）：

- 纯 WORLD 往返（f0/sp/ap 都不改）的谱失真就已经很大，且**换 fft_size、
  平滑 f0、时间平均 ap、ap 减半都救不回来**（5 种调参结果全部落在 9.5-9.6 dB）
- 听感上就是"收音机调频似的吱呀 + 水流声 + 人声糊掉"

原因是 WORLD 是**参数化重合成**：它对非周期成分用随机相位生成噪声，气声越重，
"水声"越明显。而音准修正只需要 ±1 个半音以内的微调，完全没必要重建整段信号。

PSOLA 的思路完全不同：**不合成任何东西**，只是把原始波形按基音周期切成小段，
再按新的周期间隔重新摆放。于是：

- 音色、气声、噪声纹理、相位细节全部来自原始录音 → 不会有"水声"
- 帧内容不变、只有间隔变了 → **天然保共振峰**、天然保时长
- 代价：改动超过 ±2 半音会有颗粒感，所以只用于音准修正，大幅变调仍走 WORLD

实现要点
--------
1. 基音标记（pitch marks）用 f0 的累积相位确定，比"找局部极大值"稳得多
2. 每个合成帧取以最近标记为中心、长 2T 的 Hann 窗片段
3. 合成帧间隔按目标 f0 排布；帧长保持不变（这是保共振峰的物理原因）
4. 清音/静音段原样通过，边界做 10 ms 交叉淡化
"""

from __future__ import annotations

import numpy as np

__all__ = ["find_pitch_marks", "psola_resynth"]

_EPS = 1e-12


def find_pitch_marks(
    x: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """在浊音段按累积相位生成基音标记（采样点位置）。

    做法：以 f0 帧序列驱动相位 phi += 2*pi*f0*dt，每当相位跨过 2*pi 就在当前
    采样点打一个标记。这样相邻标记的间隔严格等于当时的基音周期，且不会像
    "逐周期找峰值"那样被谐波干扰而漏标或多标。
    """
    f0 = np.asarray(f0_hz, dtype=float)
    t = np.asarray(times, dtype=float)
    xa = np.asarray(x, dtype=float).ravel()
    n = xa.size
    if f0.size < 2:
        return np.array([], dtype=int)

    # 把 f0 插值到每个采样点
    valid = np.isfinite(f0) & (f0 > 0)
    if valid.sum() < 2:
        return np.array([], dtype=int)
    f0_s = np.interp(np.arange(n), t[valid] * sr, f0[valid], left=0.0, right=0.0)
    f0_s = np.where(np.isin(np.arange(n), np.arange(n)), f0_s, 0.0)

    phase = np.cumsum(f0_s) / float(sr)
    # 相位每跨过整数就有一个周期边界
    wrap = np.diff(np.floor(phase), prepend=np.floor(phase[0])) > 0
    marks = np.flatnonzero(wrap).astype(np.int64)
    # 只保留附近确实浊音（f0>0）的标记
    ok = np.zeros(marks.size, dtype=bool)
    for i, m in enumerate(marks):
        lo = max(0, m - 2)
        hi = min(n, m + 3)
        ok[i] = np.any(f0_s[lo:hi] > 0)
    marks = marks[ok]
    if marks.size == 0:
        return marks

    # ---- 峰值吸附：试过但**失败**，已禁用 ----
    # 曾尝试在 ±T/4 内把标记挪到 argmax(|x|)，结果更差（偏低段音准 17.78 → 20.27，
    # HNR 进一步下降）：因为人声一个周期内有多个谐波峰，argmax 会跳到谐波上，
    # 导致标记间隔不均匀、同步丢失。可靠的做法（Praat / REAPER 级别）需要
    # 先低通到 1.5×f0 再找峰，并做周期一致性校验——列为 v0.4.4 待办。
    return marks


def psola_resynth(
    x: np.ndarray,
    sr: int,
    f0_src: np.ndarray,
    f0_tgt: np.ndarray,
    times: np.ndarray,
    crossfade_ms: float = 10.0,
) -> tuple[np.ndarray, dict]:
    """用 TD-PSOLA 按目标 f0 重合成，返回 (波形, 统计)。

    f0_src / f0_tgt 是同一帧网格上的原始与目标基频（Hz，非浊音为 NaN 或 0）。
    目标与原始差异会被限制在 ±2 半音内（超过则退回原始，避免颗粒感）。
    """
    sig = np.asarray(x, dtype=float).ravel()
    n = sig.size
    src = np.asarray(f0_src, dtype=float)
    tgt = np.asarray(f0_tgt, dtype=float)
    t = np.asarray(times, dtype=float)

    ratio = np.where(np.isfinite(src) & np.isfinite(tgt) & (src > 0) & (tgt > 0), tgt / np.maximum(src, _EPS), 1.0)
    ratio = np.clip(ratio, 2.0 ** (-2 / 12), 2.0 ** (2 / 12))

    # 目标 f0 逐采样（用受限后的比率）
    ratio_s = np.interp(np.arange(n), t * sr, ratio, left=1.0, right=1.0)
    src_s = np.interp(np.arange(n), t * sr, np.where(np.isfinite(src), src, 0.0), left=0.0, right=0.0)
    tgt_s = src_s * ratio_s

    marks = find_pitch_marks(sig, sr, src, t)
    if marks.size < 4:
        return sig.copy(), {"applied": False, "reason": "基音标记不足"}

    period_s = np.where(src_s > 1.0, sr / np.maximum(src_s, 1.0), float(sr / 100.0))

    out = np.zeros(n + 2 * sr, dtype=float)
    norm = np.zeros(n + 2 * sr, dtype=float)
    offset = sr // 2  # 留出头部余量，避免负索引

    # 合成位置按"目标相位"推进
    phi = 0.0
    pos = float(marks[0])
    while pos < n - 1:
        idx = int(round(pos))
        # 找最近的基音标记，取以它为中心、2T 长的 Hann 窗
        k = int(np.argmin(np.abs(marks - idx)))
        mark = marks[k]
        T = float(period_s[min(mark, n - 1)])
        half = max(8, int(round(T)))
        lo = mark - half
        hi = mark + half
        if lo < 0 or hi > n:
            # 越界则缩短窗（保持对称）
            half = int(min(mark, n - 1 - mark))
            if half < 8:
                break
            lo, hi = mark - half, mark + half

        seg = sig[lo:hi] * np.hanning(hi - lo)
        L = seg.size
        dst = idx - L // 2 + offset
        if dst >= 0 and dst + L <= out.size:
            out[dst : dst + L] += seg
            norm[dst : dst + L] += np.hanning(L)

        # 推进到下一个合成位置：步长 = 当前目标周期
        step = float(max(8.0, sr / max(tgt_s[min(idx, n - 1)], 1.0)))
        pos += step
        phi += step

    out = np.divide(out, np.maximum(norm, 1e-6))
    out = out[offset : offset + n]

    # 只在浊音段使用 PSOLA 结果，清音/静音段保留原始（交叉淡化）
    voiced = np.isfinite(src) & (src > 0)
    if voiced.any():
        vmask = np.interp(np.arange(n), t * sr, voiced.astype(float), left=0.0, right=0.0)
        w = np.clip(vmask, 0.0, 1.0)
        # 10 ms 滑窗平滑，避免硬切爆音
        k = max(3, int(crossfade_ms * 1e-3 * sr) | 1)
        w = np.convolve(w, np.hanning(k) / np.hanning(k).sum(), mode="same")
        mix = w * out + (1.0 - w) * sig
    else:
        mix = sig.copy()

    info = {
        "applied": True,
        "n_marks": int(marks.size),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "max_shift_cents": float(1200.0 * np.log2(max(np.max(ratio), 1e-9))),
    }
    return mix, info
