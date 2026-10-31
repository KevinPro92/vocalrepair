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
    """在浊音段生成基音标记（采样点位置）—— Praat 式做法。

    为什么必须"先低通再找峰"（这是上一版失败的原因）
    ----------------------------------------------
    直接在原始波形上找 ``argmax(|x|)`` 会锁到**谐波峰**：人声一个周期内有
    F1/F2/F3 好几个峰，能量最大的往往不是基音脉冲。实测那样做音准从 17.78
    恶化到 20.27 cents，谱平坦度进一步上升。

    Praat 的做法是先按 f0 把信号低通到 1.5×f0（谐波被削掉，只剩一个每周期
    一个的基音峰），再在这个"准正弦"上找峰。步骤：

    1. 用 f0 的中位数决定低通截止（1.5×f0_med，上限 0.45·sr）
    2. 零相位低通（filtfilt，避免相位延迟导致标记整体偏移）
    3. 在低通信号上找**正向局部极大** → 每周期一个
    4. 周期一致性校验：间隔落在 [0.6T, 1.6T] 之外的峰视为漏检/误检，用相位
       轨迹插值补齐
    5. 最后按时间排序并保证严格单调
    """
    f0 = np.asarray(f0_hz, dtype=float)
    t = np.asarray(times, dtype=float)
    xa = np.asarray(x, dtype=float).ravel()
    n = xa.size
    if f0.size < 2 or n < 64:
        return np.array([], dtype=int)

    valid = np.isfinite(f0) & (f0 > 0)
    if valid.sum() < 4:
        return np.array([], dtype=int)
    f0_med = float(np.median(f0[valid]))

    # --- 1) 零相位低通到 1.5 * f0 ---
    from scipy.signal import butter, filtfilt

    cutoff = min(1.5 * f0_med, 0.45 * sr)
    if cutoff <= 0:
        return np.array([], dtype=int)
    try:
        b, a = butter(4, cutoff / (sr / 2.0), btype="low")
        xl = filtfilt(b, a, xa)
    except Exception:
        xl = xa

    # --- 2) 正向局部极大 ---
    cand = np.flatnonzero((xl[1:-1] > xl[:-2]) & (xl[1:-1] >= xl[2:])) + 1
    if cand.size < 2:
        return np.array([], dtype=int)

    # --- 3) 周期一致性校验 + 补齐 ---
    # 逐采样 f0 与周期
    f0_s = np.interp(np.arange(n), t * sr, np.where(valid, f0, 0.0), left=0.0, right=0.0)
    T_s = np.where(f0_s > 1.0, sr / np.maximum(f0_s, 1.0), np.inf)

    keep = [int(cand[0])]
    for p in cand[1:]:
        prev = keep[-1]
        gap = p - prev
        T_here = T_s[min(prev, n - 1)]
        if not np.isfinite(T_here):
            # 清音/静音：直接接受（后续不在浊音段使用）
            keep.append(int(p))
            continue
        # 只丢弃"太密"的（< 0.45T，通常是谐波峰）；"太稀"的**保留**，交给下面的
        # 补齐逻辑等距插值填满。
        # 曾经用 [0.6T, 1.6T] 双向卡：在真实 KTV 素材上，后半段的峰间隔不落在这个
        # 窗口里，导致 **15 秒之后一个 mark 都不剩**（marks/s 从 247 掉到 0），
        # PSOLA 只能靠步长硬推 → 音高正确但音色退化成笛子/橡胶声。
        if gap >= 0.45 * T_here:
            keep.append(int(p))

    marks = np.asarray(keep, dtype=np.int64)

    # 补齐：浊音段内相邻标记间隔 > 1.6T 时，等距插入
    fixed: list[int] = []
    for i, m in enumerate(marks):
        if i == 0:
            fixed.append(int(m))
            continue
        prev = fixed[-1]
        T_here = T_s[min(prev, n - 1)]
        if np.isfinite(T_here) and T_here > 4:
            gap = m - prev
            n_missing = int(round(gap / T_here)) - 1
            if 1 <= n_missing <= 8:
                for j in range(1, n_missing + 1):
                    fixed.append(int(round(prev + j * gap / (n_missing + 1))))
        fixed.append(int(m))

    out = np.asarray(fixed, dtype=np.int64)
    out = out[(out > 0) & (out < n)]
    if out.size:
        out = np.unique(out)
    return out



def psola_resynth(
    x: np.ndarray,
    sr: int,
    f0_src: np.ndarray,
    f0_tgt: np.ndarray,
    times: np.ndarray,
    crossfade_ms: float = 10.0,
    min_ratio_change: float = 0.0015,
    edge_guard_ms: float = 15.0,
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

    # ---- 合成循环 ----
    # 两个曾经的致命 bug（用户实测"一闪一闪亮晶晶"整字失声）：
    #  1) 步长取了 sr / tgt_f0，而清音段 tgt_f0 = 0 → 步长 = sr 个采样 = 1 秒，
    #     一步就跳过整个字（辅音正是清音）。现在清音段固定用浊音中位周期推进。
    #  2) 边界处 `break` 会直接终止整个循环，尾部整段丢失。现在只跳过该帧。
    finite_periods = period_s[np.isfinite(period_s) & (period_s > 4)]
    period_med = float(np.median(finite_periods)) if finite_periods.size else sr / 200.0

    pos = float(marks[0])
    guard = 0
    max_iter = 4 * n  # 防御性上限，避免任何情况下死循环
    while pos < n - 1 and guard < max_iter:
        guard += 1
        idx = int(round(pos))

        # 二分查找最近的基音标记（O(log N)，原实现是 O(N) 的全表 argmin）
        k = int(np.searchsorted(marks, idx))
        k = min(max(k, 0), marks.size - 1)
        if k > 0 and abs(int(marks[k - 1]) - idx) < abs(int(marks[k]) - idx):
            k -= 1
        mark = int(marks[k])

        T = float(period_s[min(mark, n - 1)])
        if not np.isfinite(T) or T < 4.0:
            T = period_med
        half = int(round(T))
        lo = max(0, mark - half)
        hi = min(n, mark + half)
        if hi - lo >= 8:
            seg = sig[lo:hi] * np.hanning(hi - lo)
            L = seg.size
            dst = idx - L // 2 + offset
            if dst >= 0 and dst + L <= out.size:
                out[dst : dst + L] += seg
                norm[dst : dst + L] += np.hanning(L)

        # 推进：清音/静音段绝不跳整秒
        tgt_now = float(tgt_s[min(idx, n - 1)])
        step = sr / tgt_now if tgt_now > 20.0 else period_med
        step = float(np.clip(step, sr / 1000.0, sr / 40.0))
        pos += step

    out = np.divide(out, np.maximum(norm, 1e-6))
    out = out[offset : offset + n]

    # 只在浊音段使用 PSOLA 结果，清音/静音段保留原始（交叉淡化）
    voiced = np.isfinite(src) & (src > 0)
    if voiced.any():
        vmask = np.interp(np.arange(n), t * sr, voiced.astype(float), left=0.0, right=0.0)
        # 只在"确实需要改动"的区域使用 PSOLA 输出：ratio ≈ 1 的地方保留原始波形。
        # 依据是用户听感——"none（完全不动）音质最好"，那就把改动面积压到最小，
        # 避免对死区内（本来就不用修）的帧做无谓的拼接。
        need = np.abs(ratio - 1.0) > float(min_ratio_change)
        need_s = np.interp(np.arange(n), t * sr, need.astype(float), left=0.0, right=0.0)
        w = np.clip(vmask * need_s, 0.0, 1.0)
        # ---- note-boundary guard (v0.4.9) ----
        # The frames with the largest frame-wise spectral difference are clustered at
        # note junctions, and crest factor rises there (measured 8.70 s: 2.19 -> 2.57,
        # +17%). That is the perceived "too sharp / abrupt" artefact.
        # Fix: within +-15 ms of a correction jump, do not splice at all - keep the
        # original waveform. A junction is a transition region anyway, so leaving its
        # pitch uncorrected is inaudible, while the splice artefact disappears.
        if edge_guard_ms > 0:
            # NOTE: must test the FRAME-level ratio, not ratio_s. ratio_s is
            # interpolated per sample, so its first difference is always tiny and
            # the guard would silently never fire (verified: identical output).
            dr = np.abs(np.diff(ratio, prepend=ratio[0]))
            edge_f = (dr > 0.0015).astype(float)
            edge = np.interp(np.arange(n), t * sr, edge_f, left=0.0, right=0.0)
            kk = max(3, int(edge_guard_ms * 1e-3 * sr) | 1)
            edge = np.convolve(edge, np.hanning(kk) / np.hanning(kk).sum(), mode="same")
            w = w * (edge < 0.5)
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
