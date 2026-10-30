"""参考音频对齐：DTW 对齐音高走向 + LTAS 音色匹配。

两个必须遵守的工程约束（来自文献与实践）
--------------------------------------
1. **DTW 必须加 Sakoe-Chiba 带约束**。无约束 DTW 在节奏差异大时会给出
   "把所有帧对齐到同一帧"之类荒谬路径（Mueller《Fundamentals of Music
   Processing》的经典提醒）。带宽取 max(N,M) 的 25%。
2. **不能把参考的颤音硬搬过来**。对齐只用于确定"这一段应该唱到哪个音高"，
   而颤音是演唱者自己的表情，必须保留本地残差。否则会得到"听起来像别人
   在唱你的歌"的效果。

流程
----
    src cents ──┐
                ├─ DTW(带约束) ─ 融合目标 = α·参考 + (1-α)·本地量化目标
    ref cents ──┘                    └─ 加回本地颤音残差 → 新 f0

置信度：对齐后剩余的平均绝对音分差。> 150 cents 说明两条音高轨迹本质不同
（例如拿一首完全不同的歌当参考），此时**降级为纯本地量化**并如实报告。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DtwResult", "dtw_path", "align_pitch_tracks", "reference_ltas"]

_EPS = 1e-12

INF = float("inf")


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    """一维中值滤波（保边，用于剥离颤音调制）。"""
    import warnings

    k = max(3, int(k) + (1 - int(k) % 2))
    half = k // 2
    padded = np.pad(np.asarray(x, dtype=float), half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


def _vibrato_window_frames(n_frames: int, dt_s: float = 0.01, vibrato_hz: float = 5.5) -> int:
    """覆盖一个完整颤音周期的中值窗（帧）。窗长不足则退化为 5 帧。"""
    k = int(round(1.0 / max(vibrato_hz * dt_s, 1e-9)))
    return max(5, min(k, max(5, n_frames // 3)))


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """线性插值填充 NaN（用于把"只含浊音"的轨迹铺到完整时间网格）。"""
    arr = np.asarray(x, dtype=float)
    idx = np.arange(arr.size)
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return np.zeros_like(arr)
    return np.interp(idx, idx[valid], arr[valid])


@dataclass
class DtwResult:
    path: np.ndarray      # (L, 2)，每行 (i, j)
    cost: float           # 归一化总代价
    band_radius: int
    is_reliable: bool
    median_residual_cents: float

    def to_dict(self) -> dict:
        return {
            "path_length": int(self.path.shape[0]),
            "normalized_cost": round(float(self.cost), 2),
            "band_radius": int(self.band_radius),
            "median_residual_cents": round(float(self.median_residual_cents), 1),
            "is_reliable": bool(self.is_reliable),
        }


def dtw_path(
    a: np.ndarray,
    b: np.ndarray,
    radius_ratio: float = 0.10,
    max_residual_cents: float = 150.0,
) -> DtwResult:
    """带 Sakoe-Chiba 带的 DTW，返回对齐路径与置信度。

    a, b 都是一维序列（这里用音分轨迹）。代价为 |a_i - b_j|。

    radius_ratio 为什么取 0.10 而不是更宽
    -------------------------------------
    带宽越宽，DTW 越能在音高轴上"滑移"：实测 0.25（约 80 帧 ≈ 0.8 s，
    超过一个音符的时长）时，src 的某个音会被匹配到参考同音符内**颤音的
    低端**，导致目标音高偏低 20-45 cents。0.10 约等于 0.26 s，小于常见
    音符时长，既能吸收正常的速度差异，又不允许跳到别的音。
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return DtwResult(np.zeros((0, 2), dtype=int), INF, 0, False, INF)

    radius = max(1, int(round(radius_ratio * max(n, m))))
    # 每行的允许列范围（Sakoe-Chiba 带）
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    scale = m / float(n)

    for i in range(1, n + 1):
        center = i * scale
        j_lo = max(1, int(np.floor(center - radius)))
        j_hi = min(m, int(np.ceil(center + radius)))
        diag = x[i - 1]
        for j in range(j_lo, j_hi + 1):
            cost = abs(diag - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    if not np.isfinite(D[n, m]):
        return DtwResult(np.zeros((0, 2), dtype=int), INF, radius, False, INF)

    # 回溯
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev = min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        if prev == D[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif prev == D[i - 1, j]:
            i -= 1
        else:
            j -= 1
    path.reverse()
    p = np.array(path, dtype=int)

    residual = float(np.median(np.abs(x[p[:, 0]] - y[p[:, 1]]))) if p.size else INF
    return DtwResult(
        path=p,
        cost=float(D[n, m]) / max(1, n),
        band_radius=radius,
        is_reliable=bool(np.isfinite(residual) and residual <= max_residual_cents),
        median_residual_cents=residual,
    )


def align_pitch_tracks(
    src_f0: np.ndarray,
    src_voiced: np.ndarray,
    ref_f0: np.ndarray,
    ref_voiced: np.ndarray,
    alpha: float = 0.6,
    radius_ratio: float = 0.25,
    max_residual_cents: float = 150.0,
) -> tuple[np.ndarray, dict]:
    """把源音高轨迹对齐到参考的音高走向。

    返回 (aligned_cents, info)：
    - aligned_cents：与 src_f0 等长的音分轨迹（非浊音为 NaN），
      已经是 α·参考 + (1-α)·源 的融合结果
    - info：DTW 置信度与是否降级

    注意融合对象是**中心音高**而不是瞬时音高：颤音由调用方在合成前加回，
    这样参考的颤音不会污染本地表达。
    """
    src = np.asarray(src_f0, dtype=float)
    ref = np.asarray(ref_f0, dtype=float)
    sv = np.asarray(src_voiced, dtype=bool) & np.isfinite(src) & (src > 0)
    rv = np.asarray(ref_voiced, dtype=bool) & np.isfinite(ref) & (ref > 0)

    out = np.full(src.size, np.nan)
    info: dict = {"enabled": True, "alpha": float(alpha), "degraded": False}

    if sv.sum() < 5 or rv.sum() < 5:
        info.update({"degraded": True, "reason": "浊音帧不足，跳过参考对齐"})
        return out, info

    with np.errstate(divide="ignore", invalid="ignore"):
        src_full = 1200.0 * np.log2(np.maximum(np.where(sv, src, np.nan), _EPS) / 440.0)
        ref_full = 1200.0 * np.log2(np.maximum(np.where(rv, ref, np.nan), _EPS) / 440.0)

    # 在**完整时间网格**上做 DTW（非浊音帧用插值填充），而不是在"浊音帧序列"上。
    # 原因：两侧静音段的浊音帧数不同（换气长度、门限差异），若只拿浊音帧序列
    # 对齐，时间轴会整体错位——实测即使两条序列结构完全相同，残差也有 20 cents。
    src_full = _interp_nan(src_full)
    ref_full = _interp_nan(ref_full)

    # 先剥掉颤音再对齐：两侧颤音的相位互不相关，若拿瞬时音高做 DTW，
    # 算法会去对齐彼此的颤音峰值（实测残差被压到 19.7 cents，掩盖了真实的
    # 40 cents 音准差）。对齐"音高走向"必须用中心轨迹。
    src_center_full = _median_filter_1d(src_full, _vibrato_window_frames(src_full.size))
    ref_center_full = _median_filter_1d(ref_full, _vibrato_window_frames(ref_full.size))

    dtw = dtw_path(
        src_center_full,
        ref_center_full,
        radius_ratio=radius_ratio,
        max_residual_cents=max_residual_cents,
    )
    info["dtw"] = dtw.to_dict()

    if not dtw.is_reliable or dtw.path.size == 0:
        info.update(
            {
                "degraded": True,
                "reason": (
                    f"DTW 残差 {dtw.median_residual_cents:.0f} cents 超过阈值 "
                    f"{max_residual_cents:.0f}，两条音高轨迹本质不同；已降级为本地量化"
                ),
            }
        )
        return out, info

    # DTW 路径已在完整时间网格上，直接把参考中心音高映射到源帧号
    ref_for_src = np.full(src.size, np.nan)
    ref_for_src[dtw.path[:, 0]] = ref_center_full[dtw.path[:, 1]]

    # 路径未覆盖的源帧用线性插值补齐（路径连续，通常只缺首尾若干帧）
    if not np.all(np.isfinite(ref_for_src[sv])):
        valid_pos = np.flatnonzero(np.isfinite(ref_for_src))
        if valid_pos.size >= 2:
            idx_all = np.arange(src.size)
            filled = ref_for_src.copy()
            filled[sv] = np.interp(idx_all[sv], valid_pos, ref_for_src[valid_pos])
            ref_for_src = filled

    # 融合：α·参考中心 + (1-α)·**本地量化目标**
    #
    # α 自适应（关键）：当参考与本地量化目标**跨了半个音**（分歧 > 50 cents）
    # 时，说明这个音本身就落在两个半音之间、本地量化不可靠——此时参考恰恰
    # 最有信息量（它告诉你该唱哪个音级），所以把权重提到 0.85。
    # 实测固定 α=0.6 时，一个 −52 cents 的音会被融合成 −40 cents，
    # 相对两个候选音级都不像，中位偏差反而恶化。
    local_target_c = np.round(src_center_full / 100.0) * 100.0
    disagree = np.abs(ref_for_src - local_target_c) > 50.0
    alpha_eff = np.where(disagree, min(0.85, float(alpha) + 0.25), float(alpha))
    fused_center = np.where(
        np.isfinite(ref_for_src),
        alpha_eff * ref_for_src + (1.0 - alpha_eff) * local_target_c,
        src_center_full,
    )

    out[sv] = fused_center[sv]
    delta = (fused_center - src_center_full)[sv]

    info.update(
        {
            "frames_aligned": int(sv.sum()),
            "mean_abs_correction_cents": round(float(np.mean(np.abs(delta))), 2),
            "max_abs_correction_cents": round(float(np.max(np.abs(delta))), 2),
        }
    )
    return out, info


def align_pitch_by_notes(
    src_f0: np.ndarray,
    src_times: np.ndarray,
    src_voiced: np.ndarray,
    ref_f0: np.ndarray,
    ref_times: np.ndarray,
    ref_voiced: np.ndarray,
    alpha: float = 0.6,
    radius_ratio: float = 0.25,
    max_residual_semitones: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """**音符级**对齐：先把两条轨迹切成音符，再对音符序列做 DTW。

    为什么必须做到音符级（v0.4.2 的核心改动）
    ----------------------------------------
    帧级 DTW 已经证明不可靠：它允许路径在音高轴上「滑移」，会把源音匹配到
    参考同一个音符内**颤音的谷底**（实测 B3 的参考被拉到 −1046 而不是 −1000）。
    根因是帧级序列里存在大量"音高相近但属于不同音符"的帧，DTW 无法分辨
    "同一音符的波动"与"相邻音符"。

    切到音符级后，序列变成"每首歌十几到上百个离散音级"，DTW 只能做
    「音符 ↔ 音符」的整数配对，滑移在结构上被消除。

    代价函数用半音差（|Δmidi|），残差阈值 2 个半音：超过说明两条旋律不是
    同一首歌，直接降级为本地量化。
    """
    from ..analysis.notes import segment_notes

    src = np.asarray(src_f0, dtype=float)
    ref = np.asarray(ref_f0, dtype=float)
    src_t = np.asarray(src_times, dtype=float)
    ref_t = np.asarray(ref_times, dtype=float)

    out = np.full(src.size, np.nan)
    info: dict = {"enabled": True, "alpha": float(alpha), "degraded": False, "level": "note"}

    src_notes = segment_notes(src, src_t)
    ref_notes = segment_notes(ref, ref_t)
    info["n_src_notes"] = len(src_notes)
    info["n_ref_notes"] = len(ref_notes)

    if len(src_notes) < 2 or len(ref_notes) < 2:
        info.update({"degraded": True, "reason": "音符数不足（<2），跳过参考对齐"})
        return out, info

    src_midi = np.array([n.midi for n in src_notes], dtype=float)
    ref_midi = np.array([n.midi for n in ref_notes], dtype=float)

    dtw = dtw_path(
        src_midi,
        ref_midi,
        radius_ratio=radius_ratio,
        max_residual_cents=max_residual_semitones,  # 此处的"代价单位"是半音
    )
    info["dtw"] = dtw.to_dict()
    info["dtw"]["residual_unit"] = "semitones"

    if not dtw.is_reliable or dtw.path.size == 0:
        info.update(
            {
                "degraded": True,
                "reason": (
                    f"音符级 DTW 残差 {dtw.median_residual_cents:.2f} 半音超过阈值 "
                    f"{max_residual_semitones:.1f}，两条旋律不是同一首歌；已降级为本地量化"
                ),
            }
        )
        return out, info

    # 每个源音符 -> 匹配到的参考音符（同一参考音符可被多次匹配，属正常）
    src_to_ref: dict[int, int] = {}
    for i, j in dtw.path:
        src_to_ref.setdefault(int(i), int(j))

    applied: list[float] = []
    for i, note in enumerate(src_notes):
        if i not in src_to_ref:
            continue
        ref_note = ref_notes[src_to_ref[i]]
        ref_c = 1200.0 * np.log2(max(ref_note.mean_hz, _EPS) / 440.0)
        src_c = 1200.0 * np.log2(max(note.mean_hz, _EPS) / 440.0)
        local_target_c = round(src_c / 100.0) * 100.0

        # α 自适应：跨半音分歧时（|ref - local| > 50 cents）参考更有信息量
        a_eff = min(0.85, alpha + 0.25) if abs(ref_c - local_target_c) > 50.0 else alpha
        fused_c = a_eff * ref_c + (1.0 - a_eff) * local_target_c

        sel = (src_t >= note.start_s) & (src_t < note.end_s) & np.isfinite(src)
        if not np.any(sel):
            continue
        out[sel] = fused_c
        applied.append(fused_c - src_c)

    if not applied:
        info.update({"degraded": True, "reason": "没有可用的音符配对"})
        return out, info

    info.update(
        {
            "notes_matched": len(applied),
            "mean_abs_correction_cents": round(float(np.mean(np.abs(applied))), 2),
            "max_abs_correction_cents": round(float(np.max(np.abs(applied))), 2),
        }
    )
    return out, info


def reference_ltas(
    x_ref: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """参考音频的长时平均谱（用于 align_timbre 的 conservative 模式）。

    只取幅度谱的时间平均，因此对节奏与时长差异不敏感——这正适合
    "风格对齐"：对齐的是音色指纹，而不是逐帧细节。
    """
    from ..analysis.spectral import ltas
    from ..dsp.stft import stft_mag_phase

    mag, _phase, freqs, _t = stft_mag_phase(np.asarray(x_ref, float), sr, n_fft=n_fft, hop_length=hop_length)
    db, f = ltas(mag, freqs)
    return db, f
