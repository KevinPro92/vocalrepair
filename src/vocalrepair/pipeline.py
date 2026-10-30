"""完整管线：波形 → 分析 → 修复 → (波形, 报告)。

设计约束
--------
1. **不静默降级**：任何一步失败都要体现在 report 的 ``degraded_mode`` 与
   ``warnings`` 字段里，而不是悄悄跳过
2. **前后同尺度可比**：修复前后用同一个 f0 估计器与同一套指标，才能算出
   "偏差中位数从 X 降到 Y"这类可核验的结论
3. **不清音变有声**：f0 修正只作用于原本浊音的帧，非周期帧在 WORLD 里
   保持 f0 = 0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis.formant import track_formants
from .analysis.key import estimate_key
from .analysis.notes import segment_notes
from .analysis.spectral import peak_valley_summary
from .analysis.vibrato import analyze_vibrato
from .analysis.voice_quality import voice_quality_summary
from .dsp.f0 import F0Result, yin_f0
from .dsp.f0_track import robust_f0
from .dsp.octave import find_octave_suspects
from .dsp.pitch_scale import cents_deviation
from .dsp.stft import chroma_from_spectrum, stft_mag_phase

__all__ = ["AnalysisContext", "RepairResult", "analyze", "repair"]

_EPS = 1e-12


@dataclass
class AnalysisContext:
    """一次分析的完整产物（报告字段 + 绘图所需中间量）。"""

    sr: int
    duration_s: float
    f0: F0Result
    mag: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    chroma: np.ndarray
    formants: np.ndarray
    report: dict = field(default_factory=dict)

    @property
    def cents(self) -> np.ndarray:
        return self.f0.cents()

    @property
    def dev_cents(self) -> np.ndarray:
        """相对最近半音的音准偏差（NaN 处保持 NaN）。"""
        d = np.asarray(cents_deviation(np.where(np.isfinite(self.f0.f0_hz), self.f0.f0_hz, np.nan)))
        return d


def analyze(
    x: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    f0_method: str = "robust",
    forced_voiced: np.ndarray | None = None,
) -> AnalysisContext:
    """完整分析：f0、频谱、颤音、嗓音质量、音符表、调性、八度可疑段。

    f0_method
    ---------
    ``"robust"``（默认）：YIN 多候选 + Viterbi 全局路径，抗偶发八度跳变
    ``"yin"``：原始 YIN（更快，但单帧决策不可回溯）

    forced_voiced
    -------------
    外部强制的浊音掩膜（帧级）。用于"复测"阶段：把**输入**的浊音判决作为
    物理约束传给输出分析——输入静音处，输出不可能凭空出现音高。这不是
    掩盖问题，而是排除 WORLD 重建在静音边界产生的微弱周期性伪影。
    """
    sig = np.asarray(x, dtype=float).ravel()
    if f0_method == "yin":
        f0 = yin_f0(sig, sr)
    else:
        f0 = robust_f0(sig, sr)

    if forced_voiced is not None:
        mask = np.zeros(f0.f0_hz.size, dtype=bool)
        n = min(mask.size, int(np.asarray(forced_voiced).size))
        mask[:n] = np.asarray(forced_voiced, dtype=bool)[:n]
        f0.f0_hz = np.where(mask, f0.f0_hz, np.nan)
        f0.voiced = mask & np.isfinite(f0.f0_hz)
    mag, _phase, freqs, stft_times = stft_mag_phase(sig, sr, n_fft=n_fft, hop_length=hop_length)
    chroma = chroma_from_spectrum(mag, freqs)
    formants, _ft = track_formants(sig, sr, voiced=f0.voiced)

    vib = analyze_vibrato(f0.f0_hz, f0.times)
    notes = segment_notes(f0.f0_hz, f0.times)
    voice = voice_quality_summary(sig, sr, f0.f0_hz, mag, freqs)
    spec = peak_valley_summary(mag, freqs)
    key = estimate_key(chroma)
    octaves = find_octave_suspects(f0.f0_hz, f0.times)

    dev = np.asarray(cents_deviation(np.where(np.isfinite(f0.f0_hz), f0.f0_hz, np.nan)))
    voiced_dev = dev[np.isfinite(dev)]

    report = {
        "duration_s": round(sig.size / float(sr), 3),
        "voiced_frames": int(np.count_nonzero(f0.voiced)),
        "voiced_ratio": round(float(np.mean(f0.voiced)) if f0.voiced.size else 0.0, 3),
        "f0_range_hz": _f0_range(f0.f0_hz),
        "median_abs_cents": round(float(np.median(np.abs(voiced_dev))), 2) if voiced_dev.size else 0.0,
        "p95_abs_cents": round(float(np.percentile(np.abs(voiced_dev), 95)), 2) if voiced_dev.size else 0.0,
        "spectrum": spec,
        "vibrato": vib.summary(),
        "voice_quality": voice,
        "key_estimate": key,
        "n_notes": len(notes),
        "notes": [n.to_dict() for n in notes],
        "octave_suspects": [s.to_dict() for s in octaves],
    }
    return AnalysisContext(
        sr=int(sr),
        duration_s=sig.size / float(sr),
        f0=f0,
        mag=mag,
        freqs=freqs,
        times=stft_times,
        chroma=chroma,
        formants=formants,
        report=report,
    )


def _f0_range(f0: np.ndarray) -> dict:
    v = np.asarray(f0, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return {"min_hz": None, "max_hz": None}
    from .dsp.pitch_scale import hz_to_name

    return {
        "min_hz": round(float(np.percentile(v, 5)), 1),
        "max_hz": round(float(np.percentile(v, 95)), 1),
        "min_name": hz_to_name(float(np.percentile(v, 5))),
        "max_name": hz_to_name(float(np.percentile(v, 95))),
    }


def _to_world_grid(
    f0: np.ndarray,
    times: np.ndarray,
    n_frames: int,
    frame_period: float,
) -> np.ndarray:
    """把 10 ms 网格上的 f0 插值到 WORLD 的 5 ms 网格；有效区间之外填 0。"""
    t_w = np.arange(n_frames) * frame_period / 1000.0
    f = np.asarray(f0, dtype=float)
    t = np.asarray(times, dtype=float)
    valid = np.isfinite(f) & (f > 0)
    out = np.zeros(n_frames, dtype=np.float64)
    if valid.sum() >= 2:
        out = np.interp(t_w, t[valid], f[valid])
        outside = (t_w < t[valid].min()) | (t_w > t[valid].max())
        out[outside] = 0.0
    return out


def _octave_guard(
    before_f0: np.ndarray,
    after_f0: np.ndarray,
    expected_shift_semitones: float = 0.0,
    tol_cents: float = 600.0,
) -> dict:
    """输出侧八度守护：输入与输出的 f0 中位数不应相差"近一个八度"。

    为什么必须单独检查
    ------------------
    音准指标（相对最近半音的偏差）对整体八度错位**完全不敏感**：
    838 Hz 相对 G#5 只差 15 cents，看起来"很准"，实际整体错了一个八度。
    实测本项目就发生过一次（见 decision_log 教训 1）。

    判据：actual_shift = 1200·log2(median_after / median_before)，减去预期的
    变调量后，残差若超过 600 cents，说明差得接近半个八度以上——正常修音
    永远不会产生这么大的整体位移（死区 16 + 单段上限 100 cents）。
    """
    b = np.asarray(before_f0, dtype=float)
    a = np.asarray(after_f0, dtype=float)
    b = b[np.isfinite(b) & (b > 0)]
    a = a[np.isfinite(a) & (a > 0)]
    if b.size == 0 or a.size == 0:
        return {"checked": False, "is_consistent": True, "note": "无可用的浊音帧"}

    med_b = float(np.median(b))
    med_a = float(np.median(a))
    actual = 1200.0 * np.log2(med_a / med_b)
    expected = float(expected_shift_semitones) * 100.0
    residual = actual - expected
    ok = abs(residual) <= tol_cents
    return {
        "checked": True,
        "before_median_hz": round(med_b, 2),
        "after_median_hz": round(med_a, 2),
        "actual_shift_cents": round(actual, 1),
        "expected_shift_cents": round(expected, 1),
        "residual_cents": round(residual, 1),
        "is_consistent": bool(ok),
        "note": "ok" if ok else "残差接近 ±1200 cents：疑似整体八度错位，已触发安全回退",
    }


def _timbre_guard(before: AnalysisContext, after: AnalysisContext) -> dict:
    """音色守护：以**用户自己的音色**为基准，确认处理后没有漂走。

    指标是长时平均谱（LTAS）的皮尔逊相关系数。LTAS 是整首歌的平均音色指纹，
    相关系数高说明"还是这个人"。这也是本项目"默认以自己的音色为基准"的
    可核验证据——修复只改 f0 与包络的保守增益，不动谐波结构与声道特征。
    """
    from .analysis.spectral import ltas

    db_b, f = ltas(before.mag, before.freqs)
    db_a, _ = ltas(after.mag, after.freqs)
    band = (f >= 100) & (f <= 8000)
    if int(band.sum()) < 10:
        return {"checked": False, "is_preserved": True}
    corr = float(np.corrcoef(db_b[band], db_a[band])[0, 1])
    return {
        "checked": True,
        "ltas_correlation": round(corr, 4),
        "is_preserved": bool(corr >= 0.98),
        "note": "以输入自身音色为基准；相关系数 < 0.98 表示音色发生明显偏离",
    }


def _apply_reference_target(
    plan,
    target_cents: np.ndarray,
    strength: float = 0.7,
    deadzone_cents: float = 16.0,
    residual_cents: float = 8.0,
    max_cents: float = 100.0,
    max_semitones: float = 5.0,
):
    """用"参考对齐后的目标音高"替换本地量化目标，逐帧重算修正量。

    为什么单独实现而不改 plan_correction：本地量化是**段级**决策（整段一个
    目标音），而参考对齐是**帧级**的（每一帧都有自己的参考音高）。两种粒度
    混在一起会让代码难读，所以这里显式地做帧级重算，但完整沿用同样的
    死区 / 残差 / 强度 / 上限四道闸门。

    颤音残差（plan.vibrato_cents）原样加回：参考只提供"唱到哪个音"，
    颤音仍是演唱者自己的表情。
    """
    from dataclasses import replace

    center = np.asarray(plan.center_cents, dtype=float)
    resid = np.asarray(plan.vibrato_cents, dtype=float)
    tgt = np.asarray(target_cents, dtype=float)

    # 注意：这里**不做**"融合目标与本地目标差 > 50 cents 就回退"的一致性保护。
    # 实测那是错的：跨半音的分歧恰恰是参考最有信息量的情形（决定唱 C# 还是 D），
    # 回退会把参考的价值完全抹掉。真正需要保护的是 DTW 不可靠，那由
    # align_pitch_tracks 的 is_reliable 判定负责。

    with np.errstate(invalid="ignore"):
        err = tgt - center
        inside = np.abs(err) <= deadzone_cents
        applied = np.where(inside, 0.0, err - np.sign(err) * residual_cents)
    applied = np.where(np.isfinite(applied), applied, 0.0)
    applied = np.clip(applied, -max_cents, max_cents)
    applied = applied * float(np.clip(strength, 0.0, 1.0))
    applied = np.clip(applied, -float(max_semitones) * 100.0, float(max_semitones) * 100.0)

    out_cents = center + applied + resid
    corrected = 440.0 * np.power(2.0, out_cents / 1200.0)
    valid = np.isfinite(plan.f0_hz) & (plan.f0_hz > 0)
    corrected = np.where(valid & np.isfinite(out_cents), corrected, np.nan)

    return replace(
        plan,
        corrected_hz=corrected,
        target_hz=440.0 * np.power(2.0, np.where(np.isfinite(tgt), tgt, np.nan) / 1200.0),
        delta_cents=np.where(valid, applied, np.nan),
        notes=[
            {
                "source": "reference_dtw",
                "frames": int(np.count_nonzero(valid)),
                "median_applied_cents": round(float(np.median(np.abs(applied[valid]))), 2)
                if np.any(valid)
                else 0.0,
                "frames_outside_deadzone": int(np.count_nonzero(~inside & valid)),
            }
        ],
    )


@dataclass
class RepairResult:
    """修复产物：波形 + 报告 + 前后分析上下文（供绘图与 A/B 对比）。"""

    y: np.ndarray
    sample_rate: int
    report: dict
    before: AnalysisContext
    after: AnalysisContext
    target_f0: np.ndarray | None = None   # 每帧目标音高（与 before.f0.times 对齐）


def repair(
    x: np.ndarray,
    sr: int,
    strength: float = 0.7,
    transpose: float = 0.0,
    root_pc: int | None = None,
    scale: str = "major",
    clip_repair: bool = True,
    timbre: bool = True,
    timbre_match: bool = True,
    do_compress: bool = True,
    ceiling_db: float = -1.0,
    frame_period: float = 5.0,
    max_semitones: float = 5.0,
    formant_shift: float | None = None,
    fix_octave: bool = False,
    vocoder: str = "world",
    reference_audio: np.ndarray | None = None,
    reference_sr: int | None = None,
    align_pitch: bool = False,
    align_timbre: str = "off",
    reference_alpha: float = 0.6,
) -> RepairResult:
    """完整修音管线。

    顺序：削波修复 → f0 估计 → 音准修正 → WORLD 重合成（含变调/音色）
          → 压缩与软限幅 → 复测
    """
    from .repair.declip import ClippingReport, repair_clipping
    from .repair.pitch_correct import plan_correction
    from .repair.postproc import (
        _apply_silence_gate,
        compress,
        enhance_envelope,
        match_ltas,
        soft_limit,
    )
    from .repair.world_vocoder import shift_formants, world_analyze, world_synthesize

    sig = np.asarray(x, dtype=float).ravel()
    warnings_list: list[str] = []

    if vocoder.lower() != "world":
        warnings_list.append(
            f"vocoder={vocoder} 未实现，已回退到 world（保共振峰）；"
            "phase/neural 路径需额外实现，报告中标为降级"
        )

    # --- 1. 削波修复 ---
    if clip_repair:
        clean, clip_rep = repair_clipping(sig, sr)
    else:
        clean = sig.copy()
        clip_rep = ClippingReport(before_samples=int(np.count_nonzero(np.abs(sig) >= 0.98)))

    # --- 2. 分析（音准修正的基线）---
    before = analyze(clean, sr)

    # --- 3. 可选：八度错误修正（默认只报告，不修改）---
    f0_base = before.f0.f0_hz
    if fix_octave and before.report["octave_suspects"]:
        from .dsp.octave import fix_octave_errors

        f0_base = fix_octave_errors(f0_base, before.f0.times)
        warnings_list.append("已按 octave_suspects 修正疑似八度错误（可能误改真实八度跳进）")

    # --- 4. 音准修正决策 ---
    plan = plan_correction(f0_base, before.f0.times, root_pc=root_pc, scale=scale, strength=strength)

    # --- 4b. 参考音频对齐（加分项 A）---
    reference_info: dict = {"enabled": False}
    ref_ltas_db: np.ndarray | None = None
    ref_ltas_freqs: np.ndarray | None = None

    if reference_audio is not None and (align_pitch or align_timbre != "off"):
        from .audio_io import resample_audio
        from .repair.reference import align_pitch_by_notes, reference_ltas

        ref_sig = np.asarray(reference_audio, dtype=float).ravel()
        ref_sr = int(reference_sr or sr)
        if ref_sr != sr:
            ref_sig, _ = resample_audio(ref_sig, ref_sr, sr)
            warnings_list.append(f"参考音频已从 {ref_sr} Hz 重采样到 {sr} Hz")

        reference_info = {
            "enabled": True,
            "align_pitch": False,
            "align_timbre": align_timbre,
            "experimental": True,
            "note": (
                "参考对齐链路已实现（带 Sakoe-Chiba 约束的 DTW + 自适应融合 + 降级保护）。"
                "但在合成基准上，其端到端音准指标尚未优于纯本地量化：中间量（对齐后的"
                "参考音高）已接近标准音高，损失主要发生在'融合目标 → 逐帧修正 → WORLD "
                "重合成'这一段。建议 alpha <= 0.5，或仅把它用于观察参考的音高走向，"
                "不要当作音准提升手段。该结论已记入 decision_log，属 v0.4.2 待改进项。"
            ),
        }

        if align_pitch:
            ref_f0 = robust_f0(ref_sig, sr)
            # 音符级对齐（v0.4.2）：先把两条轨迹切成音符再做 DTW，
            # 从结构上消除帧级路径在音高轴上的滑移。
            aligned_cents, ref_info = align_pitch_by_notes(
                before.f0.f0_hz,
                before.f0.times,
                before.f0.voiced,
                ref_f0.f0_hz,
                ref_f0.times,
                ref_f0.voiced,
                alpha=reference_alpha,
            )
            reference_info["align_pitch"] = ref_info
            if ref_info.get("degraded"):
                warnings_list.append(
                    f"参考音高对齐已降级：{ref_info.get('reason', 'DTW 置信度不足')}"
                )
            else:
                plan = _apply_reference_target(plan, aligned_cents, strength=strength)

        if align_timbre in ("conservative", "aggressive"):
            ref_ltas_db, ref_ltas_freqs = reference_ltas(ref_sig, sr)

    # --- 5. WORLD 分解（用原始 f0，sp/ap 描述的是"内容"）---
    feat = world_analyze(clean, sr, f0_hz=before.f0.f0_hz, times=before.f0.times, frame_period=frame_period)

    applied_st = float(np.clip(transpose, -max_semitones, max_semitones))
    if abs(applied_st - float(transpose)) > 1e-9:
        warnings_list.append(
            f"transpose {transpose:+.1f} 半音超出 ±{max_semitones} 限制，已截断为 {applied_st:+.1f}"
        )

    # --- 6. 构造新的 f0 轨迹（音准修正 + 变调）---
    f0_new = _to_world_grid(plan.corrected_hz, before.f0.times, feat.f0.shape[0], frame_period)
    if abs(applied_st) > 1e-9:
        f0_new = np.where(f0_new > 0, f0_new * (2.0 ** (applied_st / 12.0)), 0.0)

    # --- 7. 音色塑形（在包络上做，不动谐波与相位）---
    sp_new = feat.sp
    timbre_info: dict = {"applied": False}
    if timbre:
        sp_new, timbre_info = enhance_envelope(feat.sp, feat.freqs, strength=strength)
        timbre_info["applied"] = True
    if formant_shift is not None and abs(float(formant_shift) - 1.0) > 1e-9:
        sp_new = shift_formants(sp_new, float(formant_shift))
        timbre_info["formant_shift_ratio"] = round(float(formant_shift), 4)

    # --- 8. 重合成 ---
    y = world_synthesize(feat, f0_new=f0_new, sp_new=sp_new)
    if y.size > sig.size:
        y = y[: sig.size]
    elif y.size < sig.size:
        y = np.pad(y, (0, sig.size - y.size))

    # --- 9. 音色回归：以输入自身的 LTAS 为基准（消除 WORLD 重合成的整体漂移）---
    ltas_info: dict = {"applied": False}
    if timbre_match:
        from .analysis.spectral import ltas

        ref_db, ref_f = ltas(before.mag, before.freqs)
        max_db = 2.0
        if ref_ltas_db is not None and ref_ltas_freqs is not None:
            # 参考音色对齐：把目标从"自己的 LTAS"混向"参考的 LTAS"
            w = 0.5 if align_timbre == "conservative" else 1.0
            max_db = 3.0 if align_timbre == "conservative" else 6.0
            ref_interp = np.interp(before.freqs, ref_ltas_freqs, ref_ltas_db)
            ref_db = (1.0 - w) * ref_db + w * ref_interp
            ltas_info["reference_weight"] = w
        y, ltas_info = match_ltas(y, sr, ref_db, before.freqs, max_db=max_db)

    # --- 9b. 动态处理 ---
    if do_compress:
        y = compress(y, sr)
    y = soft_limit(y, ceiling_db=ceiling_db)
    # 门控用**原始输入**的能量（而不是 declip 后的 clean）：削波插值会在静音
    # 边界制造微小凸起，用 clean 做门控会漏掉，导致边界帧被判成浊音。
    y = _apply_silence_gate(y, sig, sr)

    # --- 10. 复测 + 闭环守护 ---
    after = analyze(y, sr, forced_voiced=before.f0.voiced)
    guard = _octave_guard(before.f0.f0_hz, after.f0.f0_hz, applied_st)

    # 八度错位时的安全回退：宁可"不修音准"，也不能输出整体差八度的结果
    if not guard["is_consistent"]:
        warnings_list.append(
            f"输出与输入的音高中位数相差 {guard['residual_cents']:+.0f} cents"
            f"（预期 {guard['expected_shift_cents']:+.0f}），疑似整体八度错位；"
            "已回退为安全模式：保留削波修复/变调/音色，放弃音准修正"
        )
        f0_safe = _to_world_grid(before.f0.f0_hz, before.f0.times, feat.f0.shape[0], frame_period)
        if abs(applied_st) > 1e-9:
            f0_safe = np.where(f0_safe > 0, f0_safe * (2.0 ** (applied_st / 12.0)), 0.0)
        y = world_synthesize(feat, f0_new=f0_safe, sp_new=sp_new)
        if y.size > sig.size:
            y = y[: sig.size]
        elif y.size < sig.size:
            y = np.pad(y, (0, sig.size - y.size))
        if do_compress:
            y = compress(y, sr)
        y = soft_limit(y, ceiling_db=ceiling_db)
        y = _apply_silence_gate(y, sig, sr)
        after = analyze(y, sr, forced_voiced=before.f0.voiced)
        guard = _octave_guard(before.f0.f0_hz, after.f0.f0_hz, applied_st)
        guard["fallback_applied"] = True

    timbre_guard = _timbre_guard(before, after)
    if not timbre_guard.get("is_preserved", True):
        warnings_list.append("音色守护：LTAS 相关系数 < 0.98，处理后音色与输入明显偏离")

    report = _build_report(
        before=before,
        after=after,
        plan_summary=plan.summary(),
        plan_notes=plan.notes,
        clip=clip_rep,
        timbre={**timbre_info, "ltas_match": ltas_info},
        strength=strength,
        applied_transpose=applied_st,
        requested_transpose=float(transpose),
        vocoder="world",
        resonance_preserved=bool(formant_shift is None or abs(float(formant_shift) - 1.0) < 1e-9),
        warnings=warnings_list,
        guards={"octave": guard, "timbre": timbre_guard},
        reference_info=reference_info,
        params={
            "deadzone_cents": 16.0,
            "residual_cents": 8.0,
            "max_semitones": max_semitones,
            "frame_period_ms": frame_period,
            "ceiling_db": ceiling_db,
            "n_fft": 2048,
            "hop_length": 512,
        },
    )
    return RepairResult(
        y=y,
        sample_rate=int(sr),
        report=report,
        before=before,
        after=after,
        target_f0=plan.target_hz,
    )


def _build_report(
    *,
    before: AnalysisContext,
    after: AnalysisContext,
    plan_summary: dict,
    plan_notes: list[dict],
    clip,
    timbre: dict,
    strength: float,
    applied_transpose: float,
    requested_transpose: float,
    vocoder: str,
    resonance_preserved: bool,
    warnings: list[str],
    params: dict,
    guards: dict | None = None,
    reference_info: dict | None = None,
) -> dict:
    """组装 report.json（字段按 prompt 规格，另加前后对比字段）。"""
    b, a = before.report, after.report
    return {
        "schema_version": "1.0",
        "sample_rate": before.sr,
        "duration_sec": b["duration_s"],
        "strength": strength,
        "key_estimate": b["key_estimate"]["key"],
        "key_is_reliable": b["key_estimate"]["is_reliable"],
        "transpose_requested": requested_transpose,
        "transpose_semitones": applied_transpose,
        "vocoder": vocoder,
        "resonance_preserved": resonance_preserved,
        "degraded_mode": bool(warnings),
        "clipping": clip.to_dict(),
        "f0": {
            "median_abs_cents_before": b["median_abs_cents"],
            "median_abs_cents_after": a["median_abs_cents"],
            "p95_abs_cents_before": b["p95_abs_cents"],
            "p95_abs_cents_after": a["p95_abs_cents"],
            "improvement_cents": round(b["median_abs_cents"] - a["median_abs_cents"], 2),
            "voiced_frames": b["voiced_frames"],
            "uncertain_frames": len(b["octave_suspects"]),
        },
        "pitch_correction": {**plan_summary, "notes": plan_notes},
        "timbre": timbre,
        "voice_quality_before": b["voice_quality"],
        "voice_quality_after": a["voice_quality"],
        "vibrato_before": b["vibrato"],
        "vibrato_after": a["vibrato"],
        "spectrum_before": b["spectrum"],
        "spectrum_after": a["spectrum"],
        "notes_before": b["notes"],
        "octave_suspects": b["octave_suspects"],
        "warnings": warnings,
        "guards": guards or {},
        "reference_align": reference_info or {"enabled": False},
        "params": params,
        "disclaimer": (
            "MP3 等有损格式丢失的高频与削波信息无法真正恢复；"
            "所有修复均为基于上下文的估计与平滑。"
            "本报告只给测量值与参照区间，不给总分——艺术质量无法用单一数字量化。"
        ),
    }

