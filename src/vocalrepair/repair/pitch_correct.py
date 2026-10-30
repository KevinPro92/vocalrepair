"""音准修正：音符级决策 + 保护颤音 + 目标残差 8 cents。

参数为什么这样取（已与你确认）
------------------------------
- 死区 deadzone = 16 cents：|偏差| 不大于 16 就**不动**。普通听众对音准的
  分辨约 20-30 cents、受训者 5-10 cents；把每个音拉成 0 cents 正是
  "软件修音感"的来源，而 16 cents 的死区保留了演唱倾向性（导音偏高、
  三度偏高），这是"听起来像人唱的"的关键
- 目标残差 = 8 cents：超出死区的部分被拉回到只剩 8 cents。16 与 8 都是
  2 的幂，便于记忆与复现
- 单段最大修正 max_cents = 100、整体移动上限 max_semitones = 5：超过
  ±5 半音时即使共振峰保持，音色也会明显改变（声道长度是固定的物理量）

为什么必须"先分离颤音、只修中心"
--------------------------------
自然颤音是 4-7 Hz、20-80 cents 的准周期调制。逐帧吸附到半音格点会直接把
调制压平，声音立刻变机器人。正确做法：

    中心轨迹   = 中值滤波(cents)                  # 保留台阶、抑制调制
    修正后中心 = 中心 + 修正量
    输出 cents = 修正后中心 + (原始 cents - 中心)  # 把颤音原样加回去
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..dsp.pitch_scale import midi_to_hz, midi_to_name, scale_pitch_classes

__all__ = ["CorrectionPlan", "plan_correction"]

_EPS = 1e-12


@dataclass
class CorrectionPlan:
    """修正计划：既给出结果，也给出"改了什么、为什么"。"""

    f0_hz: np.ndarray            # 原始 f0（未检出为 NaN）
    corrected_hz: np.ndarray     # 修正后 f0
    target_hz: np.ndarray        # 每帧目标音高
    delta_cents: np.ndarray      # 实际施加的修正量（正 = 升高）
    center_cents: np.ndarray     # 中值中心轨迹
    vibrato_cents: np.ndarray    # 被保护下来的颤音残差
    notes: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        d = self.delta_cents[np.isfinite(self.delta_cents)]
        moved = int(np.count_nonzero(np.abs(d) > 1e-6))
        return {
            "frames_total": int(self.f0_hz.size),
            "frames_voiced": int(np.count_nonzero(np.isfinite(self.f0_hz))),
            "frames_corrected": moved,
            "median_abs_delta_cents": round(float(np.median(np.abs(d))), 2) if d.size else 0.0,
            "max_abs_delta_cents": round(float(np.max(np.abs(d))), 2) if d.size else 0.0,
        }


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """连续 True 段。"""
    out: list[tuple[int, int]] = []
    i, n = 0, valid.size
    while i < n:
        if valid[i]:
            j = i
            while j < n and valid[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _median_track(x: np.ndarray, k: int) -> np.ndarray:
    """保边中值滤波；整窗全 NaN 时静默返回 NaN（静音段的正确语义）。"""
    k = max(3, k + (1 - k % 2))
    half = k // 2
    padded = np.pad(x, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


def plan_correction(
    f0_hz: np.ndarray,
    times: np.ndarray,
    root_pc: int | None = None,
    scale: str = "major",
    strength: float = 0.7,
    deadzone_cents: float = 16.0,
    residual_cents: float = 8.0,
    max_cents: float = 100.0,
    max_semitones: float = 5.0,
    vibrato_hz: float = 5.5,
    pitch_adaptive_deadzone: bool = True,
) -> CorrectionPlan:
    """生成音准修正计划（只做 f0 层面的决策，不合成音频）。

    参数
    ----
    root_pc : 主音音级（0=C）。None = "最近半音"（chromatic）；给定则该调音级
        内的最近音（scale-aware，可避免把非调内音吸到调外）
    strength : 全局强度 0-1，逐段乘到修正量上
    """
    f0 = np.asarray(f0_hz, dtype=float).copy()
    t = np.asarray(times, dtype=float)
    n = f0.size
    if t.size != n:
        raise ValueError("f0 与 times 长度必须一致")

    valid = np.isfinite(f0) & (f0 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = np.where(valid, 1200.0 * np.log2(np.maximum(f0, _EPS) / 440.0), np.nan)

    dt = float(np.median(np.diff(t))) if n > 1 else 0.005
    # 中值窗需覆盖一个完整颤音周期，才能把调制与台阶分开
    win = int(round(1.0 / max(vibrato_hz * dt, 1e-9)))
    center = _median_track(cents, win)

    corrected = f0.copy()
    target_hz = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    resid_out = np.full(n, np.nan)
    notes: list[dict] = []

    pcs = scale_pitch_classes(root_pc, scale) if root_pc is not None else None
    limit = float(max_semitones) * 100.0

    for i0, i1 in _runs(valid):
        seg_center = center[i0:i1]
        seg_cents = cents[i0:i1]
        if not np.any(np.isfinite(seg_center)):
            continue
        c_med = float(np.nanmedian(seg_center))
        cont_midi = 69.0 + c_med / 100.0  # cents 相对 A4=440，故 MIDI = 69 + cents/100

        # 目标音高：调内最近音，或最近半音
        if pcs is None:
            tgt_midi = float(np.round(cont_midi))
        else:
            base = int(np.round(cont_midi))
            tgt_midi = float(base)
            best_d = float("inf")
            for d in range(-2, 3):
                cand = base + d
                if cand % 12 in pcs:
                    dist = abs(cand - cont_midi)
                    if dist < best_d:
                        tgt_midi, best_d = float(cand), dist

        tgt_cents = (tgt_midi - 69.0) * 100.0
        target_hz[i0:i1] = midi_to_hz(tgt_midi)

        # 分频段死区（v0.4.2）：低音区压晃动、高音区留张力。
        # 依据：低音区（< E3 ≈ 165 Hz）容易与伴奏低频争夺、也容易因气息不稳而
        # 晃动，收紧死区（12 cents）更干净；人耳对 C4-C6 的音高偏差最敏感，
        # 但高音往往带有"向上冲"的表现力张力，死区放宽到 22 cents 以免修成死板。
        # 中音区保持基准 16 cents。
        dz = deadzone_cents
        if pitch_adaptive_deadzone:
            hz = 440.0 * 2.0 ** (c_med / 1200.0)
            dz = 12.0 if hz < 165.0 else (16.0 if hz < 500.0 else 22.0)

        # 决策：只拉回超出死区的部分，残差停在 residual_cents
        err = tgt_cents - c_med
        if abs(err) <= dz:
            applied = 0.0
        else:
            applied = float(err - np.sign(err) * residual_cents)
        applied = float(np.clip(applied, -max_cents, max_cents))
        applied *= float(np.clip(strength, 0.0, 1.0))
        applied = float(np.clip(applied, -limit, limit))

        center_new = seg_center + applied
        resid = seg_cents - seg_center          # 颤音残差，原样保留
        out_cents = center_new + resid
        corrected[i0:i1] = 440.0 * np.power(2.0, out_cents / 1200.0)
        delta[i0:i1] = applied
        resid_out[i0:i1] = resid

        notes.append(
            {
                "start_s": round(float(t[i0]), 3),
                "end_s": round(float(t[i1 - 1]), 3),
                "target_midi": int(round(tgt_midi)),
                "target_name": midi_to_name(tgt_midi),
                # dev_cents：修正前相对"目标音"的偏差（正 = 偏高）。
                # 不报"相对 A4 的绝对音分"，因为后者像 -1159.5 这种数值
                # 对使用者没有意义（它只是 A3 的绝对位置）。
                "dev_cents": round(c_med - tgt_cents, 1),
                "applied_cents": round(applied, 1),
                "residual_cents": round(float(err - applied), 1),
                "inside_deadzone": bool(abs(err) <= dz),
                "deadzone_cents": dz,
            }
        )

    # 只保留原本就浊音的帧，绝不清音变有声
    corrected = np.where(valid, corrected, np.nan)

    return CorrectionPlan(
        f0_hz=f0,
        corrected_hz=corrected,
        target_hz=target_hz,
        delta_cents=delta,
        center_cents=center,
        vibrato_cents=resid_out,
        notes=notes,
    )
