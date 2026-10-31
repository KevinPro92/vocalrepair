"""发声方式检测：挤压（pressed） / 气声（breathy） / 常态。

为什么不能只用 H1-H2
--------------------
单用 H1−H2 会在 **F1 靠近 f0 或 2f0** 时严重失真——共振峰把某个谐波抬高，
测量就偏了。实测本项目的 KTV 素材：H1−H2 在 −15.7 ~ +23.2 dB 之间乱跳，
完全无法作为判据。

改用**五个指标的投票**，每个指标都有独立的物理依据：

| 指标 | 挤压（pressed） | 气声（breathy） | 依据 |
|---|---|---|---|
| H1−H2 | 小（≈0 或负） | 大（>8 dB） | 声门闭合紧密时高次谐波更强 |
| H1−A3 | 小 | 大 | 经典嗓音学指标，A3 取 F3 区最强谐波，比 H1−H2 抗 F1 干扰 |
| 谱倾斜 slope | 平（> −6 dB/oct） | 陡（< −10 dB/oct） | 挤压时高频谐波被抬升 |
| 谐波重心 | 上移 | 下移 | 同上，归一化后更稳 |
| HNR | 高（周期性强） | 低（漏气） | 挤压周期性强、气声噪声多 |

只有**多数指标同向**才给结论，并输出票数作为置信度——这比任何单指标阈值都可靠。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp.stft import stft_mag_phase

__all__ = ["VoiceModeFrame", "detect_voice_mode", "summarize_segments"]

_EPS = 1e-12


@dataclass
class VoiceModeFrame:
    time: float
    f0: float
    h1_h2: float
    h1_a3: float
    slope: float
    centroid_hz: float
    hnr: float
    verdict: str          # pressed / breathy / normal
    votes: int            # 同向指标票数（0-5）

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 2),
            "f0": round(self.f0, 1),
            "h1_h2": round(self.h1_h2, 1),
            "h1_a3": round(self.h1_a3, 1),
            "slope": round(self.slope, 2),
            "centroid_hz": round(self.centroid_hz, 0),
            "hnr": round(self.hnr, 1),
            "verdict": self.verdict,
            "votes": self.votes,
        }


def _peak_in(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """某频带内的最大幅度。"""
    sel = (freqs >= lo) & (freqs <= hi)
    return float(np.max(mag[sel])) if sel.any() else 0.0


def detect_voice_mode(
    x: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    times: np.ndarray,
    per_s: float = 0.5,
) -> list[VoiceModeFrame]:
    """按 per_s 秒聚合，输出每段的发声方式判定（含票数）。"""
    from ..analysis.spectral import frame_spectral_stats
    from ..analysis.voice_quality import hnr_db

    sig = np.asarray(x, dtype=float).ravel()
    mag, _ph, freqs, tf = stft_mag_phase(sig, sr, 2048, 512)
    f0_stft = np.interp(tf, times, np.where(np.isfinite(f0_hz), f0_hz, np.nan))
    stats = frame_spectral_stats(mag, freqs)
    hnr = hnr_db(sig, sr, f0_hz)
    hnr_stft = np.interp(tf, times, np.nan_to_num(hnr, nan=np.nan), left=np.nan, right=np.nan)

    out: list[VoiceModeFrame] = []
    dur = sig.size / sr
    lo_t = 0.0
    while lo_t < dur - per_s:
        m = (tf >= lo_t) & (tf < lo_t + per_s) & np.isfinite(f0_stft)
        if int(m.sum()) < 5:
            lo_t += per_s
            continue
        f_med = float(np.median(f0_stft[m]))
        idx = np.flatnonzero(m)

        h1 = h1_2 = h1_a3 = 0.0
        h1s, h2s, a3s = [], [], []
        for i in idx:
            a1 = _peak_in(mag[:, i], freqs, f_med * 0.97, f_med * 1.03)
            a2 = _peak_in(mag[:, i], freqs, 2 * f_med * 0.97, 2 * f_med * 1.03)
            a3 = _peak_in(mag[:, i], freqs, 2500.0, 3500.0)
            if a1 > _EPS and a2 > _EPS:
                h1s.append(20 * np.log10(a1 / a2))
            if a1 > _EPS and a3 > _EPS:
                a3s.append(20 * np.log10(a1 / a3))
        h12 = float(np.median(h1s)) if h1s else 0.0
        h1a3 = float(np.median(a3s)) if a3s else 0.0
        slope = float(np.median(stats["slope"][m]))
        centroid = float(np.median(stats["centroid"][m]))
        hn = float(np.nanmedian(hnr_stft[m])) if np.any(np.isfinite(hnr_stft[m])) else 0.0

        # 五指标投票。**缺失指标不投票**——例如 A3 频带没有峰时 H1-A3 = 0，
        # 若把它当作"0 < 25"就会投出假阳性（实测 26.5 s / 28.0 s 两段就是这样
        # 被误判成 pressed 的）。改为：只统计真正测到的指标，并按有效票数占比判定。
        p_votes: list[bool] = []
        b_votes: list[bool] = []
        if h1s:
            p_votes.append(h12 < 2.0)
            b_votes.append(h12 > 8.0)
        if a3s:
            p_votes.append(h1a3 < 25.0)
            b_votes.append(h1a3 > 40.0)
        p_votes.append(bool(slope > -6.0))
        b_votes.append(bool(slope < -10.0))
        p_votes.append(bool(centroid > 1500.0))
        b_votes.append(bool(centroid < 900.0))
        if np.isfinite(hn) and hn != 0.0:
            p_votes.append(bool(hn > 10.0))
            b_votes.append(bool(hn < 5.0))

        n_valid = len(p_votes)
        pressed = int(sum(p_votes))
        breathy = int(sum(b_votes))
        # 需要至少 3 票，且占到有效指标的 60% 才下结论
        need = max(3, int(np.ceil(0.6 * n_valid)))
        if pressed >= need and pressed > breathy:
            verdict, votes = "pressed", pressed
        elif breathy >= need and breathy > pressed:
            verdict, votes = "breathy", breathy
        else:
            verdict, votes = "normal", max(pressed, breathy)

        out.append(
            VoiceModeFrame(
                time=lo_t, f0=f_med, h1_h2=h12, h1_a3=h1a3,
                slope=slope, centroid_hz=centroid, hnr=hn,
                verdict=verdict, votes=votes,
            )
        )
        lo_t += per_s
    return out


def summarize_segments(frames: list[VoiceModeFrame]) -> dict:
    """把逐段结果压成报告（连续同判定合并）。"""
    segs: list[dict] = []
    cur: dict | None = None
    for fr in frames:
        if cur and cur["verdict"] == fr.verdict:
            cur["end_s"] = round(fr.time + 0.5, 2)
            cur["n"] += 1
            cur["votes_max"] = max(cur["votes_max"], fr.votes)
        else:
            if cur:
                segs.append(cur)
            cur = {
                "verdict": fr.verdict,
                "start_s": round(fr.time, 2),
                "end_s": round(fr.time + 0.5, 2),
                "n": 1,
                "f0_median": round(fr.f0, 1),
                "votes_max": fr.votes,
            }
    if cur:
        segs.append(cur)
    bad = [s for s in segs if s["verdict"] != "normal"]
    return {"segments": segs, "abnormal": bad, "n_abnormal": len(bad)}
