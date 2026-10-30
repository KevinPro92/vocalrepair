"""快速上手示例：合成一段跑调的歌声 → 检测 f0 → 输出波形与音高轨迹图。

运行：python examples/quickstart.py
产物：figures/quickstart_f0_track.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 允许在未 pip install 的情况下直接运行（把 src 加入搜索路径）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocalrepair import cents_between, make_note_sequence, yin_f0  # noqa: E402
from vocalrepair.dsp.pitch_scale import (  # noqa: E402
    cents_deviation,
    hz_to_name,
    midi_to_hz,
    snap_to_scale,
)


def main() -> None:
    sr = 44100
    notes = [57, 59, 60, 62]           # A3 B3 C4 D4
    detune = [40.0, -25.0, 15.0, -55.0]  # 人为跑调（cents）
    sig = make_note_sequence(
        notes,
        sr=sr,
        note_dur=0.55,
        gap=0.09,
        vibrato_rate=5.5,
        vibrato_depth_cents=35.0,
        detune_cents=detune,
        vowel="a",
        noise_snr_db=45.0,
        seed=7,
    )

    res = yin_f0(sig.x, sr)
    truth = np.interp(res.times, sig.times, sig.f0_true)
    mask = res.voiced & (res.times > 0.05) & (res.times < sig.duration - 0.05)

    print(f"时长 {sig.duration:.2f} s, 帧数 {res.f0_hz.size}, 浊音帧 {int(res.voiced.sum())}")
    print(f"检测误差(中位数绝对值): {np.nanmedian(np.abs(cents_between(truth[mask], res.f0_hz[mask]))):.2f} cents")

    # 逐帧"该修到哪个音"——调内吸附（C 大调）
    print("\n  帧时刻/ s   实测 f0/ Hz   音名   偏差/ cents   调内目标/ Hz")
    for i in range(20, res.f0_hz.size, 40):
        if not np.isfinite(res.f0_hz[i]):
            continue
        f = float(res.f0_hz[i])
        tgt = snap_to_scale(f, root_pc=0, scale="major")
        print(
            f"  {res.times[i]:9.2f}   {f:10.2f}   {hz_to_name(f):>4}"
            f"   {float(cents_deviation(f)):+8.1f}   {tgt:12.2f}"
        )

    _plot(sig, res, truth, mask, ROOT / "figures" / "quickstart_f0_track.png")


def _plot(sig, res, truth, mask, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[跳过绘图] 未安装 matplotlib：{exc}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    t = np.arange(sig.x.size) / sig.sr
    axes[0].plot(t, sig.x, lw=0.5, color="#333333")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("synthesised vocal phrase (ground-truth f0 known per sample)")

    # 音高图用 MIDI 号（对数轴）画，才符合听觉
    from vocalrepair.dsp.pitch_scale import hz_to_midi

    axes[1].plot(res.times, hz_to_midi(res.f0_hz), "o", ms=3, color="#1f77b4", label="YIN estimate")
    axes[1].plot(sig.times[::256], hz_to_midi(sig.f0_true[::256]), "-", lw=1.2, color="#d62728", alpha=0.75, label="ground truth")
    axes[1].set_ylabel("MIDI pitch (log axis)")
    axes[1].set_xlabel("time / s")
    axes[1].set_yticks(range(54, 66))
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title("f0 track: dots = detected, line = truth (vibrato visible)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n图已保存: {out_path}")


if __name__ == "__main__":
    main()
