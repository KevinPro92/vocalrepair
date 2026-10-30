"""报告输出：JSON、对比图、自包含 HTML。

对比图固定 4 个子图（prompt 第 8 节要求）：
1. 原始 STFT 频谱图
2. 修复后 STFT 频谱图（同一色标范围，才能直接比较）
3. F0 轨迹对比（原始 / 目标 / 修复）+ 削波标记
4. 长时平均谱 LTAS 对比（音色指纹是否被改变）

注意第 2 张必须与第 1 张共用 vmin/vmax：色标各自归一化会让读者误以为
"亮度变了"，那是最常见的图表误导。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["save_json", "plot_comparison", "write_html"]


def save_json(report: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return p


def _db(mag: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(mag, 1e-10))


def plot_comparison(result, path: str | Path, title: str = "vocalrepair — before / after") -> Path:
    """绘制 4 子图对比，返回保存路径。``result`` 为 pipeline.RepairResult。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    before, after = result.before, result.after
    db_b = _db(before.mag)
    db_a = _db(after.mag)
    vmax = float(np.percentile(db_b, 99.5))
    vmin = vmax - 80.0  # 统一 80 dB 显示动态范围

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    for ax, db, name in ((axes[0, 0], db_b, "before"), (axes[0, 1], db_a, "after")):
        im = ax.pcolormesh(before.times, before.freqs, db, shading="auto", vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_ylim(0, 6000)
        ax.set_title(f"{name}: STFT magnitude (dB, shared scale)")
        ax.set_xlabel("time / s")
        ax.set_ylabel("frequency / Hz")
        fig.colorbar(im, ax=ax, label="dB")

    # 3) f0 轨迹（用 MIDI 对数轴）
    from .dsp.pitch_scale import hz_to_midi

    ax = axes[1, 0]
    with np.errstate(invalid="ignore"):
        ax.plot(before.f0.times, hz_to_midi(before.f0.f0_hz), "o", ms=2.5, label="detected (before)")
        ax.plot(after.f0.times, hz_to_midi(after.f0.f0_hz), "o", ms=2.5, alpha=0.7, label="detected (after)")
        target = getattr(result, "target_f0", None)
        if target is not None:
            ax.plot(before.f0.times, hz_to_midi(target), "-", lw=0.9, alpha=0.8, label="target")
    for s in result.report.get("octave_suspects", []):
        ax.axvspan(s["start_s"], s["end_s"], color="orange", alpha=0.25)
    ax.set_xlabel("time / s")
    ax.set_ylabel("MIDI pitch")
    ax.set_title("f0 track (orange bands = octave suspects)")
    ax.legend(fontsize=8)

    # 4) LTAS 对比
    from .analysis.spectral import ltas

    ax = axes[1, 1]
    db_l_b, f_l = ltas(before.mag, before.freqs)
    db_l_a, _ = ltas(after.mag, after.freqs)
    band = (f_l >= 50) & (f_l <= 12000)
    ax.plot(f_l[band], db_l_b[band] - db_l_b[band].max(), lw=1.2, label="before")
    ax.plot(f_l[band], db_l_a[band] - db_l_a[band].max(), lw=1.2, alpha=0.8, label="after")
    ax.set_xscale("log")
    ax.set_xlabel("frequency / Hz (log)")
    ax.set_ylabel("relative level / dB")
    ax.set_title("LTAS (timbre fingerprint)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def write_html(report: dict, path: str | Path, audio_name: str | None = None, fig_name: str | None = None) -> Path:
    """生成单文件 HTML 报告（内嵌 JSON 摘要与图表链接）。

    不内嵌 base64 音频（会让文件膨胀到几十 MB），只给相对链接。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f0 = report.get("f0", {})
    clip = report.get("clipping", {})
    rows = [
        ("时长 / s", report.get("duration_sec")),
        ("采样率 / Hz", report.get("sample_rate")),
        ("估计调性", f"{report.get('key_estimate')} (reliable={report.get('key_is_reliable')})"),
        ("修音强度", report.get("strength")),
        ("变调 / 半音", f"{report.get('transpose_semitones')} (请求 {report.get('transpose_requested')})"),
        ("音准中位偏差 前 / 后 (cents)", f"{f0.get('median_abs_cents_before')} → {f0.get('median_abs_cents_after')}"),
        ("音准 P95 偏差 前 / 后 (cents)", f"{f0.get('p95_abs_cents_before')} → {f0.get('p95_abs_cents_after')}"),
        ("削波样本 前 / 后", f"{clip.get('before_samples')} → {clip.get('after_samples')}"),
        ("严重削波段数", clip.get("severe_segments")),
        ("共振峰保持", report.get("resonance_preserved")),
        ("降级模式", report.get("degraded_mode")),
    ]
    tr = "\n".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    warn = "".join(f"<li>{w}</li>" for w in report.get("warnings", [])) or "<li>无</li>"
    img = f'<img src="{fig_name}" alt="comparison">' if fig_name else ""
    audio = f'<audio controls src="{audio_name}"></audio>' if audio_name else ""
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>vocalrepair report</title>
<style>
 body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
 h1{{font-size:1.4rem}} table{{border-collapse:collapse;margin:1rem 0}}
 th,td{{border:1px solid #ddd;padding:.35rem .7rem;text-align:left;font-size:.92rem}}
 th{{background:#f6f6f6;width:16rem}} img{{max-width:100%;border:1px solid #ddd}}
 code{{background:#f3f3f3;padding:.1rem .3rem;border-radius:3px}}
 .note{{color:#666;font-size:.85rem}}
</style></head><body>
<h1>vocalrepair 诊断与修复报告</h1>
{audio}
<table>{tr}</table>
{img}
<h2>告警</h2><ul>{warn}</ul>
<h2>局限说明</h2><p class="note">{report.get('disclaimer','')}</p>
</body></html>"""
    p.write_text(html, encoding="utf-8")
    return p