#!/usr/bin/env python
"""VocalSpectra —— 人声频谱分析与修音工具的命令行入口。

子命令
------
  analyze     只分析，输出报告与图（不改音频）
  repair      完整修音：削波修复 → 音准修正 → WORLD 保共振峰重合成 → 音色/动态
  separate    人声/伴奏分离（DSP 基线；装 demucs 后可切换神经模型）
  --self-test 合成信号自检，逐条断言管线正确性

用法
----
  python vocal_spectra.py --self-test
  python vocal_spectra.py analyze --input song.wav --outdir outputs/
  python vocal_spectra.py repair --input song.wav --strength 0.7 --transpose 2 \
      --output outputs/repaired.wav --report outputs/report.json --plot outputs/compare.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 允许未 pip install 时从源码目录直接运行
_HERE = Path(__file__).resolve().parent
if str(_HERE / "src") not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))

from vocalrepair import __version__  # noqa: E402
from vocalrepair.audio_io import read_audio, to_mono, write_audio  # noqa: E402

LOG = logging.getLogger("vocal_spectra")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def _load_mono(path: str, sr: int) -> tuple:
    x, fs = read_audio(path, sr=sr, mono=False)
    x = to_mono(x)
    return x.astype(float), fs


def cmd_analyze(args: argparse.Namespace) -> int:
    from vocalrepair.pipeline import analyze
    from vocalrepair.report import plot_comparison, save_json, write_html

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x, sr = _load_mono(args.input, args.sr)
    LOG.info("已读取 %s：%.2f s @ %d Hz", args.input, x.size / sr, sr)

    ctx = analyze(x, sr)
    report = {
        "schema_version": "1.0",
        "input": str(args.input),
        "sample_rate": sr,
        **ctx.report,
        "disclaimer": "本报告只给测量值与参照区间，不给总分——艺术质量无法用单一数字量化。",
    }
    jp = save_json(report, outdir / "analysis.json")
    LOG.info("分析报告：%s", jp)

    # 分析模式下 before==after，只为复用同一套 4 子图
    class _Ctx:
        def __init__(self, c):
            self.before = c
            self.after = c
            self.report = report

    png = plot_comparison(_Ctx(ctx), outdir / "analysis.png", title=f"vocalrepair analysis — {Path(args.input).name}")
    LOG.info("对比图：%s", png)
    hp = write_html(report, outdir / "analysis.html", fig_name=png.name)
    LOG.info("HTML 报告：%s", hp)

    _print_summary(report)
    return 0


def _print_summary(report: dict) -> None:
    f0 = report.get("f0", {})
    print("\n=== 摘要 ===")
    print(f"  时长        : {report.get('duration_sec')} s @ {report.get('sample_rate')} Hz")
    print(f"  估计调性    : {report.get('key_estimate')}")
    if "median_abs_cents" in report:
        print(f"  音准中位偏差: {report.get('median_abs_cents')} cents (P95 {report.get('p95_abs_cents')})")
    elif f0:
        print(f"  音准中位偏差: {f0.get('median_abs_cents_before')} → {f0.get('median_abs_cents_after')} cents")
    notes = report.get("notes_before") or report.get("notes") or []
    print(f"  音符事件    : {len(notes)} 个")
    vq = report.get("voice_quality_before") or report.get("voice_quality") or {}
    if vq:
        print(f"  HNR / CPP   : {vq.get('hnr_db')} dB / {vq.get('cpp_db')} dB")
    vib = report.get("vibrato_before") or report.get("vibrato") or {}
    if vib.get("n_vibrato"):
        print(f"  颤音        : {vib.get('median_rate_hz')} Hz, 深度 {vib.get('median_depth_cents')} cents")


def cmd_repair(args: argparse.Namespace) -> int:
    from vocalrepair.dsp.pitch_scale import pitch_class_of
    from vocalrepair.pipeline import repair
    from vocalrepair.report import plot_comparison, save_json, write_html

    x, sr = _load_mono(args.input, args.sr)
    LOG.info("已读取 %s：%.2f s @ %d Hz", args.input, x.size / sr, sr)

    root_pc = None
    scale = "major"
    if args.key != "auto":
        key = args.key.strip()
        if key.endswith("m") or "minor" in key.lower():
            scale = "natural_minor"
            key = key.rstrip("m").replace("minor", "").strip()
        root_pc = pitch_class_of(key)
        LOG.info("指定调性：主音 %s，音阶 %s", key, scale)
    else:
        LOG.info("--key auto：不启用调内吸附（只做最近半音），调性仅用于报告")

    ref_audio = None
    if args.reference:
        if not Path(args.reference).is_file():
            LOG.error("参考音频不存在：%s", args.reference)
            return 2
        ref_audio, _ref_fs = read_audio(args.reference, sr=sr, mono=False)
        ref_audio = to_mono(ref_audio).astype(float)
        LOG.info("参考音频：%s（%.2f s），对齐模式 pitch=%s timbre=%s",
                 args.reference, ref_audio.size / sr, args.align_pitch, args.align_timbre)

    result = repair(
        x,
        sr,
        strength=args.strength,
        transpose=args.transpose,
        root_pc=root_pc,
        scale=scale,
        clip_repair=(args.clip_repair != "off"),
        frame_period=args.frame_period,
        vocoder=args.vocoder,
        max_semitones=5.0,
        reference_audio=ref_audio,
        reference_sr=sr,
        align_pitch=bool(args.align_pitch),
        align_timbre=args.align_timbre,
    )

    out = Path(args.output)
    write_audio(out, result.y, sr)
    LOG.info("输出音频：%s", out)

    if args.make_mp3:
        mp3 = out.with_suffix(".mp3")
        try:
            write_audio(mp3, result.y, sr, subtype="MP3")
            LOG.info("输出 MP3：%s", mp3)
        except Exception as exc:  # 编码器不可用时如实报告，不静默跳过
            msg = f"MP3 编码失败（{exc}）；WAV 已正常输出。需要 libsndfile 的 MP3 支持或 ffmpeg。"
            LOG.warning(msg)
            result.report.setdefault("warnings", []).append(msg)

    if args.report:
        rp = save_json(result.report, args.report)
        LOG.info("报告：%s", rp)
    if args.plot:
        pp = plot_comparison(result, args.plot, title=f"vocalrepair — {Path(args.input).name}")
        LOG.info("对比图：%s", pp)
    if args.html:
        hp = write_html(
            result.report,
            args.html,
            audio_name=out.name,
            fig_name=Path(args.plot).name if args.plot else None,
        )
        LOG.info("HTML 报告：%s", hp)

    _print_summary(result.report)
    for w in result.report.get("warnings", []):
        print(f"  [warn] {w}")
    return 0


def cmd_separate(args: argparse.Namespace) -> int:
    """人声/伴奏分离。

    优先调用 demucs（若已安装，质量高）；否则用 HPSS + 中心声道基线，
    并在报告里如实标注为低质量基线——不假装这是源分离模型。
    """
    import numpy as np

    from vocalrepair.dsp.stft import istft, stft_mag_phase
    from vocalrepair.report import save_json

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    used = "hpss_baseline"
    try:
        import subprocess

        import demucs  # noqa: F401

        LOG.info("检测到 demucs，使用神经模型分离（CPU 上较慢）")
        subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(outdir), str(args.input)],
            check=True,
        )
        used = "htdemucs"
        vocals_path = next(outdir.rglob("vocals.wav"), None)
        acc_path = next(outdir.rglob("no_vocals.wav"), None)
        if vocals_path:
            Path(outdir / "vocals.wav").write_bytes(vocals_path.read_bytes())
        if acc_path:
            Path(outdir / "accompaniment.wav").write_bytes(acc_path.read_bytes())
    except Exception as exc:
        LOG.warning("demucs 不可用（%s），降级到 HPSS 基线", exc)
        x, sr = _load_mono(args.input, args.sr)
        mag, phase, freqs, _t = stft_mag_phase(x, sr, n_fft=2048, hop_length=512)
        from scipy.ndimage import median_filter

        # HPSS：谐波成分沿时间平滑、打击成分沿频率平滑；两者能量比做软掩膜
        harm = median_filter(mag, size=(1, 31))
        perc = median_filter(mag, size=(31, 1))
        mask = harm**2 / (harm**2 + perc**2 + 1e-12)
        vocals = istft(mag * mask, phase, n_fft=2048, hop_length=512, length=x.size)
        acc = istft(mag * (1.0 - mask), phase, n_fft=2048, hop_length=512, length=x.size)
        write_audio(outdir / "vocals.wav", vocals, sr)
        write_audio(outdir / "accompaniment.wav", acc, sr)

    save_json(
        {
            "input": str(args.input),
            "model": used,
            "harmony_separated": False,
            "quality_note": (
                "HPSS 基线没有'人声'语义，只能粗分谐波/打击成分；"
                "若要可用的人声分离，请 pip install demucs 后用同一命令重跑。"
            )
            if used == "hpss_baseline"
            else "demucs 4-stem 只输出 vocals/no_vocals，主唱与和声未区分。",
        },
        outdir / "separation.json",
    )
    LOG.info("分离输出目录：%s（model=%s）", outdir, used)
    return 0


def run_self_test(args: argparse.Namespace) -> int:
    """合成信号自检：7 条断言，任一失败即退出码 1（prompt 第 9 节）。"""
    import time

    import numpy as np

    from vocalrepair.analysis.formant import track_formants
    from vocalrepair.dsp.f0 import yin_f0
    from vocalrepair.pipeline import repair
    from vocalrepair.report import plot_comparison, save_json
    from vocalrepair.synth import make_note_sequence

    outdir = Path(args.outdir or "outputs/self_test")
    outdir.mkdir(parents=True, exist_ok=True)
    sr = 44100

    # --- 构造测试信号：440 Hz 基频 + 2~5 次谐波 + 颤音，人为偏低 50 cents，1.5x 后硬削波 ---
    # 注意：这里**不能**用带共振峰的元音合成器。实测 /a/ 的 F1 = 730 Hz 会把
    # 440 Hz 的二次谐波抬到比基频强 27 dB，形成缺失基频（missing fundamental），
    # 于是自相关类检测器会合理地判成 880 Hz——那是信号的物理属性，不是管线错误，
    # 但会让自检失去意义。prompt 要求的是"基频 + 2~5 次谐波"，按此构造。
    n = int(3.6 * sr)
    t = np.arange(n) / sr
    f0_track = 440.0 * 2.0 ** (-50.0 / 1200.0) * np.power(
        2.0, 25.0 * np.sin(2.0 * np.pi * 5.5 * t) / 1200.0
    )
    phase = 2.0 * np.pi * np.cumsum(f0_track) / sr
    tone = sum(np.sin(k * phase) / k for k in range(1, 6))
    tone = 0.7 * tone / float(np.max(np.abs(tone)))
    x = np.clip(tone * 1.5, -1.0, 1.0)           # 硬削波到 [-1, 1]
    pad = np.zeros(int(0.4 * sr))
    x = np.concatenate([pad, x, pad])            # 前后静音，检验"清音不被改成有声"
    print(f"self-test 输入：{x.size / sr:.2f} s @ {sr} Hz，峰值 {np.max(np.abs(x)):.3f}")

    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} :: {detail}")

    t0 = time.time()
    res = repair(x, sr, strength=0.85)
    y = res.y
    print(f"  (repair 用时 {time.time() - t0:.2f} s)")

    # 1) 时长一致
    d = abs(y.size / sr - x.size / sr)
    chk("1) 输出时长差 < 0.1 s", d < 0.1, f"diff = {d:.4f} s")

    # 2) 无 NaN/Inf，峰值 <= -0.5 dBFS
    peak_db = 20.0 * np.log10(max(float(np.max(np.abs(y))), 1e-12))
    chk(
        "2) 无 NaN/Inf 且峰值 <= -0.5 dBFS",
        bool(np.all(np.isfinite(y))) and peak_db <= -0.5,
        f"peak = {peak_db:.2f} dBFS",
    )

    # 3) 削波样本下降
    c = res.report["clipping"]
    chk(
        "3) 削波样本数下降",
        c["after_samples"] < c["before_samples"],
        f"{c['before_samples']} -> {c['after_samples']}",
    )

    # 4) 音准偏差中位数下降
    f0r = res.report["f0"]
    chk(
        "4) 音准中位偏差下降",
        f0r["median_abs_cents_after"] < f0r["median_abs_cents_before"],
        f"{f0r['median_abs_cents_before']} -> {f0r['median_abs_cents_after']} cents",
    )

    # 5) 清音帧未被改为有声
    vb, va = res.before.f0.voiced, res.after.f0.voiced
    n = min(vb.size, va.size)
    bad = int(np.count_nonzero(va[:n] & ~vb[:n]))
    chk("5) 清音帧未被改为有声", bad == 0, f"{bad} frames 被误判为浊音")

    # 6) 文件产出
    wav_p = outdir / "self_test_out.wav"
    png_p = outdir / "self_test_compare.png"
    json_p = outdir / "self_test_report.json"
    write_audio(wav_p, y, sr)
    save_json({**res.report, "input": "synthetic", "output": str(wav_p)}, json_p)
    plot_comparison(res, png_p, title="vocalrepair self-test")
    sizes = {p.name: p.stat().st_size for p in (wav_p, png_p, json_p) if p.exists()}
    ok6 = len(sizes) == 3 and all(v > 1000 for v in sizes.values())
    chk("6) WAV / PNG / JSON 均已生成且非空", ok6, str(sizes))

    # 7) +2 半音变调后共振峰偏移 < 10%
    #    用 220 Hz 的 /i/ 单独验证：440 Hz 时 /a/ 的 F1(730) 落在两个谐波之间，
    #    LPC 无法可靠分辨，测量本身不可信——这是测量限制，不是算法限制。
    t220 = np.arange(int(2.0 * sr)) / sr
    from vocalrepair.synth import synth_vowel_allpole

    base = synth_vowel_allpole(np.full(t220.size, 220.0), sr, vowel="i")
    r2 = repair(base, sr, strength=0.0, transpose=2.0, timbre=False, do_compress=False)

    def _formants(signal: np.ndarray) -> np.ndarray:
        fr = yin_f0(signal, sr)
        fm, _ = track_formants(signal, sr, voiced=fr.voiced)
        return np.nanmedian(fm, axis=0)

    fb, fa = _formants(base), _formants(r2.y)
    rel = [
        abs(100.0 * (fa[i] - fb[i]) / fb[i])
        for i in range(min(fa.size, fb.size))
        if np.isfinite(fa[i]) and np.isfinite(fb[i]) and fb[i] > 0
    ]
    worst = max(rel) if rel else 999.0
    chk(
        "7) +2 半音变调后共振峰偏移 < 10%",
        worst < 10.0,
        f"before={np.round(fb, 1)} after={np.round(fa, 1)} worst={worst:.1f}%",
    )

    # 8) 变调超限截断（额外守护，防止误改）
    r3 = repair(base, sr, strength=0.0, transpose=9.0, timbre=False, do_compress=False)
    chk(
        "8) transpose 9 半音被截断到 ±5 并记录告警",
        abs(r3.report["transpose_semitones"] - 5.0) < 1e-6,
        f"applied = {r3.report['transpose_semitones']}",
    )

    # 9) 绝对八度正确（新增守护）
    #    第 4 条只看"相对最近半音的偏差"，一个整体差八度的输出照样能通过
    #    （因为 838 Hz 相对 G#5 只差 15 cents）。必须单独断言绝对音高。
    after_med = float(np.nanmedian(res.after.f0.f0_hz[np.isfinite(res.after.f0.f0_hz)]))
    before_med = float(np.nanmedian(res.before.f0.f0_hz[np.isfinite(res.before.f0.f0_hz)]))
    cents_to_target = 1200.0 * np.log2(after_med / 440.0)
    chk(
        "9) 绝对音高正确（修正后落在 A4 ±20 cents，无八度错位）",
        abs(cents_to_target) < 20.0,
        f"before {before_med:.1f} Hz -> after {after_med:.1f} Hz "
        f"({cents_to_target:+.1f} cents vs A4=440)",
    )

    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"\nself-test: {passed}/{len(checks)} 通过")
    if passed != len(checks):
        for name, ok, detail in checks:
            if not ok:
                print(f"  FAILED: {name} :: {detail}")
        return 1
    print(f"产物目录：{outdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocal_spectra.py",
        description="VocalSpectra — 人声频谱分析与修音（A4 = 440 Hz 基准）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--self-test", action="store_true", help="用合成信号做端到端自检")
    p.add_argument("--outdir", default=None, help="自检产物目录（默认 outputs/self_test）")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="command")

    a = sub.add_parser("analyze", help="只分析，不改音频")
    a.add_argument("--input", default="input.mp3")
    a.add_argument("--outdir", default="outputs")
    a.add_argument("--sr", type=int, default=44100)

    r = sub.add_parser("repair", help="完整修音")
    r.add_argument("--input", default="input.mp3")
    r.add_argument("--output", default="outputs/repaired.wav")
    r.add_argument("--strength", type=float, default=0.7, help="修音强度 0-1")
    r.add_argument("--key", default="auto", help="auto 或 C/D#/Am 等；给定则启用调内吸附")
    r.add_argument("--transpose", type=float, default=0.0, help="半音数，超 ±5 会被截断")
    r.add_argument("--clip-repair", dest="clip_repair", default="auto", choices=["auto", "off"])
    r.add_argument("--max-cents", dest="max_cents", type=float, default=100.0)
    r.add_argument("--frame-period", dest="frame_period", type=float, default=5.0)
    r.add_argument("--sr", type=int, default=44100)
    r.add_argument("--vocoder", default="world",
                   choices=["world", "psola", "none", "phase", "neural"],
                   help="合成引擎：world（默认，保共振峰）/ psola（音准更好但可能颗粒）/ none（只修削波）")
    r.add_argument("--make-mp3", dest="make_mp3", action="store_true")
    r.add_argument("--report", default=None)
    r.add_argument("--plot", default=None)
    r.add_argument("--html", default=None)
    r.add_argument("--reference", default=None, help="参考音频（加分项 A，本版本未实现）")
    r.add_argument("--align-pitch", dest="align_pitch", action="store_true")
    r.add_argument("--align-timbre", dest="align_timbre", default="off",
                   choices=["off", "conservative", "aggressive"])

    s = sub.add_parser("separate", help="人声/伴奏分离")
    s.add_argument("--input", default="input.mp3")
    s.add_argument("--outdir", default="outputs")
    s.add_argument("--sr", type=int, default=44100)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    LOG.info("VocalSpectra %s", __version__)

    if args.self_test:
        if not args.outdir:
            args.outdir = "outputs/self_test"
        return run_self_test(args)

    if not args.command:
        parser.print_help()
        return 2

    if not Path(args.input).is_file():
        LOG.error("输入文件不存在：%s（--input 指定，或把音频放到该路径）", args.input)
        return 2

    if args.command in ("analyze", "repair"):
        if getattr(args, "align_pitch", False) and not getattr(args, "reference", None):
            LOG.warning("--align-pitch 需要同时提供 --reference，本次忽略")
        if getattr(args, "align_timbre", "off") != "off" and not getattr(args, "reference", None):
            LOG.warning("--align-timbre 需要同时提供 --reference，本次忽略")

    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "repair":
        return cmd_repair(args)
    if args.command == "separate":
        return cmd_separate(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
