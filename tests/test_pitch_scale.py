"""音高刻度回归测试：A4 = 440 Hz 及音分换算的正确性。

这些是"数值口径"测试——它们锁死整条链路的坐标系。任何一次重构若改坏了
A4 基准、半音比或音分定义，这里必然先红。
"""

import unittest

import numpy as np

from vocalrepair.dsp.pitch_scale import (
    A4_HZ,
    cents_between,
    cents_deviation,
    hz_to_cents,
    hz_to_midi,
    hz_to_name,
    midi_to_hz,
    midi_to_name,
    note_name_to_midi,
    pitch_class_of,
    scale_pitch_classes,
    smooth_center_track,
    snap_to_semitone,
    snap_to_scale,
    soft_correct_track,
)


class TestReferencePitch(unittest.TestCase):
    """十二平均律基准点的硬性约定。"""

    def test_a4_is_440(self):
        self.assertEqual(A4_HZ, 440.0)
        self.assertAlmostEqual(float(midi_to_hz(69)), 440.0, places=9)

    def test_c4_is_261_626(self):
        # 钢琴中央 C 的标准值 261.6255653 Hz
        self.assertAlmostEqual(float(midi_to_hz(60)), 261.6255653, places=6)

    def test_semitone_ratio(self):
        ratio = float(midi_to_hz(70)) / float(midi_to_hz(69))
        self.assertAlmostEqual(ratio, 2.0 ** (1.0 / 12.0), places=12)
        self.assertAlmostEqual(ratio, 1.0594630943592953, places=12)

    def test_octave_doubles(self):
        for note in (57, 60, 69, 72):
            self.assertAlmostEqual(
                float(midi_to_hz(note + 12)) / float(midi_to_hz(note)), 2.0, places=12
            )

    def test_midi_hz_roundtrip(self):
        for note in np.arange(36.0, 96.0, 0.25):
            self.assertAlmostEqual(float(hz_to_midi(midi_to_hz(note))), float(note), places=9)


class TestCents(unittest.TestCase):
    def test_octave_is_1200_cents(self):
        self.assertAlmostEqual(float(hz_to_cents(880.0)), 1200.0, places=9)
        self.assertAlmostEqual(float(hz_to_cents(220.0)), -1200.0, places=9)

    def test_cents_between(self):
        self.assertAlmostEqual(float(cents_between(220.0, 440.0)), 1200.0, places=9)
        self.assertAlmostEqual(float(cents_between(440.0, 220.0)), -1200.0, places=9)

    def test_deviation_and_snap(self):
        self.assertAlmostEqual(float(cents_deviation(440.0)), 0.0, places=9)
        # 偏离 A4 25 cents 的音，应被识别为 +25，并吸附回 440
        f_off = 440.0 * 2.0 ** (25.0 / 1200.0)
        self.assertAlmostEqual(float(cents_deviation(f_off)), 25.0, places=6)
        self.assertAlmostEqual(float(snap_to_semitone(f_off)), 440.0, places=9)
        # 450 Hz 离 440（-38.6 cents）比离 466.16（+61 cents）更近
        self.assertAlmostEqual(float(snap_to_semitone(450.0)), 440.0, places=9)


class TestNaming(unittest.TestCase):
    def test_midi_to_name(self):
        self.assertEqual(midi_to_name(60), "C4")
        self.assertEqual(midi_to_name(69), "A4")
        self.assertEqual(midi_to_name(0), "C-1")

    def test_hz_to_name(self):
        self.assertEqual(hz_to_name(440.0), "A4")
        self.assertEqual(hz_to_name(220.0), "A3")

    def test_name_to_midi(self):
        self.assertEqual(note_name_to_midi("A4"), 69)
        self.assertEqual(note_name_to_midi("C4"), 60)
        self.assertEqual(note_name_to_midi("C#3"), 49)
        self.assertEqual(note_name_to_midi("Bb2"), 46)
        self.assertEqual(note_name_to_midi("a4"), 69)

    def test_pitch_class(self):
        self.assertEqual(pitch_class_of("A"), 9)
        self.assertEqual(pitch_class_of("C#"), 1)
        self.assertEqual(pitch_class_of("Db"), 1)
        self.assertEqual(pitch_class_of("A3"), 9)


class TestScaleSnap(unittest.TestCase):
    def test_a_natural_minor_set(self):
        self.assertEqual(scale_pitch_classes(9, "natural_minor"), frozenset({9, 11, 0, 2, 4, 5, 7}))

    def test_scale_snap_moves_out_of_key_note(self):
        # C#4 = 277.18 Hz 不在 C 大调内，应吸附到 C4（而不是留在 C#）
        self.assertAlmostEqual(float(snap_to_scale(277.18, 0, "major")), float(midi_to_hz(60)), places=6)

    def test_scale_snap_picks_nearest_not_first(self):
        """回归测试：必须选"最近"的调内音，而不是搜索顺序里第一个命中的。

        MIDI 61.43（C#4 偏高）在 C 大调下：
          C4 距 1.43 半音，D4 距 0.57 半音 -> 正确结果是 D4。
        早期实现按 (0,-1,+1) 顺序返回 C4，等于把音修到错误方向。
        """
        f = float(midi_to_hz(61.43))
        self.assertAlmostEqual(float(snap_to_scale(f, 0, "major")), float(midi_to_hz(62)), places=6)
        # 对称情况：A 小调下 MIDI 57.4（A3 偏高）应留在 A3（距 0.4），而非 G3（距 0.6）
        f2 = float(midi_to_hz(57.40))
        self.assertAlmostEqual(float(snap_to_scale(f2, 9, "natural_minor")), float(midi_to_hz(57)), places=6)

    def test_scale_snap_never_moves_more_than_half_tone_in_any_key(self):
        """任何调式、任何音高，吸附位移都不应超过 1 个半音。"""
        for root in range(12):
            for scale in ("major", "natural_minor"):
                for m in np.arange(40.0, 84.0, 0.05):
                    f = float(midi_to_hz(m))
                    out = float(hz_to_midi(snap_to_scale(f, root, scale)))
                    self.assertLessEqual(abs(out - m), 1.0 + 1e-9)


class TestSmoothingAndCorrection(unittest.TestCase):
    def test_median_filter_removes_vibrato(self):
        """中值滤波应把 40 cents 颤音压到死区以下（工程判据，非"完全消除"）。

        能力边界已实测：窗长 = 一个完整颤音周期时抑制约 70%，40 cents
        调制残差约 12 cents。只要残差小于修音死区（默认 15 cents），
        就不会被误判为跑调，这比"强行加长窗长导致换音过渡带变宽"更可取。
        """
        sr = 100.0  # 帧率 100 Hz（对应 10 ms 帧移）
        t = np.arange(120) / sr
        center = 220.0
        depth_in = 40.0
        vib = center * 2.0 ** (depth_in * np.sin(2 * np.pi * 5.0 * t) / 1200.0)
        smoothed = smooth_center_track(vib, dt_s=1.0 / sr, vibrato_hz=5.0)
        err = np.abs(cents_between(np.full_like(smoothed, center), smoothed))
        depth_out = float(np.max(err))
        self.assertLess(depth_out, 0.5 * depth_in)   # 至少压掉一半
        self.assertLess(depth_out, 15.0)             # 且落入死区，不会被误判为跑调

    def test_median_filter_preserves_step(self):
        """中值滤波是保边算子：音高台阶不能被抹成斜坡。"""
        sr = 100.0
        track = np.concatenate([np.full(60, 220.0), np.full(60, 330.0)])
        smoothed = smooth_center_track(track, dt_s=1.0 / sr, vibrato_hz=5.0)
        # 台阶两端远离过渡带处应精确保持
        self.assertAlmostEqual(float(smoothed[10]), 220.0, places=6)
        self.assertAlmostEqual(float(smoothed[110]), 330.0, places=6)

    def test_soft_correct_converges_to_deadzone(self):
        """一阶追踪应把 +40 cents 的偏差收敛到死区边界附近（而非 0）。"""
        center = 440.0
        cur = center * 2.0 ** (40.0 / 1200.0)
        for _ in range(300):
            cur = float(
                soft_correct_track([cur], [center], dt_s=0.01, tau_s=0.03, deadzone_cents=15.0)[0]
            )
        residual = float(cents_between(center, cur))  # 正 = 仍偏高
        self.assertAlmostEqual(residual, 15.0, delta=0.5)

    def test_hard_snap_when_tau_tiny(self):
        """tau -> 0 时应一步到位（等价 Auto-Tune 硬吸附）。"""
        center = 440.0
        cur = center * 2.0 ** (30.0 / 1200.0)
        out = float(soft_correct_track([cur], [center], dt_s=0.01, tau_s=1e-9, deadzone_cents=0.0)[0])
        self.assertAlmostEqual(out, center, places=6)


if __name__ == "__main__":
    unittest.main()
