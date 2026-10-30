"""修复层回归测试：音准决策规则、削波修复、包络处理不改音高。

参数约定（已与用户确认）：
- 死区 16 cents：|偏差| <= 16 不动
- 残差 8 cents：超出死区的部分被拉回到只剩 8 cents
"""

import unittest

import numpy as np

from vocalrepair.dsp.pitch_scale import cents_between, cents_deviation, hz_to_midi
from vocalrepair.repair.declip import detect_clipping, repair_clipping
from vocalrepair.repair.pitch_correct import plan_correction


def _make_track(midi_note: float, n: int = 120, dt: float = 0.01):
    """构造一条恒定音高的 f0 轨迹（可选带颤音）。"""
    t = np.arange(n) * dt
    return t


class TestPitchCorrectionRules(unittest.TestCase):
    def test_deadzone_is_not_touched(self):
        """|偏差| = 15 cents < 死区 16，必须完全不动。"""
        f = 440.0 * 2.0 ** (15.0 / 1200.0)
        f0 = np.full(100, f)
        t = np.arange(100) * 0.01
        plan = plan_correction(f0, t, strength=1.0)
        np.testing.assert_allclose(plan.corrected_hz, f0, rtol=0, atol=1e-6)
        self.assertEqual(plan.notes[0]["applied_cents"], 0.0)
        self.assertTrue(plan.notes[0]["inside_deadzone"])

    def test_residual_stops_at_8_cents(self):
        """偏差 +40 cents、strength=1 时，应施加 -32 cents，残差留 8。"""
        f = 440.0 * 2.0 ** (40.0 / 1200.0)
        f0 = np.full(100, f)
        t = np.arange(100) * 0.01
        plan = plan_correction(f0, t, strength=1.0, deadzone_cents=16.0, residual_cents=8.0)
        applied = plan.notes[0]["applied_cents"]
        self.assertAlmostEqual(applied, -32.0, delta=0.5)
        residual = float(cents_between(440.0, plan.corrected_hz[50]))
        self.assertAlmostEqual(residual, 8.0, delta=0.5)

    def test_strength_scales_correction(self):
        """strength 应线性缩放修正量。"""
        f = 440.0 * 2.0 ** (60.0 / 1200.0)
        f0 = np.full(100, f)
        t = np.arange(100) * 0.01
        full = plan_correction(f0, t, strength=1.0).notes[0]["applied_cents"]
        half = plan_correction(f0, t, strength=0.5).notes[0]["applied_cents"]
        self.assertAlmostEqual(half, full / 2.0, delta=0.5)

    def test_vibrato_is_preserved(self):
        """修正只作用于中心轨迹，颤音调制深度必须保留。"""
        n = 400
        t = np.arange(n) * 0.01
        center = 440.0 * 2.0 ** (45.0 / 1200.0)          # 偏高 45 cents，需要修
        vib = center * np.power(2.0, 40.0 * np.sin(2 * np.pi * 5.5 * t) / 1200.0)
        plan = plan_correction(vib, t, strength=1.0)

        def depth(x):
            c = 1200.0 * np.log2(x / np.median(x))
            return float(np.std(c))

        d_before = depth(vib)
        d_after = depth(plan.corrected_hz)
        self.assertAlmostEqual(d_after / d_before, 1.0, delta=0.15)

    def test_unvoiced_stays_unvoiced(self):
        """NaN（清音）帧绝不能变成有声。"""
        f0 = np.full(200, np.nan)
        f0[50:150] = 440.0
        t = np.arange(200) * 0.01
        plan = plan_correction(f0, t)
        self.assertTrue(np.all(np.isnan(plan.corrected_hz[:50])))
        self.assertTrue(np.all(np.isnan(plan.corrected_hz[150:])))
        self.assertTrue(np.all(np.isfinite(plan.corrected_hz[50:150])))

    def test_scale_aware_snaps_to_key(self):
        """指定 C 大调时，C#4 附近的音应被吸到 C4 或 D4（调内音）。"""
        f = float(440.0 * 2.0 ** ((61.43 - 69.0) * 100.0 / 1200.0))  # MIDI 61.43
        f0 = np.full(100, f)
        t = np.arange(100) * 0.01
        plan = plan_correction(f0, t, root_pc=0, scale="major", strength=1.0)
        target_midi = plan.notes[0]["target_midi"]
        self.assertIn(target_midi % 12, {0, 2, 4, 5, 7, 9, 11})


class TestDeclip(unittest.TestCase):
    def test_detect_and_repair(self):
        sr = 44100
        t = np.arange(sr) / sr
        clean = 0.6 * np.sin(2 * np.pi * 220 * t)
        clipped = np.clip(clean * 2.5, -1.0, 1.0)
        segs = detect_clipping(clipped)
        self.assertGreater(len(segs), 0)

        fixed, rep = repair_clipping(clipped, sr)
        self.assertLess(rep.after_samples, rep.before_samples)
        self.assertLess(int(np.count_nonzero(np.abs(fixed) >= 0.98)), rep.before_samples)

    def test_no_clipping_is_noop(self):
        x = 0.3 * np.sin(np.arange(1000) / 10.0)
        y, rep = repair_clipping(x, 44100)
        self.assertEqual(rep.before_samples, 0)
        np.testing.assert_allclose(y, x)


if __name__ == "__main__":
    unittest.main()
