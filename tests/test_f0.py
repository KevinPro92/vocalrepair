"""基频检测（YIN）的定量验证。

验证策略：用合成信号（真值逐采样已知）而不是真实录音，避免循环论证。
判据是"音分误差"，因为这是感知上近似线性的坐标轴。
"""

import unittest

import numpy as np

from vocalrepair.dsp.f0 import difference_function, yin_f0
from vocalrepair.dsp.pitch_scale import cents_between
from vocalrepair.synth import clipped_sample_mask, make_note_sequence


class TestDifferenceFunction(unittest.TestCase):
    def test_minimum_at_period(self):
        """纯周期信号：差分函数应在 tau = 周期 处取极小。"""
        sr = 8000
        f0 = 200.0                      # 周期 = 40 个采样点
        n = 800
        t = np.arange(n) / sr
        x = np.sin(2 * np.pi * f0 * t)
        d = difference_function(x - x.mean(), window=400, tau_max=200)
        tau_min_idx = 20                # 排除 tau 过小的平凡解
        best = tau_min_idx + int(np.argmin(d[tau_min_idx:]))
        self.assertLess(abs(best - 40), 2)


class TestYinAccuracy(unittest.TestCase):
    def test_pure_tone_a4(self):
        """440 Hz 纯音应被检出，误差 < 1 cent（0.25 Hz）。"""
        sr = 44100
        t = np.arange(sr) / sr
        x = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        res = yin_f0(x, sr)
        self.assertTrue(res.voiced.any())
        med = float(np.nanmedian(res.f0_hz))
        self.assertAlmostEqual(med, 440.0, delta=0.25)

    def test_note_sequence_with_vibrato(self):
        """带颤音的元音序列：逐帧跟踪误差中位数 < 10 cents。"""
        sig = make_note_sequence(
            [57, 59, 60, 62],
            sr=44100,
            note_dur=0.50,
            gap=0.10,
            vibrato_rate=5.5,
            vibrato_depth_cents=40.0,
            noise_snr_db=60.0,
            seed=1,
        )
        res = yin_f0(sig.x, sig.sr)
        truth = np.interp(res.times, sig.times, sig.f0_true)
        mask = res.voiced & (res.times > 0.06) & (res.times < sig.duration - 0.06)
        self.assertGreater(int(mask.sum()), 50)
        err = np.abs(cents_between(truth[mask], res.f0_hz[mask]))
        self.assertLess(float(np.nanmedian(err)), 10.0)
        self.assertLess(float(np.nanpercentile(err, 90)), 30.0)

    def test_no_octave_error_on_low_voice(self):
        """85 Hz 低音不应被判成 170 Hz（八度错误是本算法最典型的失效模式）。"""
        sig = make_note_sequence(
            [45],  # A2 = 110 Hz
            sr=44100,
            note_dur=0.8,
            gap=0.0,
            vibrato_depth_cents=0.0,
            noise_snr_db=70.0,
        )
        res = yin_f0(sig.x, sig.sr)
        med = float(np.nanmedian(res.f0_hz))
        self.assertAlmostEqual(med, 110.0, delta=1.5)

    def test_silence_is_unvoiced(self):
        res = yin_f0(np.zeros(44100), 44100)
        self.assertFalse(bool(res.voiced.any()))
        self.assertTrue(np.all(np.isnan(res.f0_hz)))

    def test_short_input_is_padded(self):
        """极短输入不应崩溃（内部会补零到一帧以上）。"""
        res = yin_f0(np.zeros(64), 44100)
        self.assertEqual(res.f0_hz.size, 1)


class TestClippingMask(unittest.TestCase):
    def test_detect_plateau(self):
        x = np.zeros(1000)
        x[400:440] = 0.999  # 40 个采样点的削波平台
        mask = clipped_sample_mask(x)
        self.assertTrue(bool(mask[400:440].all()))
        self.assertFalse(bool(mask[:390].any()))


if __name__ == "__main__":
    unittest.main()
