"""中置声道提取：立体声混音的快速人声分离（零模型依赖）。

适用场景
--------
KTV / 商业流行混音里，主唱几乎总在**正中**（左右声道等幅同相），而伴奏为了
空间感被做得更"宽"（有侧向成分、有反相成分）。于是可以用"居中程度"做一个
时频掩膜，把主唱抠出来：

    M = (L + R) / 2        居中成分（人声 + 居中的伴奏）
    S = (L - R) / 2        侧向成分（主要是伴奏）

    掩膜 = |M|^2 / (|M|^2 + |S|^2 + eps)

这不是源分离模型，只是**空间线索**，因此：
- 优点：零依赖、秒级、对"人声居中"的混音效果常常好于 HPSS
- 缺点：居中的伴奏（底鼓、贝斯、居中的吉他）会跟着人声一起被抠出来；
  如果原唱被做过立体声加宽或延迟，效果会明显变差
所以结果必须标注 model="center_extraction"、quality="spatial only"。
"""

from __future__ import annotations

import numpy as np

from ..dsp.stft import istft, stft_mag_phase

__all__ = ["extract_center"]


def extract_center(
    x: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (人声, 伴奏, 信息)。

    x 必须是立体声 (n, 2)。strength > 1 时分得更"狠"（掩膜更硬），
    但残留与伪影都会增加。
    """
    sig = np.asarray(x, dtype=float)
    if sig.ndim != 2 or sig.shape[1] != 2:
        raise ValueError("中置提取需要立体声输入 (n, 2)")

    left, right = sig[:, 0], sig[:, 1]
    ml, pl, freqs, _t = stft_mag_phase(left, sr, n_fft, hop_length)
    mr, pr, _f, _t2 = stft_mag_phase(right, sr, n_fft, hop_length)

    mid = (ml + mr) / 2.0
    side = (ml - mr) / 2.0
    eps = 1e-12
    mask = (mid**2) / (mid**2 + side**2 + eps)
    mask = np.clip(mask, 0.0, 1.0) ** float(max(strength, 1e-6))

    # 用中置成分的相位重建（M 的相位比单独 L/R 更接近人声）
    complex_mid = (ml * np.exp(1j * pl) + mr * np.exp(1j * pr)) / 2.0
    complex_side = (ml * np.exp(1j * pl) - mr * np.exp(1j * pr)) / 2.0
    phase_mid = np.angle(complex_mid)
    mid_mag = np.abs(complex_mid)
    side_mag = np.abs(complex_side)

    vocal = istft(mid_mag * mask, phase_mid, n_fft, hop_length, length=left.size)
    accomp = istft(mid_mag * (1.0 - mask), phase_mid, n_fft, hop_length, length=left.size)
    # 伴奏再补上侧向成分（否则会丢掉立体声宽度）
    accomp = accomp + istft(side_mag, np.angle(complex_side), n_fft, hop_length, length=left.size)

    info = {
        "model": "center_extraction",
        "quality": "spatial only (no source model)",
        "strength": float(strength),
        "mean_mask": round(float(np.mean(mask)), 3),
        "note": (
            "基于空间线索的快速分离：人声居中时效果好、秒级完成；"
            "居中的伴奏（底鼓/贝斯）会残留，立体声加宽的原唱会变差。"
        ),
    }
    return vocal, accomp, info
