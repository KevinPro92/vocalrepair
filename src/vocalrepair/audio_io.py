"""音频读写与格式归一化。

设计原则：核心不依赖重型库。优先用 soundfile（支持 wav/flac/ogg/mp3），
若未安装则回退到标准库级的 scipy.io.wavfile（只支持 WAV），并给出明确
提示。所有读入结果统一为 float、范围约 [-1, 1]。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_audio", "write_audio", "to_mono", "resample_audio", "peak_normalize"]


def _sf_available() -> bool:
    try:
        import soundfile  # noqa: F401
    except Exception:
        return False
    return True


def _normalize_integer(data: np.ndarray) -> np.ndarray:
    """整数 PCM -> float [-1, 1]，按位深选择正确的满量程。"""
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    if data.dtype == np.int32:
        peak = int(np.max(np.abs(data))) if data.size else 0
        # scipy 读 24-bit WAV 时给出 int32 但数值只占低 24 位
        scale = 2.0**23 if peak <= 2**23 else 2.0**31
        return data.astype(np.float64) / scale
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    return data.astype(np.float64)


def read_audio(
    path: str | Path,
    sr: int | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """读取音频文件。

    参数
    ----
    path : 输入文件路径（wav 必支持；flac/ogg/mp3 需安装 soundfile）
    sr : 目标采样率。None 表示保持原采样率
    mono : 是否下混为单声道

    返回
    ----
    (x, sr)：x 为 float 数组，形状 (n,) 或 (n, 声道数)
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到音频文件：{p}")

    data: np.ndarray
    fs: int
    if _sf_available():
        import soundfile as sf

        data, fs = sf.read(str(p), dtype="float64", always_2d=False)
        data = np.asarray(data, dtype=np.float64)
    else:
        from scipy.io import wavfile

        if p.suffix.lower() not in {".wav", ".wave"}:
            raise RuntimeError(
                f"当前未安装 soundfile，只能读取 WAV，收到 {p.suffix!r}。"
                "请先 pip install soundfile，或把文件转为 WAV。"
            )
        fs, raw = wavfile.read(str(p))
        data = _normalize_integer(np.asarray(raw))

    if mono and data.ndim == 2:
        data = data.mean(axis=1)
    if sr is not None and int(sr) != int(fs):
        data, fs = resample_audio(data, int(fs), int(sr))
    return data, int(fs)


def write_audio(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> Path:
    """写音频文件；有 soundfile 用 soundfile，否则回退 32-bit float WAV。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(x, dtype=np.float64)
    if _sf_available():
        import soundfile as sf

        sf.write(str(p), data, int(sr), subtype=subtype)
    else:
        from scipy.io import wavfile

        if p.suffix.lower() not in {".wav", ".wave"}:
            raise RuntimeError("未安装 soundfile 时只能写 WAV")
        wavfile.write(str(p), int(sr), data.astype(np.float32))
    return p


def to_mono(x: np.ndarray) -> np.ndarray:
    """多声道 -> 单声道（算术平均）。

    注意：若两声道存在反相成分，直接平均会产生抵消（mono 兼容性问题）。
    这里不擅自做相位修正，因为分析阶段需要看到真实的抵消现象。
    """
    arr = np.asarray(x, dtype=np.float64)
    return arr.mean(axis=1) if arr.ndim == 2 else arr


def resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> tuple[np.ndarray, int]:
    """有理数倍多相滤波重采样（scipy.signal.resample_poly）。

    相比线性插值，多相滤波器有抗混叠作用，对频谱分析的保真度必要。
    """
    from math import gcd

    from scipy.signal import resample_poly

    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float64), int(sr_in)
    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        out = np.stack([resample_poly(arr[:, c], up, down) for c in range(arr.shape[1])], axis=1)
    else:
        out = resample_poly(arr, up, down)
    return out, int(sr_out)


def peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """按峰值归一化（只用于比较，不用于母带）。"""
    arr = np.asarray(x, dtype=np.float64)
    m = float(np.max(np.abs(arr))) if arr.size else 0.0
    return arr if m == 0 else arr * (peak / m)
