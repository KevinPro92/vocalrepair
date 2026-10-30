# vocalrepair（中文说明）

开源的**频谱分析与人声修复**工具包（Python）。给定一个音频文件，目标是：
先**看清**人声哪里有问题，再**针对性修好**，同时保留歌手原有的音色与颤音。

> English version: [README.md](README.md)

## 当前状态（诚实标注）

版本 v0.1.0-dev。**DSP 内核已完成并通过 27 个单元测试**，其余为规划中：

| 层 | 能力 | 状态 |
|---|---|---|
| `dsp` | A4 = 440 Hz 音高刻度、音分/MIDI/音名互转、调内吸附 | **已完成** |
| `dsp` | YIN 基频跟踪（60–1000 Hz，亚音分精度）、清浊音判决 | **已完成** |
| `dsp` | 保颤音软修正（死区 + 中值中心轨迹） | **已完成** |
| `analysis` | 谱峰/谱谷、谱平坦度/对比度/倾斜、LTAS | **已完成** |
| `analysis` | 颤音 rate/depth/extent/regularity | **已完成** |
| `analysis` | HNR、CPP、jitter、shimmer、H1-H2 | **已完成** |
| `analysis` | LPC 共振峰追踪（极点法，与 f0 无关） | **已完成** |
| `analysis` | 音符事件表、调性估计（Krumhansl-Kessler） | **已完成** |
| `repair` | 削波修复（≤10ms 样条 / 长段保守）、音准修正（死区 16 / 残差 8 cents，保颤音） | **已完成** |
| `repair` | WORLD 保共振峰变调（±5 半音硬限）、包络 EQ、压缩、软限幅 | **已完成** |
| `report` | `report.json` + 4 子图对比 + 自包含 HTML | **已完成** |
| `separation` | HPSS 基线（质量低）；装了 demucs 会自动优先调用 | 仅基线 |
| `reference` | 参考音频 DTW 对齐 / 音色匹配 | 未实现 |

## 为什么全部用音分（cents）

人耳对音高近似对数感知：同样 5 Hz 偏差，在 100 Hz 处是 **84 cents**（极刺耳），
在 1000 Hz 处只有 **8.6 cents**（几乎听不出）。因此本项目所有阈值、偏差、
颤音深度统一写成：

```
cents = 1200 × log2(f / f_ref),   f_ref = A4 = 440.000 Hz
f = 440 × 2^((n - 69)/12)          （n 为 MIDI 音高号）
```

由此推出一条硬性精度要求：

| 方法 | 200 Hz 处的分辨率 | 结论 |
|---|---|---|
| STFT bin 序号（44.1 kHz / N=2048） | ≈186 cents | 不可用 |
| 谱峰抛物线插值 | ≈19 cents | 勉强 |
| 相位/瞬时频率 | ≈0.01 Hz | 可用 |
| 时域周期（YIN） | ≈0.15 cents | **本项目采用** |

所以本项目在**时域测周期**，STFT 只用于频谱**整形**。

## 安装与运行

```bash
pip install -e .            # 核心只有 numpy + scipy
pip install -e ".[dev]"     # + pytest / soundfile / matplotlib
pip install -e ".[neural]"  # + torch / demucs（可选，无 GPU 时很慢）
```

```bash
python examples/quickstart.py   # 输出 figures/quickstart_f0_track.png
python -m unittest discover -s tests -t .
```

## 三条设计原则

1. **保护颤音，而不是消除颤音**：自然颤音 4–7 Hz、深度 20–80 cents。修正只作用于
   **中值中心轨迹**，并保留死区（默认 15 cents），让演唱的倾向性活下来。
2. **要死区，不要"绝对准"**：把每个音拉到 0 cents 正是"软件修音感"的来源。
3. **用合成信号做真值**：真实录音的真实基频未知，用它验证 f0 估计是循环论证。
   所有精度声明都基于逐采样真值已知的合成信号。

## 已知能力边界（写在代码里，不藏）

- 中值滤波对颤音调制只能抑制约 70%（40 cents → 残差约 12 cents），不能完全消除。
- 削波丢失的波峰只能**估计**，不能无损恢复。
- 4-stem 分离只给"人声/鼓/贝斯/其他"，**主唱与和声的分离没有现成模型**，
  只能是启发式实验特性。

## 许可

MIT，见 [LICENSE](LICENSE)。
