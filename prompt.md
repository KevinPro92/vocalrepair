# VocalSpectra - 人声频谱分析与修音工具

一个基于 Python 的开源频谱分析 + 人声修音工具：对给定音频做波峰/波谷与频率波动分析，分离人声与伴奏，针对人声的“fallacy”（音准偏差、破音、音色单薄、颤音受损）做可解释的修正，并在**不破坏歌手音色与共鸣**的前提下实现变调与音域拓宽。支持以参考音频（原唱 / 指定翻唱）为目标的风格对齐。

> **定位**：工程可复现的分析 + 修复管线，而不是黑盒“一键电音”。所有修复都记录在 `report.json` 中，参数可追溯，频率基准严格采用 **A4 = 440Hz**。

---

## 目录

- [1. 功能范围](#1-功能范围)
- [2. 快速开始](#2-快速开始)
- [3. 目录结构](#3-目录结构)
- [4. CLI 规格](#4-cli-规格)
- [5. 算法管线与候选算法](#5-算法管线与候选算法)
- [6. 参考音频对齐（加分项 A）](#6-参考音频对齐加分项-a)
- [7. 音域拓宽与音色保持（加分项 B）](#7-音域拓宽与音色保持加分项-b)
- [8. 可视化与报告](#8-可视化与报告)
- [9. 自检](#9-自检)
- [10. Agent 执行清单](#10-agent-执行清单)
- [11. 禁止事项](#11-禁止事项)
- [12. 物理与工程边界](#12-物理与工程边界)

---

## 1. 功能范围

必须实现：

1. **频谱分析**：输出短时傅里叶变换（STFT）频谱图、基频 F0 轨迹、谐波能量分布、频谱包络、波峰/波谷统计。
2. **人声/伴奏分离**：输出 `vocals.wav`、`accompaniment.wav`；可选输出 `harmony.wav`（和声，若模型支持）。
3. **音准修正**：基于 A4=440Hz 的半音体系，将有声帧 F0 向目标音高量化；保留 4–7Hz 自然颤音；限制单帧最大修正量。
4. **破音/削波修复**：检测削波样本，插值重建峰值；严重段标记并限幅。
5. **音色饱满化**：频谱包络动态 EQ、低次谐波支撑、齿音控制、轻度压缩。
6. **共振峰保持的变调**：升/降 key 时共振峰不随之整体平移，避免“机器化尖锐”或“水牛音”。
7. **可复现**：固定默认参数，输出 `report.json`、频谱对比图、F0 对比图。

可选加分项（见第 6、7 节）：

- **参考音频对齐**：与原唱 / 指定翻唱的 F0 与频谱包络对齐。
- **音域拓宽**：在保持本人音色特征的前提下，让高音更通透、低音更扎实。

---

## 2. 快速开始

```bash
git clone <your-repo>
cd VocalSpectra
python -m pip install -r requirements.txt

# 自检（合成测试音频，必须通过）
python vocal_spectra.py --self-test

# 分离人声/伴奏
python vocal_spectra.py separate --input input.mp3 --outdir outputs/

# 完整修音
python vocal_spectra.py repair \
  --input input.mp3 \
  --output outputs/repaired.wav \
  --strength 0.7 \
  --key auto \
  --clip-repair auto \
  --make-mp3 \
  --report outputs/report.json \
  --plot outputs/compare.png

# 参考原唱对齐（加分项 A）
python vocal_spectra.py repair \
  --input input.mp3 \
  --reference reference.wav \
  --align-pitch \
  --align-timbre conservative \
  --output outputs/repaired_ref.wav
```

系统依赖：`ffmpeg`。启动时检查 `ffmpeg -version`，缺失则报错并提示安装方式。

`requirements.txt`：

```text
numpy>=1.24
scipy>=1.10
librosa>=0.10
soundfile>=0.12
pyworld>=0.3.4
matplotlib>=3.7
tqdm>=4.66
torch>=2.0          # 若使用 CREPE / Demucs / 神经声码器
torchaudio>=2.0
demucs>=4.0         # 可选：人声分离
crepe>=0.0.13       # 可选：F0 提取
```

> `demucs`、`crepe`、`torch` 为可选重依赖。若安装失败，降级到 `librosa` + `pyworld` 路径，并在报告中标记 `degraded_mode=true`，禁止静默失败。

---

## 3. 目录结构

```text
VocalSpectra/
  vocal_spectra.py          # CLI 入口
  core/
    io.py                   # 解码 / 编码 / ffmpeg 封装
    clip_repair.py          # 削波修复
    source_sep.py           # 人声分离
    pitch.py                # F0 提取与量化
    vocoder.py              # WORLD / 相位声码器 / 神经声码器
    timbre.py               # 频谱包络 EQ、齿音、共振峰
    dynamics.py             # 压缩与限幅
    reference.py            # 参考音频对齐
    analyze.py              # 频谱分析统计
  requirements.txt
  README.md
  tests/
    test_self.py
  outputs/
```

---

## 4. CLI 规格

子命令：`separate` / `analyze` / `repair`。

### `repair` 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | `input.mp3` | 输入音频；不存在且非 self-test 时退出码 2 |
| `--reference` | 无 | 参考音频（原唱 / 翻唱），触发加分项 A |
| `--output` | `outputs/repaired.wav` | PCM_16, 44.1kHz, 单声道 |
| `--strength` | `0.7` | 全局修音强度 0~1 |
| `--key` | `auto` | `auto` 或 `C`/`C#`…；auto 用 chroma 估计主音，仅用于报告与可选音阶量化 |
| `--transpose` | `0` | 半音数，正升负降；触发共振峰保持变调 |
| `--vocal-range` | 无 | 形如 `E3-G5`，用于音域拓宽边界 |
| `--clip-repair` | `auto` | `auto`/`off` |
| `--max-cents` | `100` | 单帧最大音高修正量 |
| `--frame-period` | `5` | WORLD 帧移 ms |
| `--sr` | `44100` | 处理采样率 |
| `--vocoder` | `world` | `world` / `phase` / `neural` |
| `--align-pitch` | 关闭 | 与参考 F0 对齐 |
| `--align-timbre` | `off` | `off` / `conservative` / `aggressive` |
| `--make-mp3` | 关闭 | 额外输出 320kbps MP3 |
| `--self-test` | 关闭 | 合成测试并断言管线 |

---

## 5. 算法管线与候选算法

```text
ffmpeg 解码 -> 单声道 44.1kHz float32
→ [分离] 人声 / 伴奏 / 和声
→ [分析] STFT、F0、谐波、包络、削波统计
→ [削波修复]
→ [F0 提取] -> [音准量化 + 颤音保留]
→ [变调 / 音域拓宽]（保留共振峰）
→ [重合成]（WORLD / 相位声码器 / 神经声码器）
→ [音色饱满化] 频谱包络 EQ + 齿音 + 动态
→ [参考对齐]（若提供 reference）
→ [混音回伴奏]（可选）
→ WAV / MP3 + 图 + report.json
```

### 5.1 人声 / 伴奏 / 和声分离

| 算法 | 类型 | 质量 | 速度 | 备注 |
|---|---|---|---|---|
| **Demucs v4 `htdemucs_ft`** | 时域混合 | ★★★★★ | 中 | **首选**；官方 Python 包可直接调用 |
| MDX-Net（UVR 生态） | 频谱域 U-Net | ★★★★☆ | 中 | 人声干净，需下载模型 |
| Open-Unmix (UMX) | 频谱掩码 | ★★★☆ | 快 | 轻量基线 |
| Spleeter 4stems | 频谱 U-Net | ★★☆ | 很快 | 快但残留多，仅作 fallback |
| HPSS（librosa） | 中值滤波 | ★☆ | 极快 | 无人声语义，仅演示 |
| Kim Vocal / VR 系列 | 频谱 | ★★★★ | 中 | 和声保留较好 |

**和声分离**：主流分离模型对人声整体输出一个 stem，和声与主唱混在一起。可行路径：

1. 在 `vocals.wav` 上再用 **Demucs 的 `vocals` 子stem** 或 **MDX23C** 二次分离。
2. 用 **F0 聚类**：对 vocals 的 F0 轨迹做多基频跟踪（如 `librosa.pyin` 多候选 + HMM），把同时存在的不同音高分配给不同“声部”。
3. 保守做法：报告中标注 `harmony_separated=false`，仅提供主唱 stem，避免过度分离引入伪影。

### 5.2 F0 提取

| 算法 | 特点 | 推荐场景 |
|---|---|---|
| **pYIN (`librosa.pyin`)** | 概率分布，抗八度错误 | **默认**，CPU 即可 |
| **CREPE / torchcrepe** | 神经，精度高，需 GPU 更好 | 高质量要求 |
| **FCPE** | 轻量神经，速度快 | 移动端 / 批处理 |
| **pyworld.harvest** | 鲁棒，偶有八度错误 | 与 pYIN 交叉验证 |
| SWIPE / SWIPE' | 基于频谱 | 作为对照 |
| YIN / RAPT | 经典 | 基线 |

**工程建议**：`pYIN` 作为主路径，`harvest` 做交叉校验；两路 F0 偏差 > 50 cents 的帧标记为 `uncertain`，在量化时降低修正强度。

### 5.3 音准量化（保留颤音）

1. 仅处理 `f0 > 0` 的有声帧；清音帧保持 `f0=0`。
2. 转 cents：`c = 1200 * log2(f0 / 440)`（A4=440Hz）。
3. 分离趋势与颤音：`trend = lowpass(c, 3Hz)` 或中值滤波；`vibrato = c - trend`。
4. 目标音高：`target_midi = round(69 + 12*log2(f0/440))`；若指定 `--key`，可吸附到该大调音阶音级。
5. `delta = clip(target - trend, -max_cents, +max_cents)`；`|delta| < 25 cents` 不修。
6. `trend_new = trend + strength * delta`。
7. `f0_new = 440 * 2 ** ((trend_new + vibrato) / 1200)`，仅赋给有声帧。
8. 对 `f0_new` 做短时中值滤波，禁止抹平 4–7Hz 颤音。

> **防机器化**：单帧修正上限、强度参数、颤音残差保留，三者缺一不可。

### 5.4 变调 / 重合成（共振峰保持是核心）

| 算法 | 共振峰 | 音质 | 备注 |
|---|---|---|---|
| **WORLD (`pyworld`)** | 可保持 | ★★★★ | **首选**；`cheaptrick` 取包络、`d4c` 取非周期、改 F0 后 `synthesize` |
| **相位声码器 + 包络校正** | 需额外校正 | ★★★ | 实现简单，易有相位涂抹 |
| **TD-PSOLA** | 可保持 | ★★★ | 适合单声部，需精确基音标记 |
| **Rubber Band** | 可保持 | ★★★★ | C++ 库，`rubberband-cli` 或 python binding |
| **librosa.effects.pitch_shift** | **不保持** | ★★ | 会整体移动共振峰，**不推荐用于人声** |
| **NSF-HiFiGAN / HiFi-GAN 声码器** | 可保持 | ★★★★★ | 需 F0 + 梅尔谱；对失真素材容错性最好 |
| **DDSP** | 可保持 | ★★★★ | 谐波 + 噪声建模，音色可控 |

**变调流程（WORLD）**：

```python
f0, t = pyworld.harvest(x, sr, f0_floor=80, f0_ceil=1000, frame_period=5)
sp = pyworld.cheaptrick(x, f0, t, sr)   # 频谱包络（含共振峰）
ap = pyworld.d4c(x, f0, t, sr)
f0_new = f0 * 2 ** (transpose / 12)
y = pyworld.synthesize(f0_new, sp, ap, sr, frame_period=5)
```

> 因为 `sp` 未随 F0 平移，共振峰位置保持，这就是“升 key 不尖锐、降 key 不水牛”的物理基础。

### 5.5 音色饱满化（频谱包络层面）

| 处理 | 频段 | 目的 | 实现 |
|---|---|---|---|
| 基频支撑 | 150–400Hz | 增加厚度 | 幅度谱乘平滑增益，`+1.5dB * strength` |
| 存在感 | 2–5kHz | 增加扎实感 | `+1.0dB * strength` |
| 齿音控制 | 5–10kHz | 去刺耳 | 逐帧能量比检测，动态衰减 ≤ `2dB * strength` |
| 空气感保留 | > 10kHz | 不闷 | 不做低通，仅去尖峰 |
| 共振峰微调 | F1/F2/F3 附近 | 共鸣感 | 对包络峰值做 ±1dB 平滑提升 |

边界用 Hann 窗平滑过渡，禁止硬切。所有增益在 `sp` 上做，再做逆变换。

### 5.6 破音 / 削波修复

| 算法 | 适用 | 说明 |
|---|---|---|
| **三次样条插值** | 轻微削波（≤10ms） | Audacity ClipFix 同思路 |
| **FSE（频率选择性外推）** | 中等削波 | 用未削波频谱迭代建模，文献报告 SNR 提升约 3.5dB |
| **自适应全极点模型** | 中等 | 对语音类信号有效 |
| **U-Net 音频修复（AudioSR / Demucs inpainting）** | 严重削波 | 需 GPU，质量不稳定 |
| **神经声码器重合成** | 严重失真 | 基于修正 F0 + 包络重生成，容错性最高 |

阈值：`abs(x) >= 0.98` 视为削波样本。长度 > 10ms 的连续段标记为“严重削波”，插值后限幅到 `0.95`，并在报告中记录时长。

### 5.7 动态处理

- RMS 压缩：阈值 `-18dB`，比率 `2:1`，攻击 `10ms`，释放 `100ms`，最大增益 `6dB`。
- 软限幅：仅对 `abs(y) > 0.95` 用 `tanh` 平滑，最终峰值归一化到 `-1dBFS`。

---

## 6. 参考音频对齐（加分项 A）

目标：参考原唱 / 指定翻唱的**音高走向**与**风格频谱**，让待修音频向参考靠拢，同时不过分改变音色。

### 6.1 音高对齐（`--align-pitch`）

1. 对 `input` 与 `reference` 各自提取 F0（同一算法、同一帧移）。
2. 用 **DTW（动态时间规整）** 对齐两条 F0 轨迹；代价函数用 cents 差 + 能量差。
3. 目标 F0 = `α * reference_f0 + (1-α) * 本地量化目标`，`α` 默认 `0.6`。
4. 保留本地 `vibrato` 残差，避免把参考的颤音生硬搬过来。
5. 若 DTW 路径置信度低（长段单点对齐），降级为“仅本地半音量化”，并在报告中说明。

### 6.2 音色 / 风格对齐（`--align-timbre`）

| 模式 | 做法 | 风险 |
|---|---|---|
| `off` | 不做 | — |
| `conservative` | 匹配参考的**平均频谱包络**（长时平均谱），用 EQ 逼近，最大 ±3dB | 低，推荐 |
| `aggressive` | 逐帧匹配包络 + 共振峰位置微调 | 中，可能改变音色身份 |

**实现要点**：

- 长时平均谱（LTAS）匹配：`gain(f) = LTAS_ref(f) / LTAS_src(f)`，平滑后限幅到 ±3dB，作用于 `sp`。
- 共振峰匹配：在包络上检测 F1/F2/F3 峰值位置，若偏移 > 15% 则平滑迁移，否则不动。
- **风格迁移**（如 RVC / so-vits-svc）不建议默认开启：会把音色替换成模型音色，违背“本人混声能力提升”的目标。若要提供，必须作为独立子命令并显著警告。

---

## 7. 音域拓宽与音色保持（加分项 B）

目标：让高音更通透、低音更扎实，但听感仍像**本人**。

### 7.1 分区处理

用 `--vocal-range E3-G5` 标定用户自然音域：

- **自然音域内**：仅做 5.3 的音准量化，不改变音色。
- **超出音域的上方**：变调时**轻微上移共振峰**（包络整体 × `1 + 0.03 * 超出半音数`），模拟“唱上去”的声道变化，但上限 8%。
- **超出音域的下方**：**保持或轻微下移共振峰**（≤ 5%），避免“水牛音”。

### 7.2 神经声码器重合成（可选，`--vocoder neural`）

- 用修正后的 F0 + 原始梅尔谱训练/推理 **NSF-HiFiGAN** 或 **HiFi-GAN**。
- 优点：对削波、噪声容错性高，音质上限高。
- 缺点：需 GPU；音色可能轻微偏移，需 A/B 对比确认。

### 7.3 验收指标

- **F0 偏差中位数**下降。
- **共振峰位置偏移** < 10%（相对原始）。
- **MFCC 距离**（原始 vs 修复）在合理范围，避免音色身份丢失。
- 人工试听：升 key 后不尖锐、降 key 后不沉闷。

---

## 8. 可视化与报告

输出：

- `outputs/repaired.wav` / `repaired.mp3`（320kbps）
- `outputs/vocals.wav`、`accompaniment.wav`
- `outputs/compare.png`：至少 4 子图
  1. 原始 STFT 频谱
  2. 修复后 STFT 频谱
  3. F0 轨迹对比（原始 / 修复 / 参考），叠加削波标记
  4. 频谱包络对比（原始 vs 修复）
- `outputs/report.json`：

```json
{
  "input": "...",
  "reference": null,
  "output": "...",
  "sample_rate": 44100,
  "duration_sec": 0.0,
  "strength": 0.7,
  "key_estimate": "auto",
  "transpose_semitones": 0,
  "vocoder": "world",
  "resonance_preserved": true,
  "degraded_mode": false,
  "separation": {"model": "htdemucs_ft", "harmony_separated": false},
  "clipping": {"before_samples": 0, "after_samples": 0, "severe_segments": 0},
  "f0": {
    "median_abs_cents_before": 0.0,
    "median_abs_cents_after": 0.0,
    "voiced_frames": 0,
    "uncertain_frames": 0
  },
  "timbre": {"ltas_match_db_max": 0.0, "formant_shift_pct": 0.0},
  "reference_align": {"enabled": false, "dtw_confidence": null},
  "params": {}
}
```

---

## 9. 自检

`--self-test` 生成 5 秒合成音频：基频 440Hz + 2~5 次谐波 + 轻微颤音，人为偏移 `-50 cents`，人为 `1.5x` 增益后硬削波到 `[-1,1]`。跑完整管线并断言：

1. 输出 WAV 可读，时长差 < 0.1s。
2. 无 NaN/Inf，峰值 ≤ -0.5dBFS。
3. 削波样本数下降。
4. 有声帧 F0 中位绝对 cents 偏差下降。
5. 清音帧未被改为有声。
6. `compare.png` 与 `report.json` 存在。
7. 指定 `--transpose 2` 时，共振峰位置偏移 < 10%。

自检失败时打印具体断言与中间统计，退出码 1。禁止跳过。

---

## 10. Agent 执行清单

按顺序执行，不要询问用户：

1. 创建目录与文件（见第 3 节），`README.md` 使用本文件内容。
2. 编写 `vocal_spectra.py` 与 `core/*`，含类型提示、`logging`、`argparse`、中文注释。
3. 安装依赖，检查 `ffmpeg`。重依赖安装失败时进入降级模式并记录。
4. 运行 `python vocal_spectra.py --self-test`，必须通过。
5. 若存在 `input.mp3`：
   - 运行 `separate`，输出 `vocals.wav` / `accompaniment.wav`。
   - 运行 `repair --strength 0.7`，生成 WAV、MP3、图、报告。
   - 若报告显示 F0 改善不明显或出现过度量化迹象，用 `--strength 0.5` 重跑一次，选指标更优者，并记录选择理由。
6. 若存在 `--reference` 指定文件，运行加分项 A，输出 `repaired_ref.wav` 与对比报告。
7. 校验输出文件存在、报告字段完整、图片非空。
8. 最终回复：输出路径、报告摘要（削波前后、F0 偏差前后、共振峰偏移、是否降级）、使用参数、局限性。

---

## 11. 禁止事项

- 禁止用整体频谱平移替代保共振峰变调。
- 禁止把清音帧改为有声。
- 禁止无限制半音量化导致明显电音。
- 禁止在 `--align-timbre aggressive` 之外默认做音色替换。
- 禁止静默失败或只输出计划。
- 禁止覆盖原始 mp3。
- 禁止声称 MP3 丢失信息已被“完全恢复”。

---

## 12. 物理与工程边界

- MP3 是有损压缩，已丢失的高频细节与削波信息**无法真正恢复**，所有修复都是估计与平滑。
- 人声/伴奏分离模型对人声与和声通常不做区分，强行分离可能引入伪影。
- 参考音频对齐依赖 DTW，若两者节奏差异过大，置信度会下降，应降级为本地量化。
- 神经声码器重合成会轻微改变音色身份，需 A/B 试听确认。
- 变调超过 ±5 个半音时，即使共振峰保持，音色也会明显变化，属正常物理限制。

---

## License

MIT。欢迎提交 issue 与 PR，尤其是新的分离模型、声码器与对齐算法。