# audit_log — 操作变更记录

> 只追加，不删改历史。格式依据全局规则 `audit-log`（第 11 条）。
> 无密钥、无受试者可识别信息、无敏感数据。

## 2026-10-31 00:05 — 环境事实核验（只读）

- 操作类型：运行命令（只读）
- 目标：`python -c "import numpy,scipy,matplotlib"`；`python -m pip index versions torch/demucs/soundfile`；`nvidia-smi`；`Get-PSDrive`
- 输入摘要：查询解释器版本、可用库、GPU、磁盘余量
- 输出摘要：Python 3.13.7；numpy 2.4.3 / scipy 1.17.1 / matplotlib 3.11.2 / pandas 3.0.1；无 soundfile、无 librosa；**nvidia-smi 不存在（无 GPU）**；torch 可用版本 ≥2.6.0；demucs 4.1.0；soundfile 0.14.0；E 盘剩余 85.2 GB
- 外部 API：是（PyPI 索引查询，只读元数据）
- 敏感数据：否
- 依据规则：`security-governance §5.1`（本地优先）、`paywall-handling §5`（付费调用前置评估，本例未产生费用）
- 结果：成功

## 2026-10-31 00:10 — 创建项目骨架

- 操作类型：新建
- 目标：`E:\pitch-corrector\` 及 `src\vocalrepair\{dsp,analysis,separation,repair}`、`tests`、`data\{raw,interim,processed}`、`figures`、`results`、`logs`、`docs`、`examples`、`project_memory`
- 输入摘要：按 `project-structure-naming` §1 骨架裁剪
- 输出摘要：16 个目录创建成功
- 外部 API：否
- 敏感数据：否
- 依据规则：`project-structure-naming §1`、`save-location`（位置由用户指定 E:\pitch-corrector）
- 结果：成功

## 2026-10-31 00:15 — 写入 DSP 内核与测试

- 操作类型：新建
- 目标：`src\vocalrepair\dsp\pitch_scale.py`、`dsp\f0.py`、`synth.py`、`audio_io.py`、`__init__.py`、`pyproject.toml`、`tests\{__init__,test_pitch_scale,test_f0}.py`
- 输入摘要：A4=440 音高映射、YIN 基频检测、源-滤波器合成器、WAV IO
- 输出摘要：首次运行 24 测试 3 失败（详见下条更正）
- 外部 API：否
- 敏感数据：否
- 依据规则：`iterative-development-debugging`（增量 + 每步验证）、`User_preference §7`（工作流）
- 结果：待确认 → 已修复

## 2026-10-31 00:20 — 修复三处测试失败（其中 2 处为真实代码缺陷）

- 操作类型：写（修改源文件）
- 目标：`dsp\pitch_scale.py`（`smooth_center_track`、`snap_to_scale`）、`synth.py`（`clipped_sample_mask`）、`tests\test_pitch_scale.py`
- 输入摘要：失败项 = 中值滤波抑制不足 / 削波平台首尾漏标 / 音阶吸附选错方向
- 输出摘要：窗长改为由颤音周期决定（11→19 帧）；削波改为 run-based 整体标记；吸附改为按距离取最近；1 处为测试自身 `cents_between` 方向写反（代码无误）
- 外部 API：否
- 敏感数据：否
- 依据规则：`ethics-literature-critique §6.1`（零伪造：不以改测试掩盖缺陷）
- 结果：成功（27/27 通过）

## 2026-10-31 00:25 — 端到端示例运行

- 操作类型：运行命令
- 目标：`python examples\quickstart.py`
- 输入摘要：合成 A3/B3/C4/D4 + 人为偏差 [+40,-25,+15,-55] cents + 颤音 35 cents
- 输出摘要：2.65 s / 260 帧 / 222 浊音帧；f0 检测误差中位数 **2.77 cents**；图输出 `figures\quickstart_f0_track.png`；发现吸附方向 bug 并已修复（目标 261.63 → 293.66）
- 外部 API：否
- 敏感数据：否
- 依据规则：`scientific-research-profile §4`（数据分析须验证并出图）
- 结果：成功

## 2026-10-31 00:28 — 补齐开源文件与项目记忆

- 操作类型：新建
- 目标：`README.md`、`README.zh-CN.md`、`LICENSE`（MIT）、`project_memory\decision_log.md`、`project_memory\audit_log.md`
- 输入摘要：能力矩阵、cents 精度论证、设计原则、已知边界
- 输出摘要：文档齐备；未编造作者信息（LICENSE 署名为 "vocalrepair contributors"）
- 外部 API：否
- 敏感数据：否
- 依据规则：`ethics-literature-critique §6.1`（不伪造）、`audit-log §5`（收尾留痕）
- 结果：成功

## 2026-10-31 02:10 — v0.2/v0.3 实现（分析层 + 修复层 + CLI + 自检）

- 操作类型：新建 / 写 / 运行命令
- 目标：`vocal_spectra.py`、`src\vocalrepair\{pipeline,report}.py`、
  `src\vocalrepair\dsp\{stft,octave}.py`、`src\vocalrepair\analysis\{spectral,vibrato,voice_quality,formant,notes,key}.py`、
  `src\vocalrepair\repair\{declip,pitch_correct,world_vocoder,postproc}.py`、
  `tests\test_repair.py`、`requirements.txt`、`README.md`、`README.zh-CN.md`
- 输入摘要：按 prompt.md 规格实现分析/修复/报告/自检；依赖 pyworld 0.3.5（cp313 轮子实测可用）
- 输出摘要：35 个单测全通过；`--self-test` 9/9 通过（削波 14126→0，音准 33.6→14.1 cents，
  绝对音高 427.6→436.4 Hz，+2 半音共振峰偏移 0.7%）；产物 `outputs\self_test\{wav,png,json}`
- 外部 API：是（PyPI 安装 soundfile/tqdm/pyworld）
- 敏感数据：否
- 依据规则：`iterative-development-debugging`（增量 + 每步验证）、`scientific-research-profile §4`（须验证并出图）
- 结果：成功

## 2026-10-31 02:05 — 修正过程中的两个真实缺陷（更正记录）

- 操作类型：写（修改源文件）
- 目标：`vocal_spectra.py`（自检断言与测试信号）、`src\vocalrepair\analysis\formant.py`（带宽阈值）
- 输入摘要：自检未检出"整体差一个八度"；LPC F3 被 50 Hz 下限过滤
- 输出摘要：新增第 9 条"绝对八度"断言；测试信号改为纯谐波叠加；LPC 带宽下限 50→20 Hz、上限 700→900 Hz
- 外部 API：否
- 敏感数据：否
- 依据规则：`ethics-literature-critique §6.1`（不以调测试掩盖缺陷；缺陷与根因如实记录）
- 结果：成功（详见 `decision_log.md` 同日的"三个教训"）

