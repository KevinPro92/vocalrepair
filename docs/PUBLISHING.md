# 发布到 GitHub（仓库已就绪，只差最后一步授权）

本地仓库已经初始化并完成首次提交，**只差"创建远程仓库 + 推送"**。这一步需要
你的 GitHub 授权，我无法代你完成（浏览器登录 ≠ 命令行凭据）。

## 方式 A：网页建仓库（最快，30 秒）

1. 打开 <https://github.com/new>
2. Repository name 填 **`vocalrepair`**
3. 选 **Public**（开源）
4. **不要**勾选 "Add a README / .gitignore / license"（本地已有）
5. 点 Create repository

然后在本项目根目录执行（会弹出浏览器让你授权一次）：

```powershell
git push -u origin main
```

Windows 上 Git Credential Manager 会弹窗，选 **"Sign in with your browser"**
即可，登录一次以后就不用再输。

## 方式 B：装 GitHub CLI（以后更省事）

```powershell
winget install --id GitHub.cli
gh auth login                                   # 选 GitHub.com -> HTTPS -> 浏览器
gh repo create vocalrepair --public --source=. --remote=origin --push
```

## 方式 C：用 Personal Access Token

1. <https://github.com/settings/tokens> → Generate new token (classic)
2. 勾选 `repo` 与 `workflow`（CI 需要 workflow 权限）
3. 推送时把 token 当密码：

```powershell
git push https://KevinPro92:<YOUR_TOKEN>@github.com/KevinPro92/vocalrepair.git main
```

> 注意：不要把 token 写进任何文件或 `.gitignore` 之外的地方；用完请在 GitHub 上轮换。

## 推送完成后建议做的三件事

1. **确认 CI 通过**：仓库页 → Actions → 应看到 `tests (py3.10~3.13 / ubuntu, windows)` 全绿
2. **填 About**（右上角齿轮）：
   - Description: `Open-source spectral analysis and vocal repair: pitch tracking, de-clipping, formant-preserving transpose, with an evidence-based report.`
   - Topics: `audio-dsp` `pitch-detection` `vocal` `world-vocoder` `music-technology` `python`
3. **打第一个 tag**：
   ```powershell
   git tag -a v0.4.2 -m "v0.4.2: note-level DTW alignment, F0 range 65-1100 Hz, adaptive dead zone"
   git push origin v0.4.2
   ```

## 如果 push 报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `Repository not found` | 远程仓库还没创建，或名字不对 | 按方式 A 建仓库，确认是 `vocalrepair` |
| `Authentication failed` | 没有可用凭据 | 方式 B 或 C |
| `updates were rejected` | 远程已有 commit（勾选了 Add README） | `git pull --rebase origin main` 后再 push |
| `refusing to merge unrelated histories` | 同上 | `git pull origin main --allow-unrelated-histories` |

## 首次提交里包含什么

```
vocal_spectra.py                 CLI 入口（analyze / repair / separate / --self-test）
src/vocalrepair/                 15 个模块（dsp / analysis / repair / pipeline / report）
tests/                           4 个测试文件，35 个用例
examples/quickstart.py           可复现示例
docs/                            文档（含本文件）
project_memory/                  decision_log / audit_log（9 个坑的完整记录）
.github/workflows/ci.yml         CI（py3.10-3.13 × ubuntu/windows）
README.md / README.zh-CN.md     中英双语说明
CHANGELOG.md / CONTRIBUTING.md  变更日志与贡献指南
LICENSE                          MIT
pyproject.toml / requirements.txt
```

**不入库**（已在 `.gitignore` 中排除）：`outputs/`、`data/raw/`、`__pycache__/`、
模型权重、日志。
