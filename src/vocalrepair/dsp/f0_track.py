"""稳健基频跟踪：YIN 多候选 + Viterbi 全局最优（pYIN 的轻量实现）。

原始 YIN 每帧只保留一个 tau，一旦选错（例如选了 2Δ 而不是 Δ）后续无法修正。
pYIN（Mauch & Dixon, ICASSP 2014）的做法是三段式：

1. 每帧保留**多个候选** tau —— CMND 曲线在搜索区间内的全部局部极小
2. 每个候选带一个代价（CMND 越小 = 越像周期）
3. 在所有帧的候选网格上用 **Viterbi** 求全局最小代价路径

转移代价对"相邻帧音高突变"收费，对"八度跳变"收重罚。偶发的单帧八度错误
会被邻域拉回；真实的八度跳进因为跳进后稳定持续，累计代价更低，不会被误改。

能力边界：Viterbi 只能修正**偶发**八度错误。若一段信号**全程**缺失基频
（F1 恰好落在 2f0 上、H1 比 H2 弱 20-30 dB），f0 的周期性证据不存在，
任何周期类算法都无法恢复——这是物理限制，只能靠 ``dsp.octave`` 报告与
pipeline 的输出侧守护如实告警。

搜索范围 65-1100 Hz（C2 - C6 附近）：下限 65 Hz 因为常规人声最低约 E2=82 Hz、
极端男低音到 C2；不取 C1(32.7 Hz)，那一带只有低频隆隆声。上限 1100 Hz 覆盖
女高音 C6(1046.5 Hz)，不含哨音。**范围限制只能排除"跳出人声区间"的错误，
排除不了区间内的八度错误**：实测 220 Hz 被判成 110 Hz，两个候选都在范围内。
真正的解药是下面的周期惩罚 + 多候选 Viterbi。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .f0 import F0Result, _cmnd, difference_function

__all__ = ["CandidateGrid", "yin_candidates", "viterbi_track", "robust_f0", "hps_f0"]

_EPS = 1e-12


@dataclass
class CandidateGrid:
    """逐帧候选网格。freqs/costs 形状 (帧数, K)，无效位置为 NaN/inf。"""

    freqs: np.ndarray
    costs: np.ndarray
    times: np.ndarray
    voiced: np.ndarray
    frame_length: int
    hop_length: int
    sr: int

    @property
    def dt(self) -> float:
        return self.hop_length / self.sr


def _local_minima(c: np.ndarray, tau_min: int, tau_max: int, k: int) -> list[tuple[int, float]]:
    """取 CMND 在 [tau_min, tau_max] 内的前 k 个局部极小，按 CMND 升序。"""
    hi = min(tau_max, c.size - 2)
    if hi <= tau_min:
        return [(int(np.argmin(c)), float(np.min(c)))]
    idx = np.arange(tau_min, hi + 1)
    is_min = (c[idx] < c[idx - 1]) & (c[idx] <= c[idx + 1])
    cand = idx[is_min]
    if cand.size == 0:
        cand = np.array([int(idx[np.argmin(c[idx])])])
    order = np.argsort(c[cand])
    return [(int(cand[i]), float(c[cand[i]])) for i in order[:k]]


def yin_candidates(
    x: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 1100.0,
    frame_length: int | None = None,
    hop_length: int | None = None,
    k_candidates: int = 4,
    relative_energy_db: float = -50.0,
    period_penalty: float = 0.08,
) -> CandidateGrid:
    """逐帧生成 YIN 候选（每个候选都做抛物线插值细化）。

    period_penalty（关键，不要设成 0）
    --------------------------------
    CMND 曲线随 tau 增大而**系统性下降**：累积均值
    ``c(tau) = d(tau)*tau / sum_{k<=tau} d(k)`` 在长周期处分母饱和、分子受周期
    重复而变小，于是"两倍周期"总比"一倍周期"的 CMND 更低。若直接拿 CMND 当
    Viterbi 的发射代价，就必然系统性偏向倍周期——实测 220 Hz 纯音被判成 110 Hz。

    这正是原始 YIN 坚持"取**第一个**低于阈值的局部极小"而不是"取全局最小"的原因。
    这里改成显式周期惩罚：``发射代价 = CMND(tau) + period_penalty * log2(fmax/f)``。
    默认 0.08：每低一个八度多付 0.08，而 220 与 110 的 CMND 差通常只有 0.01 量级，
    足以让"一倍周期"胜出；真实低音之间没有更短的竞争项，不会被压掉。
    """
    sig = np.asarray(x, dtype=float).ravel()
    if sig.size == 0:
        raise ValueError("输入信号为空")

    tau_min = max(1, int(np.floor(sr / float(fmax))))
    tau_max = max(tau_min + 2, int(np.ceil(sr / float(fmin))))
    if frame_length is None:
        frame_length = int(round(0.040 * sr))
    frame_length = max(int(frame_length), 2 * tau_max + 2)
    if hop_length is None:
        hop_length = int(round(0.010 * sr))
    hop_length = max(1, int(hop_length))

    span = frame_length + tau_max + 1
    if sig.size < span:
        sig = np.pad(sig, (0, span - sig.size))
    n_frames = 1 + (sig.size - span) // hop_length
    starts = np.arange(n_frames) * hop_length

    freqs = np.full((n_frames, k_candidates), np.nan)
    costs = np.full((n_frames, k_candidates), np.inf)
    rms = np.zeros(n_frames)

    for i, start in enumerate(starts):
        seg = sig[start : start + span]
        seg = seg - seg.mean()
        rms[i] = float(np.sqrt(np.mean(seg[:frame_length] ** 2)))
        d = difference_function(seg, frame_length, tau_max)
        if not np.any(d > 0):
            continue
        c = _cmnd(d)
        for k, (tau, cost) in enumerate(_local_minima(c, tau_min, tau_max, k_candidates)):
            tau_hat = float(tau)
            if 0 < tau < c.size - 1:
                y0, y1, y2 = float(c[tau - 1]), float(c[tau]), float(c[tau + 1])
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-12:
                    tau_hat = tau + float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            if tau_hat > 0:
                f = sr / tau_hat
                freqs[i, k] = f
                costs[i, k] = cost + period_penalty * np.log2(max(fmax, f) / max(f, _EPS))

    peak = float(rms.max()) if rms.size else 0.0
    if peak > 0:
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12) / peak)
    else:
        rms_db = np.full_like(rms, -np.inf)
    voiced = (rms_db > relative_energy_db) & np.any(np.isfinite(freqs), axis=1)

    times = (starts + frame_length / 2.0) / float(sr)
    return CandidateGrid(
        freqs=freqs,
        costs=costs,
        times=times,
        voiced=voiced,
        frame_length=frame_length,
        hop_length=hop_length,
        sr=int(sr),
    )

def _is_octave_jump(delta_cents: float, tol: float = 180.0) -> bool:
    """判断两个音高之间是否相差约一个（或两个）八度。"""
    if not np.isfinite(delta_cents):
        return False
    return abs(delta_cents - 1200.0) <= tol or abs(delta_cents - 2400.0) <= tol


def viterbi_track(
    grid: CandidateGrid,
    cents_per_100_jump: float = 0.20,
    octave_jump_cost: float = 3.0,
) -> np.ndarray:
    """在候选网格上求全局最优 f0 路径，返回长度 = 帧数 的数组（清音为 NaN）。

    代价 = sum 发射代价(CMND + 周期惩罚) + sum 转移代价
        转移代价 = 0.20 * |Δcents| / 100 + (3.0 若为八度跳变)

    数值依据：单帧八度错误（Δ=1200 cents）在纯突变代价下是 0.20*12 = 2.4，
    加 3.0 罚则为 5.4；而拉回正确候选通常只需多付 0.1~0.5 的发射代价，因此
    偶发错误被稳定拉回。真实八度跳进也付 5.4，但后续持续稳定，整段累计代价
    仍优于"假装没跳"。
    """
    T, K = grid.freqs.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(np.maximum(grid.freqs, _EPS) / 440.0)
    cents = np.where(np.isfinite(grid.freqs), cents, np.nan)

    out = np.full(T, np.nan)

    # 只在连续浊音段内部做 DP：跨清音段不连路径（换气/停顿本就不该被平滑）
    i = 0
    while i < T:
        if not grid.voiced[i]:
            i += 1
            continue
        j = i
        while j < T and grid.voiced[j]:
            j += 1
        _viterbi_segment(grid, cents, out, i, j, cents_per_100_jump, octave_jump_cost)
        i = j
    return out


def _viterbi_segment(
    grid: CandidateGrid,
    cents: np.ndarray,
    out: np.ndarray,
    lo: int,
    hi: int,
    cents_per_100_jump: float,
    octave_jump_cost: float,
) -> None:
    """对 [lo, hi) 一段做 Viterbi 并写入 out。"""
    n = hi - lo
    K = grid.freqs.shape[1]
    dp = np.full((n, K), np.inf)
    back = np.zeros((n, K), dtype=int)

    dp[0] = grid.costs[lo]
    for t in range(1, n):
        row_cost = grid.costs[lo + t]
        c_now = cents[lo + t]
        c_prev = cents[lo + t - 1]
        for k in range(K):
            if not np.isfinite(grid.freqs[lo + t, k]):
                continue
            best, best_j = np.inf, 0
            for j in range(K):
                if not np.isfinite(dp[t - 1, j]):
                    continue
                if not (np.isfinite(c_now[k]) and np.isfinite(c_prev[j])):
                    continue
                delta = abs(float(c_now[k] - c_prev[j]))
                cost = dp[t - 1, j] + cents_per_100_jump * delta / 100.0
                if _is_octave_jump(delta):
                    cost += octave_jump_cost
                if cost < best:
                    best, best_j = cost, j
            if np.isfinite(best):
                dp[t, k] = row_cost[k] + best
                back[t, k] = best_j

    last = int(np.argmin(dp[-1]))
    if not np.isfinite(dp[-1, last]):
        return
    path = [last]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    for t, k in enumerate(path):
        out[lo + t] = grid.freqs[lo + t, k]


def hps_f0(
    x: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 1100.0,
    n_fft: int = 4096,
    hop_length: int = 1024,
    n_harmonics: int = 4,
) -> np.ndarray:
    """谐波积谱（HPS）基频估计，作为 YIN 的**独立第二意见**。

    原理：把幅度谱按 1,2,3,4 倍抽取并相乘；只有真正的基频位置，所有次谐波
    同时有能量，乘积才会显著突出。对"奇次谐波弱导致的倍频错误"特别有效，
    因为倍频位置缺少 1f 这一项。

    局限：低频分辨率受 FFT 长度限制（4096 @ 44.1kHz -> 10.8 Hz，在 65 Hz 处
    约 270 cents），因此只用来仲裁"是否差了一个八度"，绝不直接输出为 f0。
    """
    sig = np.asarray(x, dtype=float).ravel()
    win = np.hanning(n_fft)
    n_frames = 1 + max(0, (sig.size - n_fft) // hop_length)
    out = np.full(max(1, n_frames), np.nan)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    df = float(freqs[1] - freqs[0])
    bin_lo = max(1, int(fmin / df))
    bin_hi = min(freqs.size - 1, int(fmax / df))

    for t in range(max(1, n_frames)):
        s = t * hop_length
        seg = sig[s : s + n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        mag = np.abs(np.fft.rfft(seg * win))
        hps = mag.copy()
        for k in range(2, n_harmonics + 1):
            dec = mag[::k]
            m = min(hps.size, dec.size)
            hps[:m] = hps[:m] * dec[:m]
        if bin_hi <= bin_lo:
            continue
        idx = bin_lo + int(np.argmax(hps[bin_lo:bin_hi]))
        out[t] = freqs[idx]
    return out


def robust_f0(
    x: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 1100.0,
    k_candidates: int = 4,
    octave_jump_cost: float = 3.0,
    cross_check_hps: bool = False,
    max_aperiodicity: float = 0.45,
) -> F0Result:
    """稳健 f0：YIN 多候选 + Viterbi 全局路径（+ 可选 HPS 交叉标记）。

    max_aperiodicity 的作用
    -----------------------
    Viterbi 会把最优路径一路铺满整个浊音段，包括本来没有任何候选像周期的帧
    （噪声、气声）。原始 YIN 靠 aperiodicity 门限拒绝这些帧，这里必须补回同样
    的门限，否则清音会被硬判成有声——实测自检第 5 条就是这样失败的（4 帧误判）。

    cross_check_hps
    ---------------
    True 时用 HPS 做独立第二意见：若两者中位数分歧接近 1200 cents，把整段标为
    不确定（aperiodicity 置 1），交由上层守护与报告处理，而不是静默改数。
    """
    grid = yin_candidates(x, sr, fmin=fmin, fmax=fmax, k_candidates=k_candidates)
    f0 = viterbi_track(grid, octave_jump_cost=octave_jump_cost)

    aperiodicity = np.ones(f0.size)
    finite = np.isfinite(f0)

    if cross_check_hps and finite.any():
        hps = hps_f0(x, sr, fmin=fmin, fmax=fmax)
        if hps.size > 1:
            t_hps = np.arange(hps.size) * (1024 / float(sr))
            valid_h = np.isfinite(hps)
            if valid_h.sum() >= 2:
                hps_on = np.interp(grid.times, t_hps[valid_h], hps[valid_h])
                med_yin = float(np.median(f0[finite]))
                med_hps = float(np.median(hps_on))
                if med_yin > 0 and med_hps > 0:
                    delta = 1200.0 * np.log2(med_hps / med_yin)
                    if _is_octave_jump(abs(delta)):
                        aperiodicity[:] = 1.0

    # 浊音门限：用"该帧最好候选的代价"重新过滤 Viterbi 铺满的结果。
    # 注意必须用 inf 而不是 nan 填充：np.min（非 nanmin）遇到 NaN 会**整行**
    # 返回 NaN，导致 voiced 全 False——实测 220 Hz 纯音被判定为"全段清音"。
    # 用 inf 填充既没有 NaN 传播问题，也不会触发 nanmin 的全 NaN 警告。
    costs_masked = np.where(np.isfinite(grid.costs), grid.costs, np.inf)
    best_cost = np.min(costs_masked, axis=1)
    best_cost = np.where(np.isfinite(best_cost), best_cost, np.nan)

    voiced = np.isfinite(f0) & np.isfinite(best_cost) & (best_cost < max_aperiodicity)
    if cross_check_hps:
        voiced = voiced & (aperiodicity < 0.5)

    f0 = np.where(voiced, f0, np.nan)
    aperiodicity = np.where(voiced, np.minimum(best_cost, 1.0), 1.0)

    return F0Result(
        f0_hz=f0,
        times=grid.times,
        aperiodicity=aperiodicity,
        voiced=voiced,
        frame_length=grid.frame_length,
        hop_length=grid.hop_length,
        sr=int(sr),
    )

