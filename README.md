# vocalrepair

[![CI](https://github.com/KevinPro92/vocalrepair/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinPro92/vocalrepair/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Open-source **spectral analysis and vocal repair** toolkit for Python.

Given an audio file, `vocalrepair` aims to *see* what is wrong with a vocal
take and *fix* it — while keeping the singer's own timbre and vibrato intact.

> **Status: v0.2 / v0.3 — analysis and repair layers are working.** Pitch tracking,
> spectral analysis, vibrato/voice-quality metrics, formant tracking, note-level
> pitch correction, WORLD formant-preserving transposition, de-clipping, timbre
> shaping and reporting are all implemented, and covered by 35 unit tests plus a
> 9-assertion end-to-end self-test. Source separation is a baseline only; reference
> alignment is not implemented. The table below is deliberately honest.

## Capability matrix

| Layer | Capability | Status |
|---|---|---|
| `dsp` | A4 = 440 Hz pitch scale, cents/MIDI/note-name conversion, scale-aware snapping | **done** |
| `dsp` | YIN f0 tracking (60-1000 Hz, sub-cent precision), voicing decision | **done** |
| `dsp` | STFT/ISTFT, spectral peak & valley picking (parabolic interpolation), log-frequency spectrogram, chroma | **done** |
| `dsp` | Octave-error suspects (±1 octave runs relative to local median) | **done** |
| `analysis` | Spectral peaks/valleys, flatness/contrast/slope, LTAS | **done** |
| `analysis` | Vibrato rate / depth / extent / regularity | **done** |
| `analysis` | HNR, CPP, jitter, shimmer, H1-H2 (voice quality) | **done** |
| `analysis` | LPC formant tracking (pole-based, f0 independent) | **done** |
| `analysis` | Note-event table, key estimation (Krumhansl-Kessler) | **done** |
| `repair` | De-clipping (spline for ≤10 ms, conservative for severe) | **done** |
| `repair` | Note-level pitch correction: dead zone 16 cents, residual 8 cents, vibrato preserved | **done** |
| `repair` | PSOLA engine with Praat-style pitch marks (low-pass to 1.5·f0 → peak pick → period check → fill) | **done** |
| `repair` | Glide protection (no correction when f0 slew > 2000 cents/s) — note-level decision | **done** |
| `repair` | Transient protection (spectral flux, MAD threshold) — frame-level | **done** |
| `repair` | WORLD resynthesis — formant-preserving transposition, ±5 semitone hard limit | **done** |
| `repair` | Envelope EQ (low-harmonic support / presence / sibilance), envelope matching, compression, soft limiting | **done** |
| `repair` | Pressed-voice detection (5-metric vote: H1-H2 / H1-A3 / slope / centroid / HNR) + resonance relief on high-confidence segments only | **done** |
| `repair` | Denoise: spectral subtraction / Wiener / static high-cut (off by default — see note) | **done** |
| `report` | `report.json`, 4-panel comparison figure, self-contained HTML | **done** |
| `separation` | Demucs (`--two-stems=vocals`) when installed, plus centre-channel extraction fallback | **done** |
| `reference` | Reference-audio note-level DTW alignment + LTAS timbre weighting | **experimental** (see note) |

Verified numbers from `python vocal_spectra.py --self-test` (9/9 pass):

| Check | Measured |
|---|---|
| Clipped samples | 14126 → **0** |
| Median pitch deviation | 33.6 → **14.1 cents** |
| Absolute pitch after repair | 427.6 → **436.4 Hz** (−14.1 cents vs A4) |
| Formant shift at +2 semitones | **0.7 %** (limit 10 %) |
| Unvoiced frames turned voiced | **0** |

## Command line

```bash
python vocal_spectra.py --self-test                       # end-to-end assertions
python vocal_spectra.py analyze --input song.wav --outdir outputs/
python vocal_spectra.py separate --input song.wav --outdir outputs/
```

### Repairing an arbitrary audio file

```bash
# 1. one-shot: repair + report + figure + HTML
python vocal_spectra.py repair \
    --input "my_take.m4a" \
    --output outputs/repaired.wav \
    --vocoder psola \
    --report outputs/report.json \
    --plot   outputs/compare.png \
    --html   outputs/report.html

# 2. transposition (formant-preserving, hard limit ±5 semitones)
python vocal_spectra.py repair --input a.wav --output b.wav --transpose +2   # up a whole tone
python vocal_spectra.py repair --input a.wav --output b.wav --transpose -3   # down a minor third

# 3. conservative mode: only touch notes that are actually out of tune
python vocal_spectra.py repair --input a.wav --output b.wav --min-dev 30 --strength 0.6

# 4. reference-guided: follow a version you like (runs entirely locally)
python vocal_spectra.py repair --input mine.wav --reference original.wav \
    --align-pitch --align-timbre conservative --output outputs/ref_aligned.wav

# 5. denoise (off by default; A/B it before keeping it)
python vocal_spectra.py repair --input a.wav --output b.wav --denoise specsub

# 6. vocal / accompaniment split, then repair the vocal and remix
python vocal_spectra.py separate --input ktv.m4a --outdir outputs/sep
python vocal_spectra.py repair --input outputs/sep/vocals.wav --output outputs/sep/vocals_fixed.wav --vocoder psola
```

### Key flags

| Flag | Meaning |
|---|---|
| `--vocoder psola\|world\|none` | `psola` = best pitch accuracy, time-domain (no parametric resynthesis); `world` = formant-preserving but parametric (audible "watery" artefacts on breathy material); `none` = no pitch change at all, only de-clipping |
| `--strength 0–1` | How much of the measured deviation to remove (1.0 = full correction, mechanical) |
| `--min-dev N` | Only correct notes whose median deviation exceeds N cents; everything else is left bit-identical |
| `--transpose ±N` | Semitones; clamped at ±5, the point where PSOLA/WORLD start to sound artificial |
| `--key auto\|C\|Am` | Key for scale-aware snapping; `auto` only affects the report, never the correction |
| `--reference FILE` | Reference performance for note-level DTW alignment |
| `--denoise off\|specsub\|wiener\|highcut` | Noise reduction; `off` by default (see below) |
| `--no-compress` | Disable compression + soft limiting (use this to judge whether noise comes from the recording or from make-up gain) |
| `--no-voice-relief` | Disable resonance relief on pressed-voice segments |

### Note on denoise

Measured on a real KTV recording: the noise sits **above 2 kHz and is 10–19 dB
stronger than the voice**, with a temporal CV of 1.16 (i.e. not stationary).
Static high-cut removes **voice** (−4.9 dB) faster than noise (−1.7 dB), so it is
implemented but not recommended. Spectral subtraction and Wiener reduce noise by
about 2.5 dB while leaving the voice band essentially untouched (−0.1 dB) — but
spectral gain changes break magnitude/phase consistency, which produces time-domain
overshoot (measured: peak 0.284 → 0.891 before the fix). Both methods therefore
smooth the gain in frequency *and* time and hard-limit the peak. **Default is
`off`**: a noise reduction stage is the single easiest way to damage a voice you
already like.

## Design lessons (kept here on purpose)

Five real bugs were caught by measurement, not by inspection — all documented in
`project_memory/decision_log.md` (坑 1–9):

1. **A whole octave can slip through a "pitch accuracy" check.** A check on
   *deviation from the nearest semitone* passes happily when the output is one
   octave high (838 Hz is only 15 cents from G#5). An absolute-pitch assertion
   is mandatory. Now backed by three layers: multi-candidate Viterbi, HPS
   cross-check, and an **output-side octave guard that falls back to "do not
   touch pitch"** instead of shipping a wrong-octave result.
2. **Synthetic test signals must not be built from a static spectral envelope.**
   A 440 Hz "vowel" whose 2nd harmonic is 27 dB louder than the fundamental is a
   genuine *missing fundamental* — any autocorrelation tracker will (correctly)
   report 880 Hz.
3. **CMND decreases systematically with τ.** Feeding raw CMND into Viterbi biases
   the result toward double periods (a 220 Hz tone was reported as 110 Hz). A
   period penalty `0.08·log2(fmax/f)` is required.
4. **DTW must run on the full time grid, not on the voiced-frame sequence**, and
   with a narrow band (0.10 ≈ 0.26 s). Otherwise the path slides along the pitch
   axis and locks onto vibrato troughs.
5. **Fuse with the quantised local target, not the raw pitch.** Using raw pitch
   leaks the original detuning into the new target (median error 16 → 30 cents).

## Known limitation: reference alignment

The DTW chain is complete (with degradation guards), and its intermediate values
are correct — aligned reference pitch lands within ~5 cents of the standard
pitch. However, **end-to-end pitch accuracy is still not better than pure local
quantisation** on the synthetic benchmark (27.6 vs 16.0 cents median). The loss
happens between "fused target → per-frame correction → WORLD resynthesis".
The report marks it `experimental: true` and suggests `alpha <= 0.5`, or using it
only to inspect the reference's pitch contour. Fix planned for v0.4.2
(note-level alignment instead of frame-level).


## Why everything is measured in cents

Human pitch perception is approximately logarithmic. The same 5 Hz deviation is
**84 cents** at 100 Hz (painfully out of tune) but only **8.6 cents** at 1000 Hz
(inaudible). So every threshold, deviation and vibrato depth in this project is
expressed as

```
cents = 1200 * log2(f / f_ref)
```

with `f_ref = A4 = 440.000 Hz`, and a MIDI note number `n` mapping to
`f = 440 * 2^((n - 69) / 12)`.

The corollary is a hard precision requirement:

| Method | Resolution at 200 Hz | Verdict |
|---|---|---|
| STFT bin index (44.1 kHz, N = 2048) | ~186 cents | unusable |
| Parabolic peak interpolation | ~19 cents | marginal |
| Phase / instantaneous frequency | ~0.01 Hz | good |
| Time-domain period (YIN) | ~0.15 cents | **what we use** |

A pitch corrector that reads pitch off an STFT bin grid cannot, in principle,
land within 5 cents. That is why this project measures period in the time
domain and only uses the STFT for spectral *shaping*.

## Install

```bash
git clone <your-fork-url> && cd pitch-corrector
pip install -e .            # core: numpy + scipy only
pip install -e ".[dev]"     # + pytest, soundfile, matplotlib
pip install -e ".[neural]"  # + torch, demucs (optional, slow without a GPU)
```

The core deliberately avoids heavy dependencies so that `pytest` runs on any
machine and in CI without downloading model weights.

## Quick start

```python
import numpy as np
from vocalrepair import make_note_sequence, yin_f0, cents_between

# 1. build a test phrase with known ground truth (a slightly out-of-tune A3-B3-C4)
sig = make_note_sequence([57, 59, 60], detune_cents=[40, -25, 15],
                         vibrato_depth_cents=35, sr=44100)

# 2. track the pitch
res = yin_f0(sig.x, sig.sr)
print(res.f0_hz[:5], res.voiced[:5])

# 3. how wrong is it? (ground truth is known per sample)
truth = np.interp(res.times, sig.times, sig.f0_true)
print(np.nanmedian(cents_between(truth, res.f0_hz)))   # ~+40, -25, +15
```

Run the example script for a waveform + f0-track figure:

```bash
python examples/quickstart.py
```

## Design notes

- **Vibrato is protected, not removed.** Natural singing vibrato is 4-7 Hz with
  20-80 cents depth. Correction is applied to the *median centre track* only,
  with a dead zone (default 15 cents) so expressive intonation survives.
- **Dead zone, not "perfect" tuning.** Pulling every note to 0 cents is what
  makes software-corrected vocals sound lifeless.
- **Synthetic ground truth.** Real recordings have unknown true f0, so
  validating an f0 estimator on them is circular. All accuracy claims are made
  on signals whose per-sample f0 is known exactly.
- **Honest limits are written into the docstrings** — e.g. median filtering
  suppresses ~70% of vibrato modulation, not 100%; clipped waveform peaks can
  only be estimated, never restored losslessly.

## GUI options

The DSP layer is pure functions over `numpy` arrays, so **any** GUI is a thin
wrapper around `vocalrepair.pipeline.repair()`. Ranked by effort/benefit:

| Option | Effort | Strengths | Weaknesses |
|---|---|---|---|
| **Gradio** | ~150 lines | drag-and-drop file → audio player + parameter panel; `pip install gradio`; `share=True` gives a public link instantly | limited layout control, not a "native app" |
| **Streamlit** | ~200 lines | best for showing the analysis report (HTML/matplotlib embed directly) | stateless script model; awkward for iterative A/B |
| **PySide6 / PyQt** | ~1.5k lines | real desktop app, can ship as a `.exe` | you have to draw the waveform / piano-roll yourself |
| **FastAPI + web frontend** | largest | unlimited; enables a professional waveform + piano-roll editor | frontend engineering dominates |

Recommended path: **Gradio first** (a usable tool in an afternoon), then move to
PySide6 only if you need a real waveform/piano-roll editor. A minimal Gradio sketch:

```python
import gradio as gr
from vocalrepair.audio_io import read_audio, to_mono, write_audio
from vocalrepair.pipeline import repair

def run(path, strength, semitones, denoise):
    x, sr = read_audio(path, sr=44100)
    res = repair(to_mono(x).astype(float), sr, strength=strength,
                 transpose=semitones, vocoder="psola", denoise=denoise)
    out = "repaired.wav"
    write_audio(out, res.y, sr)
    return out, res.report

gr.Interface(run,
    [gr.Audio(type="filepath"),
     gr.Slider(0, 1, 0.7, label="strength"),
     gr.Slider(-5, 5, 0, step=1, label="transpose (semitones)"),
     gr.Dropdown(["off", "specsub", "wiener"], value="off", label="denoise")],
    [gr.Audio(label="result"), gr.JSON(label="report")],
    title="vocalrepair").launch()
```

## Roadmap

- **done** spectral analysis: peaks/valleys, flatness/contrast/slope, LTAS
- **done** vibrato metrics, HNR/CPP/jitter/shimmer/H1-H2, LPC formants, note table
- **done** repair: de-clipping, note-level pitch correction, PSOLA (Praat-style pitch marks)
  and WORLD engines, glide/transient protection, pressed-voice relief, ±5 semitone transposition
- **done** report: JSON + 4-panel figure + self-contained HTML
- **done** Demucs 2-stem separation + centre-channel fallback
- **done** reference-audio note-level DTW alignment (reliable for timing, experimental for pitch)
- **next** denoise: a better noise model than "quietest 10 % of frames"
- **next** singing-vocal break detection (non-clipping "破音": HNR collapse + f0 jump +
  aperiodicity burst). Currently only **digital** clipping is repaired.
- **next** GUI (see above)
- **v0.6** microphone realtime tuner (`sounddevice`), realtime spectrogram, metronome
- **v0.7** multi-pitch (polyphonic/choir) analysis

## What this tool deliberately does not do

- No automatic scoring. It reports measured values and reference ranges, never a
  single "quality score" — artistic quality cannot be reduced to one number.
- No vocal-mode labels (chest / mix / head / falsetto) yet. The pressed/breathy axes
  are implemented and drive the resonance relief, but the labels need calibration on
  labelled real recordings before they can be shown as fact.
- Transposition is hard-limited to ±5 semitones, and the report says so — beyond that,
  PSOLA and WORLD both become audibly artificial.
- If the output f0 differs from the input by more than 600 cents (median), the pipeline
  **refuses** to return the corrected pitch and falls back to "no pitch change". A
  result that is a whole octave off is worse than no result at all.

## License

MIT. See [LICENSE](LICENSE).
