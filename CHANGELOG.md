# Changelog

All notable changes to this project are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] 2026-10-31

First version validated on **real recordings** (4 solo takes + 1 KTV track),
not just synthetic signals.

### Added
- **PSOLA engine** (`--vocoder psola`) with Praat-style pitch marks:
  zero-phase low-pass to 1.5·f0 → positive peak picking → single-sided period
  check (reject only gaps < 0.45 T) → equal-spacing fill.
- **Glide protection**: f0 slew > 2000 cents/s is never corrected (natural
  vibrato peaks at ~1257 cents/s, so it is untouched). The decision is per note
  (`glide_frac > 0.30` → skip the whole note), which eliminated the 80 ms
  processing fragments responsible for audible graininess
  (54 fragments @ 0.11 s → 16 @ 0.64 s).
- **Transient protection** (half-wave rectified spectral flux, MAD threshold
  k = 1.6): onsets, plosive consonants and voice breaks stay bit-identical.
- **Pressed-voice detection** (5-metric vote: H1−H2, H1−A3, spectral slope,
  harmonic centroid, HNR) plus `apply_resonance_relief` (−1.5 dB high shelf at
  2 kHz, applied only to ≥4/5-vote segments).
- **Denoise** (`--denoise specsub|wiener|highcut`), **off by default**.
- `--min-dev` — only correct notes whose deviation exceeds N cents.
- `--no-compress`, `--no-voice-relief`.

### Fixed
- **PSOLA dropouts**: the synthesis step advanced by `sr / target_f0`, so
  unvoiced frames (target f0 = 0) jumped one full second per window and
  consonants disappeared entirely.
- **Pitch marks vanishing after 15 s** on a real KTV vocal: the period gate was
  two-sided, and its "too sparse" branch discarded the very peaks the fill logic
  existed to handle (3573 → 9522 marks).
- **Spectral-subtraction / Wiener peak overshoot**: modifying gain while keeping
  phase breaks magnitude/phase consistency — measured peak 0.284 → **0.891**.
  Gain is now smoothed along frequency *and* time, and the output peak is
  limited relative to the input.
- `np.hanning(3) == [0, 1, 0]` — the Wiener temporal smoothing was a no-op.
- Wiener `min_gain=0.15` reduced overall RMS to 0.485; now 0.30 → **0.931**.
- Loudness pumping caused by PSOLA overlap variation → short-time RMS envelope
  matching against the input.
- Key estimation on short monophonic takes: chroma is now driven by the f0
  track instead of the spectrum (harmonic energy was being assigned to the wrong
  pitch classes).

### Changed
- Default stays `--vocoder world` for API compatibility, but **`psola` is
  recommended for solo vocals**: measured HNR *improves* (+0.17…+1.15 dB) and
  pitch accuracy is better, whereas WORLD's parametric resynthesis adds audible
  "watery" noise on breathy material (ap ≈ 0.71).
- Rolled back the 150 ms block-adaptive low-pass in `find_pitch_marks`: it fixed
  high notes but destroyed pitch-mark continuity, which garbled every consonant
  ("beep" artifacts). Recorded here so it is not attempted again.

### Real-recording results (4 solo takes, 13 s each, median |cents|)

| Take | `none` | `psola` | `world` |
|---|---|---|---|
| F major, flat | 21.71 | **19.00** | 19.11 |
| F major, sharp | 23.67 | **19.19** | 20.09 |
| G major, in tune | 12.13 | 13.86 | **12.71** |
| F major, strained high note | 24.46 | **20.91** | 22.35 |

### KTV chain (29 s, Demucs separation, CPU ≈ 24 s)

24.75 → **20.24 cents**, LTAS correlation **0.997**, output/input RMS 0.956,
transposition residual +5.1 cents (octave guard not triggered).

## [Unreleased]

### Planned
- Voice-mode classification (chest / mix / head / falsetto) with per-segment user override
- Validation against real recordings (currently all metrics come from synthetic signals)
- Microphone realtime tuner (v0.5)

## [0.4.2] — 2026-10-31

### Changed
- **Note-level DTW alignment** replaces frame-level alignment for `--reference`.
  DTW residual drops from 9.0 cents to **0.0 semitones**, note matching 5/5.
- F0 search range tightened to **65–1100 Hz** (C2–C6). C1 was rejected: nothing
  but rumble lives there, and it only invites period-doubling errors.
- Pitch-adaptive dead zone: 12 cents below E3, 16 in the middle, 22 above C5.

### Fixed
- `np.min` returns NaN for a whole row if any element is NaN, which silently
  marked every frame as unvoiced (a 220 Hz tone read as "all silence").
  Filling with `inf` instead of `nan` fixes it.
- `f0_track.py` was destroyed by a PowerShell `Get-Content | Set-Content` round
  trip (GBK decoding of UTF-8); rebuilt and now validated with `ast.parse`.

## [0.4.1] — 2026-10-31

### Added
- `dsp/f0_track.py`: multi-candidate YIN + **Viterbi** global path, with a
  period penalty (raw CMND decreases with τ and biases toward double periods)
  and an aperiodicity gate.
- `hps_f0()`: harmonic product spectrum as an independent second opinion.
- **Output-side octave guard**: if the input/output median f0 differ by more than
  600 cents (net of the requested transposition), the run falls back to
  "do not touch pitch" and records a warning. Never ship a wrong-octave result.
- `repair/reference.py`: Sakoe-Chiba banded DTW for reference alignment.
- `_timbre_guard()`: LTAS correlation between input and output ("your own timbre
  is the reference"), plus `match_ltas()` to pull the output back.
- Silence gate and `analyze(forced_voiced=...)` physical constraint.

## [0.3.0] — 2026-10-31

### Added
- `repair/`: de-clipping, note-level pitch correction, WORLD formant-preserving
  transposition (±5 semitone limit), envelope EQ / compression / soft limiting.
- `report.py`: `report.json`, four-panel comparison figure, self-contained HTML.
- `vocal_spectra.py`: `analyze` / `repair` / `separate` subcommands and
  `--self-test` with 9 assertions.

### Verified
- Clipped samples 14126 → 0; median pitch deviation 33.6 → 14.1 cents;
  formant shift at +2 semitones **0.7 %** (limit 10 %).

## [0.2.0] — 2026-10-31

### Added
- `analysis/`: spectral peaks/valleys, flatness/contrast/slope, LTAS, vibrato
  metrics (rate/depth/extent/regularity), HNR / CPP / jitter / shimmer / H1−H2,
  LPC formant tracking, note-event table, key estimation.
- `dsp/stft.py`: STFT/ISTFT, parabolic peak picking, log-frequency spectrogram,
  chroma.
- `dsp/octave.py`: octave-error suspects.

## [0.1.0] — 2026-10-31

### Added
- A4 = 440 Hz pitch scale, cents/MIDI/note-name conversion, scale-aware snapping.
- YIN f0 tracking (sub-cent precision) and a vibrato-preserving soft corrector
  (16-cent dead zone, 8-cent residual target).
- `synth.py`: synthetic signals with per-sample ground truth; all-pole
  source–filter vowel synthesizer.
- 27 unit tests.
