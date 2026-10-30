# Changelog

All notable changes to this project are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
