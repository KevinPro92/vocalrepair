# Contributing

Thanks for considering a contribution. This project is deliberately conservative
about claims, so the bar for a change is "it is measured, not asserted".

## Ground rules

1. **No fabricated numbers.** Any accuracy claim in code, docs or a PR must come
   from a reproducible measurement. If you cannot measure it, say `[unverified]`.
2. **No silent fallbacks.** If a feature degrades (missing model, unsupported
   file, failed alignment), it must be recorded in `report.json`
   (`degraded_mode`, `warnings`, `guards`) — not swallowed.
3. **Never ship a wrong-octave result.** Any change touching f0 handling must keep
   the three-layer defence intact: multi-candidate Viterbi → HPS cross-check →
   output-side octave guard with safe fallback.
4. **Unvoiced frames stay unvoiced.** Pitch correction must never invent a pitch
   where the input had none.
5. **No scoring.** The report gives measured values and reference ranges, never a
   single "quality score" — artistic quality is not a scalar.

## Development setup

```bash
git clone https://github.com/KevinPro92/vocalrepair.git
cd vocalrepair
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e ".[dev,vocoder]"                     # dev + WORLD vocoder
python -m unittest discover -s tests -t .           # 35 tests
python vocal_spectra.py --self-test                 # 9 end-to-end assertions
```

`torch` / `demucs` are **optional** and must never become hard dependencies —
CI installs only `[dev]` on purpose.

## Testing expectations

- Every bug fix gets a regression test that **fails before the fix**.
- Synthetic ground truth is preferred over real recordings for accuracy claims
  (real recordings have unknown true f0, so validating against them is circular).
- If you add a test signal, do not build it from a static spectral envelope:
  a 440 Hz "vowel" can easily have its 2nd harmonic 27 dB above the fundamental,
  which is a genuine *missing fundamental* and will (correctly) read as 880 Hz.

## Pull request checklist

- [ ] `python -m unittest discover -s tests -t .` passes
- [ ] `python vocal_spectra.py --self-test` still reports 9/9
- [ ] New numeric constants carry a comment explaining *why that value*
- [ ] New limitations are written into the docstring, not hidden
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Reporting bugs

Please include:

- The command you ran (with flags)
- `report.json` if available (it contains the guards and warnings)
- Whether the input was synthetic or a real recording
- For pitch issues: expected vs measured f0, and how you measured it

## Commit messages

```
<area>: <what changed>

Why: <the measurement or reasoning that motivated it>
```

Example:

```
f0_track: add period penalty to Viterbi emission cost

Why: raw CMND decreases with tau, so a 220 Hz tone was decoded as 110 Hz.
```
