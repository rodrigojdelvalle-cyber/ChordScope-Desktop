# ChordScope Desktop — development state

Last updated: 2026-08-27
Target hotfix: **v2.0.1**
Repository: `rodrigojdelvalle-cyber/ChordScope-Desktop`

## Product goal

Portable Windows 10/11 x64 desktop application, delivered as a single `ChordScope.exe`, preserving the existing ChordScope HTML/CSS/JS interface while using a Python MIR/audio backend. The target is offline use with no Python installation on the destination PC.

## Architecture

- Frontend: existing ChordScope HTML/CSS/JavaScript inside PySide6 + QtWebEngine.
- Audio decode / native fallback DSP: SoundFile + optional PyAV + NumPy/SciPy.
- Rhythm: Beat This! primary; native spectral-flux/autocorrelation fallback.
- Source separation: Demucs HTDemucs-FT using `demucs_infer.api.Separator` with bundled offline model repository.
- Chord engines: LV-Chordia, BTC Transformer, plus ChordScope native STFT/chroma template engine.
- Harmonic interpretation: ChordScope consensus/Viterbi + key/function/context analysis.
- Bass: independent walking-bass lane from the bass stem. Walking bass must never create automatic slash chords.
- Packaging: Nuitka onefile for Windows x64.

## Musical behavior rules

1. Prefer `N.C.` to a fabricated chord when harmonic evidence is weak/percussive.
2. Walking bass is independent from chord inversion/slash notation.
3. Acoustic chord candidates and functional/contextual interpretation are separate layers.
4. Reliability is more important than filling every beat with a chord label.
5. Guitar diagrams must be playable/verified; no impossible generated fingering merely to fill a slot.

## Reference regression: All of Me

Reference supplied by user: Key C, 140 BPM, 4/4 Medium Swing.
Expected count-in behavior: first two bars are `N.C.`; first real harmony begins with C6 on bar 3 beat 1. Typical reference progression starts around `C6/Cmaj7 -> E7 -> A7 -> Dm7 -> E7 -> Am7 -> D7 -> Dm7-G7 ...`.
Do not claim exact-track validation unless the actual audio file is processed.

## v2.0 build result

GitHub Actions run #26 compiled successfully and produced a ~793.6 MB `ChordScope.exe`. The GUI opened correctly on the user's Windows PC, but selecting an audio file raised:

`partially initialized module 'librosa.core.audio' has no attribute 'load' (most likely due to a circular import)`

The old workflow only verified that the EXE existed and had a plausible size/hash; it did not execute the packaged EXE.

## Root cause / packaging finding

Librosa 0.11 still defines `load`; this was not an API removal. Librosa uses lazy imports and its `core.audio` module imports Numba JIT/stencil/guvectorize. Nuitka documents incomplete standalone support around Numba, and an open Nuitka issue specifically tracks Librosa/Numba standalone failures. The failure therefore occurs at the compiled runtime/import boundary, not in the ChordScope frontend.

## v2.0.1 fix

ChordScope's critical path no longer uses `librosa.load`:

- `analysis/dsp.py`: native decoder, resampling, STFT chroma, HPSS, tempo fallback, bass pitch-class detector.
- `analysis/audio.py`: `probe_audio` / `load_mono` use native decoder.
- `analysis/beat_tracker.py`: Beat This! primary + native fallback.
- `analysis/template_engine.py`, `key.py`, `bass.py`: native DSP rather than Librosa loading.
- `analysis/separation.py`: Demucs decode uses ChordScope native loader; HPSS fallback is native.
- `analysis/chord_btc.py`: feeds a NumPy waveform to BTC instead of a filename.
- `analysis/librosa_compat.py`: external Librosa CQT users receive a packaging-safe `librosa.core.audio` shim backed by ChordScope/SciPy before CQT imports.
- `analysis/chord_lv.py`: applies compatibility shim before LV-Chordia import/inference.
- `runtime_smoke.py`: generates a WAV and tests the exact packaged runtime boundary including native load/STFT/HPSS/tempo/bass and an external Librosa hybrid-CQT call.
- `main.py --runtime-smoke-test`: lets GitHub Actions test the actual onefile EXE without launching the GUI.

Local source validation before commit: **5 pytest tests passed** and runtime smoke printed `CHORDSCOPE_RUNTIME_SMOKE_OK`, including `librosa_external_cqt: ok` (PyAV is additionally verified on the Windows build runner where it is installed).

## Build acceptance criteria for v2.0.1

Do not call the new EXE ready unless all are true:

1. GitHub workflow status `completed`, conclusion `success`.
2. Python compile/unit tests pass.
3. `prepare_models.py` completes and offline models are present.
4. Nuitka creates `dist/ChordScope.exe`.
5. `Verify executable` succeeds.
6. **The compiled `dist/ChordScope.exe --runtime-smoke-test` itself exits 0 and emits/writes `CHORDSCOPE_RUNTIME_SMOKE_OK`.**
7. The compiled smoke also verifies the external Librosa hybrid-CQT boundary and packaged PyAV import.
8. GitHub uploads the Windows artifact successfully.
9. User then tests a real audio file on Windows; that is the final real-world acceptance test.

## How to resume in another ChatGPT chat

Tell ChatGPT:

> Continuemos ChordScope Desktop desde el repositorio `rodrigojdelvalle-cyber/ChordScope-Desktop`. Leé primero `DEVELOPMENT_STATE.md` y revisá el último GitHub Actions build. Seguí desde ahí sin rediseñar la interfaz ni cambiar las reglas musicales del proyecto.

This repository file is the source of truth for cross-chat handoff.
