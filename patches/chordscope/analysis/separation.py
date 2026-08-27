from __future__ import annotations

import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from .resources import models_root

Progress = Callable[[int, str], None]


@dataclass(slots=True)
class SeparationResult:
    root: Path
    original: Path
    harmony: Path
    bass: Path
    drums: Optional[Path]
    vocals: Optional[Path]
    engine: str
    sample_rate: int


class SourceSeparator:
    """Separate the mixture into stems using demucs-infer's stable Separator API.

    The packaged build prefers a fully local HTDemucs-FT repository under
    ``vendor_models/demucs``. This keeps the final executable offline. If the
    local repository is missing or incomplete, demucs-infer may use its normal
    remote model resolver; if that also fails, ChordScope falls back to HPSS.

    The returned ``bass`` stem is never used as an automatic slash-chord
    source. It is only the independent walking-bass layer shown below the
    chord timeline.
    """

    _MODEL = "htdemucs_ft"
    _LOCAL_FILES = (
        "htdemucs_ft.yaml",
        "f7e0c4bc-ba3fe64a.th",
        "d12395a8-e57c48e6.th",
        "92cfc3b6-ef3bcb9c.th",
        "04573f0d-f3cf25b2.th",
    )

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._separator = None
        self._tempdirs: list[tempfile.TemporaryDirectory] = []

    def close(self) -> None:
        # demucs_infer.api.Separator has no close() lifecycle method. Releasing
        # our reference is sufficient; explicitly clear CUDA cache when used.
        self._separator = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        while self._tempdirs:
            td = self._tempdirs.pop()
            try:
                td.cleanup()
            except Exception:
                pass

    def _temp_root(self) -> Path:
        td = tempfile.TemporaryDirectory(prefix="chordscope_sep_")
        self._tempdirs.append(td)
        return Path(td.name)

    @classmethod
    def _local_repo_ready(cls, path: Path) -> bool:
        return path.is_dir() and all((path / name).is_file() for name in cls._LOCAL_FILES)

    def _build_separator(self, progress: Progress | None):
        from demucs_infer.api import Separator

        local_repo = models_root() / "demucs"
        kwargs = dict(
            model=self._MODEL,
            device=self.device,
            shifts=1,
            overlap=0.25,
            split=True,
            progress=False,
        )
        if self._local_repo_ready(local_repo):
            kwargs["repo"] = local_repo
            engine = "Demucs HTDemucs-FT · offline"
        else:
            # Development fallback. Production builds prepare and bundle the
            # local repository in prepare_models.py.
            engine = "Demucs HTDemucs-FT · runtime model"

        if progress:
            progress(6, "Cargando HTDemucs-FT")

        # Map Demucs chunk callbacks into ChordScope's separation progress.
        def callback(info: dict):
            if progress is None:
                return
            try:
                length = max(1, int(info.get("audio_length", 1)))
                offset = max(0, int(info.get("segment_offset", 0)))
                model_idx = max(0, int(info.get("model_idx_in_bag", 0)))
                models = max(1, int(info.get("models", 1)))
                frac = (model_idx + min(1.0, offset / length)) / models
                progress(18 + int(72 * frac), "Separando drums · bass · other · vocals")
            except Exception:
                pass

        kwargs["callback"] = callback
        self._separator = Separator(**kwargs)
        return self._separator, engine

    @staticmethod
    def _decode_for_separator(audio_path: Path):
        """Decode with libsndfile and return a [channels,time] float32 tensor.

        This avoids depending on a system FFmpeg executable inside the portable
        Windows build. Current soundfile wheels bundle libsndfile with WAV,
        FLAC and MPEG/MP3 support. Demucs then handles resampling and channel
        conversion in ``separate_tensor``.
        """
        import torch

        arr, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        # soundfile => [time, channels]; Demucs => [channels, time]
        wav = torch.from_numpy(np.ascontiguousarray(arr.T))
        return wav, int(sr)

    def separate(self, audio_path: str | Path, progress: Progress | None = None) -> SeparationResult:
        audio_path = Path(audio_path)
        root = self._temp_root()
        try:
            if progress:
                progress(3, "Cargando separación de fuentes")

            separator, engine = self._build_separator(progress)
            if progress:
                progress(18, "Separando drums · bass · other · vocals")

            # Prefer the self-contained soundfile decoder. If a format is not
            # supported, use demucs-infer's own loader as a secondary path.
            try:
                wav, input_sr = self._decode_for_separator(audio_path)
                _mixture, stems = separator.separate_tensor(wav, input_sr)
            except Exception:
                _mixture, stems = separator.separate_audio_file(audio_path)

            sr = int(separator.samplerate)
            written: dict[str, Path] = {}
            for name, tensor in stems.items():
                arr = tensor.detach().float().cpu().numpy()
                if arr.ndim == 2:
                    arr = arr.T
                p = root / f"{name}.wav"
                sf.write(p, arr, sr, subtype="FLOAT")
                written[name] = p

            harmony = written.get("other")
            bass = written.get("bass")
            if harmony is None or bass is None:
                raise RuntimeError("Demucs no devolvió stems other/bass")

            if progress:
                progress(100, "Separación de fuentes lista")
            return SeparationResult(
                root=root,
                original=audio_path,
                harmony=harmony,
                bass=bass,
                drums=written.get("drums"),
                vocals=written.get("vocals"),
                engine=engine,
                sample_rate=sr,
            )
        except Exception as exc:
            if progress:
                progress(26, f"Demucs no disponible · HPSS ({type(exc).__name__})")
            return self._fallback_hpss(audio_path, root, progress)

    def _fallback_hpss(self, audio_path: Path, root: Path, progress: Progress | None) -> SeparationResult:
        y, sr = librosa.load(str(audio_path), sr=44100, mono=True)
        if progress:
            progress(45, "Separación armónico/percusiva HPSS")
        harmonic, percussive = librosa.effects.hpss(y, margin=(1.0, 4.0))
        # Low-pass copy for walking bass visualization only.
        sos = butter(6, 260, btype="lowpass", fs=sr, output="sos")
        bass = sosfiltfilt(sos, y).astype(np.float32)
        hp = root / "harmony_hpss.wav"
        bp = root / "bass_lowpass.wav"
        dp = root / "drums_hpss.wav"
        sf.write(hp, harmonic, sr, subtype="FLOAT")
        sf.write(bp, bass, sr, subtype="FLOAT")
        sf.write(dp, percussive, sr, subtype="FLOAT")
        if progress:
            progress(100, "HPSS listo")
        return SeparationResult(
            root=root,
            original=audio_path,
            harmony=hp,
            bass=bp,
            drums=dp,
            vocals=None,
            engine="librosa HPSS fallback",
            sample_rate=sr,
        )
