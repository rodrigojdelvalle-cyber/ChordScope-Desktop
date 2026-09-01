from __future__ import annotations

import json
import traceback
from pathlib import Path
from threading import Event
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot, QUrl
from PySide6.QtWidgets import QFileDialog

from .analysis.audio import probe_audio
from .analysis.control import AnalysisCancelled
from .analysis.pipeline import PROFILES, analyze_file
from .analysis.resources import build_variant


class TaskWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(self, fn: Callable[[Callable[[int, str], None]], dict]) -> None:
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self) -> None:
        try:
            payload = self.fn(lambda p, m: self.progress.emit(int(p), str(m)))
            self.finished.emit(json.dumps(payload, ensure_ascii=False))
        except AnalysisCancelled as exc:
            self.cancelled.emit(json.dumps({"message": str(exc), "cancelled": True}, ensure_ascii=False))
        except Exception as exc:
            self.failed.emit(json.dumps({
                "message": str(exc),
                "type": type(exc).__name__,
                "traceback": traceback.format_exc(limit=12),
            }, ensure_ascii=False))


class DesktopBridge(QObject):
    loadProgress = Signal(int, str)
    loadReady = Signal(str)
    analysisProgress = Signal(int, str)
    analysisReady = Signal(str)
    analysisCancelled = Signal(str)
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._tasks: list[dict] = []
        self.current_audio: str | None = None
        self.analysis_profile = "balanced"

    @Slot(result=str)
    def chooseAudio(self) -> str:
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Cargar audio",
            "",
            "Audio (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.opus *.aiff *.aif);;Todos los archivos (*.*)",
        )
        return path or ""

    def _launch(self, worker: TaskWorker, kind: str, cancel_event: Event | None = None) -> None:
        thread = QThread(self)
        entry = {"thread": thread, "worker": worker, "kind": kind, "cancel_event": cancel_event}
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        if kind == "load":
            worker.progress.connect(self.loadProgress)
            worker.finished.connect(self.loadReady)
        else:
            worker.progress.connect(self.analysisProgress)
            worker.finished.connect(self.analysisReady)
            worker.cancelled.connect(self.analysisCancelled)
        worker.failed.connect(self.error)

        # Standard Qt lifecycle. v2.0.1 could call wait() from the worker's own
        # completion path, which is unsafe and can hang.
        for signal in (worker.finished, worker.failed, worker.cancelled):
            signal.connect(thread.quit)
            signal.connect(worker.deleteLater)

        def cleanup():
            try:
                self._tasks.remove(entry)
            except ValueError:
                pass
            thread.deleteLater()

        thread.finished.connect(cleanup)
        self._tasks.append(entry)
        thread.start()

    @Slot(str)
    def setAnalysisProfile(self, profile: str) -> None:
        p = str(profile or "balanced").strip().lower()
        self.analysis_profile = p if p in PROFILES else "balanced"

    @Slot(result=bool)
    def cancelAnalysis(self) -> bool:
        cancelled = False
        for entry in list(self._tasks):
            if entry.get("kind") != "analysis":
                continue
            event = entry.get("cancel_event")
            if event is not None:
                event.set()
                cancelled = True
        if cancelled:
            self.analysisProgress.emit(0, "Cancelación solicitada · cerrando etapa activa")
        return cancelled

    @Slot(str)
    def loadAudio(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self.error.emit(json.dumps({"message": "El archivo no existe", "path": path}, ensure_ascii=False))
            return
        self.current_audio = str(p)

        def task(progress):
            meta = probe_audio(p, progress=progress)
            meta.update({
                "path": str(p),
                "name": p.name,
                "file_url": QUrl.fromLocalFile(str(p)).toString(),
            })
            return meta

        self._launch(TaskWorker(task), "load")

    @Slot(str)
    def analyzeAudio(self, path: str = "") -> None:
        target = Path(path or self.current_audio or "")
        if not target.exists():
            self.error.emit(json.dumps({"message": "No hay un archivo de audio válido para analizar"}, ensure_ascii=False))
            return
        if any(entry.get("kind") == "analysis" for entry in self._tasks):
            self.error.emit(json.dumps({"message": "Ya hay un análisis en ejecución"}, ensure_ascii=False))
            return

        cancel_event = Event()
        profile = self.analysis_profile

        def task(progress):
            result = analyze_file(target, progress=progress, profile=profile, cancel_event=cancel_event)
            return result.to_dict()

        self._launch(TaskWorker(task), "analysis", cancel_event=cancel_event)

    @Slot(result=str)
    def systemInfo(self) -> str:
        info = {"desktop": True, "version": "2.0.2", "build_variant": build_variant()}
        try:
            import torch
            info.update({
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            })
        except Exception as exc:
            info["torch_error"] = str(exc)
        return json.dumps(info, ensure_ascii=False)
