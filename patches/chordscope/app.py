from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QMainWindow

from .bridge import DesktopBridge

AUDIO_EXTS={".mp3",".wav",".flac",".ogg",".m4a",".aac",".opus",".aiff",".aif"}


class AudioWebView(QWebEngineView):
    audioDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        urls=event.mimeData().urls()
        if urls:
            p=Path(urls[0].toLocalFile())
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                event.acceptProposedAction();return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls=event.mimeData().urls()
        if urls:
            p=Path(urls[0].toLocalFile())
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                self.audioDropped.emit(str(p))
                event.acceptProposedAction();return
        super().dropEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChordScope Desktop 2.0.2")
        self.resize(1540, 940)
        self.setMinimumSize(1100, 720)

        view = AudioWebView(self)
        self.setCentralWidget(view)
        self.view = view

        settings = view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)

        channel = QWebChannel(view.page())
        self.bridge = DesktopBridge(self)
        channel.registerObject("backend", self.bridge)
        view.page().setWebChannel(channel)
        view.audioDropped.connect(self.bridge.loadAudio)

        html = Path(__file__).resolve().parent / "frontend" / "index.html"
        view.setUrl(QUrl.fromLocalFile(str(html)))


def run() -> int:
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--autoplay-policy=no-user-gesture-required")
    app = QApplication(sys.argv)
    app.setApplicationName("ChordScope Desktop")
    app.setApplicationVersion("2.0.2")
    app.setOrganizationName("ChordScope")
    win = MainWindow()
    win.show()
    return app.exec()
