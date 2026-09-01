from __future__ import annotations

from threading import Event


class AnalysisCancelled(RuntimeError):
    """Raised when the user asks ChordScope to stop the active analysis."""


def is_cancelled(cancel_event: Event | None) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def check_cancel(cancel_event: Event | None) -> None:
    if is_cancelled(cancel_event):
        raise AnalysisCancelled("Análisis cancelado por el usuario")
