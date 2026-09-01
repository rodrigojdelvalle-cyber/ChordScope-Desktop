from threading import Event

import pytest

from chordscope.analysis.control import AnalysisCancelled, check_cancel


def test_cancel_token_is_cooperative():
    e = Event()
    check_cancel(e)
    e.set()
    with pytest.raises(AnalysisCancelled):
        check_cancel(e)
