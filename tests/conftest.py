"""Temporary diagnostics for the Linux CI teardown hang."""

from __future__ import annotations

import os
import threading
import time


_watchdog_stop = threading.Event()
_current_test: tuple[str, float] | None = None
_reported_test: str | None = None


def _watch_stalled_test() -> None:
    global _reported_test
    while not _watchdog_stop.wait(1.0):
        current = _current_test
        if current is None:
            continue
        node_id, started_at = current
        if time.monotonic() - started_at < 15.0 or _reported_test == node_id:
            continue
        _reported_test = node_id
        os.write(2, f"[DEBUG-ci-hang] stalled test: {node_id}\n".encode())


def pytest_sessionstart(session) -> None:
    del session
    threading.Thread(target=_watch_stalled_test, daemon=True).start()


def pytest_runtest_logstart(nodeid: str, location) -> None:
    global _current_test, _reported_test
    del location
    _reported_test = None
    _current_test = (nodeid, time.monotonic())


def pytest_runtest_logfinish(nodeid: str, location) -> None:
    global _current_test
    del nodeid, location
    _current_test = None


def pytest_sessionfinish(session, exitstatus) -> None:
    del session, exitstatus
    _watchdog_stop.set()
