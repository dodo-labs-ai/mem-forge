"""A local model endpoint that takes a request and does not finish answering it."""

from __future__ import annotations

import socketserver
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

# ``silent`` never answers. ``headers`` sends the status line and headers of an
# answer, then nothing more. ``trickle`` sends the headers, then one byte of the
# body at a time, often enough that no single read waits long.
StallMode = Literal["silent", "headers", "trickle"]
STALL_MODES: tuple[StallMode, ...] = ("silent", "headers", "trickle")

# The endpoint gives up and closes the connection after this long, so a
# request that is not ended at its timeout fails with another error instead of
# keeping a test waiting.
GIVES_UP_AFTER_S = 5.0
_TRICKLE_INTERVAL_S = 0.05


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, mode: StallMode) -> None:
        super().__init__(("127.0.0.1", 0), _StallingHandler)
        self.mode = mode
        self.closing = threading.Event()


class _StallingHandler(socketserver.BaseRequestHandler):
    server: _Server

    def handle(self) -> None:
        received = b""
        while b"\r\n\r\n" not in received:
            chunk = self.request.recv(65536)
            if not chunk:
                return
            received += chunk
        mode = self.server.mode
        gives_up_at = time.monotonic() + GIVES_UP_AFTER_S
        try:
            if mode == "headers":
                self.request.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 1000\r\n\r\n{"
                )
            elif mode == "trickle":
                self.request.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 1000000\r\n\r\n"
                )
                while time.monotonic() < gives_up_at and not self.server.closing.is_set():
                    self.request.sendall(b" ")
                    time.sleep(_TRICKLE_INTERVAL_S)
                return
            self.server.closing.wait(max(0.0, gives_up_at - time.monotonic()))
        except OSError:
            return


@contextmanager
def stalling_endpoint(mode: StallMode) -> Iterator[str]:
    """Serve ``mode`` on a local port for the length of the block; yield its address."""
    server = _Server(mode)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.closing.set()
        server.shutdown()
        server.server_close()
        thread.join()
