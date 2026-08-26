from __future__ import annotations

import json

import pytest

import memforge.capability_discovery as discovery
from memforge.api_target import Edition, TargetConfigurationError, capability_document


class _Response:
    def __init__(self, body: bytes, *, content_type: str = "application/json") -> None:
        self.body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


def test_discovery_fetches_exact_origin_well_known_document(monkeypatch):
    captured: dict[str, object] = {}
    body = json.dumps(capability_document(Edition.CLOUD)).encode("utf-8")

    class _Opener:
        def open(self, request, timeout):
            captured.update(url=request.full_url, accept=request.get_header("Accept"), timeout=timeout)
            return _Response(body)

    monkeypatch.setattr(discovery, "build_opener", lambda *_handlers: _Opener())

    target = discovery.discover_target("https://memory.example.com/")

    assert target.edition is Edition.CLOUD
    assert captured == {
        "url": "https://memory.example.com/.well-known/memforge",
        "accept": "application/json",
        "timeout": 10.0,
    }


def test_discovery_fails_closed_on_wrong_media_type_or_oversized_body(monkeypatch):
    class _Opener:
        response = _Response(b"{}", content_type="text/html")

        def open(self, request, timeout):
            return self.response

    opener = _Opener()
    monkeypatch.setattr(discovery, "build_opener", lambda *_handlers: opener)

    with pytest.raises(TargetConfigurationError, match="memforge_capability_content_type_invalid"):
        discovery.discover_target("https://memory.example.com")

    opener.response = _Response(b"x" * (discovery.MAX_CAPABILITY_BYTES + 1))
    with pytest.raises(TargetConfigurationError, match="memforge_capability_too_large"):
        discovery.discover_target("https://memory.example.com")
