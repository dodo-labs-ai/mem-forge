"""Opt-in development diagnostics for failed LLM attempts and validation.

Production deployments must leave content capture disabled. Normal telemetry is
content-free; these artifacts intentionally contain authorized development data.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from functools import wraps
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class FailureTraceSink(Protocol):
    """Persist one complete JSON artifact; return its retrievable handle."""

    def write(self, trace_id: str, payload: bytes) -> str: ...


class NoOpFailureTraceSink:
    def write(self, trace_id: str, payload: bytes) -> str:
        return "disabled"


class LocalFailureTraceSink:
    def __init__(self, directory: Path):
        self.directory = directory

    def write(self, trace_id: str, payload: bytes) -> str:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = self.directory / f"{trace_id}.json"
        with NamedTemporaryFile(dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return str(target.resolve())


def failure_trace_enabled() -> bool:
    return os.environ.get("MEMFORGE_LLM_FAILURE_CAPTURE_ENABLED", "").lower() in {"1", "true", "yes"}


def local_failure_trace_sink_from_env() -> FailureTraceSink | None:
    if not failure_trace_enabled():
        return None
    directory = os.environ.get("MEMFORGE_LLM_FAILURE_TRACE_DIR", "~/.memforge/llm-failure-traces")
    return LocalFailureTraceSink(Path(directory).expanduser())


_LINEAGE: ContextVar[dict] = ContextVar("llm_failure_trace_lineage", default={})
_CAPTURE: ContextVar[FailureCapture | None] = ContextVar("llm_failure_capture", default=None)


@contextmanager
def failure_trace_context(**lineage):
    """Bind explicit caller identities, isolated across concurrent tasks."""
    token = _LINEAGE.set({**_LINEAGE.get(), **lineage})
    try:
        yield
    finally:
        _LINEAGE.reset(token)


def enrich_failure_trace_context(**lineage):
    """Add identities discovered inside an already scoped caller operation."""
    if _LINEAGE.get():
        _LINEAGE.set({**_LINEAGE.get(), **lineage})


def _json_value(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value.model_json_schema()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _best_effort(method):
    @wraps(method)
    def record(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            logger.warning("llm_failure_trace_capture_failed capture_id=%s", self.capture_id)
    return record


class FailureCapture:
    """Own request/response lifetime, failure persistence and recovery updates."""

    def __init__(self, sink, *, prompt, schema, operation):
        self.sink = sink
        self.capture_id = uuid4().hex
        self.record = dict(version=1, capture_id=self.capture_id,
            trace_id=_LINEAGE.get().get("trace_id") or self.capture_id,
            lineage=dict(_LINEAGE.get()), operation=operation, prompt=prompt,
            schema=schema.model_json_schema(), attempts=[], failures=[], outcome="pending",
            created_at=datetime.now(timezone.utc).isoformat())

    @_best_effort
    def begin_attempt(self, request):
        self.record["attempts"].append(dict(request=_json_value(request), response=None))

    @_best_effort
    def response(self, response):
        # Provider credentials and internal transport headers are not LLM output.
        body = _json_value(response)
        self.record["attempts"][-1]["response"] = {
            key: body[key] for key in ("choices", "usage", "id", "model", "created", "object") if key in body
        }

    @_best_effort
    def failed(self, error, *, stage, location=None, context=None):
        failure = dict(stage=stage, error_class=type(error).__name__,
            attempt=len(self.record["attempts"]),
            message=str(error) if stage in {"schema_validation", "business_validation"} else type(error).__name__)
        if isinstance(error, ValidationError):
            failure["validation_errors"] = error.errors(include_url=False, include_input=False)
        for key in ("location", "received", "allowed_refs"):
            if hasattr(error, key):
                failure[key] = getattr(error, key)
        if location is not None:
            failure["location"] = location
        if context:
            failure["context"] = context
        code = getattr(error, "error_code", None) or getattr(error, "code", None)
        if code is not None:
            failure["error_code"] = code
        self.record["failures"].append(_json_value(failure))
        self.record["outcome"] = "failed"

    async def persist(self):
        if not self.record["failures"]:
            return
        try:
            payload = json.dumps(self.record, ensure_ascii=False, separators=(",", ":")).encode()
            handle = await asyncio.to_thread(self.sink.write, self.capture_id, payload)
            logger.warning("llm_failure_trace trace_id=%s capture_id=%s artifact=%s outcome=%s",
                self.record["trace_id"], self.capture_id, handle, self.record["outcome"])
        except Exception:
            logger.warning("llm_failure_trace_persistence_failed capture_id=%s", self.capture_id)

    async def recovered(self):
        self.record["outcome"] = "recovered"
        await self.persist()


@asynccontextmanager
async def capture_call(sink, *, prompt, schema, operation):
    capture = None
    if sink is not None and not isinstance(sink, NoOpFailureTraceSink):
        try:
            capture = FailureCapture(sink, prompt=prompt, schema=schema, operation=operation)
        except Exception:
            logger.warning("llm_failure_trace_initialization_failed")
    token = _CAPTURE.set(capture)
    try:
        yield capture
    except BaseException as error:
        if capture is not None:
            capture.failed(error, stage="logical_call")
        raise
    else:
        if capture is not None and capture.record["failures"]:
            capture.record["outcome"] = "recovered"
    finally:
        _CAPTURE.reset(token)
        if capture is not None:
            await capture.persist()


def current_capture():
    return _CAPTURE.get()


async def record_validation_failure(response, error, *, location=None, persist=True, **context):
    """Capture a rejected proposal even when the caller handles the rejection."""
    capture = getattr(response, "_llm_failure_capture", None)
    if capture is not None:
        capture.failed(error, stage="business_validation", location=location, context=context)
        if persist:
            await capture.persist()
    return capture


@asynccontextmanager
async def validation_trace(response, **lineage):
    """Keep the raw provider reply available through business validation."""
    capture = getattr(response, "_llm_failure_capture", None)
    if capture is not None:
        capture.record["lineage"].update(lineage)
    try:
        yield capture
    except Exception as error:
        await record_validation_failure(response, error)
        raise
    finally:
        if capture is not None:
            object.__delattr__(response, "_llm_failure_capture")
