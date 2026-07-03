"""Lightweight process-memory observation for source sync.

This module intentionally emits structured log lines instead of binding to the
cloud metrics seam. RSS is process-wide and high-cardinality dimensions such as
doc_id are useful in logs but inappropriate as metric labels.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger("memforge.pipeline.sync.memory")


@dataclass(frozen=True)
class MemorySample:
    rss_mb: float | None
    peak_rss_mb: float | None


class ProcessMemorySampler:
    """Read current and peak process RSS without adding runtime dependencies."""

    def __init__(self, proc_status_path: str = "/proc/self/status") -> None:
        self._proc_status_path = proc_status_path
        self._proc_status_supported = os.path.exists(proc_status_path)

    def sample(self) -> MemorySample:
        if self._proc_status_supported:
            try:
                with open(self._proc_status_path, encoding="utf-8") as handle:
                    sample = self._sample_from_proc_status(handle.read())
                if sample.rss_mb is not None or sample.peak_rss_mb is not None:
                    return sample
                self._proc_status_supported = False
            except OSError:
                self._proc_status_supported = False

        return self._sample_from_getrusage()

    @staticmethod
    def _sample_from_proc_status(text: str) -> MemorySample:
        rss_kb: int | None = None
        peak_kb: int | None = None
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = ProcessMemorySampler._parse_status_kb(line)
            elif line.startswith("VmHWM:"):
                peak_kb = ProcessMemorySampler._parse_status_kb(line)
        return MemorySample(
            rss_mb=round(rss_kb / 1024, 3) if rss_kb is not None else None,
            peak_rss_mb=round(peak_kb / 1024, 3) if peak_kb is not None else None,
        )

    @staticmethod
    def _parse_status_kb(line: str) -> int | None:
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    @staticmethod
    def _sample_from_getrusage() -> MemorySample:
        try:
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            return MemorySample(rss_mb=None, peak_rss_mb=None)

        # Linux reports kilobytes; macOS reports bytes.
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return MemorySample(rss_mb=None, peak_rss_mb=round(maxrss / divisor, 3))


class SyncMemoryObserver:
    """Emit compact JSON RSS samples for source sync stages."""

    _INFO_STAGES = {
        "sync_run_start",
        "after_discovery",
        "document_lifecycle_enter",
        "document_lifecycle_exit",
        "sync_run_end",
    }
    _FORBIDDEN_FIELDS = {
        "title",
        "source_url",
        "content",
        "document",
        "prompt",
        "excerpt",
        "source_config",
        "token",
    }

    def __init__(
        self,
        *,
        sampler: Any | None = None,
        logger: Any | None = None,
        info_delta_threshold_mb: float = 64.0,
    ) -> None:
        self._sampler = sampler or ProcessMemorySampler()
        self._logger = logger or globals()["logger"]
        self._info_delta_threshold_mb = float(info_delta_threshold_mb)
        self._lock = threading.Lock()
        self._sample_seq = 0
        self._started_at = time.perf_counter()
        self._last_rss_mb: float | None = None
        self._run_max_rss_mb: float | None = None
        self._run_peak_rss_mb: float | None = None
        self._run_max_active_seen = 0
        self._doc_start_rss: dict[str, float] = {}

    def sample(
        self,
        stage: str,
        *,
        source_id: str,
        run_id: str,
        doc_id: str | None = None,
        ok: bool = True,
        error_class: str | None = None,
        active_lifecycles: int | None = None,
        max_active_seen: int | None = None,
        level: str | None = None,
        **fields: Any,
    ) -> None:
        try:
            event, chosen_level = self._build_event(
                stage,
                source_id=source_id,
                run_id=run_id,
                doc_id=doc_id,
                ok=ok,
                error_class=error_class,
                active_lifecycles=active_lifecycles,
                max_active_seen=max_active_seen,
                level=level,
                fields=fields,
            )
            payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
            log = getattr(self._logger, chosen_level, None) or getattr(self._logger, "info")
            log(payload)
        except Exception:
            return

    def _build_event(
        self,
        stage: str,
        *,
        source_id: str,
        run_id: str,
        doc_id: str | None,
        ok: bool,
        error_class: str | None,
        active_lifecycles: int | None,
        max_active_seen: int | None,
        level: str | None,
        fields: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        sample = self._read_sample()
        with self._lock:
            self._sample_seq += 1
            sample_seq = self._sample_seq
            rss_delta = self._delta(sample.rss_mb, self._last_rss_mb)
            if sample.rss_mb is not None:
                self._last_rss_mb = sample.rss_mb
                self._run_max_rss_mb = self._max(self._run_max_rss_mb, sample.rss_mb)
            if sample.peak_rss_mb is not None:
                self._run_peak_rss_mb = self._max(self._run_peak_rss_mb, sample.peak_rss_mb)

            doc_delta = None
            if doc_id:
                if stage == "document_lifecycle_enter" and sample.rss_mb is not None:
                    self._doc_start_rss[doc_id] = sample.rss_mb
                doc_delta = self._delta(sample.rss_mb, self._doc_start_rss.get(doc_id))
                if stage == "document_lifecycle_exit":
                    self._doc_start_rss.pop(doc_id, None)

            active = max(0, active_lifecycles) if active_lifecycles is not None else None
            max_seen = max(0, max_active_seen) if max_active_seen is not None else None
            if max_seen is not None:
                self._run_max_active_seen = max(self._run_max_active_seen, max_seen)

            event: dict[str, Any] = {
                "event": "sync_memory",
                "stage": stage,
                "source_id": source_id,
                "run_id": run_id,
                "sample_seq": sample_seq,
                "ok": ok,
                "rss_mb": sample.rss_mb,
                "rss_delta_mb": rss_delta,
                "doc_rss_delta_mb": doc_delta,
                "peak_rss_mb": sample.peak_rss_mb,
                "elapsed_ms_from_run_start": int((time.perf_counter() - self._started_at) * 1000),
            }
            if doc_id is not None:
                event["doc_id"] = doc_id
            if error_class is not None:
                event["error_class"] = error_class
            if active is not None:
                event["active_lifecycles"] = active
            if max_seen is not None:
                event["max_active_seen"] = max_seen
            for key, value in fields.items():
                if key in self._FORBIDDEN_FIELDS:
                    continue
                event[key] = value
            if stage == "sync_run_end":
                event["run_max_rss_mb"] = self._run_max_rss_mb
                event["run_peak_rss_mb"] = self._run_peak_rss_mb
                event["run_max_active_seen"] = self._run_max_active_seen

            chosen_level = self._choose_level(stage, rss_delta, level)
            return event, chosen_level

    def _read_sample(self) -> MemorySample:
        try:
            sample = self._sampler.sample()
        except Exception:
            return MemorySample(rss_mb=None, peak_rss_mb=None)
        if not isinstance(sample, MemorySample):
            return MemorySample(rss_mb=None, peak_rss_mb=None)
        return sample

    def _choose_level(self, stage: str, rss_delta: float | None, level: str | None) -> str:
        if level in {"info", "debug"}:
            return level
        if stage in self._INFO_STAGES:
            return "info"
        if rss_delta is not None and abs(rss_delta) >= self._info_delta_threshold_mb:
            return "info"
        return "debug"

    @staticmethod
    def _delta(current: float | None, previous: float | None) -> float | None:
        if current is None or previous is None:
            return None
        return round(current - previous, 3)

    @staticmethod
    def _max(current: float | None, candidate: float) -> float:
        return candidate if current is None else max(current, candidate)
