from __future__ import annotations

import json

from memforge.pipeline.sync_memory import MemorySample, ProcessMemorySampler, SyncMemoryObserver


class SequenceSampler:
    def __init__(self, samples):
        self.samples = list(samples)

    def sample(self):
        sample = self.samples.pop(0)
        if isinstance(sample, Exception):
            raise sample
        return sample


class RecordingLogger:
    def __init__(self):
        self.records: list[tuple[str, dict]] = []

    def info(self, message):
        self.records.append(("info", json.loads(message)))

    def debug(self, message):
        self.records.append(("debug", json.loads(message)))


def test_process_memory_sampler_parses_proc_status_rss_and_hwm():
    sample = ProcessMemorySampler._sample_from_proc_status(
        "Name:\tpython\n"
        "VmRSS:\t2048 kB\n"
        "VmHWM:\t4096 kB\n"
    )

    assert sample == MemorySample(rss_mb=2.0, peak_rss_mb=4.0)


def test_sync_memory_observer_emits_compact_json_with_deltas_and_sequence():
    logger = RecordingLogger()
    observer = SyncMemoryObserver(
        sampler=SequenceSampler(
            [
                MemorySample(rss_mb=100.0, peak_rss_mb=120.0),
                MemorySample(rss_mb=180.0, peak_rss_mb=190.0),
            ]
        ),
        logger=logger,
        info_delta_threshold_mb=64.0,
    )

    observer.sample(
        "sync_run_start",
        source_id="src-a",
        run_id="run-a",
        item_count=2,
    )
    observer.sample(
        "after_extract",
        source_id="src-a",
        run_id="run-a",
        doc_id="DOC-1",
        raw_memory_count=3,
        content_chars=2048,
    )

    assert [level for level, _event in logger.records] == ["info", "info"]
    first = logger.records[0][1]
    second = logger.records[1][1]
    assert first["event"] == "sync_memory"
    assert first["stage"] == "sync_run_start"
    assert first["sample_seq"] == 1
    assert first["rss_mb"] == 100.0
    assert first["rss_delta_mb"] is None
    assert first["item_count"] == 2
    assert second["stage"] == "after_extract"
    assert second["sample_seq"] == 2
    assert second["rss_mb"] == 180.0
    assert second["rss_delta_mb"] == 80.0
    assert second["doc_rss_delta_mb"] is None
    assert second["raw_memory_count"] == 3
    assert "title" not in second
    assert "source_url" not in second


def test_sync_memory_observer_swallow_sampler_errors_and_still_logs_event():
    logger = RecordingLogger()
    observer = SyncMemoryObserver(
        sampler=SequenceSampler([RuntimeError("proc read failed")]),
        logger=logger,
    )

    observer.sample("after_fetch", source_id="src-a", run_id="run-a", doc_id="DOC-1")

    assert len(logger.records) == 1
    level, event = logger.records[0]
    assert level == "debug"
    assert event["event"] == "sync_memory"
    assert event["stage"] == "after_fetch"
    assert event["source_id"] == "src-a"
    assert event["run_id"] == "run-a"
    assert event["doc_id"] == "DOC-1"
    assert event["sample_seq"] == 1
    assert event["ok"] is True
    assert event["rss_mb"] is None
    assert event["rss_delta_mb"] is None
    assert event["doc_rss_delta_mb"] is None
    assert event["peak_rss_mb"] is None


def test_sync_memory_observer_adds_run_summary_on_sync_run_end():
    logger = RecordingLogger()
    observer = SyncMemoryObserver(
        sampler=SequenceSampler(
            [
                MemorySample(rss_mb=100.0, peak_rss_mb=120.0),
                MemorySample(rss_mb=140.0, peak_rss_mb=160.0),
                MemorySample(rss_mb=130.0, peak_rss_mb=170.0),
            ]
        ),
        logger=logger,
    )

    observer.sample("sync_run_start", source_id="src-a", run_id="run-a")
    observer.sample("document_lifecycle_enter", source_id="src-a", run_id="run-a", doc_id="DOC-1")
    observer.sample("sync_run_end", source_id="src-a", run_id="run-a", status="success")

    summary = logger.records[-1][1]
    assert summary["stage"] == "sync_run_end"
    assert summary["run_max_rss_mb"] == 140.0
    assert summary["run_peak_rss_mb"] == 170.0
    assert summary["status"] == "success"
